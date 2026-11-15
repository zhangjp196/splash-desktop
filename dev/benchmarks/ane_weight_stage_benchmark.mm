// Time the one-time GGUF row-scale preparation separately from the repeated
// per-chunk conversion into ANE int8 weight surfaces. Compare the latter with
// ane-projection-benchmark for the same N/K/rows to see how much of the ANE
// path is weight staging versus evaluation.
//
//   ane-weight-stage-benchmark <metallib> <N> <K> [rows] [rounds] [format]

#include "../tests/engine/GgufFormatReference.hpp"
#include "../tests/engine/TestBuffers.hpp"
#include "ane/Program.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/AneFfn.h"
#include "ops/Linear.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace splash;
using namespace splash::metal;
using namespace splash::ops;
using gguf_reference::Fmt;
using gguf_reference::fmtNamed;
using gguf_reference::makeNative;
using gguf_reference::repack;

constexpr uint32_t kSegment = 2560;
constexpr uint32_t kSegmentStep = ANE_FFN_INTERMEDIATE_BLOCK;
constexpr uint32_t kWeightRows = ANE_FFN_WEIGHT_ROWS;
constexpr uint32_t kDefaultRows = 2048, kDefaultRounds = 11;

MetalBuffer upload(MetalBackend &backend, const std::vector<uint8_t> &bytes) {
  MetalBuffer buffer = backend.allocateBuffer(bytes.size(), BufferStorage::Shared, "ANE weight stage source");
  std::memcpy(buffer.contents(), bytes.data(), bytes.size());
  return buffer;
}

template <class Run> double medianOf(uint32_t rounds, Run run) {
  static_cast<void>(run()); // Warmup.
  std::vector<double> samples;
  samples.reserve(rounds);
  for (uint32_t i = 0; i < rounds; ++i) samples.push_back(run());
  std::ranges::nth_element(samples, samples.begin() + samples.size() / 2);
  return samples[samples.size() / 2];
}

uint32_t argument(int argc, char **argv, int index, uint32_t fallback) {
  return argc > index ? static_cast<uint32_t>(std::strtoul(argv[index], nullptr, 10)) : fallback;
}

uint32_t segmentFor(uint32_t hidden) {
  for (uint32_t width = kSegment; width >= kSegmentStep; width -= kSegmentStep)
    if (hidden % width == 0) return width;
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: ane-weight-stage-benchmark <metallib> <N> <K> [rows] [rounds] [format]\n");
    return 2;
  }
  const uint32_t columns = argument(argc, argv, 2, 0), hidden = argument(argc, argv, 3, 0);
  const uint32_t rows = argument(argc, argv, 4, kDefaultRows), rounds = argument(argc, argv, 5, kDefaultRounds);
  const uint32_t segmentSize = segmentFor(hidden);
  if (!columns || columns % kWeightRows || !hidden || !segmentSize || !rows || !rounds) {
    std::fprintf(stderr,
                 "N must be a positive multiple of %u, K must be divisible by an ANE segment (512 to 2560), "
                 "rows/rounds positive\n",
                 kWeightRows);
    return 2;
  }
  try {
    MetalBackend backend(argv[1]);
    const std::string formatName = argc > 6 ? argv[6] : "iq3s";
    const Fmt format = fmtNamed(formatName);
    std::mt19937 rng(7);
    const std::vector<uint8_t> native = makeNative(format, columns, hidden, rng);
    const gguf_reference::Packed packed = repack(format, native, columns, hidden, nullptr);
    const MetalBuffer plane0 = upload(backend, packed.w0);
    const MetalBuffer meta = upload(backend, packed.meta);
    // Match QuantizedSegment::plane1Slot(): formats without a second data
    // plane still bind a nonempty buffer in that argument slot; the kernel
    // ignores it according to `format`.
    const MetalBuffer plane1 = kQuantFormats[format].plane1_bytes ? upload(backend, packed.w1) : meta;

    MetalBuffer signs = backend.allocateBuffer(ANE_FFN_INTERMEDIATE_BLOCK * sizeof(float), BufferStorage::Shared,
                                                "ANE weight stage signs");
    std::uniform_int_distribution<int> bit(0, 1);
    auto *signValues = static_cast<float *>(signs.contents());
    for (uint32_t i = 0; i < ANE_FFN_INTERMEDIATE_BLOCK; ++i) signValues[i] = bit(rng) ? 1.0f : -1.0f;

    MetalBuffer rowScales = backend.allocateBuffer(uint64_t{columns} * sizeof(_Float16), BufferStorage::Shared,
                                                    "ANE weight stage row scales");
    const uint32_t groups = hidden / 32;
    const AneFfnWeightParams scaleParams{groups, 0, 0, hidden, 0, 0, static_cast<uint32_t>(format)};
    CommandGraph scaleGraph;
    scaleGraph.add("ane_ffn_row_scale_gguf_inputs", {plane0, plane1, meta, rowScales, signs},
                   scaleParams, {columns / kWeightRows, 1, 1}, {ANE_FFN_WEIGHT_THREADS, 1, 1});
    const double rowScaleMs = medianOf(rounds, [&] {
      return backend.submitCommandAsync(scaleGraph.dispatches()).wait().gpuSeconds * 1000.0;
    });

    std::vector<ane::Surface> staged;
    for (uint32_t segment = 0; segment < hidden / segmentSize; ++segment)
      staged.push_back(ane::Surface::create(backend, columns, segmentSize, ane::Surface::Element::Int8));
    ane::Surface stagedScale = ane::Surface::create(backend, columns, 1, ane::Surface::Element::Float16);
    CommandGraph stageGraph;
    for (uint32_t segment = 0; segment < hidden / segmentSize; ++segment) {
      const AneFfnWeightParams params{groups, 0, segment * segmentSize, segmentSize, staged[segment].strideBytes,
                                      stagedScale.strideBytes / uint32_t{sizeof(_Float16)},
                                      static_cast<uint32_t>(format)};
      stageGraph.add("ane_ffn_weights_gguf_inputs",
                     {plane0, plane1, meta, rowScales, staged[segment].buffer,
                      stagedScale.buffer, signs},
                     params, {columns / kWeightRows, segmentSize / ANE_FFN_INPUT_BLOCK, 1},
                     {ANE_FFN_WEIGHT_THREADS, 1, 1});
    }
    const double stageMs = medianOf(rounds, [&] {
      return backend.submitCommandAsync(stageGraph.dispatches()).wait().gpuSeconds * 1000.0;
    });

    const uint64_t stagedBytes = uint64_t{columns} * hidden + uint64_t{columns} * sizeof(_Float16);
    std::printf("GGUF %s %ux%u, %u rows, median of %u samples\n", formatName.c_str(), columns, hidden, rows,
                rounds);
    std::printf("  row scales (one-time per loaded projection): %.3f ms\n", rowScaleMs);
    std::printf("  weight staging (repeated per prefill chunk): %.3f ms, %.2f MiB written, %.2f GB/s\n", stageMs,
                double(stagedBytes) / (1024.0 * 1024.0), double(stagedBytes) / (stageMs * 1e6));
    std::printf("  Compare ANE evaluation: ane-projection-benchmark <metallib> %u %u %u %u\n", columns, hidden, rows,
                rounds);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "ane-weight-stage-benchmark: %s\n", error.what());
    return 1;
  }
}
