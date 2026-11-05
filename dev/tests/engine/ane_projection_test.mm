// The ANE projection split (ops/AneProjection): its rows against the GPU's
// whole projection at a mixer's shape, and its rejection of the projections it
// does not take. Runs at IQ3_S weights, 2048 rows, the 35B's QKV width.
//
//   ane-projection-split <metallib> [rows] [share] [hidden] [output] [rounds]

#include "GgufFormatReference.hpp"
#include "TestBuffers.hpp"
#include "AwakeClock.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Gguf.h"
#include "ops/AneProjection.hpp"
#include "ops/Linear.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace splash;
using namespace splash::test;
using namespace splash::metal;
using namespace splash::ops;
using gguf_reference::Fmt;
using gguf_reference::Packed;
using gguf_reference::fmtNamed;
using gguf_reference::makeNative;
using gguf_reference::repack;

constexpr uint32_t kLayers = 2;

struct Geometry final {
  uint32_t rows = 2048, hidden = 5120, output = 8192, rounds = 3;
  double share = 0.5;
};

std::mt19937 rng(11);

struct Tensor {
  Fmt format;
  uint32_t N, K;
  std::vector<uint8_t> native;
  QuantizedSegment segment;
};

MetalBuffer upload(MetalBackend &backend, const std::vector<uint8_t> &bytes) {
  MetalBuffer buffer = sharedBuffer(backend, bytes.size());
  std::memcpy(buffer.contents(), bytes.data(), bytes.size());
  return buffer;
}

Tensor tensor(MetalBackend &backend, Fmt format, uint32_t N, uint32_t K) {
  Tensor t{format, N, K, makeNative(format, N, K, rng), {}};
  const Packed planes = repack(format, t.native, N, K, nullptr);
  t.segment = QuantizedSegment::planes(format, N, K, upload(backend, planes.w0),
                                       kQuantFormats[format].plane1_bytes ? upload(backend, planes.w1) : MetalBuffer{},
                                       upload(backend, planes.meta));
  return t;
}

Projection projection(const Tensor &t) {
  BlockWeights blocks;
  blocks.segments.push_back(t.segment);
  return Projection(t.N, t.K, std::move(blocks));
}

void fillRows(const MetalBuffer &buffer, uint32_t rows, uint32_t hidden) {
  std::normal_distribution<float> normal;
  auto *values = static_cast<uint16_t *>(buffer.contents());
  for (uint64_t i = 0; i < uint64_t{rows} * hidden; ++i) values[i] = floatToBf16(normal(rng));
}

float bf16At(const MetalBuffer &buffer, uint64_t index) {
  return bf16ToFloat(static_cast<const uint16_t *>(buffer.contents())[index]);
}

uint32_t argument(int argc, char **argv, int index, uint32_t fallback) {
  return argc > index ? static_cast<uint32_t>(std::strtoul(argv[index], nullptr, 10)) : fallback;
}

// The RMS of the ANE's rows of `actual` against the GPU's whole projection.
double rms(const MetalBuffer &actual, const MetalBuffer &expected, uint32_t firstRow, uint32_t rows,
           uint32_t width) {
  double noise = 0.0, signal = 0.0;
  for (uint32_t row = firstRow; row < rows; ++row)
    for (uint32_t c = 0; c < width; ++c) {
      const uint64_t at = uint64_t{row} * width + c;
      const double value = double(bf16At(actual, at)), expectedValue = double(bf16At(expected, at));
      const double difference = value - expectedValue;
      noise += difference * difference;
      signal += expectedValue * expectedValue;
    }
  return std::sqrt(noise / std::max(signal, 1e-9));
}

template <class Run> double medianOf(uint32_t rounds, Run run) {
  run();
  std::vector<double> times;
  for (uint32_t round = 0; round < rounds; ++round) times.push_back(run());
  std::ranges::nth_element(times, times.begin() + times.size() / 2);
  return times[times.size() / 2];
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: ane-projection-split <metallib> [rows] [share] [hidden] [output] [rounds]\n");
    return 2;
  }
  Geometry geometry;
  geometry.rows = argument(argc, argv, 2, geometry.rows);
  if (argc > 3) geometry.share = std::strtod(argv[3], nullptr);
  geometry.hidden = argument(argc, argv, 4, geometry.hidden);
  geometry.output = argument(argc, argv, 5, geometry.output);
  geometry.rounds = argument(argc, argv, 6, geometry.rounds);
  try {
    MetalBackend backend(argv[1]);
    const Fmt format = fmtNamed("iq3s");
    std::vector<Tensor> tensors;
    for (uint32_t layer = 0; layer < kLayers; ++layer)
      tensors.push_back(tensor(backend, format, geometry.output, geometry.hidden));
    std::vector<Projection> projections;
    for (const Tensor &t : tensors) projections.push_back(projection(t));
    std::vector<const Projection *> layers;
    for (const Projection &p : projections) layers.push_back(&p);

    // What the split does not take: none. A fused projection is taken, its
    // segments staged into their offsets of the full-width surfaces.
    if (!AneProjection::unsupported(std::span<const Projection *const>{}))
      throw std::runtime_error("the split takes no layers");
    {
      // The GDN's split of 10240 and 6144 of 16384, and an even split for
      // the other widths.
      const uint32_t firstWidth = geometry.output == 16384 ? 10240 : geometry.output / 2;
      const uint32_t secondWidth = geometry.output - firstWidth;
      const Tensor first = tensor(backend, format, firstWidth, geometry.hidden);
      const Tensor second = tensor(backend, format, secondWidth, geometry.hidden);
      BlockWeights fused;
      fused.segments.push_back(first.segment);
      fused.segments.push_back(second.segment);
      fused.segments.back().columnOffset = firstWidth;
      const Projection whole(geometry.output, geometry.hidden, std::move(fused));
      const Projection *one = &whole;
      if (const char *reason = AneProjection::unsupported(std::span<const Projection *const>(&one, 1)))
        throw std::runtime_error(std::string("the split does not take a fused projection: ") + reason);
      // The fused projection's split: its segments go to their rows' offsets
      // of the ANE's full-width weight surfaces.
      const MetalBuffer input = sharedBuffer(backend, uint64_t{geometry.rows} * geometry.hidden * 2);
      const MetalBuffer reference = sharedBuffer(backend, uint64_t{geometry.rows} * geometry.output * 2);
      const MetalBuffer splitOut = sharedBuffer(backend, uint64_t{geometry.rows} * geometry.output * 2);
      fillRows(input, geometry.rows, geometry.hidden);
      Linear fusedLinear(backend.capabilities());
      {
        CommandGraph graph;
        fusedLinear.addPrefill(graph, input, whole, reference, geometry.rows);
        static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
      }
      AneProjection fusedSplit(backend, std::span<const Projection *const>(&one, 1), geometry.share, nullptr);
      {
        if (!fusedSplit.begin()) throw std::runtime_error("the fused split stopped: " + fusedSplit.reason());
        CommandGraph graph;
        const uint32_t gpuRows = fusedSplit.gpuRows(geometry.rows);
        fusedSplit.add(graph, 0, input, splitOut, geometry.rows,
                       [&] { fusedLinear.addPrefill(graph, input, whole, splitOut, gpuRows); });
        static_cast<void>(fusedSplit.commit(graph, {}).wait());
        if (!fusedSplit.finish()) throw std::runtime_error("the fused split stopped: " + fusedSplit.reason());
      }
      const double error = rms(splitOut, reference, fusedSplit.gpuRows(geometry.rows), geometry.rows, geometry.output);
      std::printf("fused %ux%u in %zu segments: the ANE's rows %.3f%% from the GPU's whole projection\n",
                  geometry.output, geometry.hidden, whole.blocks().segments.size(), error * 100.0);
      if (!(error < 0.05)) throw std::runtime_error("the fused split's ANE rows are not within 5%");
    }
    if (const char *reason = AneProjection::unsupported(layers)) {
      std::fprintf(stderr, "the split does not take this shape: %s\n", reason);
      return 1;
    }

    const MetalBuffer input = sharedBuffer(backend, uint64_t{geometry.rows} * geometry.hidden * 2);
    const MetalBuffer whole = sharedBuffer(backend, uint64_t{geometry.rows} * geometry.output * 2);
    const MetalBuffer splitOut = sharedBuffer(backend, uint64_t{geometry.rows} * geometry.output * 2);
    fillRows(input, geometry.rows, geometry.hidden);
    Linear linear(backend.capabilities());

    // The GPU alone over the whole projection, as the reference and the time
    // the split is measured against.
    const auto timeGpu = [&] {
      CommandGraph graph;
      for (uint32_t layer = 0; layer < kLayers; ++layer)
        linear.addPrefill(graph, input, projections[layer], whole, geometry.rows);
      const auto start = AwakeClock::now();
      static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
      return millisecondsSince(start);
    };
    const double gpuMilliseconds = medianOf(geometry.rounds, timeGpu);

    AneProjection split(backend, layers, geometry.share, nullptr);
    const uint32_t aneRows = split.aneRows(geometry.rows), gpuRows = split.gpuRows(geometry.rows);
    const auto gpuPart = [&](metal::CommandGraph &graph) {
      for (uint32_t layer = 0; layer < kLayers; ++layer)
        linear.addPrefill(graph, input, projections[layer], splitOut, gpuRows);
    };
    const auto timeSplit = [&] {
      if (!split.begin()) throw std::runtime_error("the split stopped: " + split.reason());
      CommandGraph graph;
      for (uint32_t layer = 0; layer < kLayers; ++layer)
        split.add(graph, layer, input, splitOut, geometry.rows, [&] { gpuPart(graph); });
      const auto start = AwakeClock::now();
      static_cast<void>(split.commit(graph, {}).wait());
      const double milliseconds = millisecondsSince(start);
      if (!split.finish()) throw std::runtime_error("the split stopped: " + split.reason());
      return milliseconds;
    };
    const double milliseconds = medianOf(geometry.rounds, timeSplit);

    // A residual's split, as an attention output projection takes it: the
    // ANE's rows are the layer's input plus the projection.
    {
      const MetalBuffer residual = sharedBuffer(backend, uint64_t{geometry.rows} * geometry.output * 2);
      const MetalBuffer reference = sharedBuffer(backend, uint64_t{geometry.rows} * geometry.output * 2);
      const MetalBuffer out = sharedBuffer(backend, uint64_t{geometry.rows} * geometry.output * 2);
      fillRows(residual, geometry.rows, geometry.output);
      {
        CommandGraph graph;
        linear.addPrefillResidual(graph, input, projections[0], residual, reference, geometry.rows, {});
        static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
      }
      AneProjection residualSplit(backend, std::span<const Projection *const>(&layers[0], 1), geometry.share,
                                  nullptr);
      if (!residualSplit.begin()) throw std::runtime_error("the residual split stopped: " + residualSplit.reason());
      CommandGraph graph;
      const uint32_t gpuRows = residualSplit.gpuRows(geometry.rows);
      residualSplit.add(graph, 0, input, out, geometry.rows,
                        [&] { linear.addPrefillResidual(graph, input, projections[0], residual, out, gpuRows, {}); },
                        residual);
      static_cast<void>(residualSplit.commit(graph, {}).wait());
      if (!residualSplit.finish()) throw std::runtime_error("the residual split stopped: " + residualSplit.reason());
      const double error = rms(out, reference, gpuRows, geometry.rows, geometry.output);
      std::printf("residual %ux%u: the ANE's rows %.3f%% from the GPU's whole projection\n", geometry.output,
                  geometry.hidden, error * 100.0);
      if (!(error < 0.05)) throw std::runtime_error("the residual split's ANE rows are not within 5%");
    }

    // The split's halves alone: the GPU's leading rows, and the ANE's
    // trailing ones with the rotate and output around them.
    const double gpuPartMilliseconds = medianOf(geometry.rounds, [&] {
      CommandGraph graph;
      gpuPart(graph);
      const auto start = AwakeClock::now();
      static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
      return millisecondsSince(start);
    });
    const double anePartMilliseconds = medianOf(geometry.rounds, [&] {
      if (!split.begin()) throw std::runtime_error("the split stopped: " + split.reason());
      CommandGraph graph;
      for (uint32_t layer = 0; layer < kLayers; ++layer) {
        split.beginLayer(graph, layer, input, geometry.rows);
        split.join(graph, splitOut, geometry.rows);
      }
      const auto start = AwakeClock::now();
      static_cast<void>(split.commit(graph, {}).wait());
      const double milliseconds = millisecondsSince(start);
      if (!split.finish()) throw std::runtime_error("the split stopped: " + split.reason());
      return milliseconds;
    });

    // The GPU's leading rows against the whole, and the ANE's trailing ones.
    const double leadingError = rms(splitOut, whole, 0, gpuRows, geometry.output);
    const double aneError = rms(splitOut, whole, gpuRows, geometry.rows, geometry.output);
    std::printf("%ux%u, %u rows (%u GPU, %u ANE), %u layers, %.2f share, median of %u\n", geometry.output,
                geometry.hidden, geometry.rows, gpuRows, aneRows, kLayers, geometry.share, geometry.rounds);
    std::printf("  GPU alone: %.2f ms; the split: %.2f ms (%.2fx)\n", gpuMilliseconds, milliseconds,
                gpuMilliseconds / milliseconds);
    std::printf("  parts alone: GPU leading %.2f ms, ANE trailing %.2f ms\n", gpuPartMilliseconds,
                anePartMilliseconds);
    std::printf("  GPU leading rows: %.3f%% from the whole projection\n", leadingError * 100.0);
    std::printf("  ANE trailing rows: %.3f%% from the whole projection\n", aneError * 100.0);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "ane-projection-split: %s\n", error.what());
    return 1;
  }
}
