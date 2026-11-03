// The mixer's projection geometry (hidden 5120, intermediate 8192) through
// the split's own program: three layers of AneFfn over synthetic weights, run
// once on the split and once on the GPU alone, timed and compared. It answers
// the spike's question — what the Neural Engine's rate is on a QKV-shaped
// matmul — with the program and request shape the service accepts, rather
// than with a hand-written MIL.
//
//   ane-mixer-probe <metallib> [rows] [ane units]

#include "AwakeClock.hpp"
#include "GgufFormatReference.hpp"
#include "TestBuffers.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Gguf.h"
#include "ops/AneFfn.hpp"
#include "ops/Linear.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
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

// The mixer's geometry: a QKV projection of 8192 columns over 5120 hidden
// values, and the down projection a split wants of the same shape class.
constexpr uint32_t kHidden = 5120, kIntermediate = 8192, kLayers = 3;

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

Projection projection(const std::vector<const Tensor *> &parts, uint32_t columns) {
  BlockWeights blocks;
  uint32_t offset = 0;
  for (const Tensor *t : parts) {
    blocks.segments.push_back(t->segment);
    blocks.segments.back().columnOffset = offset;
    offset += t->N;
  }
  return Projection(columns, parts.front()->K, std::move(blocks));
}

struct Chunk {
  PrefillFfnBuffers ffn;
  std::array<MetalBuffer, 2> hidden;
};

Chunk chunk(MetalBackend &backend, uint32_t rows) {
  const auto bf16Rows = [&](uint32_t width) { return sharedBuffer(backend, uint64_t{rows} * width * 2); };
  return {{bf16Rows(kHidden), bf16Rows(kIntermediate), bf16Rows(kIntermediate), {}},
          {bf16Rows(kHidden), bf16Rows(kHidden)}};
}

void fillRows(const Chunk &chunk, uint32_t rows) {
  std::normal_distribution<float> normal;
  auto *values = static_cast<uint16_t *>(chunk.ffn.normalized.contents());
  for (uint64_t i = 0; i < uint64_t{rows} * kHidden; ++i) values[i] = floatToBf16(normal(rng));
}

float bf16At(const MetalBuffer &buffer, uint64_t index) {
  return bf16ToFloat(static_cast<const uint16_t *>(buffer.contents())[index]);
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: ane-mixer-probe <metallib> [rows] [ane units]\n");
    return 2;
  }
  const uint32_t rows = argc > 2 ? static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10)) : 2048;
  const uint32_t aneUnits = argc > 3 ? static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10)) : 8;
  try {
    MetalBackend backend(argv[1]);
    const Fmt format = fmtNamed("iq3s");
    std::vector<Tensor> tensors;
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
      tensors.push_back(tensor(backend, format, kIntermediate, kHidden)); // gate
      tensors.push_back(tensor(backend, format, kIntermediate, kHidden)); // up
      tensors.push_back(tensor(backend, format, kHidden, kIntermediate)); // down
    }
    std::vector<Projection> projections;
    projections.reserve(tensors.size());
    for (size_t layer = 0; layer < kLayers; ++layer) {
      projections.push_back(projection({&tensors[3 * layer]}, kIntermediate));
      projections.push_back(projection({&tensors[3 * layer + 1]}, kIntermediate));
      projections.push_back(projection({&tensors[3 * layer + 2]}, kHidden));
    }
    std::vector<SwiGluProjections> layers;
    for (uint32_t layer = 0; layer < kLayers; ++layer)
      layers.push_back({&projections[3 * layer], &projections[3 * layer + 1], &projections[3 * layer + 2]});
    if (const char *reason = AneFfn::unsupported(layers)) {
      std::fprintf(stderr, "the split does not take this shape: %s\n", reason);
      return 1;
    }
    const Chunk buffers = chunk(backend, rows);
    fillRows(buffers, rows);
    Linear linear(backend.capabilities());

    // The GPU alone, then the split, over the same rows and weights.
    std::memset(buffers.hidden[0].contents(), 0, buffers.hidden[0].sizeBytes());
    CommandGraph alone;
    for (uint32_t layer = 0; layer < kLayers; ++layer)
      linear.addPrefillSwiGlu(alone, layers[layer], buffers.ffn, buffers.hidden[layer & 1],
                              buffers.hidden[(layer & 1) ^ 1], rows);
    auto start = AwakeClock::now();
    static_cast<void>(backend.submitCommandAsync(alone.command()).wait());
    const double gpuMilliseconds = millisecondsSince(start);
    std::vector<float> gpu(kHidden * rows);
    for (uint32_t index = 0; index < rows; ++index)
      for (uint32_t channel = 0; channel < kHidden; ++channel)
        gpu[uint64_t{index} * kHidden + channel] =
            bf16At(buffers.hidden[kLayers & 1], uint64_t{index} * kHidden + channel);

    std::memset(buffers.hidden[0].contents(), 0, buffers.hidden[0].sizeBytes());
    AneFfn split(backend, layers, aneUnits, nullptr);
    if (!split.begin()) throw std::runtime_error("the split stopped: " + split.reason());
    CommandGraph graph;
    for (uint32_t layer = 0; layer < kLayers; ++layer)
      split.add(graph, layer, buffers.ffn, buffers.hidden[layer & 1], buffers.hidden[(layer & 1) ^ 1], rows);
    start = AwakeClock::now();
    static_cast<void>(split.commit(graph, {}).wait());
    const double splitMilliseconds = millisecondsSince(start);

    double signal = 0.0, noise = 0.0, worst = 0.0;
    for (uint32_t index = 0; index < rows; ++index)
      for (uint32_t channel = 0; channel < kHidden; ++channel) {
        const float expected = gpu[uint64_t{index} * kHidden + channel];
        const float actual =
            bf16At(buffers.hidden[kLayers & 1], uint64_t{index} * kHidden + channel);
        const double difference = double(actual) - expected;
        signal += double(expected) * expected;
        noise += difference * difference;
        worst = std::max<double>(worst, std::fabs(difference) / std::max<double>(std::fabs(expected), 1e-3));
      }
    const double flops = 3.0 * kLayers * kHidden * kIntermediate * rows * 2.0;
    std::printf("mixer geometry %ux%u, %u rows, %u of %u units on the ANE\n", kIntermediate, kHidden, rows, aneUnits,
                kIntermediate / 512);
    std::printf("  GPU alone: %.2f ms (%.2f TFLOPS)\n", gpuMilliseconds, flops / gpuMilliseconds / 1e12 * 1e3);
    std::printf("  split:     %.2f ms (%.2fx), %.2f%% RMS from the GPU alone, worst %.2f%%\n", splitMilliseconds,
                gpuMilliseconds / splitMilliseconds, std::sqrt(noise / std::max(signal, 1e-9)) * 100.0, worst * 100.0);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "ane-mixer-probe: %s\n", error.what());
    return 1;
  }
}
