// A dense FFN's geometry through the split's own program: three layers of
// AneFfn over synthetic weights, run once on the split and once on the GPU
// alone, timed and compared. It answers the spike's question — what the
// Neural Engine's rate is on a projection-shaped matmul — with the program and
// request shape the service accepts, rather than with a hand-written MIL.
//
// The default geometry is the mixer's (hidden 5120, intermediate 8192). A MoE
// target's dense parts are the shared expert's width (hidden 2048, 1024 or
// 512) and its leading dense layers' (hidden 2048, intermediate 6144), which
// `--ane-moe` would split if the engine offered it.
//
//   ane-mixer-probe <metallib> [rows] [ane units] [hidden] [intermediate] [rounds]

#include "AwakeClock.hpp"
#include "../tests/engine/GgufFormatReference.hpp"
#include "../tests/engine/TestBuffers.hpp"
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

constexpr uint32_t kLayers = 3;

// The geometry the probe runs, read from its arguments.
struct Geometry final {
  uint32_t rows = 2048, units = 8, hidden = 5120, intermediate = 8192, rounds = 5;
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

Chunk chunk(MetalBackend &backend, const Geometry &geometry) {
  const auto bf16Rows = [&](uint32_t width) {
    return sharedBuffer(backend, uint64_t{geometry.rows} * width * 2);
  };
  return {{bf16Rows(geometry.hidden), bf16Rows(geometry.intermediate), bf16Rows(geometry.intermediate), {}},
          {bf16Rows(geometry.hidden), bf16Rows(geometry.hidden)}};
}

void fillRows(const Chunk &chunk, const Geometry &geometry) {
  std::normal_distribution<float> normal;
  auto *values = static_cast<uint16_t *>(chunk.ffn.normalized.contents());
  for (uint64_t i = 0; i < uint64_t{geometry.rows} * geometry.hidden; ++i) values[i] = floatToBf16(normal(rng));
}

float bf16At(const MetalBuffer &buffer, uint64_t index) {
  return bf16ToFloat(static_cast<const uint16_t *>(buffer.contents())[index]);
}

uint32_t argument(int argc, char **argv, int index, uint32_t fallback) {
  return argc > index ? static_cast<uint32_t>(std::strtoul(argv[index], nullptr, 10)) : fallback;
}

// The median of `run` over `rounds` runs, the first of which is a warm-up
// that wires the command.
template <class Run> double medianOf(uint32_t rounds, Run run) {
  run();
  std::vector<double> times;
  times.reserve(rounds);
  for (uint32_t round = 0; round < rounds; ++round) times.push_back(run());
  std::ranges::nth_element(times, times.begin() + times.size() / 2);
  return times[times.size() / 2];
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: ane-mixer-probe <metallib> [rows] [ane units] [hidden] [intermediate] [rounds]\n");
    return 2;
  }
  Geometry geometry;
  geometry.rows = argument(argc, argv, 2, geometry.rows);
  geometry.units = argument(argc, argv, 3, geometry.units);
  geometry.hidden = argument(argc, argv, 4, geometry.hidden);
  geometry.intermediate = argument(argc, argv, 5, geometry.intermediate);
  geometry.rounds = argument(argc, argv, 6, geometry.rounds);
  try {
    MetalBackend backend(argv[1]);
    const Fmt format = fmtNamed("iq3s");
    std::vector<Tensor> tensors;
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
      tensors.push_back(tensor(backend, format, geometry.intermediate, geometry.hidden)); // gate
      tensors.push_back(tensor(backend, format, geometry.intermediate, geometry.hidden)); // up
      tensors.push_back(tensor(backend, format, geometry.hidden, geometry.intermediate)); // down
    }
    std::vector<Projection> projections;
    projections.reserve(tensors.size());
    for (size_t layer = 0; layer < kLayers; ++layer) {
      projections.push_back(projection({&tensors[3 * layer]}, geometry.intermediate));
      projections.push_back(projection({&tensors[3 * layer + 1]}, geometry.intermediate));
      projections.push_back(projection({&tensors[3 * layer + 2]}, geometry.hidden));
    }
    std::vector<SwiGluProjections> layers;
    for (uint32_t layer = 0; layer < kLayers; ++layer)
      layers.push_back({&projections[3 * layer], &projections[3 * layer + 1], &projections[3 * layer + 2]});
    if (const char *reason = AneFfn::unsupported(layers)) {
      std::fprintf(stderr, "the split does not take this shape: %s\n", reason);
      return 1;
    }
    const Chunk buffers = chunk(backend, geometry);
    fillRows(buffers, geometry);
    Linear linear(backend.capabilities());

    // The GPU alone, then the split, over the same rows and weights, the
    // median of `rounds` timed commands after a warm-up.
    const auto gpuCommand = [&] {
      std::memset(buffers.hidden[0].contents(), 0, buffers.hidden[0].sizeBytes());
      CommandGraph alone;
      for (uint32_t layer = 0; layer < kLayers; ++layer)
        linear.addPrefillSwiGlu(alone, layers[layer], buffers.ffn, buffers.hidden[layer & 1],
                                buffers.hidden[(layer & 1) ^ 1], geometry.rows);
      return alone;
    };
    const auto timeGpu = [&] {
      CommandGraph alone = gpuCommand();
      const auto start = AwakeClock::now();
      static_cast<void>(backend.submitCommandAsync(alone.command()).wait());
      return millisecondsSince(start);
    };
    const double gpuMilliseconds = medianOf(geometry.rounds, timeGpu);
    std::vector<float> gpu(geometry.hidden * geometry.rows);
    for (uint32_t index = 0; index < geometry.rows; ++index)
      for (uint32_t channel = 0; channel < geometry.hidden; ++channel)
        gpu[uint64_t{index} * geometry.hidden + channel] =
            bf16At(buffers.hidden[kLayers & 1], uint64_t{index} * geometry.hidden + channel);

    std::memset(buffers.hidden[0].contents(), 0, buffers.hidden[0].sizeBytes());
    AneFfn split(backend, layers, geometry.units, nullptr);
    const auto splitCommand = [&] {
      std::memset(buffers.hidden[0].contents(), 0, buffers.hidden[0].sizeBytes());
      if (!split.begin()) throw std::runtime_error("the split stopped: " + split.reason());
      CommandGraph graph;
      for (uint32_t layer = 0; layer < kLayers; ++layer)
        split.add(graph, layer, buffers.ffn, buffers.hidden[layer & 1], buffers.hidden[(layer & 1) ^ 1],
                  geometry.rows);
      return graph;
    };
    const auto timeSplit = [&] {
      const CommandGraph graph = splitCommand();
      const auto start = AwakeClock::now();
      static_cast<void>(split.commit(graph, {}).wait());
      const double milliseconds = millisecondsSince(start);
      if (!split.finish()) throw std::runtime_error("the split stopped: " + split.reason());
      return milliseconds;
    };
    const AneFfn::Served before = split.served();
    const double splitMilliseconds = medianOf(geometry.rounds, timeSplit);
    const AneFfn::Served after = split.served();
    const uint64_t evaluations = after.evaluations - before.evaluations;
    const double aneMilliseconds = after.milliseconds - before.milliseconds;
    // One evaluation is one layer's ANE part: gate and up over the ANE's
    // channels, then down over them.
    const double aneChannels = double(geometry.units) * 512.0;
    const double aneFlops = 3.0 * geometry.hidden * aneChannels * geometry.rows * 2.0;

    double signal = 0.0, noise = 0.0, worst = 0.0;
    for (uint32_t index = 0; index < geometry.rows; ++index)
      for (uint32_t channel = 0; channel < geometry.hidden; ++channel) {
        const float expected = gpu[uint64_t{index} * geometry.hidden + channel];
        const float actual =
            bf16At(buffers.hidden[kLayers & 1], uint64_t{index} * geometry.hidden + channel);
        const double difference = double(actual) - expected;
        signal += double(expected) * expected;
        noise += difference * difference;
        worst = std::max<double>(worst, std::fabs(difference) / std::max<double>(std::fabs(expected), 1e-3));
      }
    const double flops = 3.0 * kLayers * geometry.hidden * geometry.intermediate * geometry.rows * 2.0;
    const double aneRate = aneMilliseconds > 0.0
                                ? aneFlops * double(evaluations) / (aneMilliseconds / 1000.0) / 1e12
                                : 0.0;
    std::printf("geometry %ux%u, %u rows, %u of %u units on the ANE, median of %u\n", geometry.intermediate,
                geometry.hidden, geometry.rows, geometry.units, geometry.intermediate / 512, geometry.rounds);
    std::printf("  GPU alone: %.2f ms (%.2f TFLOPS)\n", gpuMilliseconds, flops / gpuMilliseconds / 1e12 * 1e3);
    std::printf("  split:     %.2f ms (%.2fx), %.2f%% RMS from the GPU alone, worst %.2f%%\n", splitMilliseconds,
                gpuMilliseconds / splitMilliseconds, std::sqrt(noise / std::max(signal, 1e-9)) * 100.0, worst * 100.0);
    if (evaluations)
      std::printf("  ANE:       %.2f ms over %llu evaluations (%.2f ms a layer), %.2f TFLOPS on its share\n",
                  aneMilliseconds, static_cast<unsigned long long>(evaluations),
                  aneMilliseconds / double(evaluations), aneRate);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "ane-mixer-probe: %s\n", error.what());
    return 1;
  }
}
