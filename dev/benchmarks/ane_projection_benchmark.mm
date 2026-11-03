// A single projection on the Neural Engine, for the shape a mixer's QKV
// projection has (K = 5120, N = 8192, 2048 rows): how long one evaluation
// takes, the throughput it reaches and how far its int8 result is from the
// CPU's. It is the AneFfn program without the FFN's rotation, silu and GPU
// join: an int8 activation surface per hidden segment times int8 weights,
// both dequantized by 1 / ANE_FFN_INT8_UNIT on the engine, summed, and scaled
// by the weights' row scales and the tokens' scales.
//
//   ane-projection-benchmark <metallib> <N> <K> [rows] [rounds]

#include "AwakeClock.hpp"
#include "ane/Program.hpp"
#include "metal/abi/AneFfn.h"
#include "metal/MetalBackend.hpp"

#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <string>
#include <vector>

namespace {

using splash::ane::Surface;
using splash::metal::MetalBackend;

// The hidden columns the AneFfn program splits a projection's inputs into
// (ops/AneFfn.cpp kSegment): a matmul reads one segment at a time.
constexpr uint32_t kSegment = 2560;
constexpr uint32_t kDefaultRows = 2048, kDefaultRounds = 3;

std::string fp16(double value) {
  char text[32];
  std::snprintf(text, sizeof text, "fp16(%a)", value);
  return text;
}

std::string dimensions(uint32_t height, uint32_t width) {
  return "[1, 1, " + std::to_string(height) + ", " + std::to_string(width) + "]";
}

// The MIL of the projection: one matmul per input segment, their sum scaled by
// the weights' row scales and the tokens' scales.
std::string mil(uint32_t rows, uint32_t columns, uint32_t segments, uint32_t stage,
                const std::vector<Surface> &activations, const std::vector<Surface> &weights,
                const Surface &weightScale, const Surface &tokenScale, const Surface &output) {
  const std::string unit = fp16(1.0 / ANE_FFN_INT8_UNIT);
  std::string parameters, body;
  const auto line = [&](const std::string &text) { body += "        " + text + ";\n"; };
  const auto f16 = [&](const std::string &name, uint32_t height, uint32_t width, const std::string &expression) {
    line("tensor<fp16, " + dimensions(height, width) + "> " + name + " = " + expression);
  };
  const auto buffer = [&](const std::string &name, const Surface &surface, uint32_t height, uint32_t width) {
    parameters += (parameters.empty() ? "" : ", ") + surface.bufferType(height, width) + " " + name;
    line(std::string("tensor<") + Surface::milType(surface.element) + ", " + dimensions(height, width) + "> " + name +
         "_t = tensor_buffer_to_tensor<ios17>(input = " + name + ")");
  };
  for (uint32_t segment = 0; segment < segments; ++segment) {
    const std::string index = std::to_string(segment);
    buffer("x" + index, activations[segment], kSegment, rows);
    buffer("w" + index, weights[segment], columns, kSegment);
  }
  buffer("sw", weightScale, columns, 1);
  buffer("sx", tokenScale, 1, rows);
  for (uint32_t segment = 0; segment < segments; ++segment) {
    const std::string index = std::to_string(segment);
    f16("x" + index + "_d", kSegment, rows, "dequantize(input = x" + index + "_t, scale = " + unit + ")");
    f16("w" + index + "_d", columns, kSegment, "dequantize(input = w" + index + "_t, scale = " + unit + ")");
    line("tensor<fp16, " + dimensions(columns, rows) + "> m" + index +
         " = matmul(transpose_x = bool(false), transpose_y = bool(false), x = w" + index + "_d, y = x" + index +
         "_d)");
  }
  std::string total = "m0";
  for (uint32_t segment = 1; segment < segments && stage > 1; ++segment) {
    const std::string next = "sum" + std::to_string(segment);
    f16(next, columns, rows, "add(x = " + total + ", y = m" + std::to_string(segment) + ")");
    total = next;
  }
  std::string last = total;
  if (stage > 2) {
    f16("rows", columns, rows, "mul(x = " + total + ", y = sw_t)");
    f16("tokens", columns, rows, "mul(x = rows, y = sx_t)");
    last = "tokens";
  }
  line(output.bufferType(columns, rows) +
       " y = tensor_to_tensor_buffer<ios17>(input = " + last + ", interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), "
       "strides = tensor<int64, [4]>(" +
       output.strides(columns) + "))");
  return "program(1.3)\n{\n    func f" + std::to_string(rows) + "<ios18>(" + parameters + ") {\n" + body +
         "    } -> (y);\n}\n";
}

template <class T> T *contents(const Surface &surface) {
  return static_cast<T *>(surface.buffer.contents());
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: ane-projection-benchmark <metallib> <N> <K> [rows] [rounds]\n");
    return 2;
  }
  const uint32_t columns = static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10));
  const uint32_t hidden = static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10));
  const uint32_t rows = argc > 4 ? static_cast<uint32_t>(std::strtoul(argv[4], nullptr, 10)) : kDefaultRows;
  const uint32_t rounds = argc > 5 ? static_cast<uint32_t>(std::strtoul(argv[5], nullptr, 10)) : kDefaultRounds;
  const uint32_t stage = argc > 6 ? static_cast<uint32_t>(std::strtoul(argv[6], nullptr, 10)) : 3;
  if (!columns || !hidden || hidden % kSegment || !rows || !rounds) {
    std::fprintf(stderr, "N and rows must be positive and K a multiple of %u\n", kSegment);
    return 2;
  }
  const uint32_t segments = hidden / kSegment;
  try {
    MetalBackend backend(argv[1]);
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> codes(-127, 127);
    std::vector<Surface> activations, weights;
    std::vector<std::vector<int8_t>> hostActivations(segments), hostWeights(segments);
    for (uint32_t segment = 0; segment < segments; ++segment) {
      activations.push_back(Surface::create(backend, kSegment, rows, Surface::Element::Int8));
      weights.push_back(Surface::create(backend, columns, kSegment, Surface::Element::Int8));
      hostActivations[segment].resize(uint64_t{kSegment} * rows);
      hostWeights[segment].resize(uint64_t{columns} * kSegment);
      for (int8_t &value : hostActivations[segment]) value = static_cast<int8_t>(codes(rng));
      for (int8_t &value : hostWeights[segment]) value = static_cast<int8_t>(codes(rng));
      std::memcpy(contents<int8_t>(activations.back()), hostActivations[segment].data(),
                  uint64_t{kSegment} * rows);
      std::memcpy(contents<int8_t>(weights.back()), hostWeights[segment].data(), uint64_t{columns} * kSegment);
    }
    Surface weightScale = Surface::create(backend, columns, 1, Surface::Element::Float16);
    Surface tokenScale = Surface::create(backend, 1, rows, Surface::Element::Float16);
    Surface output = Surface::create(backend, columns, rows, Surface::Element::Float16);
    const float weightStep = 1.0f / (ANE_FFN_INT8_UNIT * 512.0f), tokenStep = 1.0f / (ANE_FFN_INT8_UNIT * 512.0f);
    for (uint32_t column = 0; column < columns; ++column) contents<_Float16>(weightScale)[column] = weightStep;
    for (uint32_t token = 0; token < rows; ++token) contents<_Float16>(tokenScale)[token] = tokenStep;

    const std::string source =
        mil(rows, columns, segments, stage, activations, weights, weightScale, tokenScale, output);
    // The MIL names no BLOBFILE, but the service loads a program with its
    // blob beside it: a zero blob keeps that path the one AneFfn takes.
    const std::vector<uint8_t> blob(64 * 1024, 0);
    splash::ane::Program program(
        source, blob, splash::ane::Program::Limits{std::chrono::seconds(180), [] { return false; }},
        "build/ane-cache");
    const uint32_t procedure = program.procedure("f" + std::to_string(rows));
    // The program's inputs in the order its function declares them: each
    // segment's activations, then its weights, then the scales.
    std::vector<Surface> bound;
    for (uint32_t segment = 0; segment < segments; ++segment) {
      bound.push_back(activations[segment]);
      bound.push_back(weights[segment]);
    }
    bound.push_back(weightScale);
    bound.push_back(tokenScale);
    const splash::ane::Program::Binding binding = program.bind(procedure, bound, output);

    splash::metal::SharedEvent event = backend.newSharedEvent();
    uint64_t value = 1;
    std::mutex mutex;
    std::condition_variable changed;
    double milliseconds = 0.0;
    for (uint32_t round = 0; round < rounds; ++round) {
      bool done = false, succeeded = false;
      const uint64_t wait = value, signal = ++value;
      const auto start = splash::AwakeClock::now();
      program.enqueue(binding, event, wait, signal, [&](bool success) {
        std::lock_guard lock(mutex);
        done = true;
        succeeded = success;
        changed.notify_one();
      });
      // The GPU signals a job's `ready` after the request is queued: raising it
      // first could be what the service rejects.
      event.signal(wait);
      std::unique_lock lock(mutex);
      if (!changed.wait_for(lock, std::chrono::seconds(120), [&] { return done; }))
        throw std::runtime_error("the Neural Engine did not report the evaluation");
      if (!succeeded) throw std::runtime_error("the evaluation failed");
      milliseconds = splash::millisecondsSince(start);
      std::printf("  round %u: %.3f ms\n", round + 1, milliseconds);
    }

    // The CPU's projection of a few rows: what the engine's int8 path should
    // hold within a few percent.
    double worst = 0.0;
    for (uint32_t column = 0; column < columns; column += 977)
      for (uint32_t token = 0; token < rows; token += 683) {
        double reference = 0.0;
        for (uint32_t segment = 0; segment < segments; ++segment)
          for (uint32_t k = 0; k < kSegment; ++k)
            reference += double(hostWeights[segment][uint64_t{column} * kSegment + k]) *
                         double(hostActivations[segment][uint64_t{k} * rows + token]);
        reference *= double(weightStep) * double(tokenStep) / (double(ANE_FFN_INT8_UNIT) * ANE_FFN_INT8_UNIT);
        const _Float16 *values = contents<_Float16>(output);
        const double actual = double(values[uint64_t{column} * (output.strideBytes / sizeof(_Float16)) + token]);
        const double scale = std::max(std::fabs(reference), 1e-6);
        worst = std::max(worst, std::fabs(actual - reference) / scale);
      }
    const double flops = 2.0 * columns * hidden * rows;
    std::printf("%s %ux%u %u rows: %.3f ms, %.2f TFLOPS, %.2f%% from the CPU\n", "ANE", columns, hidden, rows,
                milliseconds, flops / (milliseconds / 1000.0) / 1e12, worst * 100.0);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "ane-projection-benchmark: %s\n", error.what());
    return 1;
  }
}
