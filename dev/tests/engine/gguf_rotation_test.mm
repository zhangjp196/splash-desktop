// Prism ML's and Gyro's input rotation (kernels/shared/gguf_rotation.metal): gguf_rotate's H (D x) and the
// rotated PQ2_0 token gather's D (H r) are bitwise the bf16 rounding of the fp32 butterflies in source order,
// which lie within one bf16 step of the fp64 transform, for the largest rotation block (1024, Prism's) and the
// smallest (128, Gyro's).
//   gguf-rotation <splash.metallib>
#include "GgufFormatReference.hpp"
#include "TestBuffers.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Gguf.h"
#include "metal/abi/MoE.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace gguf_reference;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::test::sharedBuffer;

namespace {

float bf16ToFloat(uint16_t bits) {
  const uint32_t word = uint32_t{bits} << 16;
  float value;
  std::memcpy(&value, &word, 4);
  return value;
}
// Round to nearest even, as the kernels' bfloat conversion.
uint16_t floatToBf16(float value) {
  uint32_t word;
  std::memcpy(&word, &value, 4);
  return uint16_t((word + 0x7FFF + ((word >> 16) & 1)) >> 16);
}

// The unnormalized transform of one block in fp32, stage by stage as the kernel computes it.
void butterflies(float *v, uint32_t block) {
  for (uint32_t stride = 1; stride < block; stride *= 2)
    for (uint32_t a = 0; a < block; ++a)
      if (!(a & stride)) {
        const float x = v[a], y = v[a + stride];
        v[a] = x + y;
        v[a + stride] = x - y;
      }
}
void butterflies(double *v, uint32_t block) {
  for (uint32_t stride = 1; stride < block; stride *= 2)
    for (uint32_t a = 0; a < block; ++a)
      if (!(a & stride)) {
        const double x = v[a], y = v[a + stride];
        v[a] = x + y;
        v[a + stride] = x - y;
      }
}

template <class T> MetalBuffer upload(MetalBackend &backend, const std::vector<T> &values) {
  MetalBuffer buffer = sharedBuffer(backend, values.size() * sizeof(T));
  std::memcpy(buffer.contents(), values.data(), values.size() * sizeof(T));
  return buffer;
}

// Compares got with the fp32 reference bitwise and with the fp64 values within one bf16 step.
int check(const char *what, const uint16_t *got, const std::vector<float> &fp32, const std::vector<double> &fp64) {
  size_t differ = 0, far = 0;
  for (size_t i = 0; i < fp32.size(); ++i) {
    differ += got[i] != floatToBf16(fp32[i]);
    const double step = std::ldexp(1.0, std::ilogb(std::max(std::fabs(fp64[i]), 1e-30)) - 7);
    far += std::fabs(double(bf16ToFloat(got[i])) - fp64[i]) > step;
  }
  const bool ok = !differ && !far;
  std::cout << what << ": " << differ << " of " << fp32.size() << " values differ from bf16(fp32 butterflies), "
            << far << " lie beyond a bf16 step of fp64 " << (ok ? "ok" : "FAIL") << '\n';
  return ok ? 0 : 1;
}

// One block size's cases: gguf_rotate over three blocks of five rows, and the rotated PQ2_0 gather of four tokens.
int checkBlock(MetalBackend &backend, std::mt19937 &rng, uint32_t block) {
  const float scale = float(1.0 / std::sqrt(double(block)));
  const uint32_t threads = block / 4;
  std::vector<int8_t> signs(3 * block);
  for (int8_t &s : signs) s = rng() & 1 ? 1 : -1;
  const MetalBuffer signBuffer = upload(backend, signs);
  const std::string suffix = " (block " + std::to_string(block) + ")";
  int failures = 0;

  {  // gguf_rotate: rows of three blocks.
    const uint32_t rows = 5, width = 3 * block;
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<uint16_t> input(rows * width);
    for (uint16_t &x : input) x = floatToBf16(normal(rng));
    std::vector<float> fp32(input.size());
    std::vector<double> fp64(input.size());
    for (uint32_t r = 0; r < rows; ++r)
      for (uint32_t b = 0; b < width; b += block) {
        float *f = fp32.data() + r * width + b;
        double *d = fp64.data() + r * width + b;
        for (uint32_t i = 0; i < block; ++i) {
          f[i] = bf16ToFloat(input[r * width + b + i]) * float(signs[b + i]);
          d[i] = f[i];
        }
        butterflies(f, block);
        butterflies(d, block);
        for (uint32_t i = 0; i < block; ++i) f[i] *= scale, d[i] *= scale;
      }
    const MetalBuffer in = upload(backend, input), out = sharedBuffer(backend, input.size() * 2);
    CommandGraph graph;
    graph.add("gguf_rotate", {in, signBuffer, out}, GgufRotationParams{width, block, scale, 0},
              {width / block, rows, 1}, {threads, 1, 1});
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    failures += check(("gguf_rotate H (D x)" + suffix).c_str(), static_cast<const uint16_t *>(out.contents()), fp32,
                      fp64);
  }

  {  // moe_rotate_intermediate: the routed tiles' live rows, in place, and the shared tile's untouched.
    const uint32_t tileRows = 8, routedTiles = 2, tiles = 3, intermediate = 2 * block;
    const std::vector<MoeTileDescriptor> descriptors{{7, 3}, {7, 8}, {128, 5}};
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<uint16_t> input(tiles * tileRows * intermediate);
    for (uint16_t &x : input) x = floatToBf16(normal(rng));
    std::vector<float> fp32(input.size());
    std::vector<double> fp64(input.size());
    for (uint32_t tile = 0; tile < tiles; ++tile)
      for (uint32_t row = 0; row < tileRows; ++row)
        for (uint32_t b = 0; b < intermediate; b += block) {
          const size_t at = (size_t{tile} * tileRows + row) * intermediate + b;
          float *f = fp32.data() + at;
          double *d = fp64.data() + at;
          const bool live = tile < routedTiles && row < descriptors[tile].rows;
          for (uint32_t i = 0; i < block; ++i) {
            f[i] = live ? bf16ToFloat(input[at + i]) * float(signs[b + i]) : bf16ToFloat(input[at + i]);
            d[i] = f[i];
          }
          if (live) {
            butterflies(f, block);
            butterflies(d, block);
            for (uint32_t i = 0; i < block; ++i) f[i] *= scale, d[i] *= scale;
          }
        }
    const MetalBuffer data = upload(backend, input);
    const MetalBuffer tileBuffer = upload(backend, descriptors);
    CommandGraph graph;
    graph.add("moe_rotate_intermediate", {data, tileBuffer, signBuffer},
              GgufRotationParams{intermediate, block, scale, tileRows}, {intermediate / block, routedTiles, tileRows},
              {threads, 1, 1});
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    failures += check(("moe_rotate_intermediate H (D e)" + suffix).c_str(),
                      static_cast<const uint16_t *>(data.contents()), fp32, fp64);
  }

  {  // moe_gather_rows_rotated: the routed tile's rows of H (D x), the shared tile's of the input.
    const uint32_t tileRows = 8, routesPerRow = 2, rows = 3, tiles = 2, hidden = 2 * block;
    const std::vector<MoeTileDescriptor> descriptors{{7, rows}, {128, rows}};
    std::vector<uint32_t> groupedRoutes(tiles * tileRows, ~0u), routeRows(rows * routesPerRow);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<uint16_t> input(rows * hidden), rotated(rows * hidden);
    for (uint16_t &x : input) x = floatToBf16(normal(rng));
    for (uint16_t &x : rotated) x = floatToBf16(normal(rng));
    for (uint32_t r = 0; r < rows; ++r)
      for (uint32_t slot = 0; slot < routesPerRow; ++slot) {
        const uint32_t route = r * routesPerRow + slot;
        groupedRoutes[(slot == 0 ? 0 : 1) * tileRows + r] = route;
        routeRows[route] = (slot == 0 ? 0 : 1) * tileRows + r;
      }
    // The gather's expected rows: 8 per tile, the 3 live ones then zeros.
    std::vector<uint16_t> want(tiles * tileRows * hidden, 0);
    for (uint32_t t = 0; t < tiles; ++t)
      for (uint32_t local = 0; local < rows; ++local) {
        const uint32_t route = groupedRoutes[t * tileRows + local];
        const uint32_t row = route / routesPerRow;
        const auto *source = (t == 0 ? rotated : input).data() + row * hidden;
        std::copy(source, source + hidden, want.data() + (t * tileRows + local) * hidden);
      }
    const MetalBuffer in = upload(backend, input), rot = upload(backend, rotated);
    const MetalBuffer routes = upload(backend, groupedRoutes), tileBuffer = upload(backend, descriptors);
    const std::vector<uint32_t> live{tiles};
    const MetalBuffer count = upload(backend, live);
    MetalBuffer gathered = sharedBuffer(backend, uint64_t{tiles} * tileRows * hidden * sizeof(uint16_t));
    std::memset(gathered.contents(), 0xFF, tiles * tileRows * hidden * sizeof(uint16_t));
    CommandGraph graph;
    graph.add("moe_gather_rows_rotated", {in, rot, routes, tileBuffer, count, gathered},
              MoeGatherParams{tileRows, hidden, routesPerRow}, {tiles, hidden / 256, 1}, {256, 1, 1});
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    const auto *got = static_cast<const uint16_t *>(gathered.contents());
    size_t differ = 0;
    for (size_t i = 0; i < want.size(); ++i) differ += got[i] != want[i];
    std::cout << "moe_gather_rows_rotated (block " << block << "): " << differ << " of " << want.size()
              << " values differ from the routed and shared rows " << (differ ? "FAIL" : "ok") << '\n';
    failures += differ != 0;
  }

  {  // moe_rotate_intermediate: the routed tiles' live rows, in place, and the shared tile's untouched.
    const uint32_t vocabulary = 7, hidden = 2 * block;
    std::vector<uint8_t> table = makeNative(PQ20, vocabulary, hidden, rng);
    const std::vector<uint32_t> tokens{3, 0, 6, 3};
    std::vector<float> fp32(tokens.size() * hidden);
    std::vector<double> fp64(fp32.size());
    std::vector<float> row(hidden);
    for (size_t t = 0; t < tokens.size(); ++t) {
      rowValues(PQ20, table.data() + size_t{tokens[t]} * rowBytes(PQ20, hidden), hidden, row.data());
      for (uint32_t b = 0; b < hidden; b += block) {
        float *f = fp32.data() + t * hidden + b;
        double *d = fp64.data() + t * hidden + b;
        for (uint32_t i = 0; i < block; ++i) f[i] = row[b + i], d[i] = row[b + i];
        butterflies(f, block);
        butterflies(d, block);
        for (uint32_t i = 0; i < block; ++i) f[i] = f[i] * scale * float(signs[b + i]), d[i] = d[i] * scale * signs[b + i];
      }
    }
    const MetalBuffer tokenBuffer = upload(backend, tokens), rows = upload(backend, table);
    const MetalBuffer out = sharedBuffer(backend, fp32.size() * 2);
    CommandGraph graph;
    const GgufEmbedParams embed{uint32_t(tokens.size()), vocabulary, hidden};
    graph.add("gguf_embed_rotated_pq20", {tokenBuffer, rows, signBuffer, out},
              GgufEmbedRotationParams{embed, {hidden, block, scale, 0}},
              {hidden / block, uint32_t(tokens.size()), 1}, {threads, 1, 1});
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    failures += check(("gguf_embed_rotated_pq20 D (H r)" + suffix).c_str(),
                      static_cast<const uint16_t *>(out.contents()), fp32, fp64);
  }
  return failures;
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc != 2) {
      std::cerr << "usage: gguf-rotation <splash.metallib>\n";
      return 2;
    }
    std::mt19937 rng(7);
    int failures = 0;
    try {
      MetalBackend backend(argv[1]);
      for (const uint32_t block : {GGUF_ROTATION_BLOCK, GGUF_ROTATION_MIN_BLOCK})
        failures += checkBlock(backend, rng, block);
    } catch (const std::exception &e) {
      std::cerr << "gguf-rotation: FAIL: " << e.what() << '\n';
      return 1;
    }
    std::cout << (failures ? "GGUF rotation tests FAILED" : "GGUF rotation tests passed") << '\n';
    return failures ? 1 : 0;
  }
}
