// Verify the Metal stem consumes protocol tile-major pixels and emits the
// image-wide raster patch order used by the vision tower.
#include "TestBuffers.hpp"
#include "LinearNumerics.hpp"
#include "metal/abi/Inkling2Vision.h"
#include "ops/Inkling2VisionStem.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#import <Foundation/Foundation.h>

#include <iostream>
#include <stdexcept>

using namespace splash;

#include "inkling2_vision_numerics.hpp"

static void run(metal::MetalBackend &backend, uint32_t height, uint32_t width) {
  const uint32_t count = height * width * 40 * 40 * 3;
  auto input = test::sharedBuffer(backend, uint64_t(count) * sizeof(float));
  auto output = test::sharedBuffer(backend, uint64_t(count) * sizeof(uint16_t));
  auto *pixels = static_cast<float *>(input.contents());
  auto *patches = static_cast<uint16_t *>(output.contents());
  // Values are exactly representable as BF16 and depend on tile and position.
  for (uint32_t tile = 0; tile < height * width; ++tile)
    for (uint32_t y = 0; y < 40; ++y)
      for (uint32_t x = 0; x < 40; ++x)
        for (uint32_t c = 0; c < 3; ++c)
          pixels[((tile * 40 + y) * 40 + x) * 3 + c] = float((tile * 17 + y * 3 + x + c) % 128);
  metal::CommandGraph graph;
  graph.add("inkling2_vision_patchify", {input, output},
            Inkling2VisionStemParams{1024, 20, height, width}, {height * width * 4, 1, 1});
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  // Construct the expected patch sequence independently by walking the
  // global raster patch grid and looking up its pixel within a source tile.
  for (uint32_t py = 0; py < height * 2; ++py)
    for (uint32_t px = 0; px < width * 2; ++px)
      for (uint32_t y = 0; y < 20; ++y)
        for (uint32_t x = 0; x < 20; ++x)
          for (uint32_t c = 0; c < 3; ++c) {
            const uint32_t tile = (py / 2) * width + px / 2;
            const uint32_t src = ((tile * 40 + py % 2 * 20 + y) * 40 + px % 2 * 20 + x) * 3 + c;
            const uint32_t dst = (((py * width * 2 + px) * 20 + y) * 20 + x) * 3 + c;
            test::require(patches[dst] == test::floatToBf16(pixels[src]), "patchify layout mismatch");
          }
}

// Reference the HF Inkling2VisionEmbeddings equations independently: each
// 40x40 tile yields four 20x20 patches, and positions are bilinearly sampled
// over the complete image grid. GGUF convolution weights are stored CHW.
static void stemNumerics(metal::MetalBackend &backend) {
  constexpr uint32_t H = 1024, tiles = 2, patches = 8;
  auto bf = [&](uint64_t n) { return test::sharedBuffer(backend, n * 2); };
  ops::Inkling2VisionStemBuffers b;
  b.pixels = test::sharedBuffer(backend, tiles * 40 * 40 * 3 * sizeof(float));
  b.patches = bf(patches * 1200); b.patchEmbedding = bf(H * 1200);
  b.patchBias = bf(H); b.classEmbedding = bf(H); b.classPosition = bf(H);
  b.positions = bf(32 * 32 * H);
  b.hidden = test::sharedBuffer(backend, (patches + 1) * H * sizeof(float));
  auto *pix = static_cast<float *>(b.pixels.contents());
  auto *w = static_cast<uint16_t *>(b.patchEmbedding.contents());
  auto *bias = static_cast<uint16_t *>(b.patchBias.contents());
  auto *cls = static_cast<uint16_t *>(b.classEmbedding.contents());
  auto *cp = static_cast<uint16_t *>(b.classPosition.contents());
  auto *pos = static_cast<uint16_t *>(b.positions.contents());
  for (uint32_t i = 0; i < tiles * 40 * 40 * 3; ++i)
    pix[i] = float(int((i * 13 + i / 47) % 31) - 15) / 32;
  for (uint32_t d = 0; d < H; ++d) {
    bias[d] = 0;
    cls[d] = test::floatToBf16(float(int(d % 13) - 6) / 32);
    cp[d] = test::floatToBf16(float(int(d % 7) - 3) / 64);
    for (uint32_t k = 0; k < 1200; ++k)
      w[d * 1200 + k] = test::floatToBf16(float(int((d * 7 + k * 3) % 17) - 8) / 2048);
  }
  for (uint32_t r = 0; r < 32; ++r)
    for (uint32_t c = 0; c < 32; ++c)
      for (uint32_t d = 0; d < H; ++d)
        pos[(r * 32 + c) * H + d] = test::floatToBf16(float(int((r * 11 + c * 7 + d * 3) % 41) - 20) / 128);
  metal::CommandGraph graph;
  ops::Inkling2VisionStem::encode(graph, b, H, 1, 2);
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  auto *actual = static_cast<float *>(b.hidden.contents());
  for (uint32_t i = 0; i < (patches + 1) * H; ++i)
    test::require(std::isfinite(actual[i]), "non-finite stem output");
  auto round = [](float x) { return test::bf16ToFloat(test::floatToBf16(x)); };
  float maxError = 0;
  for (uint32_t d = 0; d < H; ++d)
    maxError = std::max(maxError, std::abs(actual[d] - round(test::bf16ToFloat(cls[d]) + test::bf16ToFloat(cp[d]))));
  for (uint32_t py = 0; py < 2; ++py) for (uint32_t px = 0; px < 4; ++px)
    for (uint32_t d = 0; d < H; ++d) {
      float sum = 0;
      for (uint32_t y = 0; y < 20; ++y) for (uint32_t x = 0; x < 20; ++x)
        for (uint32_t c = 0; c < 3; ++c) {
          uint32_t src = (((px / 2) * 40 + py * 20 + y) * 40 + (px % 2) * 20 + x) * 3 + c;
          sum = std::fma(round(pix[src]), test::bf16ToFloat(w[d * 1200 + c * 400 + y * 20 + x]), sum);
        }
      const float fy = (py + .5f) * 16 - .5f, fx = (px + .5f) * 8 - .5f;
      const uint32_t y0 = uint32_t(fy), x0 = uint32_t(fx);
      auto p = [&](uint32_t y, uint32_t x) { return test::bf16ToFloat(pos[(y * 32 + x) * H + d]); };
      const float wy = fy - y0, wx = fx - x0;
      float a = p(y0, x0) * (1 - wx) + p(y0, x0 + 1) * wx;
      float c = p(y0 + 1, x0) * (1 - wx) + p(y0 + 1, x0 + 1) * wx;
      float expected = round(round(a * (1 - wy) + c * wy) + round(sum));
      maxError = std::max(maxError, std::abs(actual[(1 + py * 4 + px) * H + d] - expected));
    }
  std::cout << "inkling2 stem 1x2 max_abs_error=" << maxError << '\n';
  test::require(std::isfinite(maxError) && maxError <= 0.002f, "stem numerical mismatch");
}

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      test::require(argc == 2, "usage: inkling2-vision-patchify METALLIB");
      metal::MetalBackend backend(argv[1]);
      for (const auto [h, w] : {std::pair{1u, 1u}, std::pair{1u, 3u}, std::pair{2u, 1u}, std::pair{2u, 3u}})
        run(backend, h, w);
      stemNumerics(backend);
      inkAttention(backend, 3);
      inkAttention(backend, 19);
      inkBlocks(backend, 1, 2, 1);
      inkBlocks(backend, 2, 3, 23);
      inkMerger(backend, 1, 2);
      inkMerger(backend, 2, 3);
      std::cout << "inkling2_vision_patchify_test: PASS\n";
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "inkling2_vision_patchify_test: FAIL: " << error.what() << '\n';
      return 1;
    }
  }
}
