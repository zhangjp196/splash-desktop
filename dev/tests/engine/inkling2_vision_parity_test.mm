// Real-projector parity gate for the Inkling2 vision tower.
//
//   inkling2-vision-parity METALLIB SOURCE OUT_DIR [--safetensors]
//
// Binds the projector's 422 tensors through Inkling2VisionLoader (the same
// BF16 image the engine loads, as MetalBuffer views), encodes deterministic
// tile-major 40x40 HWC pixels through the image-wide graph of
// Inkling2VisionGraphResources (stem, 23 blocks, merger), and writes per grid
//
//   pixels_<h>x<w>.f32   the pixels the graph consumed, tile-major HWC
//   output_<h>x<w>.f32   one 2304-wide row per tile, from the merger output
//
// dev/tools/inkling2_vision_reference.py re-derives the rows in torch on the
// CPU from the same GGUF and reports the difference.
#include "LinearNumerics.hpp"
#include "TestChecks.hpp"

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "model/Inkling2VisionLoader.hpp"
#include "model/WeightImages.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::model::Inkling2VisionBufferWeights;
using splash::model::Inkling2VisionGraphResources;
using splash::test::bf16ToFloat;
using splash::test::require;

struct Grid final {
  uint32_t height, width;
  [[nodiscard]] uint32_t tiles() const { return height * width; }
  [[nodiscard]] std::string name() const {
    return std::to_string(height) + "x" + std::to_string(width);
  }
};

void writeFloats(const std::filesystem::path &path, const std::vector<float> &values) {
  std::ofstream file(path, std::ios::binary);
  require(bool(file), "cannot open " + path.string());
  file.write(reinterpret_cast<const char *>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(float)));
  require(bool(file), "cannot write " + path.string());
}

// Normalized RGB in (-1, 1), a function of the tile and of the position in it:
// every grid sees its own patch sequence, while each pixel is exactly a BF16,
// so patchify's conversion is the only rounding the pixels take.
std::vector<float> pixelsFor(Grid grid) {
  std::vector<float> pixels(size_t{grid.tiles()} * 40 * 40 * 3);
  for (uint32_t tile = 0; tile < grid.tiles(); ++tile)
    for (uint32_t y = 0; y < 40; ++y)
      for (uint32_t x = 0; x < 40; ++x)
        for (uint32_t c = 0; c < 3; ++c) {
          const int numerator = int((tile * 29 + y * 3 + x * 7 + c * 11) % 251) - 125;
          pixels[(size_t(tile) * 40 + y) * 40 * 3 + size_t(x) * 3 + c] =
              float(numerator) / 128.0f;
        }
  return pixels;
}

// The merger's per-tile rows in FP32 after one submission of the graph.
std::vector<float> rowsOf(MetalBackend &backend, const Inkling2VisionGraphResources &resources,
                          uint32_t tiles, double *gpuSeconds) {
  CommandGraph graph;
  resources.encode(graph);
  *gpuSeconds = backend.submitCommandAsync(graph.dispatches()).wait().gpuSeconds;
  const auto *words = static_cast<const uint16_t *>(resources.output().contents());
  std::vector<float> rows(size_t{tiles} * 2304);
  for (size_t index = 0; index < rows.size(); ++index) {
    const float value = bf16ToFloat(words[index]);
    require(std::isfinite(value), "non-finite parity output");
    rows[index] = value;
  }
  return rows;
}

void report(const std::string &name, const std::vector<float> &rows, double gpuSeconds) {
  double squares = 0.0, scale = 0.0;
  for (float value : rows) {
    squares += double(value) * value;
    scale = std::max(scale, std::fabs(double(value)));
  }
  std::cout << "grid " << name << ": rows=" << rows.size() / 2304
            << " rms=" << std::sqrt(squares / double(rows.size()))
            << " max_abs=" << scale << " gpu_ms=" << gpuSeconds * 1e3 << '\n';
}

} // namespace

int main(int argc, char **argv) {
  const bool safetensors = argc == 5 && std::string_view(argv[4]) == "--safetensors";
  if ((argc != 4 && !safetensors) || (argc == 5 && !safetensors)) {
    std::cerr << "usage: inkling2-vision-parity METALLIB SOURCE OUT_DIR [--safetensors]\n";
    return 2;
  }
  @autoreleasepool {
    try {
      const std::filesystem::path output = argv[3];
      std::filesystem::create_directories(output);
      MetalBackend backend(argv[1]);
      splash::model::Inkling2VisionLoader loader(
          argv[2], safetensors ? splash::model::Inkling2VisionSourceFormat::Safetensors
                               : splash::model::Inkling2VisionSourceFormat::Gguf);
      std::cout << "image " << loader.image().bytes << " bytes for "
                << splash::model::Inkling2VisionLoader::tensorCount() << " tensors\n";
      splash::model::WeightImages images(backend, "inkling2 vision parity");
      const Inkling2VisionBufferWeights weights = loader.load(images);
      require(weights.blocks.size() == 23, "the projector bound another layer count");
      for (const Grid grid : {Grid{1, 1}, Grid{1, 2}, Grid{2, 2}}) {
        const std::vector<float> pixels = pixelsFor(grid);
        MetalBuffer input = backend.allocateBuffer(uint64_t(pixels.size()) * sizeof(float),
                                                   BufferStorage::Shared, "parity pixels");
        std::memcpy(input.contents(), pixels.data(), pixels.size() * sizeof(float));
        const Inkling2VisionGraphResources resources(backend, weights, input, grid.height,
                                                     grid.width);
        std::cout << "grid " << grid.name() << ": tokens=" << 1 + 4 * grid.tiles()
                  << " scratch=" << resources.scratchBytes() << " bytes\n";
        double gpuSeconds = 0.0;
        const std::vector<float> rows = rowsOf(backend, resources, grid.tiles(), &gpuSeconds);
        // A second execution on the same arena: the graph must not depend on
        // what an earlier image left behind in its scratch.
        double repeatedSeconds = 0.0;
        const std::vector<float> repeated =
            rowsOf(backend, resources, grid.tiles(), &repeatedSeconds);
        require(rows == repeated, "the vision graph is not deterministic");
        writeFloats(output / ("pixels_" + grid.name() + ".f32"), pixels);
        writeFloats(output / ("output_" + grid.name() + ".f32"), rows);
        report(grid.name(), rows, gpuSeconds);
      }
      std::cout << "inkling2_vision_parity_test: PASS\n";
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "inkling2_vision_parity_test: FAIL: " << error.what() << '\n';
      return 1;
    }
  }
}
