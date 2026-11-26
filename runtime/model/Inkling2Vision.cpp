#include "model/Inkling2Vision.hpp"

#include <stdexcept>
#include <string>

namespace splash::model {
namespace {

const GgufTensor &tensor(const GgufFile &file, const std::string &name,
                         std::initializer_list<uint64_t> shape) {
  const GgufTensor &found = file.require(name);
  if (found.dims != std::vector<uint64_t>(shape))
    throw GgufError("Inkling2 vision tensor shape mismatch: " + name);
  return found;
}

Inkling2VisionTensor affine(const GgufFile &file, const std::string &stem,
                            uint32_t input, uint32_t output) {
  const auto &weight = tensor(file, stem + ".weight", {input, output});
  const GgufTensor *bias = nullptr;
  if (const auto *candidate = file.find(stem + ".bias")) {
    if (candidate->dims != std::vector<uint64_t>{output})
      throw GgufError("Inkling2 vision bias shape mismatch: " + stem);
    bias = candidate;
  }
  return {&weight, bias};
}

void requireMetadata(const GgufFile &file, const Inkling2VisionLayout &l) {
  const auto &m = file.metadata();
  if (m.architecture() != "clip" || m.stringValue("clip.vision.projector_type") != "inkling2v")
    throw GgufError("unsupported Inkling2 vision GGUF architecture");
  const auto require = [&](const char *key, uint64_t value) {
    if (m.unsignedValue(key) != value)
      throw GgufError(std::string("Inkling2 vision metadata mismatch: ") + key);
  };
  require("clip.vision.embedding_length", l.hidden);
  require("clip.vision.feed_forward_length", l.intermediate);
  require("clip.vision.block_count", l.layers);
  require("clip.vision.attention.head_count", l.heads);
  require("clip.vision.patch_size", l.patch);
  require("clip.vision.image_size", l.patch * l.positionSide);
  require("clip.vision.projection_dim", l.output);
}

} // namespace

Inkling2VisionWeights bindInkling2Vision(const GgufFile &gguf, Inkling2VisionLayout l) {
  if (!l.hidden || !l.intermediate || !l.layers || !l.heads || !l.headDimension ||
      l.heads * l.headDimension != l.hidden || l.patch != 20 || l.positionSide != 32 ||
      l.output != 2304)
    throw GgufError("invalid Inkling2 vision layout");
  requireMetadata(gguf, l);
  Inkling2VisionWeights result;
  result.layout = l;
  result.patchEmbedding.weight = &tensor(gguf, "v.patch_embd.weight", {l.patch, l.patch, 3, l.hidden});
  result.classEmbedding = &tensor(gguf, "v.class_embd", {l.hidden});
  result.classPositionEmbedding = &tensor(gguf, "v.class_position_embd.weight", {l.hidden});
  result.positionEmbedding = &tensor(gguf, "v.position_embd.weight", {l.hidden, l.positionSide * l.positionSide});
  result.preNorm = {&tensor(gguf, "v.pre_ln.weight", {l.hidden}),
                    &tensor(gguf, "v.pre_ln.bias", {l.hidden})};
  result.blocks.reserve(l.layers);
  for (uint32_t i = 0; i < l.layers; ++i) {
    const std::string p = "v.blk." + std::to_string(i) + ".";
    Inkling2VisionBlockWeights b;
    b.norm1 = {&tensor(gguf, p + "ln1.weight", {l.hidden}), &tensor(gguf, p + "ln1.bias", {l.hidden})};
    b.query = affine(gguf, p + "attn_q", l.hidden, l.hidden);
    b.key = affine(gguf, p + "attn_k", l.hidden, l.hidden);
    b.value = affine(gguf, p + "attn_v", l.hidden, l.hidden);
    b.attentionOutput = affine(gguf, p + "attn_out", l.hidden, l.hidden);
    b.norm2 = {&tensor(gguf, p + "ln2.weight", {l.hidden}), &tensor(gguf, p + "ln2.bias", {l.hidden})};
    b.up = affine(gguf, p + "ffn_up", l.hidden, l.intermediate);
    b.down = affine(gguf, p + "ffn_down", l.intermediate, l.hidden);
    b.scale1 = &tensor(gguf, p + "ls1.weight", {l.hidden});
    b.scale2 = &tensor(gguf, p + "ls2.weight", {l.hidden});
    result.blocks.push_back(b);
  }
  result.mergerUp = affine(gguf, "mm.1", 4 * l.hidden, 4 * l.hidden);
  result.mergerDown = affine(gguf, "mm.2", 4 * l.hidden, l.output);
  return result;
}

std::array<float, 3 * 20 * 20 * 4> inkling2VisionPatchify40(std::span<const float> rgbTile) {
  constexpr size_t extent = 40, patch = 20, channels = 3;
  if (rgbTile.size() != extent * extent * channels)
    throw std::invalid_argument("Inkling2 vision tile must be 40x40 RGB");
  std::array<float, channels * patch * patch * 4> result{};
  size_t out = 0;
  for (size_t py = 0; py < 2; ++py)
    for (size_t px = 0; px < 2; ++px)
      for (size_t y = 0; y < patch; ++y)
        for (size_t x = 0; x < patch; ++x)
          for (size_t c = 0; c < channels; ++c) {
            const size_t source = (((py * patch + y) * extent + px * patch + x) * channels) + c;
            result[out++] = rgbTile[source];
          }
  return result;
}

} // namespace splash::model
