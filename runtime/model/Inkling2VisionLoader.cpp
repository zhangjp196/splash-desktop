#include "model/Inkling2VisionLoader.hpp"

#include "model/WeightLayout.hpp"
#include "model/SafetensorsCheckpoint.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <set>
#include <vector>

namespace splash::model {
namespace {
constexpr std::string_view kMagic = "MDI20001";
constexpr std::string_view kComponent = "inkling2/vision.bin";
constexpr uint64_t kBf16Bytes = sizeof(uint16_t);
struct Entry {
  std::string name;
  const GgufTensor *ggufTensor = nullptr;
  const SourceTensor *safetensorsTensor = nullptr;
  uint64_t offset = 0;
  uint64_t storedBytes = 0;
  bool patchHwcToChw = false;
};

struct SafetensorsSpec {
  std::string imageName, sourceName;
  std::vector<uint64_t> shape;
  bool patchHwcToChw = false;
};

std::vector<SafetensorsSpec> safetensorsSpecs(const Inkling2VisionLayout &l) {
  std::vector<SafetensorsSpec> specs;
  const auto add = [&](std::string imageName, std::string sourceName, std::vector<uint64_t> shape,
                       bool patchHwcToChw = false) {
    specs.push_back({std::move(imageName), std::move(sourceName), std::move(shape), patchHwcToChw});
  };
  const auto affine = [&](const std::string &imageStem, const std::string &sourceStem,
                          std::vector<uint64_t> weightShape, uint64_t biasSize) {
    add(imageStem + ".weight", sourceStem + ".weight", std::move(weightShape));
    if (biasSize) add(imageStem + ".bias", sourceStem + ".bias", {biasSize});
  };
  add("v.patch_embd.weight", "vision_tower.backbone.embeddings.patch_embedding.weight",
      {l.hidden, uint64_t{l.patch} * l.patch * 3}, true);
  add("v.class_embd", "vision_tower.backbone.embeddings.class_embedding", {l.hidden});
  add("v.class_position_embd.weight", "vision_tower.backbone.embeddings.cls_position_embedding", {l.hidden});
  add("v.position_embd.weight", "vision_tower.backbone.embeddings.position_embedding",
      {uint64_t{l.positionSide} * l.positionSide, l.hidden});
  affine("v.pre_ln", "vision_tower.backbone.ln_pre", {l.hidden}, l.hidden);
  for (uint32_t i = 0; i < l.layers; ++i) {
    const std::string image = "v.blk." + std::to_string(i) + ".";
    const std::string source = "vision_tower.backbone.layers." + std::to_string(i) + ".";
    affine(image + "ln1", source + "ln_1", {l.hidden}, l.hidden);
    affine(image + "attn_q", source + "self_attn.q_proj", {l.hidden, l.hidden}, l.hidden);
    affine(image + "attn_k", source + "self_attn.k_proj", {l.hidden, l.hidden}, l.hidden);
    affine(image + "attn_v", source + "self_attn.v_proj", {l.hidden, l.hidden}, l.hidden);
    affine(image + "attn_out", source + "self_attn.out_proj", {l.hidden, l.hidden}, l.hidden);
    affine(image + "ln2", source + "ln_2", {l.hidden}, l.hidden);
    affine(image + "ffn_up", source + "mlp.c_fc", {l.intermediate, l.hidden}, l.intermediate);
    affine(image + "ffn_down", source + "mlp.c_proj", {l.hidden, l.intermediate}, l.hidden);
    add(image + "ls1.weight", source + "ls_1.gamma", {l.hidden});
    add(image + "ls2.weight", source + "ls_2.gamma", {l.hidden});
  }
  affine("mm.1", "vision_tower.proj_in", {l.intermediate, uint64_t{4} * l.hidden}, 0);
  affine("mm.2", "vision_tower.proj_out", {l.output, l.intermediate}, 0);
  return specs;
}

void addName(std::set<std::string, std::less<>> &names, const std::string &stem, bool bias) {
  names.insert(stem + ".weight");
  if (bias) names.insert(stem + ".bias");
}

std::set<std::string, std::less<>> inventory(const Inkling2VisionLayout &l,
                                             const GgufFile &gguf) {
  std::set<std::string, std::less<>> n{"v.patch_embd.weight", "v.class_embd",
      "v.class_position_embd.weight", "v.position_embd.weight", "v.pre_ln.weight",
      "v.pre_ln.bias"};
  for (uint32_t i = 0; i < l.layers; ++i) {
    const std::string p = "v.blk." + std::to_string(i) + ".";
    addName(n, p + "ln1", false);
    addName(n, p + "attn_q", true);
    addName(n, p + "attn_k", true);
    addName(n, p + "attn_v", true);
    addName(n, p + "attn_out", true);
    addName(n, p + "ln2", false);
    addName(n, p + "ffn_up", true);
    addName(n, p + "ffn_down", true);
    n.insert(p + "ls1.weight");
    n.insert(p + "ls2.weight");
  }
  addName(n, "mm.1", false);
  addName(n, "mm.2", false);
  // Inkling2's projector has no merger biases; include optional block affine
  // biases only when the source declares them.
  for (const auto &t : gguf.tensors())
    if (t.name.ends_with(".bias") && n.contains(t.name.substr(0, t.name.size() - 5) + ".weight"))
      n.insert(t.name);
  return n;
}

uint64_t storedBytes(const GgufTensor &tensor) {
  if (tensor.elements() > std::numeric_limits<uint64_t>::max() / kBf16Bytes)
    throw WeightStoreError("Inkling2 vision tensor size overflow: " + tensor.name);
  return tensor.elements() * kBf16Bytes;
}

void writeBf16(const SourceTensor &tensor, std::span<uint8_t> destination, bool patchHwcToChw,
               const Inkling2VisionLayout &layout) {
  if (tensor.dtype != "BF16" || tensor.bytes != destination.size())
    throw WeightStoreError("Inkling2 Safetensors tensor must be BF16 with its declared byte size");
  if (!patchHwcToChw) {
    tensor.read(0, destination);
    return;
  }
  const uint64_t features = uint64_t{layout.patch} * layout.patch * 3;
  if (tensor.shape != std::vector<uint64_t>{layout.hidden, features} ||
      destination.size() != uint64_t{layout.hidden} * features * kBf16Bytes)
    throw WeightStoreError("Inkling2 patch embedding cannot be reordered from HWC to CHW");
  std::vector<uint8_t> raw(destination.size());
  tensor.read(0, raw);
  for (uint64_t out = 0; out < layout.hidden; ++out)
    for (uint64_t y = 0; y < layout.patch; ++y)
      for (uint64_t x = 0; x < layout.patch; ++x)
        for (uint64_t channel = 0; channel < 3; ++channel) {
          const uint64_t src = (out * features + (y * layout.patch + x) * 3 + channel) * kBf16Bytes;
          const uint64_t dst = (out * features + channel * layout.patch * layout.patch +
                                y * layout.patch + x) * kBf16Bytes;
          std::memcpy(destination.data() + dst, raw.data() + src, kBf16Bytes);
        }
}

void writeBf16(const WeightSource &source, const GgufTensor &tensor,
               std::span<uint8_t> destination) {
  if (destination.size() != storedBytes(tensor))
    throw WeightStoreError("Inkling2 vision tensor output size mismatch: " + tensor.name);
  if (tensor.type == ggml::kBF16) {
    if (tensor.bytes != destination.size())
      throw WeightStoreError("Inkling2 BF16 tensor byte size mismatch: " + tensor.name);
    source.readData(tensor.offset, destination);
    return;
  }
  if (tensor.type != ggml::kF32 || tensor.bytes != tensor.elements() * sizeof(float))
    throw WeightStoreError("Inkling2 vision tensor must be BF16 or exactly representable F32: " + tensor.name);

  std::vector<uint8_t> raw(static_cast<size_t>(tensor.bytes));
  source.readData(tensor.offset, raw);
  for (uint64_t i = 0; i < tensor.elements(); ++i) {
    uint32_t bits;
    std::memcpy(&bits, raw.data() + i * sizeof(bits), sizeof(bits));
    if ((bits & 0xffffu) != 0)
      throw WeightStoreError("Inkling2 F32 tensor is not exactly representable as BF16: " + tensor.name);
    const uint16_t bf16 = static_cast<uint16_t>(bits >> 16);
    std::memcpy(destination.data() + i * sizeof(bf16), &bf16, sizeof(bf16));
  }
}
} // namespace

struct Inkling2VisionLoader::Planned {
  std::unique_ptr<WeightSource> source;
  std::unique_ptr<GgufFile> gguf;
  std::unique_ptr<SafetensorsCheckpoint> safetensors;
  Inkling2VisionSourceFormat format = Inkling2VisionSourceFormat::Gguf;
  Inkling2VisionLayout layout;
  std::vector<Entry> entries;
  uint64_t bytes = 0;
  void check() const {
    if (gguf) source->checkUnchanged();
    else safetensors->checkUnchanged();
  }
};

Inkling2VisionLoader::Inkling2VisionLoader(const std::filesystem::path &path, Inkling2VisionLayout layout)
    : Inkling2VisionLoader(path, Inkling2VisionSourceFormat::Gguf, layout) {}

Inkling2VisionLoader::Inkling2VisionLoader(const std::filesystem::path &path, Inkling2VisionSourceFormat format,
                                           Inkling2VisionLayout layout) {
  auto p = std::make_shared<Planned>();
  p->layout = layout;
  p->format = format;
  uint64_t at = kWeightFileAlignment;
  if (format == Inkling2VisionSourceFormat::Gguf) {
    p->source = std::make_unique<WeightSource>(path);
    p->gguf = std::make_unique<GgufFile>(*p->source);
    (void)bindInkling2Vision(*p->gguf, layout);
    const auto names = inventory(layout, *p->gguf);
    if (names.size() != tensorCount() || p->gguf->tensors().size() != tensorCount())
      throw WeightStoreError("Inkling2 mmproj tensor inventory must contain exactly 422 tensors");
    for (const auto &t : p->gguf->tensors())
      if (!names.contains(t.name)) throw WeightStoreError("unexpected Inkling2 mmproj tensor: " + t.name);
    if (p->gguf->tensors().size() != names.size())
      throw WeightStoreError("missing Inkling2 mmproj tensor");

    for (const auto &name : names) {
      const auto &t = p->gguf->require(name);
      if (t.type != ggml::kBF16 && t.type != ggml::kF32)
        throw WeightStoreError("unsupported Inkling2 mmproj tensor type: " + name);
      const uint64_t bytes = storedBytes(t);
      at = alignWeightOffset(at);
      p->entries.push_back({name, &t, nullptr, at, bytes});
      if (bytes > UINT64_MAX - at) throw WeightStoreError("Inkling2 vision image size overflow");
      at += bytes;
    }
  } else if (format == Inkling2VisionSourceFormat::Safetensors) {
    p->safetensors = std::make_unique<SafetensorsCheckpoint>(path);
    auto specs = safetensorsSpecs(layout);
    std::sort(specs.begin(), specs.end(),
              [](const SafetensorsSpec &a, const SafetensorsSpec &b) { return a.imageName < b.imageName; });
    if (specs.size() != tensorCount())
      throw WeightStoreError("Inkling2 Safetensors vision inventory must contain exactly 422 tensors");
    for (const auto &spec : specs) {
      const auto &tensor = p->safetensors->require(spec.sourceName);
      if (tensor.dtype != "BF16" || tensor.shape != spec.shape)
        throw WeightStoreError("Inkling2 Safetensors tensor has an unsupported dtype or shape: " + spec.sourceName);
      uint64_t elements = 1;
      for (uint64_t extent : tensor.shape) {
        if (!extent || elements > UINT64_MAX / extent)
          throw WeightStoreError("Inkling2 Safetensors tensor size overflow: " + spec.sourceName);
        elements *= extent;
      }
      if (elements > UINT64_MAX / kBf16Bytes || tensor.bytes != elements * kBf16Bytes)
        throw WeightStoreError("Inkling2 Safetensors tensor byte size mismatch: " + spec.sourceName);
      const uint64_t bytes = tensor.bytes;
      at = alignWeightOffset(at);
      p->entries.push_back({spec.imageName, nullptr, &tensor, at, bytes, spec.patchHwcToChw});
      if (bytes > UINT64_MAX - at) throw WeightStoreError("Inkling2 vision image size overflow");
      at += bytes;
    }
  } else {
    throw WeightStoreError("unsupported Inkling2 vision source format");
  }
  std::sort(p->entries.begin(), p->entries.end(),
            [](const Entry &a, const Entry &b) { return a.name < b.name; });
  p->bytes = alignWeightOffset(at);
  p->check();
  planned_ = std::move(p);
}

Inkling2VisionLoader::~Inkling2VisionLoader() = default;

ImagePlan Inkling2VisionLoader::image() const {
  auto p = planned_;
  return {std::string(kComponent), std::string(kMagic), 0, 0, p->bytes,
      [p](std::span<uint8_t> dst, const metal::MetalBuffer &) {
        if (dst.size() != p->bytes) throw WeightStoreError("Inkling2 vision image size mismatch");
        std::fill(dst.begin(), dst.end(), uint8_t{0});
        auto header = weightFileHeader(kMagic, 0, 0);
        std::memcpy(dst.data(), header.data(), header.size());
        for (const auto &e : p->entries) {
          if (e.offset > dst.size() || e.storedBytes > dst.size() - e.offset)
            throw WeightStoreError("Inkling2 tensor outside image");
          auto tensorBytes = dst.subspan(e.offset, e.storedBytes);
          if (e.ggufTensor) writeBf16(*p->source, *e.ggufTensor, tensorBytes);
          else writeBf16(*e.safetensorsTensor, tensorBytes, e.patchHwcToChw, p->layout);
        }
        p->check();
      }};
}

std::vector<std::pair<std::string, metal::MetalBuffer>>
Inkling2VisionLoader::bind(WeightFile &file) const {
  std::vector<std::pair<std::string, metal::MetalBuffer>> result;
  result.reserve(planned_->entries.size());
  for (const auto &e : planned_->entries)
    result.emplace_back(e.name, file.section(e.storedBytes, e.name));
  file.finish();
  return result;
}

Inkling2VisionBufferWeights Inkling2VisionLoader::bindWeights(WeightFile &file) const {
  const auto sections = bind(file);
  const auto get = [&](const std::string &name, bool required = true) -> metal::MetalBuffer {
    const auto it = std::lower_bound(sections.begin(), sections.end(), name,
        [](const auto &entry, const auto &key) { return entry.first < key; });
    if (it != sections.end() && it->first == name) return it->second;
    if (required) throw WeightStoreError("missing Inkling2 vision section: " + name);
    return {};
  };
  const auto affine = [&](const std::string &stem) {
    return Inkling2VisionBufferTensor{get(stem + ".weight"), get(stem + ".bias", false)};
  };
  Inkling2VisionBufferWeights result;
  result.layout = planned_->layout;
  result.patchEmbedding = get("v.patch_embd.weight");
  result.classEmbedding = get("v.class_embd");
  result.classPositionEmbedding = get("v.class_position_embd.weight");
  result.positionEmbedding = get("v.position_embd.weight");
  result.preNorm = affine("v.pre_ln");
  result.blocks.reserve(result.layout.layers);
  for (uint32_t i = 0; i < result.layout.layers; ++i) {
    const std::string p = "v.blk." + std::to_string(i) + ".";
    result.blocks.push_back({affine(p + "ln1"), affine(p + "attn_q"), affine(p + "attn_k"),
        affine(p + "attn_v"), affine(p + "attn_out"), affine(p + "ln2"),
        affine(p + "ffn_up"), affine(p + "ffn_down"), get(p + "ls1.weight"), get(p + "ls2.weight")});
  }
  result.mergerUp = affine("mm.1");
  result.mergerDown = affine("mm.2");
  return result;
}

Inkling2VisionBufferWeights Inkling2VisionLoader::load(WeightImages &images) const {
  const auto plan = image();
  auto file = images.load(plan);
  auto weights = bindWeights(file);
  weights.files.push_back(file.record());
  return weights;
}

} // namespace splash::model

namespace splash::model {

uint64_t Inkling2VisionGraphResources::requiredScratchBytes(uint32_t tileHeight, uint32_t tileWidth) {
  const uint64_t tiles = uint64_t(tileHeight) * tileWidth;
  if (!tiles || tiles > 16384 || tileHeight > UINT32_MAX / 40 || tileWidth > UINT32_MAX / 40)
    throw std::invalid_argument("invalid Inkling2 vision tile grid");
  const uint64_t tokens = 1 + 4 * tiles;
  const uint64_t rows = (tokens + 63) / 64 * 64;
  const uint64_t padded = (tokens + 127) / 128 * 128;
  const uint64_t mergerRows = (tiles + 31) / 32 * 32;
  uint64_t total = 0;
  const auto add = [&](uint64_t size) { total = (total + 255) & ~uint64_t(255); total += size; };
  add(tiles * 4 * 1200 * 2);
  add(tokens * 1024 * sizeof(float));
  add(tokens * 1024 * sizeof(float));
  add(4096 * 2);
  add(mergerRows * 4096 * 2);
  add(mergerRows * 4096 * 2);
  add(mergerRows * 2304 * 2);
  for (int i = 0; i < 8; ++i) add(rows * 1024 * 2);
  for (int i = 0; i < 4; ++i) add(padded * 1024 * 2);
  add(rows * 4096 * 2);
  return total;
}

Inkling2VisionGraphResources::Inkling2VisionGraphResources(
    metal::MetalBackend &backend, const Inkling2VisionBufferWeights &w,
    metal::MetalBuffer pixels, uint32_t tileHeight, uint32_t tileWidth)
    : tileHeight_(tileHeight), tileWidth_(tileWidth) {
  const auto &l = w.layout;
  if (l.hidden != 1024 || l.intermediate != 4096 || l.layers != 23 ||
      l.heads != 16 || l.headDimension != 64 || l.patch != 20 ||
      l.positionSide != 32 || l.output != 2304 || w.blocks.size() != 23)
    throw std::invalid_argument("unsupported Inkling2 vision graph layout");
  const uint64_t tiles = uint64_t(tileHeight) * tileWidth;
  if (!tiles || tiles > (uint64_t(UINT32_MAX) / 4096 / 32) * 32 ||
      tileHeight > UINT32_MAX / 40 || tileWidth > UINT32_MAX / 40)
    throw std::invalid_argument("invalid Inkling2 vision tile grid");
  if (!pixels || pixels.sizeBytes() < tiles * 40 * 40 * 3 * sizeof(float))
    throw std::invalid_argument("undersized Inkling2 vision pixel buffer");
  if (w.mergerUp.bias || w.mergerDown.bias)
    throw std::invalid_argument("Inkling2 vision merger biases are unsupported");
  const uint64_t tokens = 1 + 4 * tiles;
  const uint64_t rows = (tokens + 63) / 64 * 64;
  const uint64_t padded = (tokens + 127) / 128 * 128;
  const uint64_t mergerRows = (tiles + 31) / 32 * 32;
  const auto multiply = [](uint64_t a, uint64_t b) {
    if (b && a > UINT64_MAX / b) throw std::overflow_error("Inkling2 vision scratch size overflow");
    return a * b;
  };
  struct Slice { metal::MetalBuffer *buffer; uint64_t offset, size; };
  std::vector<Slice> slices;
  uint64_t total = 0;
  const auto add = [&](metal::MetalBuffer &buffer, uint64_t count, uint64_t width) {
    const uint64_t size = multiply(count, width);
    if (total > UINT64_MAX - 255) throw std::overflow_error("Inkling2 vision scratch alignment overflow");
    total = (total + 255) & ~uint64_t(255);
    if (size > UINT64_MAX - total) throw std::overflow_error("Inkling2 vision scratch size overflow");
    slices.push_back({&buffer, total, size});
    total += size;
  };
  auto &b = buffers_;
  b.stem.pixels = std::move(pixels);
  add(b.stem.patches, tiles, 4 * 1200 * 2);
  add(b.stem.hidden, tokens, 1024 * sizeof(float));
  add(b.preNormalized, tokens, 1024 * sizeof(float));
  add(b.zeroBias, 4096, 2);
  add(b.mergerInput, mergerRows, 4096 * 2);
  add(b.mergerUp, mergerRows, 4096 * 2);
  add(b.mergerDown, mergerRows, 2304 * 2);
  auto &s = b.blocks.front();
  for (auto *buffer : {&s.normalized, &s.query, &s.key, &s.value, &s.context,
                       &s.projected, &s.normalized2, &s.down}) add(*buffer, rows, 1024 * 2);
  for (auto *buffer : {&s.queries, &s.keys, &s.values, &s.attention}) add(*buffer, padded, 1024 * 2);
  add(s.intermediate, rows, 4096 * 2);
  if (total != requiredScratchBytes(tileHeight, tileWidth))
    throw std::logic_error("Inkling2 vision scratch accounting mismatch");
  arena_ = backend.allocateBuffer(total, metal::BufferStorage::Shared, "Inkling2 vision scratch");
  if (!arena_ || !arena_.contents()) throw std::runtime_error("could not allocate Inkling2 vision scratch");
  std::memset(arena_.contents(), 0, static_cast<size_t>(total));
  for (const auto &slice : slices) *slice.buffer = backend.view(arena_, slice.offset, slice.size);
  b.stem.patchEmbedding = w.patchEmbedding;
  b.stem.classEmbedding = w.classEmbedding;
  b.stem.classPosition = w.classPositionEmbedding;
  b.stem.positions = w.positionEmbedding;
  b.preNormWeight = w.preNorm.weight;
  b.preNormBias = w.preNorm.bias;
  b.mergerUpWeight = w.mergerUp.weight;
  b.mergerDownWeight = w.mergerDown.weight;
  s.residual = b.preNormalized;
  for (size_t i = 0; i < b.blocks.size(); ++i) {
    if (i) b.blocks[i] = s;
    auto &dst = b.blocks[i];
    const auto &src = w.blocks[i];
    const auto bias = [&](const metal::MetalBuffer &value) { return value ? value : b.zeroBias; };
    dst.norm1Weight = src.norm1.weight; dst.norm1Bias = bias(src.norm1.bias);
    dst.queryWeight = src.query.weight; dst.queryBias = bias(src.query.bias);
    dst.keyWeight = src.key.weight; dst.keyBias = bias(src.key.bias);
    dst.valueWeight = src.value.weight; dst.valueBias = bias(src.value.bias);
    dst.outputWeight = src.attentionOutput.weight; dst.outputBias = bias(src.attentionOutput.bias);
    dst.scale1 = src.scale1;
    dst.norm2Weight = src.norm2.weight; dst.norm2Bias = bias(src.norm2.bias);
    dst.upWeight = src.up.weight; dst.upBias = bias(src.up.bias);
    dst.downWeight = src.down.weight; dst.downBias = bias(src.down.bias);
    dst.scale2 = src.scale2;
  }
}

void Inkling2VisionGraphResources::encode(metal::CommandGraph &graph) const {
  ops::Inkling2VisionPlan::encode(graph, buffers_, tileHeight_, tileWidth_);
}

} // namespace splash::model
