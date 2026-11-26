// Checks that Inkling2VisionLoader binds the 422-tensor tower from each
// naming a checkpoint may use, the release's and a transformers one's (an
// MLX conversion's), and refuses a tower with a tensor missing or unused.
//
// A checkpoint names the tower by its embeddings' patch embedding, and the
// blocks follow that naming: a release puts the backbone under `backbone.`
// and names its norms ln_1, its scales ls_1.gamma and its MLP c_fc; a
// transformers checkpoint drops `backbone.` and names them layernorm_before,
// lambda_1 and mlp.fc1. /Users/zhangjp/Documents/ai-models/mlx/Inkling-2-Mini-MLX-4bit
// is such a conversion, and the engine could not start on it when the loader
// knew only the release's.
#include "TestChecks.hpp"
#include "TestCheckpoint.hpp"
#include "TestFiles.hpp"
#include "model/Inkling2VisionLoader.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace splash::model;
using splash::test::rejects;
using splash::test::require;
using splash::test::SyntheticTensor;
using splash::test::writeSyntheticShard;

namespace {

constexpr uint32_t kHidden = 1024, kIntermediate = 4096, kLayers = 23, kPatch = 20, kSide = 32, kOutput = 2304;

// The tower's 422 tensors under one naming, as inkling2/vision.bin reads them
// but for the image's patch order: the patch embedding is a linear
// [1024, 1200] here, and written again by the loader.
std::vector<SyntheticTensor> towerTensors(bool release) {
  std::vector<SyntheticTensor> result;
  const auto add = [&](std::string name, std::vector<uint64_t> shape) {
    result.push_back({std::move(name), "BF16", std::move(shape)});
  };
  const auto affine = [&](const std::string &name, uint64_t rows, uint64_t columns) {
    add(name + ".weight", {rows, columns});
    add(name + ".bias", {rows});
  };
  const auto norm = [&](const std::string &name) {
    add(name + ".weight", {kHidden});
    add(name + ".bias", {kHidden});
  };
  const std::string tower = release ? "vision_tower.backbone." : "vision_tower.";
  add(tower + "embeddings.patch_embedding.weight", {kHidden, kPatch * kPatch * 3});
  add(tower + "embeddings.class_embedding", {kHidden});
  add(tower + "embeddings.cls_position_embedding", {kHidden});
  add(tower + "embeddings.position_embedding", {kSide * kSide, kHidden});
  norm(tower + (release ? "ln_pre" : "pre_layernorm"));
  for (uint32_t i = 0; i < kLayers; ++i) {
    const std::string layer = tower + "layers." + std::to_string(i) + ".";
    norm(layer + (release ? "ln_1" : "layernorm_before"));
    affine(layer + "self_attn.q_proj", kHidden, kHidden);
    affine(layer + "self_attn.k_proj", kHidden, kHidden);
    affine(layer + "self_attn.v_proj", kHidden, kHidden);
    affine(layer + "self_attn.out_proj", kHidden, kHidden);
    norm(layer + (release ? "ln_2" : "layernorm_after"));
    affine(layer + (release ? "mlp.c_fc" : "mlp.fc1"), kIntermediate, kHidden);
    affine(layer + (release ? "mlp.c_proj" : "mlp.fc2"), kHidden, kIntermediate);
    add(layer + (release ? "ls_1.gamma" : "lambda_1"), {kHidden});
    add(layer + (release ? "ls_2.gamma" : "lambda_2"), {kHidden});
  }
  // The projector is the tower's, under vision_tower. in both namings.
  add("vision_tower.proj_in.weight", {kIntermediate, 4 * kHidden});
  add("vision_tower.proj_out.weight", {kOutput, kIntermediate});
  return result;
}

uint64_t plannedBytes(const std::filesystem::path &directory) {
  const Inkling2VisionLoader loader(directory, Inkling2VisionSourceFormat::Safetensors);
  return loader.image().bytes;
}

} // namespace

int main() {
  try {
    const splash::test::TemporaryDirectory temporary("inkling2-vision-inventory");
    const std::filesystem::path root = temporary.path();

    // A release's naming and a transformers one's plan the same image: the
    // loader's names are the image's, whatever the checkpoint calls them.
    const std::filesystem::path release = root / "release", converted = root / "converted";
    writeSyntheticShard(release / "model.safetensors", towerTensors(true));
    writeSyntheticShard(converted / "model.safetensors", towerTensors(false));
    const uint64_t releaseBytes = plannedBytes(release);
    const uint64_t convertedBytes = plannedBytes(converted);
    require(releaseBytes == convertedBytes, "the two namings plan different images");
    require(releaseBytes > 0, "the tower planned an empty image");

    // A tensor the tower needs, under neither naming, is refused by name.
    const std::filesystem::path missing = root / "missing";
    std::vector<SyntheticTensor> tensors = towerTensors(false);
    tensors.erase(tensors.begin());  // the patch embedding
    writeSyntheticShard(missing / "model.safetensors", tensors);
    rejects([&] { (void)plannedBytes(missing); }, "missing source tensor",
            "a tower without its patch embedding was accepted");

    // A naming the loader does not know is refused by the name it sought,
    // not by an image written from the wrong tensors.
    const std::filesystem::path unknown = root / "unknown";
    std::vector<SyntheticTensor> renamed = towerTensors(false);
    for (SyntheticTensor &tensor : renamed)
      if (tensor.name.find("layernorm_before") != std::string::npos)
        tensor.name.replace(tensor.name.find("layernorm_before"), 16, "ln_1");
    writeSyntheticShard(unknown / "model.safetensors", renamed);
    rejects([&] { (void)plannedBytes(unknown); }, "missing source tensor",
            "a tower mixing two block namings was accepted");

    std::cout << "inkling2_vision_inventory_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "inkling2_vision_inventory_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
