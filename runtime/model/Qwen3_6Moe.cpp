#include "Qwen3_6Moe.hpp"
#include "model/QwenTargetLoader.hpp"

#include <utility>

namespace splash::model {

// Block images keep the routed experts of each projection as one tensor, the
// shared expert's as another, and the router and the shared-expert scalar
// gate in F32 (readMoeBlock).
Qwen3_6MoeWeights
loadQwen3_6MoeWeights(metal::MetalBackend &backend, Qwen3_6MoeLayout layout,
                      const QwenTargetFiles &files) {
  return loadQwenTarget<Qwen3_6MoeWeights>(
      backend, layout, files, [&](auto &loader, const BlockTargetFormat &format, const MoeExperts &experts) {
        Qwen3_6MoeWeights weights = readQwenTargetWeights<Qwen3_6MoeWeights>(
            layout, loader, format,
            [&](WeightFile &file, Qwen3_6MoeLayerWeights &layer, const BlockTargetFormat &, uint32_t index) {
              layer.ffn = readMoeBlock(file, layout, experts, index);
            });
        if (experts.store)
          for (uint32_t index = 0; index < weights.layers.size(); ++index)
            experts.store->attach(index, weights.layers[index].ffn);
        return weights;
      });
}

} // namespace splash::model
