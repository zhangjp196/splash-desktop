#include "ops/Inkling2VisionBlock.hpp"
#include "metal/abi/Inkling2Vision.h"
#include "metal/abi/Vision.h"
#include <cmath>
#include <stdexcept>

namespace splash::ops {
namespace {
using metal::MetalBuffer;
constexpr uint32_t kHidden=1024, kIntermediate=4096, kHeads=16, kHeadDim=64;
void addGemm(metal::CommandGraph &g, const char *name, const MetalBuffer &in,
             const MetalBuffer &w, const MetalBuffer &b, const MetalBuffer &out,
             uint32_t n, uint32_t k, uint32_t rows) {
  g.add(name,{in,w,b,out,out},VisionGemmParams{n,k},{(rows+63)/64,n/128,1});
}
}
void Inkling2VisionBlock::encode(metal::CommandGraph &g,
  const Inkling2VisionBlockBuffers &b,uint32_t tokens,uint32_t padded,
  uint32_t tileHeight,uint32_t tileWidth,bool skipClsRope) {
  const uint64_t gh64 = uint64_t(tileHeight) * 2, gw64 = uint64_t(tileWidth) * 2;
  if (gh64 > UINT32_MAX || gw64 > UINT32_MAX)
    throw std::invalid_argument("invalid Inkling2 vision tile grid");
  const uint32_t gh = uint32_t(gh64), gw = uint32_t(gw64);
  if (!tokens || !padded || padded%128 || tokens>padded || !gh || !gw ||
      uint64_t(gh)*gw + (skipClsRope?1u:0u) != tokens)
    throw std::invalid_argument("invalid Inkling2 vision block token geometry");
  const MetalBuffer *all[]={&b.residual,&b.normalized,&b.query,&b.key,&b.value,
    &b.queries,&b.keys,&b.values,&b.attention,&b.context,&b.projected,&b.normalized2,
    &b.intermediate,&b.down,&b.norm1Weight,&b.norm1Bias,&b.queryWeight,&b.queryBias,
    &b.keyWeight,&b.keyBias,&b.valueWeight,&b.valueBias,&b.outputWeight,&b.outputBias,
    &b.scale1,&b.norm2Weight,&b.norm2Bias,&b.upWeight,&b.upBias,&b.downWeight,&b.downBias,&b.scale2};
  for (auto *buffer:all) if (!*buffer) throw std::invalid_argument("unbound Inkling2 vision block buffer");
  // These buffers are simultaneously live at various points in the block;
  // require disjoint views so a partial alias cannot silently corrupt a graph.
  const MetalBuffer *scratch[] = {&b.residual, &b.normalized, &b.query, &b.key, &b.value,
    &b.queries, &b.keys, &b.values, &b.attention, &b.context, &b.projected,
    &b.normalized2, &b.intermediate, &b.down};
  for (size_t i = 0; i < std::size(scratch); ++i)
    for (size_t j = i + 1; j < std::size(scratch); ++j)
      if (scratch[i]->overlaps(*scratch[j]))
        throw std::invalid_argument("overlapping Inkling2 vision block scratch");
  const uint64_t rows64 = (uint64_t(tokens) + 63) / 64 * 64;
  const auto extent = [](const MetalBuffer &buffer, uint64_t bytes) {
    if (buffer.sizeBytes() < bytes)
      throw std::invalid_argument("undersized Inkling2 vision block buffer");
  };
  extent(b.residual, uint64_t(tokens) * kHidden * sizeof(float));
  for (const MetalBuffer *buffer : {&b.normalized, &b.query, &b.key, &b.value,
                                   &b.context, &b.projected, &b.normalized2, &b.down})
    extent(*buffer, rows64 * kHidden * 2);
  extent(b.intermediate, rows64 * kIntermediate * 2);
  for (const MetalBuffer *buffer : {&b.queries, &b.keys, &b.values, &b.attention})
    extent(*buffer, uint64_t(padded) * kHidden * 2);
  for (const MetalBuffer *buffer : {&b.norm1Weight, &b.norm1Bias, &b.queryBias,
                                   &b.keyBias, &b.valueBias, &b.outputBias,
                                   &b.scale1, &b.norm2Weight, &b.norm2Bias,
                                   &b.downBias, &b.scale2})
    extent(*buffer, kHidden * 2);
  extent(b.upBias, kIntermediate * 2);
  for (const MetalBuffer *buffer : {&b.queryWeight, &b.keyWeight, &b.valueWeight, &b.outputWeight})
    extent(*buffer, uint64_t(kHidden) * kHidden * 2);
  for (const MetalBuffer *buffer : {&b.upWeight, &b.downWeight})
    extent(*buffer, uint64_t(kHidden) * kIntermediate * 2);
  Inkling2VisionBlockParams p{tokens,padded,kHidden,kIntermediate,kHeads,kHeadDim,gh,gw,skipClsRope?1u:0u,1e-5f};
  g.add("inkling2_vision_layer_norm",{b.residual,b.norm1Weight,b.norm1Bias,b.normalized},p,{tokens,1,1});
  addGemm(g,"vision_gemm_m64n128",b.normalized,b.queryWeight,b.queryBias,b.query,kHidden,kHidden,tokens);
  addGemm(g,"vision_gemm_m64n128",b.normalized,b.keyWeight,b.keyBias,b.key,kHidden,kHidden,tokens);
  addGemm(g,"vision_gemm_m64n128",b.normalized,b.valueWeight,b.valueBias,b.value,kHidden,kHidden,tokens);
  Inkling2VisionRopeParams rp{tokens,padded,kHidden,kHeads,kHeadDim,gh,gw,skipClsRope?1u:0u,10000.0f};
  g.add("inkling2_vision_axial_qkv",{b.query,b.key,b.value,b.queries,b.keys,b.values},rp,{padded,1,1});
  VisionAttentionParams ap{tokens,padded,1.0f/std::sqrt(float(kHeadDim))};
  g.add("vision_attention_h64",{b.queries,b.keys,b.values,b.attention},ap,
        {(tokens+SPLASH_VISION_ATTENTION_ROWS-1)/SPLASH_VISION_ATTENTION_ROWS,kHeads,1},
        {SPLASH_VISION_ATTENTION_THREADS,1,1});
  VisionQkvParams qp{tokens,padded,kHidden,kHeadDim,kHeadDim,kHeads};
  g.add("vision_attention_gather",{b.attention,b.context},qp,{tokens,1,1});
  addGemm(g,"vision_gemm_m64n128",b.context,b.outputWeight,b.outputBias,b.projected,kHidden,kHidden,tokens);
  g.add("inkling2_vision_scaled_residual",{b.residual,b.projected,b.scale1,b.residual},p,{tokens,1,1});
  g.add("inkling2_vision_layer_norm",{b.residual,b.norm2Weight,b.norm2Bias,b.normalized2},p,{tokens,1,1});
  g.add("vision_gemm_m32n256_gelu_erf",
        {b.normalized2,b.upWeight,b.upBias,b.intermediate,b.intermediate},
        VisionGemmParams{kIntermediate,kHidden},
        {(tokens+31)/32,kIntermediate/256,1});
  addGemm(g,"vision_gemm_m64n128",b.intermediate,b.downWeight,b.downBias,b.down,kHidden,kIntermediate,tokens);
  g.add("inkling2_vision_scaled_residual",{b.residual,b.down,b.scale2,b.residual},p,{tokens,1,1});
}
} // namespace splash::ops
