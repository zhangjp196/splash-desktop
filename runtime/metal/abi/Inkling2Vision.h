#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct Inkling2VisionStemParams {
  uint32_t hidden_size;
  uint32_t patch_size;
  uint32_t tile_height;
  uint32_t tile_width;
};
static_assert(sizeof(Inkling2VisionStemParams) == 16,
              "Inkling2 vision stem parameters must be 16 bytes");

// One native Inkling2 block, with caller-owned row-major BF16 scratch.
struct Inkling2VisionBlockParams {
  uint32_t tokens;
  uint32_t padded_tokens;
  uint32_t hidden_size;
  uint32_t intermediate_size;
  uint32_t heads;
  uint32_t head_dim;
  uint32_t grid_height;
  uint32_t grid_width;
  uint32_t skip_cls_rope;
  float epsilon;
};
static_assert(sizeof(Inkling2VisionBlockParams) == 40,
              "Inkling2 vision block parameters must be 40 bytes");

// Axial frequencies are laid out as width then height, each repeated for
// adjacent interleaved pairs. A token index of zero is CLS.
struct Inkling2VisionRopeParams {
  uint32_t tokens;
  uint32_t padded_tokens;
  uint32_t hidden_size;
  uint32_t heads;
  uint32_t head_dim;
  uint32_t grid_height;
  uint32_t grid_width;
  uint32_t skip_cls_rope;
  float theta;
};
static_assert(sizeof(Inkling2VisionRopeParams) == 36,
              "Inkling2 vision RoPE parameters must be 36 bytes");

struct Inkling2VisionMergerParams {
  uint32_t tile_height;
  uint32_t tile_width;
  uint32_t padded_rows;
};
