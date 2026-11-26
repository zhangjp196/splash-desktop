#pragma once

// Additive Inkling-2 attention ABI carried by the ported Metal kernels.
//
// The fork's `PagedAttention.h` owns the existing four-lane prefill/decode
// attention ABI and is used by the shipped kernels; it cannot be replaced.
// This header adds the upstream 1.7.2 structs that the ported
// `relative_attention` kernels need, without touching those existing types.
// They eventually come from the L4 framework port (ops::PagedAttention).

#include "metal/abi/KvExtent.h"
#include "metal/abi/VerifyTree.h"

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#define SPLASH_ATTENTION_TILE_ROWS 8u
#define SPLASH_ATTENTION_MAXIMUM_SPLITS 128u

#define SPLASH_ATTENTION_TILE_PARTS(group) \
  ((group) <= 8u ? 1u : (group) <= 16u ? 2u : (group) <= 32u ? 4u : 8u)

// A sequence of an attention dispatch: at most row_bound rows, positions from
// committed_tokens, layout entry first_sequence + s; staged from stage_row on
// of each staging, split into first_slot-based partial slots.
struct SplashAttentionSequence {
  uint32_t committed_tokens;
  uint32_t row_bound;
  uint32_t layout_row;
  uint32_t stage_row;
  uint32_t chunk_stride;
  uint32_t page_table_entries;
  uint32_t split_count;
  uint32_t first_tile;
  uint32_t first_slot;
  uint32_t block_stride;
  uint32_t block_list;
};
static_assert(sizeof(SplashAttentionSequence) == 44,
              "Inkling-2 attention sequences are 44 bytes on both sides");

struct SplashAttentionParams {
  uint32_t sequences;
  uint32_t first_sequence;
  uint32_t rows;
  uint32_t tiles;
  uint32_t trees;
  uint32_t last_rows;
  SplashKvLayer kv;
  SplashAttentionSequence sequence[SPLASH_MAXIMUM_BATCH_WIDTH];
};
static_assert(sizeof(SplashAttentionParams) == 32 + 44 * SPLASH_MAXIMUM_BATCH_WIDTH,
              "Inkling-2 attention parameters are 384 bytes on both sides");

struct SplashKvPathCommitParams {
  uint32_t extent_pages;
  uint32_t regions;
  uint32_t committed_tokens[SPLASH_MAXIMUM_BATCH_WIDTH];
};
static_assert(sizeof(SplashKvPathCommitParams) == 8 + 4 * SPLASH_MAXIMUM_BATCH_WIDTH,
              "Inkling-2 path commit parameters are 24 bytes on both sides");
