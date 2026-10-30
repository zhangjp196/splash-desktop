#pragma once

#include "metal/abi/PagedAttention.h"
#include "metal/kernels/common/kv_extent.h"
#include "metal/kernels/common/kv_paging.h"
#include <metal_stdlib>

using namespace metal;

constant uint SplashChunkMaximumRows = SPLASH_PREFILL_TOKEN_BUDGET;
constant uint SplashChunkMaximumPhysicalTokens =
    SPLASH_MAXIMUM_PHYSICAL_KV_TOKENS;

inline bool
splash_chunk_contract_valid(constant SplashChunkedPrefillParams &params) {
  uint required_pages = (params.committed_tokens + params.chunk_tokens +
                         SplashKvPageTokens - 1) /
                        SplashKvPageTokens;
  return params.committed_tokens <= SplashChunkMaximumPhysicalTokens &&
         params.chunk_tokens > 0 &&
         params.chunk_tokens <= SplashChunkMaximumRows &&
         params.committed_tokens + params.chunk_tokens <=
             SplashChunkMaximumPhysicalTokens &&
         params.chunk_stride >= params.chunk_tokens &&
         params.chunk_stride <= SplashChunkMaximumRows &&
         params.chunk_stride % SPLASH_TARGET_KV_BLOCK_TOKENS == 0 &&
         params.page_table_entries >= required_pages &&
         params.kv.extent_pages > 0;
}

template <uint HeadDim>
inline ulong splash_current_key_index(uint stride, uint head, uint token,
                                        uint dimension) {
  return (ulong(head) * stride + token) * HeadDim + dimension;
}

template <uint HeadDim>
inline ulong splash_current_value_index(uint stride, uint head, uint token,
                                          uint dimension) {
  return (ulong(head) * HeadDim + dimension) * stride + token;
}

// The scratch of a store entry's scale reduction, by CacheElement: a maximum
// per simdgroup of the HeadDim threads for INT8; BF16 copies the source bits
// and has none. The packed 4-bit format uses INT8's scratch; it reduces over
// the dimension pairs instead (paged_store_row.h).
#define SPLASH_STORE_MAXIMA_int8_t(HeadDim) threadgroup float maxima[(HeadDim) / 32];
#define SPLASH_STORE_MAXIMA_SplashKvInt4(HeadDim) threadgroup float maxima[(HeadDim) / 32];
#define SPLASH_STORE_MAXIMA_bfloat(HeadDim) threadgroup float *maxima = nullptr;

// One lane per dimension stores a current row in its final page slot.
// INT8 derives a per-row scale; BF16 copies the original bits; the packed
// 4-bit format derives the same per-row scale as INT8 and writes two
// dimensions' nibbles per byte. Slots are addressed inside the head's slab of
// the page: the page index functions at page zero and head zero.
template <uint KVHeads, uint HeadDim, typename CacheElement, bool ValuesTokenMajor = false>
__attribute__((always_inline)) inline void splash_store_kv_row(
    device const bfloat *chunk_keys, device const bfloat *chunk_values,
    device const SplashKvPage *page_table,
    constant SplashChunkedPrefillParams &params, threadgroup float *maxima,
    bool value_tensor, uint head, uint chunk_token, uint dimension,
    uint simd_lane, uint simd_group) {
  uint logical_token = params.committed_tokens + chunk_token;
  const SplashKvPage page = page_table[logical_token / SplashKvPageTokens];
  const SplashKvPageTensors<CacheElement> slab =
      SplashKvAddressing<KVHeads, HeadDim, CacheElement>(params.kv, head).page(page);
  uint page_token = logical_token % SplashKvPageTokens;
  ulong source_index =
      value_tensor ? splash_current_value_index<HeadDim>(params.chunk_stride, head,
                                                           chunk_token, dimension)
                   : splash_current_key_index<HeadDim>(params.chunk_stride, head,
                                                         chunk_token, dimension);
  bfloat source = value_tensor ? chunk_values[source_index] : chunk_keys[source_index];

  if constexpr (is_same<CacheElement, bfloat>::value) {
    if (value_tensor)
      slab.values[splash_kv_value_index<KVHeads, HeadDim, ValuesTokenMajor>(0, 0, page_token, dimension)] = source;
    else
      slab.keys[splash_kv_key_index<KVHeads, HeadDim>(0, 0, page_token, dimension)] = source;
    return;
  } else {

    float value = float(source);
    float local_maximum = simd_max(abs(value));
    if (simd_lane == 0)
      maxima[simd_group] = local_maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
      for (uint offset = HeadDim / 64; offset != 0; offset >>= 1) {
        if (simd_lane < offset)
          maxima[simd_lane] = max(maxima[simd_lane], maxima[simd_lane + offset]);
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum = maxima[0];
    // The packed format spends the same 7 signed bits INT8 does, so its
    // per-row scale is unchanged; only the step count of the rounding differs.
    constexpr int QuantizedMaximum = is_same<CacheElement, SplashKvInt4>::value ? 7 : 127;
    constexpr int QuantizedMinimum = -QuantizedMaximum;
    float scale = maximum == 0.0f ? 0.0f : maximum / float(QuantizedMaximum);
    int quantized = maximum == 0.0f
                        ? 0
                        : clamp(int(rint(value * float(QuantizedMaximum) / maximum)),
                                QuantizedMinimum, QuantizedMaximum);

    if constexpr (is_same<CacheElement, SplashKvInt4>::value) {
      // Two elements share a byte, and a byte packs a pair of *neighbouring
      // element indices*: the even index in the low nibble, the odd one in the
      // high nibble. That is how MPP's int4b_format operand reads a slab,
      // verified at runtime against a CPU reference.
      //
      // Which two (token, dimension) an element index pairs follows from the
      // slab's order, so it is worked out from the index itself rather than
      // assumed:
      //
      // - keys are token-major, element = (head * tokens + token) * D +
      //   dimension, so a byte pairs two dimensions of one token and both
      //   come from this threadgroup's own chunk rows;
      // - values are dimension-major by default, element = (head * D +
      //   dimension) * tokens + token, so a byte pairs two tokens of one
      //   dimension, and the partner token can be in another chunk.
      //
      // Only the thread owning the even index writes the byte, so no two
      // threads race for it; its partner's source is read here.
      const bool pairAlongDimension = !value_tensor || ValuesTokenMajor;
      // A value byte pairs two tokens of one dimension, and either token of the
      // pair can be the one this chunk ends on: the even token when the chunk
      // ends right after it (its partner opens the next chunk), the odd token
      // when it opens this chunk (its partner closed the previous one). A token
      // whose partner is in another chunk writes its own nibble, so no nibble
      // waits on a chunk that never returns for it. Two dimensions of one token
      // always meet in one chunk, as HeadDim is even.
      const bool partnerInChunk =
          pairAlongDimension ||
          (page_token % 2 == 0 ? chunk_token + 1 < params.chunk_tokens
                               : chunk_token != 0);
      const bool firstOfPair = pairAlongDimension ? (dimension % 2 == 0)
                                                  : (page_token % 2 == 0);
      const bool ownsByte = firstOfPair || !partnerInChunk;
      if (!ownsByte) {
        // The partner owns the byte; this thread only contributes its scale.
        if (dimension == 0) {
          device float *scales = value_tensor ? slab.value_scales : slab.key_scales;
          scales[splash_q8_scale_index<KVHeads>(0, 0, page_token)] = scale;
        }
        return;
      }
      float pairValue = 0.0F;
      bool pairPresent = false;
      if (pairAlongDimension) {
        const uint pairDimension = dimension + 1;
        pairPresent = pairDimension < HeadDim;
        if (pairPresent) {
          const ulong pairIndex =
              value_tensor
                  ? splash_current_value_index<HeadDim>(params.chunk_stride, head,
                                                        chunk_token, pairDimension)
                  : splash_current_key_index<HeadDim>(params.chunk_stride, head,
                                                      chunk_token, pairDimension);
          pairValue = float(value_tensor ? chunk_values[pairIndex] : chunk_keys[pairIndex]);
        }
      } else {
        // The partner token is the other one of the pair: the next token for an
        // even one, the previous for an odd one, and both sit in this page.
        pairPresent = partnerInChunk;
        if (pairPresent) {
          const uint pairToken = page_token % 2 == 0 ? chunk_token + 1 : chunk_token - 1;
          const ulong pairIndex = splash_current_value_index<HeadDim>(
              params.chunk_stride, head, pairToken, dimension);
          pairValue = float(chunk_values[pairIndex]);
        }
      }
      const int pairQuantized =
          !pairPresent || maximum == 0.0f
              ? 0
              : clamp(int(rint(pairValue * float(QuantizedMaximum) / maximum)),
                      QuantizedMinimum, QuantizedMaximum);
      // Bytes are read-modified-written: a chunk that ends between the two
      // elements of one byte leaves the other nibble to the chunk that stores
      // it, which must find its neighbour's bits intact. The even element's
      // nibble is the low one, the odd element's the high one.
      const ulong element =
          value_tensor
              ? splash_kv_value_index<KVHeads, HeadDim, ValuesTokenMajor>(
                    0, 0, page_token, dimension)
              : splash_kv_key_index<KVHeads, HeadDim>(0, 0, page_token, dimension);
      device uchar *byte =
          (value_tensor ? slab.values : slab.keys) + splash_kv_stored<CacheElement>(element);
      const uint own = uint(quantized) & 0xFu;
      const uint partner = uint(pairQuantized) & 0xFu;
      if (pairPresent)
        *byte = uchar(own | (partner << 4));
      else if (firstOfPair)
        *byte = uchar((*byte & 0xF0u) | own);
      else
        *byte = uchar((*byte & 0x0Fu) | (own << 4));
      if (dimension == 0) {
        device float *scales = value_tensor ? slab.value_scales : slab.key_scales;
        scales[splash_q8_scale_index<KVHeads>(0, 0, page_token)] = scale;
      }
      return;
    }

    if (value_tensor) {
      slab.values[splash_kv_value_index<KVHeads, HeadDim, ValuesTokenMajor>(0, 0, page_token, dimension)] =
          char(quantized);
      if (dimension == 0)
        slab.value_scales[splash_q8_scale_index<KVHeads>(0, 0, page_token)] = scale;
    } else {
      slab.keys[splash_kv_key_index<KVHeads, HeadDim>(0, 0, page_token, dimension)] =
          char(quantized);
      if (dimension == 0)
        slab.key_scales[splash_q8_scale_index<KVHeads>(0, 0, page_token)] = scale;
    }
  }
}
