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
// Writes one element of a packed 4-bit slab: the even element's nibble is
// the low one, the odd element's the high one. When the pair's other
// element is in this chunk its nibble is written too; otherwise only this
// element's nibble is replaced, leaving the other chunk's bits intact.
inline void splash_store_packed_pair(device uchar *slab, ulong element,
                                     int quantized, int pairQuantized,
                                     bool pairPresent, bool firstOfPair) {
  device uchar *byte = slab + splash_kv_stored<SplashKvInt4>(element);
  const uint own = uint(quantized) & 0xFu;
  const uint partner = uint(pairQuantized) & 0xFu;
  if (pairPresent)
    *byte = uchar(own | (partner << 4));
  else if (firstOfPair)
    *byte = uchar((*byte & 0xF0u) | own);
  else
    *byte = uchar((*byte & 0x0Fu) | (own << 4));
}

template <uint KVHeads, uint HeadDim, typename KeyElement,
          typename ValueElement, bool ValuesTokenMajor = false>
__attribute__((always_inline)) inline void splash_store_kv_row(
    device const bfloat *chunk_keys, device const bfloat *chunk_values,
    device const SplashKvPage *page_table,
    constant SplashChunkedPrefillParams &params, threadgroup float *maxima,
    bool value_tensor, uint head, uint chunk_token, uint dimension,
    uint simd_lane, uint simd_group) {
  uint logical_token = params.committed_tokens + chunk_token;
  const SplashKvPage page = page_table[logical_token / SplashKvPageTokens];
  const SplashKvPageTensors<KeyElement, ValueElement> slab =
      SplashKvAddressing<KVHeads, HeadDim, KeyElement, ValueElement>(params.kv, head).page(page);
  uint page_token = logical_token % SplashKvPageTokens;
  ulong source_index =
      value_tensor ? splash_current_value_index<HeadDim>(params.chunk_stride, head,
                                                           chunk_token, dimension)
                   : splash_current_key_index<HeadDim>(params.chunk_stride, head,
                                                         chunk_token, dimension);
  bfloat source = value_tensor ? chunk_values[source_index] : chunk_keys[source_index];

  if constexpr (is_same<KeyElement, bfloat>::value) {
    if (!value_tensor) {
      // BF16 keys: one element per byte, a copy without a scale.
      slab.keys[splash_kv_key_index<KVHeads, HeadDim>(0, 0, page_token, dimension)] = source;
      return;
    }
  }
  if constexpr (is_same<ValueElement, bfloat>::value) {
    if (value_tensor) {
      // BF16 values: one element per byte, a copy without a scale.
      slab.values[splash_kv_value_index<KVHeads, HeadDim, ValuesTokenMajor>(
          0, 0, page_token, dimension)] = source;
      return;
    }
  }

  // Quantize the row: one fp32 scale per (head, token), the same 7 signed
  // bits INT8 spends for the packed format.
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
  constexpr int KeyQuantizedMaximum = is_same<KeyElement, SplashKvInt4>::value ? 7 : 127;
  constexpr int ValueQuantizedMaximum =
      is_same<ValueElement, SplashKvInt4>::value ? 7 : 127;
  const int QuantizedMaximum = value_tensor ? ValueQuantizedMaximum : KeyQuantizedMaximum;
  const int QuantizedMinimum = -QuantizedMaximum;
  float scale = maximum == 0.0f ? 0.0f : maximum / float(QuantizedMaximum);
  int quantized = maximum == 0.0f
                      ? 0
                      : clamp(int(rint(value * float(QuantizedMaximum) / maximum)),
                              QuantizedMinimum, QuantizedMaximum);

  if (value_tensor) {
    if constexpr (is_same<ValueElement, SplashKvInt4>::value) {
      // Values are dimension-major unless the layout stores them token-major.
      // Dimension-major values pair two tokens of one dimension, so a byte's
      // partner can be in another chunk; token-major values pair two
      // dimensions of one token, which always meet in one chunk.
      const bool pairAlongDimension = ValuesTokenMajor;
      const bool partnerInChunk =
          pairAlongDimension ||
          (page_token % 2 == 0 ? chunk_token + 1 < params.chunk_tokens
                               : chunk_token != 0);
      const bool firstOfPair = pairAlongDimension ? (dimension % 2 == 0)
                                                  : (page_token % 2 == 0);
      const bool ownsByte = firstOfPair || !partnerInChunk;
      if (!ownsByte) {
        // The partner owns the byte; this thread only contributes its scale.
        if (dimension == 0)
          slab.value_scales[splash_q8_scale_index<KVHeads>(0, 0, page_token)] = scale;
        return;
      }
      float pairValue = 0.0F;
      bool pairPresent = false;
      if (pairAlongDimension) {
        const uint pairDimension = dimension + 1;
        pairPresent = pairDimension < HeadDim;
        if (pairPresent)
          pairValue = float(chunk_values[splash_current_value_index<HeadDim>(
              params.chunk_stride, head, chunk_token, pairDimension)]);
      } else {
        pairPresent = partnerInChunk;
        if (pairPresent) {
          const uint pairToken = page_token % 2 == 0 ? chunk_token + 1 : chunk_token - 1;
          pairValue = float(chunk_values[splash_current_value_index<HeadDim>(
              params.chunk_stride, head, pairToken, dimension)]);
        }
      }
      const int pairQuantized =
          !pairPresent || maximum == 0.0f
              ? 0
              : clamp(int(rint(pairValue * float(QuantizedMaximum) / maximum)),
                      QuantizedMinimum, QuantizedMaximum);
      const ulong element =
          splash_kv_value_index<KVHeads, HeadDim, ValuesTokenMajor>(
              0, 0, page_token, dimension);
      splash_store_packed_pair(slab.values, element, quantized, pairQuantized,
                               pairPresent, firstOfPair);
      if (dimension == 0)
        slab.value_scales[splash_q8_scale_index<KVHeads>(0, 0, page_token)] = scale;
    } else if constexpr (is_same<ValueElement, int8_t>::value) {
      // INT8 values: one element per byte, a scale per (head, token).
      slab.values[splash_kv_value_index<KVHeads, HeadDim, ValuesTokenMajor>(
          0, 0, page_token, dimension)] = char(quantized);
      if (dimension == 0)
        slab.value_scales[splash_q8_scale_index<KVHeads>(0, 0, page_token)] = scale;
    }
  } else {
    if constexpr (is_same<KeyElement, SplashKvInt4>::value) {
      // Keys are token-major, element = (head * tokens + token) * D +
      // dimension, so a byte pairs two dimensions of one token and both come
      // from this threadgroup's own chunk rows.
      const bool pairPresent = dimension + 1 < HeadDim;
      const bool firstOfPair = dimension % 2 == 0;
      float pairValue = 0.0F;
      if (pairPresent)
        pairValue = float(chunk_keys[splash_current_key_index<HeadDim>(
            params.chunk_stride, head, chunk_token, dimension + 1)]);
      const int pairQuantized =
          !pairPresent || maximum == 0.0f
              ? 0
              : clamp(int(rint(pairValue * float(QuantizedMaximum) / maximum)),
                      QuantizedMinimum, QuantizedMaximum);
      const ulong element =
          splash_kv_key_index<KVHeads, HeadDim>(0, 0, page_token, dimension);
      splash_store_packed_pair(slab.keys, element, quantized, pairQuantized,
                               pairPresent, firstOfPair);
      if (dimension == 0)
        slab.key_scales[splash_q8_scale_index<KVHeads>(0, 0, page_token)] = scale;
    } else if constexpr (is_same<KeyElement, int8_t>::value) {
      // INT8 keys: one element per byte, a scale per (head, token).
      slab.keys[splash_kv_key_index<KVHeads, HeadDim>(0, 0, page_token, dimension)] =
          char(quantized);
      if (dimension == 0)
        slab.key_scales[splash_q8_scale_index<KVHeads>(0, 0, page_token)] = scale;
    }
  }
}
