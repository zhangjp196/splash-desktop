#pragma once

#include "metal/abi/KvExtent.h"
#include "metal/kernels/common/kv_paging.h"
#include <metal_stdlib>

using namespace metal;

// Kernels reach a KV page through its entry (abi/KvExtent.h): the GPU address
// of the extent that holds it, with the page's index in the low bits. Every
// kernel that touches KV pages forms the pointers to one KV head's slab of
// each of a page's tensors in one layer the same way, with
// SplashKvAddressing: splash_kv_offset places the layer's regions in the
// extent, whose 64-bit offsets it adds to the extent's address, and the page
// index functions of kv_paging.h place the page and the head in each region.

inline uint splash_kv_page_index(SplashKvPage page) {
  return uint(page) & SPLASH_KV_PAGE_INDEX_MASK;
}

inline device uchar *splash_kv_extent(SplashKvPage page, uint index) {
  return reinterpret_cast<device uchar *>(page - index);
}

// One page's bytes of keys (or values) and of their scales in one layer.
template <uint KVHeads, uint HeadDim, typename CacheElement> struct SplashKvPageBytes {
  static constexpr constant bool Quantized =
      is_same<CacheElement, int8_t>::value || is_same<CacheElement, SplashKvInt4>::value;
  static constexpr constant uint Data =
      uint(KVHeads * SplashKvPageTokens * HeadDim) * SplashKvBits<CacheElement>::Value / 8;
  static constexpr constant uint Scale =
      Quantized ? KVHeads * SplashKvPageTokens * sizeof(float) : 0;
};

// What a slab's pointer addresses: bytes for the packed 4-bit format, the
// element itself otherwise. Only the packed one is read through an MPP tensor.
template <typename CacheElement> struct SplashKvStorage {
  using Type = CacheElement;
};
template <> struct SplashKvStorage<SplashKvInt4> {
  using Type = uchar;
};

// One KV head's slab of each tensor of a page; BF16 has no scales. With the
// 4-bit element the pointers address bytes: two elements share one, so the
// slab is read through a tensor<device int4b_format, ...> built from them
// (paged_attention_tile.h), never indexed element by element.
template <typename CacheElement> struct SplashKvPageTensors {
  device typename SplashKvStorage<CacheElement>::Type *keys;
  device typename SplashKvStorage<CacheElement>::Type *values;
  device float *key_scales;
  device float *value_scales;
};

template <uint KVHeads, uint HeadDim, typename CacheElement> struct SplashKvAddressing {
  using Bytes = SplashKvPageBytes<KVHeads, HeadDim, CacheElement>;
  uint layer_offset;
  uint head;
  ulong key_scales_offset;
  ulong values_offset;
  ulong value_scales_offset;

  SplashKvAddressing(SplashKvLayer kv, uint kv_head)
      : layer_offset(kv.offset), head(kv_head),
        key_scales_offset(splash_kv_offset(kv.extent_pages, Bytes::Data, Bytes::Scale,
                                           0, SPLASH_KV_KEY_SCALES, 0)),
        values_offset(splash_kv_offset(kv.extent_pages, Bytes::Data, Bytes::Scale,
                                       0, SPLASH_KV_VALUES, 0)),
        value_scales_offset(splash_kv_offset(kv.extent_pages, Bytes::Data,
                                             Bytes::Scale, 0,
                                             SPLASH_KV_VALUE_SCALES, 0)) {}

  SplashKvPageTensors<CacheElement> page(SplashKvPage entry) const {
    const uint index = splash_kv_page_index(entry);
    device uchar *region = splash_kv_extent(entry, index) + layer_offset;
    using Stored = typename SplashKvStorage<CacheElement>::Type;
    SplashKvPageTensors<CacheElement> tensors{};
    // The packed 4-bit slab advances by raw bytes (its stored type is uchar),
    // the others by elements; splash_kv_stored gives each the right step.
    tensors.keys = reinterpret_cast<device Stored *>(region) +
                   splash_kv_stored<CacheElement>(
                       splash_kv_key_index<KVHeads, HeadDim>(index, head, 0, 0));
    tensors.values = reinterpret_cast<device Stored *>(region + values_offset) +
                     splash_kv_stored<CacheElement>(
                         splash_kv_value_index<KVHeads, HeadDim>(index, head, 0, 0));
    if constexpr (Bytes::Quantized) {
      const ulong scale_index = splash_q8_scale_index<KVHeads>(index, head, 0);
      tensors.key_scales =
          reinterpret_cast<device float *>(region + key_scales_offset) + scale_index;
      tensors.value_scales =
          reinterpret_cast<device float *>(region + value_scales_offset) + scale_index;
    }
    return tensors;
  }
};
