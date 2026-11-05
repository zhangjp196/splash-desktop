#pragma once

// Where KV pages live, shared by the host and the kernels. A pool's pages sit
// in extents: ordinary shared Metal buffers that all hold the same number of
// pages, allocated when the pool grows; an empty one is released by the next
// reclaim pass between commands. With a block index, an extent holds its
// pages' index after every layer's K/V region, as its index plane, and an
// extent of pages that hold only their index (kv::PageForm::Index) holds only
// that plane. Kernels reach extents only through the page entries of a
// request's tables, never through a bound buffer: a page table's entries reach
// the pages' K/V, an index table's their index plane.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#include "metal/abi/ExecutionGeometry.h"

// One KV page as kernels address it: the GPU address of the extent that holds
// it, or of its index plane, 16 KiB-aligned, with the page's index in the
// extent in the low bits.
typedef uint64_t SplashKvPage;
#define SPLASH_KV_PAGE_INDEX_BITS 14u
#define SPLASH_KV_PAGE_INDEX_MASK ((1u << SPLASH_KV_PAGE_INDEX_BITS) - 1u)
// The K/V entry of a page that holds only its blocks' index (kv::PageForm::
// Index): no K/V to reach. A kernel that reads through it faults, loudly, on
// the null address; the target's dense-history rule keeps every read that
// goes through a page table off such pages.
#define SPLASH_KV_PAGE_ABSENT ((SplashKvPage)0)

// The dimension of the KV heads of Qwen3.5's pages, the default of the
// element functions below. A layout holds its own (kv::Layout::
// headDimension); a kernel takes its own as a template value, HeadDim
// (kernels/common/kv_paging.h).
#define SPLASH_KV_HEAD_DIMENSION 256u

// The tensors of a layer, in the order they sit in its region of an extent.
// BF16 has no scales: their bytes are zero.
#define SPLASH_KV_KEYS 0u
#define SPLASH_KV_KEY_SCALES 1u
#define SPLASH_KV_VALUES 2u
#define SPLASH_KV_VALUE_SCALES 3u

// The bytes one page holds of one tensor of one layer: the data run of keys
// and values, the scale run of whichever of them is quantized.
inline uint32_t splash_kv_tensor_bytes(uint32_t k_data_bytes, uint32_t k_scale_bytes,
                                       uint32_t v_data_bytes, uint32_t v_scale_bytes,
                                       uint32_t tensor) {
  switch (tensor) {
  case SPLASH_KV_KEYS: return k_data_bytes;
  case SPLASH_KV_KEY_SCALES: return k_scale_bytes;
  case SPLASH_KV_VALUES: return v_data_bytes;
  case SPLASH_KV_VALUE_SCALES: return v_scale_bytes;
  default: return 0;
  }
}

// Where one element of one KV head sits in a page's slab of a tensor, in
// elements, for `token` of the page's SPLASH_TARGET_KV_BLOCK_TOKENS, in heads
// of head_dimension dimensions: keys token-major, values dimension-major, and
// one scale per (head, token) for either of them.
inline uint64_t splash_kv_key_element(uint32_t head, uint32_t token,
                                      uint32_t dimension,
                                      uint32_t head_dimension = SPLASH_KV_HEAD_DIMENSION) {
  return (uint64_t(head) * SPLASH_TARGET_KV_BLOCK_TOKENS + token) *
             head_dimension +
         dimension;
}

inline uint64_t splash_kv_value_element(uint32_t head, uint32_t token,
                                        uint32_t dimension,
                                        uint32_t head_dimension = SPLASH_KV_HEAD_DIMENSION) {
  return (uint64_t(head) * head_dimension + dimension) *
             SPLASH_TARGET_KV_BLOCK_TOKENS +
         token;
}

inline uint64_t splash_kv_scale_element(uint32_t head, uint32_t token) {
  return uint64_t(head) * SPLASH_TARGET_KV_BLOCK_TOKENS + token;
}

// Where one tensor of one page of one layer sits in its extent, in bytes.
// Every extent of a pool holds extent_pages pages. Per attention layer a
// region holds the keys of every page, then their key scales, values and
// value scales; a region is k_data_bytes + k_scale_bytes + v_data_bytes +
// v_scale_bytes, the data and scale runs of each of keys and values in the
// order of the tensors above.
inline uint64_t splash_kv_offset(uint32_t extent_pages, uint32_t k_data_bytes,
                                 uint32_t k_scale_bytes, uint32_t v_data_bytes,
                                 uint32_t v_scale_bytes, uint32_t layer,
                                 uint32_t tensor, uint32_t index) {
  const uint64_t region =
      uint64_t(layer) * extent_pages *
      (uint64_t(k_data_bytes) + k_scale_bytes + v_data_bytes + v_scale_bytes);
  // The tensors' runs within a region, in the order of the tensors above.
  const uint32_t region_offset =
      tensor == SPLASH_KV_KEYS ? 0u
                               : tensor == SPLASH_KV_KEY_SCALES ? k_data_bytes
                               : tensor == SPLASH_KV_VALUES ? k_data_bytes + k_scale_bytes
                                                          : k_data_bytes + k_scale_bytes + v_data_bytes;
  return region + uint64_t(extent_pages) * region_offset +
         uint64_t(splash_kv_tensor_bytes(k_data_bytes, k_scale_bytes,
                                         v_data_bytes, v_scale_bytes, tensor)) *
             uint64_t(index);
}

// Where one tensor of one page of one layer sits in its extent, in bytes, for
// a layout that holds one data run and one scale run per layer.
inline uint64_t splash_kv_offset(uint32_t extent_pages, uint32_t data_bytes,
                                 uint32_t scale_bytes, uint32_t layer,
                                 uint32_t tensor, uint32_t index) {
  return splash_kv_offset(extent_pages, data_bytes, scale_bytes, data_bytes,
                          scale_bytes, layer, tensor, index);
}

// Where one attention layer sits in every extent of a pool: its region
// begins `offset` bytes into each extent.
struct SplashKvLayer {
  uint32_t extent_pages;
  uint32_t offset;
};

static_assert(sizeof(SplashKvLayer) == 8, "KV layer placement is 8 bytes on both sides");

// The placement of `layer` in a pool of extents of extent_pages pages, for
// the data and scale runs of one layer.
inline SplashKvLayer splash_kv_layer(uint32_t extent_pages, uint32_t data_bytes,
                                     uint32_t scale_bytes, uint32_t layer) {
  return {extent_pages, uint32_t(splash_kv_offset(extent_pages, data_bytes, scale_bytes,
                                                 layer, SPLASH_KV_KEYS, 0))};
}

// The placement of `layer` in a pool of extents of extent_pages pages, with
// the data and scale runs of keys and of values kept apart.
inline SplashKvLayer splash_kv_layer(uint32_t extent_pages, uint32_t k_data_bytes,
                                     uint32_t k_scale_bytes, uint32_t v_data_bytes,
                                     uint32_t v_scale_bytes, uint32_t layer) {
  return {extent_pages,
          uint32_t(splash_kv_offset(extent_pages, k_data_bytes, k_scale_bytes,
                                    v_data_bytes, v_scale_bytes, layer,
                                    SPLASH_KV_KEYS, 0))};
}

// The entry of the page at `index` of the extent whose GPU address is
// extent_address.
inline SplashKvPage splash_kv_page_entry(uint64_t extent_address, uint32_t index) {
  return extent_address | index;
}
