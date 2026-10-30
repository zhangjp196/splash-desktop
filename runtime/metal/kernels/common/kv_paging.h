#pragma once

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/KvExtent.h"
#include <metal_stdlib>

using namespace metal;

// KV page geometry of both formats, shared by the attention and store
// kernels. A page holds SplashKvPageTokens tokens of every KV head of HeadDim
// dimensions, a value each kernel is compiled for: keys token-major, values
// dimension-major, or token-major like the keys where a layout's are
// (ValuesTokenMajor, kv::Layout::valuesTokenMajor): a block of a head's
// tokens then holds its values contiguously. INT8 and INT4 add one fp32 scale
// per (KV head, token) for each tensor.
constant uint SplashKvPageTokens = SPLASH_TARGET_KV_BLOCK_TOKENS;

// The 4-bit KV element. It has no addressable storage of its own: two elements
// share a byte, the even element the low nibble, and kernels read a slab
// through a tensor<device int4b_format, ...> operand rather than a pointer.
// SplashKvBits makes every index below byte-accurate for it.
struct SplashKvInt4 {};

// Bits of one KV element: 4 for the packed format, the stored width otherwise.
template <typename CacheElement> struct SplashKvBits {
  static constexpr constant uint Value = sizeof(CacheElement) * 8;
};
template <> struct SplashKvBits<SplashKvInt4> {
  static constexpr constant uint Value = 4;
};

// The byte an element index addresses. INT8 and BF16 keep one element per
// byte or two; INT4 pairs neighbouring elements in one byte.
template <typename CacheElement> inline ulong splash_kv_byte(ulong element) {
  return element * SplashKvBits<CacheElement>::Value / 8;
}

// The offset in units of a slab's stored type for an element index: bytes for
// the packed 4-bit format (its stored type is uchar), elements otherwise, so
// that pointer arithmetic on SplashKvStorage<CacheElement>::Type stays right.
template <typename CacheElement> inline ulong splash_kv_stored(ulong element) {
  return element;
}
template <> inline ulong splash_kv_stored<SplashKvInt4>(ulong element) {
  return element / 2;
}

// The type an MPP tensor operand of a format's slab is declared with: the
// stored element itself, except for the packed 4-bit format, which has no
// addressable element of its own (SplashKvInt4).
template <typename CacheElement> struct SplashKvTensorElement {
  using Type = CacheElement;
};
template <> struct SplashKvTensorElement<SplashKvInt4> {
  using Type = int4b_format;
};

// How many elements one stored unit of a format holds: two for the packed
// 4-bit format, whose stored unit is a byte, one for the others.
template <typename CacheElement> struct SplashKvElementsPerStored {
  static constexpr constant uint Value = 1;
};
template <> struct SplashKvElementsPerStored<SplashKvInt4> {
  static constexpr constant uint Value = 2;
};

// An element's index in a layer's region of one tensor: its page's slab,
// then its place in the slab (abi/KvExtent.h).
template <uint KVHeads, uint HeadDim>
inline ulong splash_kv_key_index(uint page, uint head, uint token,
                                   uint dimension) {
  constexpr ulong ElementsPerPage = ulong(KVHeads) * SplashKvPageTokens * HeadDim;
  return ulong(page) * ElementsPerPage +
         splash_kv_key_element(head, token, dimension, HeadDim);
}

template <uint KVHeads, uint HeadDim, bool ValuesTokenMajor = false>
inline ulong splash_kv_value_index(uint page, uint head, uint token,
                                     uint dimension) {
  constexpr ulong ElementsPerPage = ulong(KVHeads) * SplashKvPageTokens * HeadDim;
  return ulong(page) * ElementsPerPage +
         (ValuesTokenMajor ? splash_kv_key_element(head, token, dimension, HeadDim)
                           : splash_kv_value_element(head, token, dimension, HeadDim));
}

template <uint KVHeads>
inline ulong splash_q8_scale_index(uint page, uint head, uint token) {
  constexpr ulong ScalesPerPage = ulong(KVHeads) * SplashKvPageTokens;
  return ulong(page) * ScalesPerPage + splash_kv_scale_element(head, token);
}
