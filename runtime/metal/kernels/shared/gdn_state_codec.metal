#include "metal/abi/KernelABI.h"

template <typename State>
inline float load_state(device const State *values, uint index) {
  return float(values[index]);
}

template <typename State>
inline void store_state(device State *values, uint index, float value) {
  values[index] = State(value);
}

template <typename State>
inline void quantize_block(device const State *state, device char *codes,
                           constant GdnStateCodecParams &p, uint block,
                           uint layer) {
  const uint blocksPerRow = p.columns / SPLASH_GDN_CACHE_BLOCK;
  const uint row = block / blocksPerRow;
  const uint column = (block % blocksPerRow) * SPLASH_GDN_CACHE_BLOCK;
  device const State *values = reinterpret_cast<device const State *>(
      reinterpret_cast<device const char *>(state) + ulong(layer) * p.state_layer_bytes) +
      ulong(row) * p.columns + column;
  float maximum = 0.0f;
  for (uint i = 0; i < SPLASH_GDN_CACHE_BLOCK; ++i) {
    const float value = load_state(values, i);
    if (isnan(value)) { maximum = NAN; break; }
    maximum = max(maximum, abs(value));
  }
  const bfloat storedScale = bfloat(maximum == 0.0f ? 1.0f : maximum / 127.0f);
  const float scale = float(storedScale);
  device char *layerCodes = codes + ulong(layer) * p.codes_layer_bytes;
  device char *blockCodes = layerCodes + ulong(row) * p.columns + column;
  for (uint i = 0; i < SPLASH_GDN_CACHE_BLOCK; ++i) {
    const float normalized = load_state(values, i) / scale;
    const float rounded = isfinite(normalized) ? rint(normalized) : 0.0f;
    blockCodes[i] = char(clamp(rounded, -127.0f, 127.0f));
  }
  const uint scaleIndex = row * blocksPerRow + block % blocksPerRow;
  reinterpret_cast<device bfloat *>(layerCodes + p.scales_offset)[scaleIndex] = storedScale;
}

template <typename State>
inline void dequantize_block(device const char *codes, device State *state,
                             constant GdnStateCodecParams &p, uint block,
                             uint layer) {
  const uint blocksPerRow = p.columns / SPLASH_GDN_CACHE_BLOCK;
  const uint row = block / blocksPerRow;
  const uint column = (block % blocksPerRow) * SPLASH_GDN_CACHE_BLOCK;
  device const char *layerCodes = codes + ulong(layer) * p.codes_layer_bytes;
  const uint scaleIndex = row * blocksPerRow + block % blocksPerRow;
  const float scale = float(reinterpret_cast<device const bfloat *>(
      layerCodes + p.scales_offset)[scaleIndex]);
  device const char *blockCodes = layerCodes + ulong(row) * p.columns + column;
  device State *values = reinterpret_cast<device State *>(
      reinterpret_cast<device char *>(state) + ulong(layer) * p.state_layer_bytes) +
      ulong(row) * p.columns + column;
  for (uint i = 0; i < SPLASH_GDN_CACHE_BLOCK; ++i)
    store_state(values, i, float(blockCodes[i]) * scale);
}

#define GDN_CACHE_KERNELS(Suffix, State) \
  kernel void gdn_cache_quantize##Suffix( \
      device const State *state [[buffer(0)]], device char *codes [[buffer(1)]], \
      constant GdnStateCodecParams &p [[buffer(2)]], \
      uint2 position [[thread_position_in_grid]]) { \
    const uint blocks = p.rows * (p.columns / SPLASH_GDN_CACHE_BLOCK); \
    if (position.x < blocks) quantize_block(state, codes, p, position.x, position.y); \
  } \
  kernel void gdn_cache_dequantize##Suffix( \
      device const char *codes [[buffer(0)]], device State *state [[buffer(1)]], \
      constant GdnStateCodecParams &p [[buffer(2)]], \
      uint2 position [[thread_position_in_grid]]) { \
    const uint blocks = p.rows * (p.columns / SPLASH_GDN_CACHE_BLOCK); \
    if (position.x < blocks) dequantize_block(codes, state, p, position.x, position.y); \
  }

GDN_CACHE_KERNELS(, float)
GDN_CACHE_KERNELS(_bf16_state, bfloat)
#undef GDN_CACHE_KERNELS
