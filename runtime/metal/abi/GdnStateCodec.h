#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#define SPLASH_GDN_CACHE_BLOCK 32u

struct GdnStateCodecParams {
  uint64_t state_layer_bytes;
  uint64_t codes_layer_bytes;
  uint64_t scales_offset;
  uint32_t rows;
  uint32_t columns;
};

static_assert(sizeof(GdnStateCodecParams) == 32,
              "GDN state codec parameters are 32 bytes on host and Metal");
