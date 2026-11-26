#pragma once

#include <bit>
#include <cstdint>
#include <optional>

namespace splash::model {

// The bf16 bits of value when it is exactly a bf16: preparation never rounds
// a weight.
[[nodiscard]] inline std::optional<uint16_t> exactBfloat16(float value) {
  const auto bits = std::bit_cast<uint32_t>(value);
  if (bits & 0xFFFFu) return std::nullopt;
  return static_cast<uint16_t>(bits >> 16);
}

// The float a bf16's bits hold.
[[nodiscard]] inline float widenBfloat16(uint16_t bits) { return std::bit_cast<float>(uint32_t{bits} << 16); }

// The float a half's bits hold, as a safetensors checkpoint stores its norms
// in F16 (an "oQ4e-fp16" release).
[[nodiscard]] inline float widenFloat16(uint16_t bits) {
  const uint32_t sign = uint32_t(bits & 0x8000) << 16;
  const uint32_t exponent = (bits >> 10) & 0x1F, mantissa = bits & 0x3FF;
  uint32_t value;
  if (exponent == 0) {
    if (mantissa == 0) return std::bit_cast<float>(sign);
    uint32_t shifted = mantissa;
    int shift = 0;
    while ((shifted & 0x400) == 0) {
      shifted <<= 1;
      ++shift;
    }
    value = sign | (uint32_t(127 - 15 - shift) << 23) | ((shifted & 0x3FF) << 13);
  } else if (exponent == 31) {
    value = sign | 0x7F800000u | (mantissa << 13);
  } else {
    value = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  }
  return std::bit_cast<float>(value);
}

// The bf16 bits nearest to value, ties to even, as a bf16 conversion rounds:
// the one place preparation rounds a weight, a quantized vision tower's values
// whose own quantization already lost more than the rounding does.
[[nodiscard]] inline uint16_t narrowBfloat16(float value) {
  const auto bits = std::bit_cast<uint32_t>(value);
  if ((bits & 0x7FFFFFFFu) > 0x7F800000u) return static_cast<uint16_t>((bits >> 16) | 0x0040u); // NaN stays one
  return static_cast<uint16_t>((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
}

} // namespace splash::model
