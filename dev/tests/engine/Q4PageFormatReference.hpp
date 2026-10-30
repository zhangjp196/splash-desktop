#pragma once

#include "metal/abi/KvExtent.h"
#include "ops/PagedKv.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace splash::kv {

// A CPU oracle for the packed 4-bit KV page, the counterpart of
// Q8PageFormatReference.hpp for kv::Format::Int4.
//
// The format: one fp32 scale per (KV head, token), exactly as INT8, and two
// elements packed in one byte: element `e` of a tensor sits in byte e / 2, the
// even element in the low nibble and the odd one in the high nibble. MPP's
// int4b_format operand reads a slab that way (verified at runtime against a
// CPU reference: see .atlas/kv-quant/SPIKE.md), and splash_store_kv_row
// writes it the same way.
//
// Which two (token, dimension) an element index pairs is a property of the
// tensor's order, not of the format: keys are token-major, so a byte pairs two
// dimensions of one token; dimension-major values pair two tokens of one
// dimension. This oracle packs by element index alone, which is why it agrees
// with the kernels for either order (splash_kv_key_element and
// splash_kv_value_element give the index the kernel packs at).
//
// The quantizer spends 7 signed bits: values are rounded to
// [-7, 7] over the row maximum, so the scale is maximum / 7 rather than
// INT8's maximum / 127.
inline constexpr int32_t kPackedQuantizedMaximum = 7;
inline constexpr int32_t kPackedQuantizedMinimum = -7;
static constexpr uint32_t kPackedElementsPerByte = 2;

// This oracle targets the Qwen3.8 kernel specialization.
inline constexpr Layout kPackedOracleLayout{16, 4, 256, Format::Int4};
inline constexpr uint32_t kPackedKvHeads = kPackedOracleLayout.kvHeads;
inline constexpr uint32_t kPackedHeadDimension = kPackedOracleLayout.headDimension;
inline constexpr uint64_t kPackedElementsPerLayerPage =
    kPackedOracleLayout.elementsPerLayerPage();
inline constexpr uint64_t kPackedScalesPerTensorLayerPage =
    kPackedOracleLayout.scalesPerTensorLayerPage();
// Two elements to a byte, so a page's data region halves.
inline constexpr uint64_t kPackedDataBytesPerLayerPage =
    kPackedElementsPerLayerPage / kPackedElementsPerByte;
inline constexpr uint64_t kPackedScaleBytesPerLayerPage =
    kPackedOracleLayout.scaleBytesPerLayerPage();
inline constexpr uint64_t kPackedBytesPerLayerPage =
    kPackedOracleLayout.bytesPerLayerPage();
inline constexpr uint64_t kPackedBytesPerModelPage =
    kPackedOracleLayout.bytesPerModelPage();

static_assert(kPackedDataBytesPerLayerPage * kPackedElementsPerByte ==
                  kPackedElementsPerLayerPage,
              "the 4-bit page packs whole bytes");
static_assert(kPackedBytesPerLayerPage ==
                  2 * (kPackedDataBytesPerLayerPage + kPackedScaleBytesPerLayerPage),
              "keys and values share one geometry");

struct Q4LayerPage final {
  std::array<uint8_t, kPackedDataBytesPerLayerPage> keys{};
  std::array<float, kPackedScalesPerTensorLayerPage> keyScales{};
  std::array<uint8_t, kPackedDataBytesPerLayerPage> values{};
  std::array<float, kPackedScalesPerTensorLayerPage> valueScales{};
};

static_assert(sizeof(Q4LayerPage) == kPackedBytesPerLayerPage);
static_assert(std::is_standard_layout_v<Q4LayerPage>);

namespace packed_reference_detail {

constexpr uint64_t logicalElements(uint32_t validTokens) {
  return uint64_t{validTokens} * kPackedKvHeads * kPackedHeadDimension;
}

constexpr uint64_t logicalIndex(uint32_t token, uint32_t head,
                                uint32_t dimension) {
  return (uint64_t{token} * kPackedKvHeads + head) * kPackedHeadDimension +
         dimension;
}

inline void validateInput(std::span<const float> values, uint32_t validTokens,
                          const char *name) {
  if (validTokens > kPageTokens)
    throw std::invalid_argument("4-bit KV valid token count exceeds one page");
  if (values.size() != logicalElements(validTokens))
    throw std::invalid_argument(std::string(name) +
                                " has the wrong logical page size");
  if (std::any_of(values.begin(), values.end(),
                  [](float value) { return !std::isfinite(value); }))
    throw std::invalid_argument(std::string(name) +
                                " contains a non-finite value");
}

inline float storedScale(float maximum) {
  return maximum == 0.0F ? 0.0F
                         : maximum / float(kPackedQuantizedMaximum);
}

inline int quantize(float value, float maximum) {
  if (maximum == 0.0F)
    return 0;
  long quantized = static_cast<long>(
      std::nearbyint(value * float(kPackedQuantizedMaximum) / maximum));
  return static_cast<int>(
      std::clamp<long>(quantized, kPackedQuantizedMinimum,
                       kPackedQuantizedMaximum));
}

// Packs one element into its byte, keeping the neighbouring nibble.
inline void packNibble(std::span<uint8_t> bytes, uint64_t element, int value) {
  const uint8_t nibble = uint8_t(value) & 0x0FU;
  uint8_t &byte = bytes[element / kPackedElementsPerByte];
  if (element % kPackedElementsPerByte == 0)
    byte = uint8_t((byte & 0xF0U) | nibble);
  else
    byte = uint8_t((byte & 0x0FU) | (nibble << 4));
}

// The signed 4-bit value a nibble holds.
inline int unpackNibble(const uint8_t *bytes, uint64_t element) {
  const uint8_t byte = bytes[element / kPackedElementsPerByte];
  const uint8_t nibble = element % kPackedElementsPerByte == 0
                             ? uint8_t(byte & 0x0FU)
                             : uint8_t(byte >> 4);
  return nibble < 8 ? int(nibble) : int(nibble) - 16;
}

inline void checkElement(uint32_t head, uint32_t token, uint32_t dimension) {
  if (head >= kPackedKvHeads || token >= kPageTokens ||
      dimension >= kPackedHeadDimension)
    throw std::out_of_range("4-bit KV element is outside the page");
}

} // namespace packed_reference_detail

inline float dequantizeKey(const Q4LayerPage &source, uint32_t head,
                           uint32_t token, uint32_t dimension) {
  packed_reference_detail::checkElement(head, token, dimension);
  return float(packed_reference_detail::unpackNibble(
             source.keys.data(),
             splash_kv_key_element(head, token, dimension))) *
         source.keyScales[splash_kv_scale_element(head, token)];
}

inline float dequantizeValue(const Q4LayerPage &source, uint32_t head,
                             uint32_t token, uint32_t dimension) {
  packed_reference_detail::checkElement(head, token, dimension);
  return float(packed_reference_detail::unpackNibble(
             source.values.data(),
             splash_kv_value_element(head, token, dimension))) *
         source.valueScales[splash_kv_scale_element(head, token)];
}

inline void quantizeLayerPage(std::span<const float> logicalKeys,
                              std::span<const float> logicalValues,
                              uint32_t validTokens, Q4LayerPage &destination) {
  using namespace packed_reference_detail;
  validateInput(logicalKeys, validTokens, "logical keys");
  validateInput(logicalValues, validTokens, "logical values");
  destination = {};
  for (uint32_t head = 0; head < kPackedKvHeads; ++head) {
    for (uint32_t token = 0; token < validTokens; ++token) {
      float keyMaximum = 0.0F;
      float valueMaximum = 0.0F;
      for (uint32_t dimension = 0; dimension < kPackedHeadDimension;
           ++dimension) {
        keyMaximum = std::max(
            keyMaximum,
            std::abs(logicalKeys[logicalIndex(token, head, dimension)]));
        valueMaximum = std::max(
            valueMaximum,
            std::abs(logicalValues[logicalIndex(token, head, dimension)]));
      }
      destination.keyScales[splash_kv_scale_element(head, token)] =
          storedScale(keyMaximum);
      destination.valueScales[splash_kv_scale_element(head, token)] =
          storedScale(valueMaximum);
      for (uint32_t dimension = 0; dimension < kPackedHeadDimension;
           ++dimension) {
        packNibble(destination.keys,
                   splash_kv_key_element(head, token, dimension),
                   quantize(logicalKeys[logicalIndex(token, head, dimension)],
                            keyMaximum));
        packNibble(destination.values,
                   splash_kv_value_element(head, token, dimension),
                   quantize(logicalValues[logicalIndex(token, head, dimension)],
                            valueMaximum));
      }
    }
  }
}

inline void dequantizeLayerPage(const Q4LayerPage &source,
                                uint32_t validTokens,
                                std::span<float> logicalKeys,
                                std::span<float> logicalValues) {
  using namespace packed_reference_detail;
  if (validTokens > kPageTokens)
    throw std::invalid_argument("4-bit KV valid token count exceeds one page");
  if (logicalKeys.size() != kPackedElementsPerLayerPage)
    throw std::invalid_argument("logical keys has the wrong logical page size");
  if (logicalValues.size() != kPackedElementsPerLayerPage)
    throw std::invalid_argument(
        "logical values has the wrong logical page size");
  std::fill(logicalKeys.begin(), logicalKeys.end(), 0.0F);
  std::fill(logicalValues.begin(), logicalValues.end(), 0.0F);
  for (uint32_t token = 0; token < validTokens; ++token) {
    for (uint32_t head = 0; head < kPackedKvHeads; ++head) {
      for (uint32_t dimension = 0; dimension < kPackedHeadDimension;
           ++dimension) {
        const uint64_t output = logicalIndex(token, head, dimension);
        logicalKeys[output] = dequantizeKey(source, head, token, dimension);
        logicalValues[output] =
            dequantizeValue(source, head, token, dimension);
      }
    }
  }
}

} // namespace splash::kv
