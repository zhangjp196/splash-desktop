#pragma once

#include "metal/abi/KvExtent.h"
#include "ops/PagedKv.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace splash::kv {

// A CPU oracle for the k8v4 KV page: INT8 keys with one fp32 scale per
// (KV head, token), and packed 4-bit values with one fp32 scale per
// (KV head, token) as well. It is the counterpart of
// Q8PageFormatReference.hpp (INT8) and Q4PageFormatReference.hpp (packed
// 4-bit) for a layout whose two tensors carry different element formats.
//
// The keys spend INT8's 7 signed bits over the row maximum (scale
// maximum / 127), the values the packed format's 7 signed bits (scale
// maximum / 7). A value byte holds two neighbouring element indices, the
// even index in the low nibble and the odd one in the high nibble, which is
// how MPP's int4b_format operand reads a slab (verified at runtime against a
// CPU reference: see .atlas/kv-quant/SPIKE.md). Keys are token-major and
// values dimension-major, so a key byte is one element and a value byte pairs
// two tokens of one dimension.
//
// The page's regions follow splash_kv_offset: keys' data, their scales,
// values' data, their scales.
inline constexpr Layout kK8V4OracleLayout{16, 4, 256, Format::Int8, 0, false, 0, 0,
                                          KvStorage{{8, false}, {4, false}}};
inline constexpr uint32_t kK8V4KvHeads = kK8V4OracleLayout.kvHeads;
inline constexpr uint32_t kK8V4HeadDimension = kK8V4OracleLayout.headDimension;
inline constexpr uint64_t kK8V4ElementsPerLayerPage =
    kK8V4OracleLayout.elementsPerLayerPage();
inline constexpr uint64_t kK8V4ScalesPerTensorLayerPage =
    kK8V4OracleLayout.scalesPerTensorLayerPageK();
inline constexpr uint64_t kK8V4KeyDataBytesPerLayerPage =
    kK8V4OracleLayout.dataBytesPerLayerPageK();
inline constexpr uint64_t kK8V4KeyScaleBytesPerLayerPage =
    kK8V4OracleLayout.scaleBytesPerLayerPageK();
inline constexpr uint64_t kK8V4ValueDataBytesPerLayerPage =
    kK8V4OracleLayout.dataBytesPerLayerPageV();
inline constexpr uint64_t kK8V4ValueScaleBytesPerLayerPage =
    kK8V4OracleLayout.scaleBytesPerLayerPageV();
inline constexpr uint64_t kK8V4BytesPerLayerPage =
    kK8V4OracleLayout.bytesPerLayerPage();
inline constexpr int32_t kK8V4KeyQuantizedMaximum = 127;
inline constexpr int32_t kK8V4KeyQuantizedMinimum = -127;
inline constexpr int32_t kK8V4ValueQuantizedMaximum = 7;
inline constexpr int32_t kK8V4ValueQuantizedMinimum = -7;

static_assert(kK8V4KeyDataBytesPerLayerPage == kK8V4ElementsPerLayerPage,
              "INT8 keys are one byte an element");
static_assert(kK8V4ValueDataBytesPerLayerPage * 2 == kK8V4ElementsPerLayerPage,
              "packed 4-bit values are two elements a byte");
static_assert(kK8V4BytesPerLayerPage ==
                  kK8V4KeyDataBytesPerLayerPage + kK8V4KeyScaleBytesPerLayerPage +
                      kK8V4ValueDataBytesPerLayerPage + kK8V4ValueScaleBytesPerLayerPage,
              "the page's regions are the keys' and values' data and scales");

struct K8V4LayerPage final {
  std::array<int8_t, kK8V4KeyDataBytesPerLayerPage> keys{};
  std::array<float, kK8V4ScalesPerTensorLayerPage> keyScales{};
  std::array<uint8_t, kK8V4ValueDataBytesPerLayerPage> values{};
  std::array<float, kK8V4ScalesPerTensorLayerPage> valueScales{};
};

static_assert(sizeof(K8V4LayerPage) == kK8V4BytesPerLayerPage);
static_assert(std::is_standard_layout_v<K8V4LayerPage>);

namespace k8v4_reference_detail {

constexpr uint64_t logicalElements(uint32_t validTokens) {
  return uint64_t{validTokens} * kK8V4KvHeads * kK8V4HeadDimension;
}

constexpr uint64_t logicalIndex(uint32_t token, uint32_t head,
                                uint32_t dimension) {
  return (uint64_t{token} * kK8V4KvHeads + head) * kK8V4HeadDimension +
         dimension;
}

inline void validateInput(std::span<const float> values, uint32_t validTokens,
                          const char *name) {
  if (validTokens > kPageTokens)
    throw std::invalid_argument("k8v4 KV valid token count exceeds one page");
  if (values.size() != logicalElements(validTokens))
    throw std::invalid_argument(std::string(name) +
                                " has the wrong logical page size");
  if (std::any_of(values.begin(), values.end(),
                  [](float value) { return !std::isfinite(value); }))
    throw std::invalid_argument(std::string(name) +
                                " contains a non-finite value");
}

inline float storedScale(float maximum, int32_t quantizedMaximum) {
  return maximum == 0.0F ? 0.0F : maximum / float(quantizedMaximum);
}

inline int quantize(float value, float maximum, int32_t quantizedMaximum,
                    int32_t quantizedMinimum) {
  if (maximum == 0.0F)
    return 0;
  long quantized = static_cast<long>(
      std::nearbyint(value * float(quantizedMaximum) / maximum));
  return static_cast<int>(
      std::clamp<long>(quantized, quantizedMinimum, quantizedMaximum));
}

// Packs one value element into its byte, keeping the neighbouring nibble.
inline void packNibble(std::span<uint8_t> bytes, uint64_t element, int value) {
  const uint8_t nibble = uint8_t(value) & 0x0FU;
  uint8_t &byte = bytes[element / 2];
  if (element % 2 == 0)
    byte = uint8_t((byte & 0xF0U) | nibble);
  else
    byte = uint8_t((byte & 0x0FU) | (nibble << 4));
}

// The signed 4-bit value a nibble holds.
inline int unpackNibble(const uint8_t *bytes, uint64_t element) {
  const uint8_t byte = bytes[element / 2];
  const uint8_t nibble = element % 2 == 0 ? uint8_t(byte & 0x0FU)
                                          : uint8_t(byte >> 4);
  return nibble < 8 ? int(nibble) : int(nibble) - 16;
}

inline void checkElement(uint32_t head, uint32_t token, uint32_t dimension) {
  if (head >= kK8V4KvHeads || token >= kPageTokens ||
      dimension >= kK8V4HeadDimension)
    throw std::out_of_range("k8v4 KV element is outside the page");
}

} // namespace k8v4_reference_detail

inline float dequantizeK8V4Key(const K8V4LayerPage &source, uint32_t head,
                               uint32_t token, uint32_t dimension) {
  k8v4_reference_detail::checkElement(head, token, dimension);
  return float(source.keys[splash_kv_key_element(head, token, dimension)]) *
         source.keyScales[splash_kv_scale_element(head, token)];
}

inline float dequantizeK8V4Value(const K8V4LayerPage &source, uint32_t head,
                                 uint32_t token, uint32_t dimension) {
  k8v4_reference_detail::checkElement(head, token, dimension);
  return float(k8v4_reference_detail::unpackNibble(
             source.values.data(),
             splash_kv_value_element(head, token, dimension))) *
         source.valueScales[splash_kv_scale_element(head, token)];
}

inline void quantizeK8V4LayerPage(std::span<const float> logicalKeys,
                                  std::span<const float> logicalValues,
                                  uint32_t validTokens,
                                  K8V4LayerPage &destination) {
  using namespace k8v4_reference_detail;
  validateInput(logicalKeys, validTokens, "logical keys");
  validateInput(logicalValues, validTokens, "logical values");
  destination = {};
  for (uint32_t head = 0; head < kK8V4KvHeads; ++head) {
    for (uint32_t token = 0; token < validTokens; ++token) {
      float keyMaximum = 0.0F;
      float valueMaximum = 0.0F;
      for (uint32_t dimension = 0; dimension < kK8V4HeadDimension; ++dimension) {
        keyMaximum = std::max(
            keyMaximum, std::abs(logicalKeys[logicalIndex(token, head, dimension)]));
        valueMaximum = std::max(
            valueMaximum, std::abs(logicalValues[logicalIndex(token, head, dimension)]));
      }
      destination.keyScales[splash_kv_scale_element(head, token)] =
          storedScale(keyMaximum, kK8V4KeyQuantizedMaximum);
      destination.valueScales[splash_kv_scale_element(head, token)] =
          storedScale(valueMaximum, kK8V4ValueQuantizedMaximum);
      for (uint32_t dimension = 0; dimension < kK8V4HeadDimension; ++dimension) {
        destination.keys[splash_kv_key_element(head, token, dimension)] = int8_t(
            quantize(logicalKeys[logicalIndex(token, head, dimension)], keyMaximum,
                     kK8V4KeyQuantizedMaximum, kK8V4KeyQuantizedMinimum));
        packNibble(destination.values,
                   splash_kv_value_element(head, token, dimension),
                   quantize(logicalValues[logicalIndex(token, head, dimension)],
                            valueMaximum, kK8V4ValueQuantizedMaximum,
                            kK8V4ValueQuantizedMinimum));
      }
    }
  }
}

inline void dequantizeK8V4LayerPage(const K8V4LayerPage &source,
                                    uint32_t validTokens,
                                    std::span<float> logicalKeys,
                                    std::span<float> logicalValues) {
  using namespace k8v4_reference_detail;
  if (validTokens > kPageTokens)
    throw std::invalid_argument("k8v4 KV valid token count exceeds one page");
  if (logicalKeys.size() != kK8V4ElementsPerLayerPage)
    throw std::invalid_argument("logical keys has the wrong logical page size");
  if (logicalValues.size() != kK8V4ElementsPerLayerPage)
    throw std::invalid_argument("logical values has the wrong logical page size");
  std::fill(logicalKeys.begin(), logicalKeys.end(), 0.0F);
  std::fill(logicalValues.begin(), logicalValues.end(), 0.0F);
  for (uint32_t token = 0; token < validTokens; ++token) {
    for (uint32_t head = 0; head < kK8V4KvHeads; ++head) {
      for (uint32_t dimension = 0; dimension < kK8V4HeadDimension; ++dimension) {
        const uint64_t output = logicalIndex(token, head, dimension);
        logicalKeys[output] = dequantizeK8V4Key(source, head, token, dimension);
        logicalValues[output] = dequantizeK8V4Value(source, head, token, dimension);
      }
    }
  }
}

} // namespace splash::kv
