// The packed 4-bit KV page: its geometry in Layout, the CPU oracle's
// round trip, and the nibble packing that MPP's int4b_format operand reads.
#include "Q4PageFormatReference.hpp"
#include "TestChecks.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace splash::kv;
using splash::test::rejects;
using splash::test::require;

namespace {

// The 4-bit page is INT8's with the data regions halved: same number of
// fp32 scales, two dimensions per byte.
void testByteAccounting() {
  constexpr Layout int8{16, 4, 256, Format::Int8};
  constexpr Layout int4{16, 4, 256, Format::Int4};
  static_assert(int4.valid());
  static_assert(int8.valid());
  static_assert(kPackedOracleLayout.valid());
  static_assert(int4.elementsPerLayerPage() == int8.elementsPerLayerPage());
  static_assert(int4.scalesPerTensorLayerPage() ==
                int8.scalesPerTensorLayerPage());
  static_assert(int4.elementsPerScale() == int8.elementsPerScale());
  static_assert(int4.scaleBytesPerLayerPage() ==
                int8.scaleBytesPerLayerPage());
  // The data regions are INT8's, halved.
  static_assert(2 * int4.dataBytesPerLayerPage() ==
                int8.dataBytesPerLayerPage());
  static_assert(int4.dataBytesPerLayerPage() == 4 * 32 * 256 / 2);
  // So a layer page is 2 * (32768/2 + 512) = 33792 bytes against INT8's
  // 2 * (32768 + 512) = 66560: half the data, the same scales, so 50.8% of
  // INT8's page rather than 50%.
  static_assert(int4.bytesPerLayerPage() == 33'792);
  static_assert(int8.bytesPerLayerPage() == 66'560);
  static_assert(int4.bytesPerLayerPage() == int8.dataBytesPerLayerPage() / 2 * 2 +
                                                2 * int8.scaleBytesPerLayerPage());
  static_assert(int4.bytesPerModelPage() == 540'672);
  static_assert(int8.bytesPerModelPage() == 1'064'960);
  // The scales are kept, so a page shrinks by a little under half: the
  // 4-bit page is 50.8% of the INT8 one.
  static_assert(int4.bytesPerModelPage() * 1000 / int8.bytesPerModelPage() == 507);
  // BF16 keeps two bytes an element and has no scales, so its data region is
  // four times the 4-bit one.
  constexpr Layout bf16{16, 4, 256, Format::BFloat16};
  static_assert(int4.dataBytesPerLayerPage() * 4 ==
                bf16.dataBytesPerLayerPage());
  static_assert(bf16.scaleBytesPerLayerPage() == 0);
}

// Names reach the persistent cache namespace, so a settled format cannot
// change its spelling.
void testNames() {
  require(formatName(Format::Int4) == "int4", "int4 format name");
  require(storageFormatName(Format::Int4, false) ==
              "q4s4_f32_scale_per_token_head_k_token_major_v_dimension_major",
          "int4 dimension-major storage name");
  require(storageFormatName(Format::Int4, true) ==
              "q4s4_f32_scale_per_token_head_k_token_major_v_token_major",
          "int4 token-major storage name");
  require(validFormat(Format::Int4), "int4 is a valid format");
  require(quantizedFormat(Format::Int4), "int4 keeps scales");
  require(!quantizedFormat(Format::BFloat16), "bf16 keeps no scales");
  require(elementsPerByte(Format::Int4) == 2, "int4 packs two per byte");
  require(elementsPerByte(Format::Int8) == 1, "int8 keeps one per byte");
}

// The oracle packs the even element of a pair in the low nibble, which is
// what the MPP operand reads.
void testNibblePacking() {
  Q4LayerPage page{};
  // Element 0 of a slab is the low nibble of byte 0, element 1 the high one.
  packed_reference_detail::packNibble(page.keys, 0, 5);
  packed_reference_detail::packNibble(page.keys, 1, -3);
  require(page.keys[0] == uint8_t(((-3 & 0xF) << 4) | 5), "pair packs into one byte");
  require(packed_reference_detail::unpackNibble(page.keys.data(), 0) == 5,
          "low nibble round trip");
  require(packed_reference_detail::unpackNibble(page.keys.data(), 1) == -3,
          "high nibble round trip");
  // The signed range is [-7, 7]: 8 and 9 read back as -8 and -7.
  require(packed_reference_detail::unpackNibble(page.keys.data(), 0) >=
              kPackedQuantizedMinimum &&
          packed_reference_detail::unpackNibble(page.keys.data(), 0) <=
              kPackedQuantizedMaximum,
          "packed value within range");
  // Packing one element leaves its neighbour alone.
  packed_reference_detail::packNibble(page.keys, 0, -1);
  require(packed_reference_detail::unpackNibble(page.keys.data(), 1) == -3,
          "repacking keeps the neighbour");
}

// A page of pseudo-random values survives the oracle's quantize and
// dequantize within the error a 7-bit symmetric quantizer has.
void testRoundTrip() {
  constexpr uint32_t kTokens = kPageTokens;
  const uint64_t elements = uint64_t{kTokens} * kPackedKvHeads * kPackedHeadDimension;
  std::vector<float> keys(elements), values(elements);
  uint32_t state = 0x12345678U;
  const auto next = [&state]() {
    state = state * 1664525U + 1013904223U;
    return float(int32_t(state >> 8) - (1 << 23)) / float(1 << 23);
  };
  for (uint64_t i = 0; i < elements; ++i) {
    keys[i] = next() * 4.0F;
    values[i] = next() * 4.0F;
  }

  Q4LayerPage page{};
  quantizeLayerPage(keys, values, kTokens, page);

  std::vector<float> restoredKeys(elements), restoredValues(elements);
  dequantizeLayerPage(page, kTokens, restoredKeys, restoredValues);

  // One step of a 7-bit symmetric quantizer is at most half a scale off,
  // and a scale is the row maximum over 7.
  for (uint32_t head = 0; head < kPackedKvHeads; ++head) {
    for (uint32_t token = 0; token < kTokens; ++token) {
      float keyMaximum = 0.0F, valueMaximum = 0.0F;
      for (uint32_t dimension = 0; dimension < kPackedHeadDimension; ++dimension) {
        const uint64_t index =
            packed_reference_detail::logicalIndex(token, head, dimension);
        keyMaximum = std::max(keyMaximum, std::abs(keys[index]));
        valueMaximum = std::max(valueMaximum, std::abs(values[index]));
      }
      const float keyStep = keyMaximum == 0.0F ? 0.0F : keyMaximum / 7.0F;
      const float valueStep = valueMaximum == 0.0F ? 0.0F : valueMaximum / 7.0F;
      for (uint32_t dimension = 0; dimension < kPackedHeadDimension; ++dimension) {
        const uint64_t index =
            packed_reference_detail::logicalIndex(token, head, dimension);
        require(std::abs(restoredKeys[index] - keys[index]) <= keyStep * 0.5F + 1e-5F,
                "key round trip within half a step");
        require(std::abs(restoredValues[index] - values[index]) <=
                    valueStep * 0.5F + 1e-5F,
                "value round trip within half a step");
      }
    }
  }
  // Every byte of both data regions was written: a page of nonzero values
  // leaves no zero pairs unless it rounds to them.
  require(std::any_of(page.keys.begin(), page.keys.end(),
                      [](uint8_t byte) { return byte != 0; }),
          "keys were packed");
  require(std::any_of(page.values.begin(), page.values.end(),
                      [](uint8_t byte) { return byte != 0; }),
          "values were packed");
  // The scales are the per-(head, token) maxima over 7.
  for (uint32_t head = 0; head < kPackedKvHeads; ++head) {
    for (uint32_t token = 0; token < kTokens; ++token) {
      float keyMaximum = 0.0F;
      for (uint32_t dimension = 0; dimension < kPackedHeadDimension; ++dimension) {
        keyMaximum = std::max(
            keyMaximum,
            std::abs(keys[packed_reference_detail::logicalIndex(token, head, dimension)]));
      }
      require(page.keyScales[splash_kv_scale_element(head, token)] ==
                  keyMaximum / 7.0F,
              "key scale is the row maximum over 7");
    }
  }
}

// A byte holds two neighbouring element indices, so both nibbles carry their
// own value and neither overwrites the other. The dimension a byte pairs
// depends on the tensor's order (testNames notes it): keys pair dimensions of
// one token, dimension-major values pair tokens of one dimension.
void testDistinctNeighbours() {
  // The packing itself, without quantization in the way: both nibbles of a
  // byte are independent, and each keeps what was written to it.
  Q4LayerPage page{};
  for (uint64_t element = 0; element < 64; element += 2) {
    packed_reference_detail::packNibble(page.keys, element, int(element % 7) - 3);
    packed_reference_detail::packNibble(page.keys, element + 1, int(element % 5) - 2);
  }
  for (uint64_t element = 0; element < 64; element += 2) {
    require(packed_reference_detail::unpackNibble(page.keys.data(), element) ==
                int(element % 7) - 3,
            "every even element keeps its low nibble");
    require(packed_reference_detail::unpackNibble(page.keys.data(), element + 1) ==
                int(element % 5) - 2,
            "every odd element keeps its high nibble");
  }
  // Which element indices a byte pairs is fixed by the slab's order, so the
  // index functions must give neighbours for a pair the kernel packs together:
  // keys pair two dimensions of a token, dimension-major values two tokens of
  // a dimension.
  require(splash_kv_key_element(0, 0, 0) + 1 == splash_kv_key_element(0, 0, 1),
          "key dimensions are adjacent element indices");
  require(splash_kv_value_element(0, 0, 0) + 1 == splash_kv_value_element(0, 1, 0),
          "dimension-major value tokens are adjacent element indices");
  // ...and a token's own key region starts at its first dimension, while a
  // dimension's value region starts at its first token.
  require(splash_kv_key_element(0, 1, 0) == kPackedHeadDimension,
          "the next token's keys start a head dimension later");
  require(splash_kv_value_element(0, 0, 1) == kPageTokens,
          "the next dimension's values start a page of tokens later");
  // A logical page of one token starts at the page's first element.
  require(splash_kv_key_element(0, 0, 0) == 0, "the first key is the first element");
  require(splash_kv_value_element(0, 0, 0) == 0, "the first value is the first element");
}

// The oracle refuses what it cannot represent.
void testRefusals() {
  std::vector<float> keys(kPackedElementsPerLayerPage, 0.0F);
  std::vector<float> values(kPackedElementsPerLayerPage, 0.0F);
  Q4LayerPage page{};
  rejects([&] { quantizeLayerPage(keys, values, kPageTokens + 1, page); },
          "4-bit KV valid token count exceeds one page",
          "a page of more tokens than it holds");
  std::vector<float> shortKeys(kPackedElementsPerLayerPage - 1, 0.0F);
  rejects([&] { quantizeLayerPage(shortKeys, values, kPageTokens, page); },
          "logical keys has the wrong logical page size",
          "a short key page");
  keys[0] = std::numeric_limits<float>::quiet_NaN();
  rejects([&] { quantizeLayerPage(keys, values, kPageTokens, page); },
          "logical keys contains a non-finite value", "a non-finite key");
  keys[0] = 0.0F;
  rejects(
      [&] { dequantizeLayerPage(page, kPageTokens + 1, keys, values); },
      "4-bit KV valid token count exceeds one page",
      "a dequantize past the page");
  std::vector<float> shortOut(kPackedElementsPerLayerPage - 1, 0.0F);
  rejects([&] { dequantizeLayerPage(page, kPageTokens, shortOut, values); },
          "logical keys has the wrong logical page size",
          "a short dequantize target");
}

} // namespace

int main() {
  testByteAccounting();
  testNames();
  testNibblePacking();
  testRoundTrip();
  testDistinctNeighbours();
  testRefusals();
  std::cout << "q4 paged kv tests passed\n";
  return 0;
}
