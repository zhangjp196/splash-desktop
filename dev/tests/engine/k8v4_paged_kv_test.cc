// The k8v4 KV page: INT8 keys with packed 4-bit values, its geometry in
// Layout, the CPU oracle's round trip, and the byte order of the page's
// regions.
#include "K8V4PageFormatReference.hpp"
#include "TestChecks.hpp"

#include <algorithm>
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

// The page is INT8's keys with INT4's values: the key data is INT8's whole
// region, the value data half of it, and both keep a scale per (head, token).
void testByteAccounting() {
  constexpr Layout int8{16, 4, 256, Format::Int8};
  constexpr Layout int4{16, 4, 256, Format::Int4};
  static_assert(kK8V4OracleLayout.valid());
  static_assert(kK8V4OracleLayout.format == Format::Int8);
  static_assert(kK8V4OracleLayout.effectiveStorage() ==
                KvStorage{{8, false}, {4, false}});
  static_assert(kK8V4KeyDataBytesPerLayerPage == 4 * 32 * 256);
  static_assert(kK8V4ValueDataBytesPerLayerPage == 4 * 32 * 256 / 2);
  static_assert(kK8V4KeyScaleBytesPerLayerPage == 4 * 32 * sizeof(float));
  static_assert(kK8V4ValueScaleBytesPerLayerPage == 4 * 32 * sizeof(float));
  // A 27B page: keys 32 KiB and their scales 512 B, values 16 KiB and their
  // scales 512 B, so 49 KiB against INT8's 65 KiB: 75.4% of the page.
  static_assert(kK8V4BytesPerLayerPage == 50'176);
  static_assert(kK8V4BytesPerLayerPage ==
                kK8V4KeyDataBytesPerLayerPage + kK8V4KeyScaleBytesPerLayerPage +
                    kK8V4ValueDataBytesPerLayerPage + kK8V4ValueScaleBytesPerLayerPage);
  static_assert(kK8V4OracleLayout.bytesPerLayerPage() == kK8V4BytesPerLayerPage);
  static_assert(kK8V4OracleLayout.bytesPerModelPage() == 802'816);
  static_assert(int8.bytesPerModelPage() == 1'064'960);
  static_assert(kK8V4OracleLayout.bytesPerModelPage() * 1000 /
                        int8.bytesPerModelPage() ==
                    753);
  // The keys are INT8's, the values INT4's: a page is INT8's keys plus half
  // of its values, scales included.
  static_assert(kK8V4KeyDataBytesPerLayerPage ==
                int8.dataBytesPerLayerPage());
  static_assert(kK8V4ValueDataBytesPerLayerPage ==
                int4.dataBytesPerLayerPage());
  static_assert(kK8V4ValueDataBytesPerLayerPage * 2 ==
                kK8V4KeyDataBytesPerLayerPage);
  // Every one of the four regions is a whole number of 64 KiB units in an
  // extent: the key data needs two pages, the value data four and either
  // scale's 512 bytes 128.
  static_assert(kK8V4OracleLayout.extentAlignmentPages() == 128);
  const uint64_t aligned =
      uint64_t{kK8V4OracleLayout.extentAlignmentPages()} * kK8V4BytesPerLayerPage;
  require(aligned % 65'536 == 0, "a k8v4 extent's page is not 64 KiB aligned");
  require(uint64_t{kK8V4OracleLayout.extentAlignmentPages()} *
                  kK8V4KeyDataBytesPerLayerPage % 65'536 == 0 &&
              uint64_t{kK8V4OracleLayout.extentAlignmentPages()} *
                  kK8V4KeyScaleBytesPerLayerPage % 65'536 == 0 &&
              uint64_t{kK8V4OracleLayout.extentAlignmentPages()} *
                  kK8V4ValueDataBytesPerLayerPage % 65'536 == 0 &&
              uint64_t{kK8V4OracleLayout.extentAlignmentPages()} *
                  kK8V4ValueScaleBytesPerLayerPage % 65'536 == 0,
          "a k8v4 extent region is not 64 KiB aligned");
}

// The names reach the persistent cache namespace and the status JSON, so the
// two-element format cannot answer with its keys' single-format name.
void testNames() {
  require(formatName(kK8V4OracleLayout) == "k8v4", "k8v4 layout name");
  require(formatName(Format::Int8) == "int8", "int8 format name");
  require(storageFormatName(kK8V4OracleLayout) ==
              "q8s4_f32_scale_per_token_head_k_token_major_v_dimension_major",
          "k8v4 dimension-major storage name");
  Layout tokenMajor = kK8V4OracleLayout;
  tokenMajor.valuesTokenMajor = true;
  tokenMajor.indexDimension = 128;
  require(storageFormatName(tokenMajor) ==
              "q8s4_f32_scale_per_token_head_k_token_major_v_token_major",
          "k8v4 token-major storage name");
  require(storageFormatName(Format::Int8, false) ==
              "q8s8_f32_scale_per_token_head_k_token_major_v_dimension_major",
          "int8 keeps its storage name");
  require(singleFormat(KvStorage{{8, false}, {4, false}}) == Format::Int8,
          "k8v4 is an INT8 layout with packed values");
  require(storageOf(Format::Int8) == KvStorage{{8, false}, {8, false}},
          "a single format's storage is both tensors'");
  require(validKvStorage(KvStorage{{8, false}, {4, false}}),
          "k8v4 storage is valid");
  require(!validKvStorage(KvStorage{{3, false}, {4, false}}),
          "a 3-bit key element is invalid");
  require(!validKvStorage(KvStorage{{8, true}, {4, false}}),
          "an 8-bit float element is invalid");
  Layout inverted{16, 4, 256, Format::Int8, 0, false, 0, 0,
                  KvStorage{{4, false}, {8, false}}};
  require(!inverted.valid(), "a 4-bit key with 8-bit values is invalid");
  Layout mislabelled{16, 4, 256, Format::BFloat16, 0, false, 0, 0,
                     KvStorage{{8, false}, {4, false}}};
  require(!mislabelled.valid(),
          "a k8v4 storage under a BF16 format is invalid");
}

// splash_kv_offset places the keys' data, their scales, the values' data and
// their scales in that order within every extent.
void testRegionOrder() {
  constexpr uint32_t extentPages = 3;
  const uint64_t region = uint64_t{extentPages} * kK8V4KeyDataBytesPerLayerPage +
                          uint64_t{extentPages} * kK8V4KeyScaleBytesPerLayerPage +
                          uint64_t{extentPages} * kK8V4ValueDataBytesPerLayerPage +
                          uint64_t{extentPages} * kK8V4ValueScaleBytesPerLayerPage;
  const auto offset = [&](uint32_t layer, uint32_t tensor, uint32_t index) {
    return splash_kv_offset(extentPages, kK8V4KeyDataBytesPerLayerPage,
                            kK8V4KeyScaleBytesPerLayerPage,
                            kK8V4ValueDataBytesPerLayerPage,
                            kK8V4ValueScaleBytesPerLayerPage, layer, tensor,
                            index);
  };
  require(offset(0, SPLASH_KV_KEYS, 0) == 0, "the keys open a region");
  require(offset(0, SPLASH_KV_KEY_SCALES, 0) ==
              uint64_t{extentPages} * kK8V4KeyDataBytesPerLayerPage,
          "the key scales follow the keys");
  require(offset(0, SPLASH_KV_VALUES, 0) ==
              uint64_t{extentPages} * (kK8V4KeyDataBytesPerLayerPage +
                                       kK8V4KeyScaleBytesPerLayerPage),
          "the values follow the key scales");
  require(offset(0, SPLASH_KV_VALUE_SCALES, 0) ==
              uint64_t{extentPages} * (kK8V4KeyDataBytesPerLayerPage +
                                       kK8V4KeyScaleBytesPerLayerPage +
                                       kK8V4ValueDataBytesPerLayerPage),
          "the value scales follow the values");
  require(offset(1, SPLASH_KV_KEYS, 0) == region,
          "the next layer's region follows the first");
  require(offset(0, SPLASH_KV_KEYS, 2) == 2 * kK8V4KeyDataBytesPerLayerPage,
          "a key page advances by a key page's bytes");
  require(offset(0, SPLASH_KV_VALUES, 2) ==
              uint64_t{extentPages} * (kK8V4KeyDataBytesPerLayerPage +
                                       kK8V4KeyScaleBytesPerLayerPage) +
                  2 * kK8V4ValueDataBytesPerLayerPage,
          "a value page advances by a value page's bytes");
  require(splash_kv_tensor_bytes(kK8V4KeyDataBytesPerLayerPage,
                                 kK8V4KeyScaleBytesPerLayerPage,
                                 kK8V4ValueDataBytesPerLayerPage,
                                 kK8V4ValueScaleBytesPerLayerPage,
                                 SPLASH_KV_KEYS) ==
              kK8V4KeyDataBytesPerLayerPage,
          "the keys' tensor bytes");
  require(splash_kv_tensor_bytes(kK8V4KeyDataBytesPerLayerPage,
                                 kK8V4KeyScaleBytesPerLayerPage,
                                 kK8V4ValueDataBytesPerLayerPage,
                                 kK8V4ValueScaleBytesPerLayerPage,
                                 SPLASH_KV_VALUES) ==
              kK8V4ValueDataBytesPerLayerPage,
          "the values' tensor bytes");
  require(region == uint64_t{extentPages} * kK8V4BytesPerLayerPage,
          "a region is a layer page per extent page");
}

// A page of pseudo-random values survives the oracle's quantize and
// dequantize: the keys within half a step of their 127-level scale, the
// values within half a step of theirs at 7, which are different errors.
void testRoundTrip() {
  constexpr uint32_t kTokens = kPageTokens;
  const uint64_t elements =
      uint64_t{kTokens} * kK8V4KvHeads * kK8V4HeadDimension;
  std::vector<float> keys(elements), values(elements);
  uint32_t state = 0x9e3779b9U;
  const auto next = [&state]() {
    state = state * 1664525U + 1013904223U;
    return float(int32_t(state >> 8) - (1 << 23)) / float(1 << 23);
  };
  for (uint64_t i = 0; i < elements; ++i) {
    keys[i] = next() * 4.0F;
    values[i] = next() * 4.0F;
  }

  K8V4LayerPage page{};
  quantizeK8V4LayerPage(keys, values, kTokens, page);

  std::vector<float> restoredKeys(elements), restoredValues(elements);
  dequantizeK8V4LayerPage(page, kTokens, restoredKeys, restoredValues);

  for (uint32_t head = 0; head < kK8V4KvHeads; ++head) {
    for (uint32_t token = 0; token < kTokens; ++token) {
      float keyMaximum = 0.0F, valueMaximum = 0.0F;
      for (uint32_t dimension = 0; dimension < kK8V4HeadDimension;
           ++dimension) {
        const uint64_t index =
            k8v4_reference_detail::logicalIndex(token, head, dimension);
        keyMaximum = std::max(keyMaximum, std::abs(keys[index]));
        valueMaximum = std::max(valueMaximum, std::abs(values[index]));
      }
      const float keyStep = keyMaximum == 0.0F ? 0.0F : keyMaximum / 127.0F;
      const float valueStep =
          valueMaximum == 0.0F ? 0.0F : valueMaximum / 7.0F;
      require(page.keyScales[splash_kv_scale_element(head, token)] ==
                  keyStep,
              "the key scale is the row maximum over 127");
      require(page.valueScales[splash_kv_scale_element(head, token)] ==
                  valueStep,
              "the value scale is the row maximum over 7");
      for (uint32_t dimension = 0; dimension < kK8V4HeadDimension;
           ++dimension) {
        const uint64_t index =
            k8v4_reference_detail::logicalIndex(token, head, dimension);
        require(std::abs(restoredKeys[index] - keys[index]) <=
                    keyStep * 0.5F + 1e-5F,
                "a key round trip is within half a step of 127");
        require(std::abs(restoredValues[index] - values[index]) <=
                    valueStep * 0.5F + 1e-5F,
                "a value round trip is within half a step of 7");
      }
    }
  }
  // The keys are bytes of the full 7-bit range, the values nibbles of the
  // narrower one: neither region is left empty by a page of nonzero data.
  require(std::any_of(page.keys.begin(), page.keys.end(),
                      [](int8_t byte) { return byte != 0; }),
          "the keys were stored");
  require(std::any_of(page.values.begin(), page.values.end(),
                      [](uint8_t byte) { return byte != 0; }),
          "the values were packed");
  int8_t largestKey = 0;
  for (const int8_t byte : page.keys)
    largestKey = std::max(largestKey, int8_t(std::abs(int(byte))));
  require(largestKey > 7, "the keys were narrowed to the values' range");
}

// The packed value byte holds two neighbouring elements, the even one in the
// low nibble; the keys' element is one byte, so the two tensors' element
// indices agree only where their layouts do.
void testPacking() {
  K8V4LayerPage page{};
  k8v4_reference_detail::packNibble(page.values, 0, 5);
  k8v4_reference_detail::packNibble(page.values, 1, -3);
  require(page.values[0] == uint8_t(((-3 & 0xF) << 4) | 5),
          "a value pair packs into one byte");
  require(k8v4_reference_detail::unpackNibble(page.values.data(), 0) == 5,
          "the even value keeps the low nibble");
  require(k8v4_reference_detail::unpackNibble(page.values.data(), 1) == -3,
          "the odd value keeps the high nibble");
  k8v4_reference_detail::packNibble(page.values, 0, -1);
  require(k8v4_reference_detail::unpackNibble(page.values.data(), 1) == -3,
          "repacking a value keeps its neighbour");
  require(splash_kv_value_element(0, 0, 0) + 1 ==
              splash_kv_value_element(0, 1, 0),
          "dimension-major value tokens are adjacent elements");
  require(splash_kv_key_element(0, 0, 0) + 1 ==
              splash_kv_key_element(0, 0, 1),
          "key dimensions are adjacent elements");
  require(splash_kv_key_element(0, 1, 0) == kK8V4HeadDimension,
          "a token's keys start a head dimension later");
  require(splash_kv_value_element(0, 0, 1) == kPageTokens,
          "a dimension's values start a page of tokens later");
}

// The oracle refuses what it cannot represent.
void testRefusals() {
  std::vector<float> keys(kK8V4ElementsPerLayerPage, 0.0F);
  std::vector<float> values(kK8V4ElementsPerLayerPage, 0.0F);
  K8V4LayerPage page{};
  rejects([&] { quantizeK8V4LayerPage(keys, values, kPageTokens + 1, page); },
          "k8v4 KV valid token count exceeds one page",
          "a page of more tokens than it holds");
  std::vector<float> shortKeys(kK8V4ElementsPerLayerPage - 1, 0.0F);
  rejects([&] { quantizeK8V4LayerPage(shortKeys, values, kPageTokens, page); },
          "logical keys has the wrong logical page size",
          "a short key page");
  keys[0] = std::numeric_limits<float>::quiet_NaN();
  rejects([&] { quantizeK8V4LayerPage(keys, values, kPageTokens, page); },
          "logical keys contains a non-finite value", "a non-finite key");
  keys[0] = 0.0F;
  rejects([&] { dequantizeK8V4LayerPage(page, kPageTokens + 1, keys, values); },
          "k8v4 KV valid token count exceeds one page",
          "a dequantize past the page");
  std::vector<float> shortOut(kK8V4ElementsPerLayerPage - 1, 0.0F);
  rejects([&] { dequantizeK8V4LayerPage(page, kPageTokens, shortOut, values); },
          "logical keys has the wrong logical page size",
          "a short dequantize target");
}

} // namespace

int main() {
  testByteAccounting();
  testNames();
  testRegionOrder();
  testRoundTrip();
  testPacking();
  testRefusals();
  std::cout << "k8v4 paged kv tests passed\n";
  return 0;
}
