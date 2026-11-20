#include "ops/CpuExpertPrototype.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

std::vector<std::byte> bf16(std::initializer_list<float> values) {
  std::vector<std::byte> result;
  for (float value : values) {
    const uint16_t bits = static_cast<uint16_t>(std::bit_cast<uint32_t>(value) >> 16);
    const auto *bytes = reinterpret_cast<const std::byte *>(&bits);
    result.insert(result.end(), bytes, bytes + sizeof(bits));
  }
  return result;
}

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

} // namespace

int main() {
  using splash::ops::cpu_prototype::Bf16Matrix;
  using splash::ops::cpu_prototype::runBf16Expert;
  // 2-wide input, 2-wide intermediate, 1-wide output; all weights are 1.
  const auto gate = bf16({1, 1, 1, 1});
  const auto up = bf16({1, 1, 1, 1});
  const auto down = bf16({1, 1});
  const auto output = runBf16Expert({gate, 2, 2}, {up, 2, 2}, {down, 1, 2},
                                    std::vector<float>{1, 2});
  const float expected = 6.0f * (3.0f / (1.0f + std::exp(-3.0f)));
  require(output.size() == 1 && std::fabs(output[0] - expected) < 1e-6f,
          "expert GEMM/SiLU result differs");
  bool rejected = false;
  try {
    static_cast<void>(runBf16Expert({gate, 2, 2}, {up, 2, 2}, {down, 1, 2},
                                    std::vector<float>{1}));
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected, "mismatched input dimensions were accepted");
  auto bits = [](float value) {
    return static_cast<uint16_t>(std::bit_cast<uint32_t>(value) >> 16);
  };
  using splash::ops::cpu_prototype::Bf16Expert;
  using splash::ops::cpu_prototype::runBf16GroupedExperts;
  const auto identity = bf16({1, 0, 0, 1});
  const auto doubled = bf16({2, 2, 2, 2});
  const std::vector<Bf16Expert> experts = {
      {{gate, 2, 2}, {up, 2, 2}, {identity, 2, 2}},
      {{doubled, 2, 2}, {doubled, 2, 2}, {identity, 2, 2}}};
  const std::vector<MoeTileDescriptor> tiles = {{0, 2}, {1, 1}};
  const std::vector<uint32_t> groupedRoutes = {0, 2, 1, UINT32_MAX};
  const std::vector<uint32_t> groupedRouteRows = {0, 2, 1, UINT32_MAX};
  const std::vector<uint16_t> groupedInput = {bits(1), bits(2), bits(2), bits(1),
                                               bits(3), bits(1), bits(0), bits(0)};
  const auto groupedOutput = runBf16GroupedExperts(
      experts, tiles, groupedRoutes, groupedRouteRows, groupedInput,
      2, 2, 2, 2);
  const float expert0Row0 = 9.0f / (1.0f + std::exp(-3.0f));
  const float expert0Row1 = 9.0f / (1.0f + std::exp(-3.0f));
  const float expert1Row = 64.0f / (1.0f + std::exp(-8.0f));
  require(groupedOutput.size() == 8 &&
              std::fabs(groupedOutput[0] - expert0Row0) < 1e-6f &&
              std::fabs(groupedOutput[1] - expert0Row0) < 1e-6f &&
              std::fabs(groupedOutput[2] - expert0Row1) < 1e-6f &&
              std::fabs(groupedOutput[4] - expert1Row) < 1e-6f &&
              groupedOutput[6] == 0.0f && groupedOutput[7] == 0.0f,
          "multi-expert grouped BF16 rows differ");
  rejected = false;
  try {
    const std::vector<MoeTileDescriptor> invalidTiles = {{2, 1}};
    static_cast<void>(runBf16GroupedExperts(experts, invalidTiles,
        std::vector<uint32_t>{0, UINT32_MAX},
        std::vector<uint32_t>{0, UINT32_MAX},
        std::vector<uint16_t>{bits(1), bits(2), bits(0), bits(0)},
        2, 1, 2, 2));
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected, "invalid grouped expert id was accepted");
  using splash::ops::cpu_prototype::mergeBf16Experts;
  auto value = [](uint16_t encoded) {
    return std::bit_cast<float>(uint32_t{encoded} << 16);
  };
  // Two output rows, two routes each, width two. The UINT32_MAX route is
  // absent; the other routes use non-unit weights and preserve residual.
  const std::vector<uint16_t> grouped = {bits(2), bits(4), bits(10), bits(-2)};
  const std::vector<uint32_t> routeRows = {1, UINT32_MAX, 0, 1};
  const std::vector<float> weights = {0.5f, 99.0f, 2.0f, -0.25f};
  const std::vector<uint16_t> residual = {bits(1), bits(-1), bits(3), bits(5)};
  std::vector<uint16_t> merged(4);
  mergeBf16Experts(grouped, routeRows, weights, residual, merged, 2, 2, 2);
  require(value(merged[0]) == 6.0f && value(merged[1]) == -2.0f &&
              value(merged[2]) == 4.5f && value(merged[3]) == 13.5f,
          "BF16 MoE merge differs from Metal residual/route semantics");
  rejected = false;
  try {
    const std::vector<uint32_t> invalidRoutes = {2, UINT32_MAX, 0, 1};
    mergeBf16Experts(grouped, invalidRoutes, weights, residual, merged, 2, 2, 2);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected, "out-of-range grouped route row was accepted");
  using splash::ops::cpu_prototype::Q4_0Matrix;
  using splash::ops::cpu_prototype::decodeQ4_0Row;
  // GGUF Q4_0 known vector: d=0.5, packed 0x98 gives q0=8, q16=9.
  std::vector<std::byte> q4(18, std::byte{0});
  const uint16_t half = 0x3800; // 0.5
  std::memcpy(q4.data(), &half, 2);
  q4[2] = std::byte{0x98};
  const auto decoded = decodeQ4_0Row({q4, 1, 32}, 0);
  require(decoded[0] == 0.0f && decoded[16] == 0.5f && decoded[1] == -4.0f,
          "native GGUF Q4_0 known vector differs");
  using splash::ops::cpu_prototype::Q4_0Expert;
  using splash::ops::cpu_prototype::runQ4_0GroupedExperts;
  auto q4HalfWeights = [] {
    std::vector<std::byte> bytes(32 * 18, std::byte{0});
    const uint16_t scale = 0x3800;
    for (size_t row = 0; row < 32; ++row) {
      std::memcpy(bytes.data() + row * 18, &scale, sizeof(scale));
      for (size_t i = 0; i < 16; ++i) bytes[row * 18 + 2 + i] = std::byte{0x99};
    }
    return bytes;
  };
  const auto q4Weights = q4HalfWeights();
  const splash::ops::cpu_prototype::Q4_0Matrix q4Matrix{q4Weights, 32, 32};
  const std::vector<Q4_0Expert> q4Experts = {{q4Matrix, q4Matrix, q4Matrix},
                                             {q4Matrix, q4Matrix, q4Matrix}};
  const std::vector<MoeTileDescriptor> q4Tiles = {{0, 1}, {1, 1}};
  const std::vector<uint32_t> q4GroupedRoutes = {0, UINT32_MAX, 1, UINT32_MAX};
  const std::vector<uint32_t> q4RouteRows = {0, 2};
  std::vector<uint16_t> q4Input(4 * 32, bits(0));
  for (size_t i = 0; i < 32; ++i) q4Input[i] = bits(1);
  for (size_t i = 64; i < 96; ++i) q4Input[i] = bits(1);
  const auto q4Output = runQ4_0GroupedExperts(q4Experts, q4Tiles,
      q4GroupedRoutes, q4RouteRows, q4Input, 1, 2, 2, 32);
  const float q4Expected = 16.0f * (16.0f / (1.0f + std::exp(-16.0f))) * 16.0f;
  require(q4Output.size() == 128 && std::fabs(q4Output[0] - q4Expected) < 1e-3f &&
              std::fabs(q4Output[64] - q4Expected) < 1e-3f &&
              q4Output[32] == 0.0f && q4Output[96] == 0.0f,
          "native Q4_0 grouped gate/up/down, expert rows, or padding differs");
  std::cout << "cpu_expert_prototype_test: PASS\n";
}
