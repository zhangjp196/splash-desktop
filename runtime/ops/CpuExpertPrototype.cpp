#include "ops/CpuExpertPrototype.hpp"

#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace splash::ops::cpu_prototype {
namespace {

void validate(Bf16Matrix matrix, const char *name) {
  if (!matrix.outputSize || !matrix.inputSize ||
      uint64_t{matrix.outputSize} * matrix.inputSize >
          std::numeric_limits<size_t>::max() / sizeof(uint16_t) ||
      matrix.bytes.size() != size_t{matrix.outputSize} * matrix.inputSize * sizeof(uint16_t))
    throw std::invalid_argument(std::string("invalid BF16 ") + name + " matrix");
}

float decodeBf16(const std::byte *bytes) noexcept {
  uint16_t bits;
  std::memcpy(&bits, bytes, sizeof(bits));
  return std::bit_cast<float>(uint32_t{bits} << 16);
}

float decodeBf16(uint16_t bits) noexcept {
  return std::bit_cast<float>(uint32_t{bits} << 16);
}

uint16_t encodeBf16(float value) noexcept {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  // Round to nearest, ties to even, as for Metal's bfloat conversion.
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

void multiply(Bf16Matrix weights, std::span<const float> input,
              std::span<float> output) {
  for (uint32_t row = 0; row < weights.outputSize; ++row) {
    float sum = 0.0f;
    const std::byte *weight = weights.bytes.data() + size_t{row} * weights.inputSize * 2;
    for (uint32_t column = 0; column < weights.inputSize; ++column)
      sum += decodeBf16(weight + size_t{column} * 2) * input[column];
    output[row] = sum;
  }
}

} // namespace

Q4_0Matrix Q4_0Matrix::fromRows(const QuantizedSegment &segment) {
  if (segment.layout != QuantizedSegment::Layout::Rows || segment.formatId != GGUF_FMT_Q40 ||
      segment.inputSize % 32 || segment.plane0.storage() != metal::BufferStorage::Shared)
    throw std::invalid_argument("CPU expert requires Shared native Q4_0 rows");
  const uint64_t rowBytes = uint64_t{segment.inputSize / 32} * 18;
  const uint64_t bytes = uint64_t{segment.outputSize} * rowBytes;
  const void *contents = segment.plane0.contents();
  if (!segment.outputSize || !contents || segment.plane0.sizeBytes() < bytes ||
      bytes > std::numeric_limits<size_t>::max())
    throw std::invalid_argument("Q4_0 expert weight rows are truncated or inaccessible");
  return {{static_cast<const std::byte *>(contents), static_cast<size_t>(bytes)},
          segment.outputSize, segment.inputSize};
}

std::vector<float> decodeQ4_0Row(Q4_0Matrix matrix, uint32_t row) {
  if (!matrix.outputSize || !matrix.inputSize || matrix.inputSize % 32 || row >= matrix.outputSize ||
      matrix.bytes.size() != uint64_t{matrix.outputSize} * (matrix.inputSize / 32) * 18)
    throw std::invalid_argument("invalid Q4_0 row matrix");
  std::vector<float> result(matrix.inputSize);
  const std::byte *block = matrix.bytes.data() + uint64_t{row} * (matrix.inputSize / 32) * 18;
  for (uint32_t b = 0; b < matrix.inputSize / 32; ++b, block += 18) {
    uint16_t scaleBits;
    std::memcpy(&scaleBits, block, sizeof(scaleBits));
    const float scale = static_cast<float>(std::bit_cast<_Float16>(scaleBits));
    for (uint32_t i = 0; i < 16; ++i) {
      const uint8_t packed = std::to_integer<uint8_t>(block[2 + i]);
      result[b * 32 + i] = scale * (static_cast<int>(packed & 15) - 8);
      result[b * 32 + i + 16] = scale * (static_cast<int>(packed >> 4) - 8);
    }
  }
  return result;
}

std::vector<float> runBf16Expert(Bf16Matrix gate, Bf16Matrix up,
                                 Bf16Matrix down,
                                 std::span<const float> input) {
  validate(gate, "gate");
  validate(up, "up");
  validate(down, "down");
  if (gate.inputSize != input.size() || up.inputSize != input.size() ||
      gate.outputSize != up.outputSize || down.inputSize != gate.outputSize)
    throw std::invalid_argument("BF16 expert matrix dimensions do not compose");

  std::vector<float> gated(gate.outputSize), upper(up.outputSize);
  multiply(gate, input, gated);
  multiply(up, input, upper);
  for (size_t i = 0; i < gated.size(); ++i) {
    const float value = gated[i];
    gated[i] = (value / (1.0f + std::exp(-value))) * upper[i];
  }
  std::vector<float> output(down.outputSize);
  multiply(down, gated, output);
  return output;
}

std::vector<Bf16Expert> makeBf16ExpertViews(
    const QuantizedSegment &rg, const QuantizedSegment &ru,
    const QuantizedSegment &rd, std::span<const uint32_t> slots,
    const QuantizedSegment &sg, const QuantizedSegment &su,
    const QuantizedSegment &sd) {
  auto validateRows = [](const QuantizedSegment &s) {
    if (!s.isFloat() || s.layout != QuantizedSegment::Layout::Rows ||
        s.formatId != GGUF_FMT_BF16 || s.columnOffset != 0)
      throw std::invalid_argument("CPU MoE adapter supports contiguous BF16 row layout only");
    (void)Bf16Matrix::fromRows(s);
  };
  for (const auto *s : {&rg, &ru, &rd, &sg, &su, &sd}) validateRows(*s);
  if (slots.empty() || rg.outputSize % slots.size() || ru.outputSize % slots.size() ||
      rd.outputSize % slots.size()) throw std::invalid_argument("invalid routed BF16 slot dimensions");
  const uint32_t gn = rg.outputSize / static_cast<uint32_t>(slots.size());
  const uint32_t un = ru.outputSize / static_cast<uint32_t>(slots.size());
  const uint32_t dn = rd.outputSize / static_cast<uint32_t>(slots.size());
  if (!gn || gn != un || !dn || rg.inputSize != ru.inputSize || sg.outputSize != gn ||
      su.outputSize != un || sd.outputSize != dn || sg.inputSize != rg.inputSize ||
      su.inputSize != ru.inputSize || sd.inputSize != rd.inputSize)
    throw std::invalid_argument("routed and shared BF16 expert shapes differ");
  auto slice = [](Bf16Matrix m, uint32_t index, uint32_t rows) {
    const size_t bytes = size_t{rows} * m.inputSize * sizeof(uint16_t);
    return Bf16Matrix{m.bytes.subspan(size_t{index} * bytes, bytes), rows, m.inputSize};
  };
  std::vector<Bf16Expert> result(slots.size() + 1);
  std::vector<uint8_t> used(slots.size());
  for (size_t expert=0; expert<slots.size(); ++expert) {
    const uint32_t slot=slots[expert];
    if (slot>=slots.size() || used[slot]++) throw std::invalid_argument("invalid routed expert slot table");
    result[expert]={slice(Bf16Matrix::fromRows(rg),slot,gn),slice(Bf16Matrix::fromRows(ru),slot,un),slice(Bf16Matrix::fromRows(rd),slot,dn)};
  }
  result.back()={Bf16Matrix::fromRows(sg),Bf16Matrix::fromRows(su),Bf16Matrix::fromRows(sd)};
  return result;
}

std::vector<Q4_0Expert> makeQ4_0ExpertViews(
    const QuantizedSegment &rg, const QuantizedSegment &ru,
    const QuantizedSegment &rd, std::span<const uint32_t> slots,
    const QuantizedSegment &sg, const QuantizedSegment &su,
    const QuantizedSegment &sd) {
  auto validateRows = [](const QuantizedSegment &s) {
    if (s.layout != QuantizedSegment::Layout::Rows || s.formatId != GGUF_FMT_Q40 ||
        s.columnOffset != 0 || s.inputSize % 32)
      throw std::invalid_argument("CPU MoE adapter supports native Q4_0 rows only");
    (void)Q4_0Matrix::fromRows(s);
  };
  for (const auto *s : {&rg, &ru, &rd, &sg, &su, &sd}) validateRows(*s);
  if (slots.empty() || rg.outputSize % slots.size() || ru.outputSize % slots.size() ||
      rd.outputSize % slots.size()) throw std::invalid_argument("invalid routed Q4_0 slot dimensions");
  const uint32_t gn = rg.outputSize / static_cast<uint32_t>(slots.size());
  const uint32_t un = ru.outputSize / static_cast<uint32_t>(slots.size());
  const uint32_t dn = rd.outputSize / static_cast<uint32_t>(slots.size());
  if (!gn || gn != un || !dn || rg.inputSize != ru.inputSize || sg.outputSize != gn ||
      su.outputSize != un || sd.outputSize != dn || sg.inputSize != rg.inputSize ||
      su.inputSize != ru.inputSize || sd.inputSize != rd.inputSize)
    throw std::invalid_argument("routed and shared Q4_0 expert shapes differ");
  auto slice = [](Q4_0Matrix m, uint32_t index, uint32_t rows) {
    const size_t bytes = size_t{rows} * (m.inputSize / 32) * 18;
    return Q4_0Matrix{m.bytes.subspan(size_t{index} * bytes, bytes), rows, m.inputSize};
  };
  const Q4_0Matrix gate = Q4_0Matrix::fromRows(rg), up = Q4_0Matrix::fromRows(ru),
                   down = Q4_0Matrix::fromRows(rd);
  std::vector<Q4_0Expert> result(slots.size() + 1);
  std::vector<uint8_t> used(slots.size());
  for (size_t expert = 0; expert < slots.size(); ++expert) {
    const uint32_t slot = slots[expert];
    if (slot >= slots.size() || used[slot]++)
      throw std::invalid_argument("invalid routed expert slot table");
    result[expert] = {slice(gate, slot, gn), slice(up, slot, un), slice(down, slot, dn)};
  }
  result.back() = {Q4_0Matrix::fromRows(sg), Q4_0Matrix::fromRows(su), Q4_0Matrix::fromRows(sd)};
  return result;
}

std::vector<float> runBf16GroupedExperts(
    std::span<const Bf16Expert> experts,
    std::span<const MoeTileDescriptor> tileDescriptors,
    std::span<const uint32_t> groupedRoutes,
    std::span<const uint32_t> routeRows,
    std::span<const uint16_t> groupedInput, uint32_t inputRows,
    uint32_t routesPerRow, uint32_t tileRows, uint32_t hiddenSize) {
  if (!inputRows || !routesPerRow || !tileRows || !hiddenSize || experts.empty())
    throw std::invalid_argument("BF16 grouped experts require nonzero dimensions");
  const uint64_t routeCount = uint64_t{inputRows} * routesPerRow;
  const uint64_t groupedCapacity = uint64_t{tileDescriptors.size()} * tileRows;
  const uint64_t inputElements = groupedCapacity * hiddenSize;
  if (routeCount > std::numeric_limits<size_t>::max() ||
      groupedCapacity > std::numeric_limits<uint32_t>::max() ||
      inputElements > std::numeric_limits<size_t>::max() ||
      groupedRoutes.size() != groupedCapacity || routeRows.size() != routeCount ||
      groupedInput.size() != inputElements)
    throw std::invalid_argument("invalid BF16 grouped expert buffer dimensions");

  for (const Bf16Expert &expert : experts) {
    validate(expert.gate, "gate");
    validate(expert.up, "up");
    validate(expert.down, "down");
    if (expert.gate.inputSize != hiddenSize || expert.up.inputSize != hiddenSize ||
        expert.gate.outputSize != expert.up.outputSize ||
        expert.down.inputSize != expert.gate.outputSize ||
        expert.down.outputSize != hiddenSize)
      throw std::invalid_argument("BF16 grouped expert dimensions do not compose");
  }

  std::vector<uint8_t> seen(routeRows.size(), 0);
  std::vector<float> output(static_cast<size_t>(inputElements), 0.0f);
  for (size_t tile = 0; tile < tileDescriptors.size(); ++tile) {
    const MoeTileDescriptor descriptor = tileDescriptors[tile];
    if (descriptor.expert >= experts.size() || !descriptor.rows || descriptor.rows > tileRows ||
        (tile && descriptor.expert < tileDescriptors[tile - 1].expert))
      throw std::invalid_argument("invalid BF16 MoE tile descriptor");
    const Bf16Expert &expert = experts[descriptor.expert];
    const size_t base = tile * tileRows;
    const uint32_t intermediateSize = expert.gate.outputSize;
    std::vector<float> gated(size_t{descriptor.rows} * intermediateSize);
    std::vector<float> upper(gated.size());
    for (uint32_t row = 0; row < descriptor.rows; ++row) {
      const uint32_t route = groupedRoutes[base + row];
      if (route >= routeCount || routeRows[route] != base + row || seen[route])
        throw std::invalid_argument("invalid BF16 MoE grouped route mapping");
      seen[route] = 1;
      const size_t inputBase = (base + row) * hiddenSize;
      for (uint32_t out = 0; out < intermediateSize; ++out) {
        float gateSum = 0.0f;
        float upSum = 0.0f;
        const std::byte *gateWeight = expert.gate.bytes.data() + size_t{out} * hiddenSize * 2;
        const std::byte *upWeight = expert.up.bytes.data() + size_t{out} * hiddenSize * 2;
        for (uint32_t column = 0; column < hiddenSize; ++column) {
          const float value = decodeBf16(groupedInput[inputBase + column]);
          gateSum += decodeBf16(gateWeight + size_t{column} * 2) * value;
          upSum += decodeBf16(upWeight + size_t{column} * 2) * value;
        }
        const size_t index = size_t{row} * intermediateSize + out;
        gated[index] = gateSum;
        upper[index] = upSum;
      }
    }
    for (uint32_t row = 0; row < descriptor.rows; ++row) {
      for (uint32_t column = 0; column < intermediateSize; ++column) {
        const size_t index = size_t{row} * intermediateSize + column;
        const float value = gated[index];
        gated[index] = (value / (1.0f + std::exp(-value))) * upper[index];
      }
      for (uint32_t out = 0; out < hiddenSize; ++out) {
        float sum = 0.0f;
        const std::byte *weight = expert.down.bytes.data() + size_t{out} * intermediateSize * 2;
        for (uint32_t column = 0; column < intermediateSize; ++column)
          sum += decodeBf16(weight + size_t{column} * 2) *
                 gated[size_t{row} * intermediateSize + column];
        output[(base + row) * hiddenSize + out] = sum;
      }
    }
    for (uint32_t row = descriptor.rows; row < tileRows; ++row)
      if (groupedRoutes[base + row] != UINT32_MAX)
        throw std::invalid_argument("BF16 MoE padded grouped row is not empty");
  }
  for (size_t route = 0; route < routeRows.size(); ++route) {
    const uint32_t grouped = routeRows[route];
    if (grouped == UINT32_MAX) continue;
    if (grouped >= groupedCapacity || !seen[route])
      throw std::invalid_argument("BF16 MoE route row is out of range");
  }
  return output;
}

std::vector<float> runQ4_0GroupedExperts(
    std::span<const Q4_0Expert> experts,
    std::span<const MoeTileDescriptor> tileDescriptors,
    std::span<const uint32_t> groupedRoutes,
    std::span<const uint32_t> routeRows,
    std::span<const uint16_t> groupedInput, uint32_t inputRows,
    uint32_t routesPerRow, uint32_t tileRows, uint32_t hiddenSize) {
  if (!inputRows || !routesPerRow || !tileRows || !hiddenSize || experts.empty())
    throw std::invalid_argument("Q4_0 grouped experts require nonzero dimensions");
  const uint64_t routeCount = uint64_t{inputRows} * routesPerRow;
  const uint64_t groupedCapacity = uint64_t{tileDescriptors.size()} * tileRows;
  const uint64_t inputElements = groupedCapacity * hiddenSize;
  if (routeCount > std::numeric_limits<size_t>::max() ||
      groupedCapacity > std::numeric_limits<uint32_t>::max() ||
      inputElements > std::numeric_limits<size_t>::max() ||
      groupedRoutes.size() != groupedCapacity || routeRows.size() != routeCount ||
      groupedInput.size() != inputElements)
    throw std::invalid_argument("invalid Q4_0 grouped expert buffer dimensions");
  auto validateQ4 = [](Q4_0Matrix matrix, const char *name) {
    if (!matrix.outputSize || !matrix.inputSize || matrix.inputSize % 32 ||
        matrix.bytes.size() != uint64_t{matrix.outputSize} * (matrix.inputSize / 32) * 18)
      throw std::invalid_argument(std::string("invalid Q4_0 ") + name + " matrix");
  };
  for (const Q4_0Expert &expert : experts) {
    validateQ4(expert.gate, "gate");
    validateQ4(expert.up, "up");
    validateQ4(expert.down, "down");
    if (expert.gate.inputSize != hiddenSize || expert.up.inputSize != hiddenSize ||
        expert.gate.outputSize != expert.up.outputSize ||
        expert.down.inputSize != expert.gate.outputSize ||
        expert.down.outputSize != hiddenSize)
      throw std::invalid_argument("Q4_0 grouped expert dimensions do not compose");
  }
  auto dot = [](Q4_0Matrix matrix, uint32_t row, const uint16_t *input) {
    const std::byte *block = matrix.bytes.data() +
        uint64_t{row} * (matrix.inputSize / 32) * 18;
    float sum = 0.0f;
    for (uint32_t b = 0; b < matrix.inputSize / 32; ++b, block += 18) {
      uint16_t scaleBits;
      std::memcpy(&scaleBits, block, sizeof(scaleBits));
      const float scale = static_cast<float>(std::bit_cast<_Float16>(scaleBits));
      for (uint32_t i = 0; i < 16; ++i) {
        const uint8_t packed = std::to_integer<uint8_t>(block[2 + i]);
        sum += scale * (static_cast<int>(packed & 15) - 8) * decodeBf16(input[b * 32 + i]);
        sum += scale * (static_cast<int>(packed >> 4) - 8) * decodeBf16(input[b * 32 + i + 16]);
      }
    }
    return sum;
  };
  std::vector<uint8_t> seen(routeRows.size(), 0);
  std::vector<float> output(static_cast<size_t>(inputElements), 0.0f);
  for (size_t tile = 0; tile < tileDescriptors.size(); ++tile) {
    const MoeTileDescriptor descriptor = tileDescriptors[tile];
    if (descriptor.expert >= experts.size() || !descriptor.rows || descriptor.rows > tileRows ||
        (tile && descriptor.expert < tileDescriptors[tile - 1].expert))
      throw std::invalid_argument("invalid Q4_0 MoE tile descriptor");
    const Q4_0Expert &expert = experts[descriptor.expert];
    const size_t base = tile * tileRows;
    const uint32_t intermediateSize = expert.gate.outputSize;
    std::vector<float> gated(size_t{descriptor.rows} * intermediateSize);
    std::vector<float> upper(gated.size());
    for (uint32_t row = 0; row < descriptor.rows; ++row) {
      const uint32_t route = groupedRoutes[base + row];
      if (route >= routeCount || routeRows[route] != base + row || seen[route])
        throw std::invalid_argument("invalid Q4_0 MoE grouped route mapping");
      seen[route] = 1;
      const uint16_t *input = groupedInput.data() + (base + row) * hiddenSize;
      for (uint32_t out = 0; out < intermediateSize; ++out) {
        const size_t index = size_t{row} * intermediateSize + out;
        gated[index] = dot(expert.gate, out, input);
        upper[index] = dot(expert.up, out, input);
      }
    }
    for (uint32_t row = 0; row < descriptor.rows; ++row) {
      for (uint32_t column = 0; column < intermediateSize; ++column) {
        const size_t index = size_t{row} * intermediateSize + column;
        const float value = gated[index];
        gated[index] = (value / (1.0f + std::exp(-value))) * upper[index];
      }
      for (uint32_t out = 0; out < hiddenSize; ++out) {
        const std::byte *block = expert.down.bytes.data() +
            uint64_t{out} * (intermediateSize / 32) * 18;
        float sum = 0.0f;
        for (uint32_t b = 0; b < intermediateSize / 32; ++b, block += 18) {
          uint16_t scaleBits;
          std::memcpy(&scaleBits, block, sizeof(scaleBits));
          const float scale = static_cast<float>(std::bit_cast<_Float16>(scaleBits));
          for (uint32_t i = 0; i < 16; ++i) {
            const uint8_t packed = std::to_integer<uint8_t>(block[2 + i]);
            sum += scale * (static_cast<int>(packed & 15) - 8) * gated[size_t{row} * intermediateSize + b * 32 + i];
            sum += scale * (static_cast<int>(packed >> 4) - 8) * gated[size_t{row} * intermediateSize + b * 32 + i + 16];
          }
        }
        output[(base + row) * hiddenSize + out] = sum;
      }
    }
    for (uint32_t row = descriptor.rows; row < tileRows; ++row)
      if (groupedRoutes[base + row] != UINT32_MAX)
        throw std::invalid_argument("Q4_0 MoE padded grouped row is not empty");
  }
  for (size_t route = 0; route < routeRows.size(); ++route) {
    const uint32_t grouped = routeRows[route];
    if (grouped == UINT32_MAX) continue;
    if (grouped >= groupedCapacity || !seen[route])
      throw std::invalid_argument("Q4_0 MoE route row is out of range");
  }
  return output;
}

void mergeBf16Experts(std::span<const uint16_t> expertOutput,
                      std::span<const uint32_t> routeRows,
                      std::span<const float> routingWeights,
                      std::span<const uint16_t> residual,
                      std::span<uint16_t> output, uint32_t rows,
                      uint32_t routesPerRow, uint32_t hiddenSize) {
  if (!rows || !hiddenSize)
    throw std::invalid_argument("BF16 MoE merge requires nonzero rows and hidden size");
  const uint64_t outputElements = uint64_t{rows} * hiddenSize;
  const uint64_t routeCount = uint64_t{rows} * routesPerRow;
  if (outputElements > std::numeric_limits<size_t>::max() ||
      routeCount > std::numeric_limits<size_t>::max() ||
      residual.size() != outputElements || output.size() != outputElements ||
      routeRows.size() != routeCount || routingWeights.size() != routeCount ||
      expertOutput.size() % hiddenSize != 0)
    throw std::invalid_argument("invalid BF16 MoE merge buffer dimensions");
  const size_t groupedRows = expertOutput.size() / hiddenSize;
  for (size_t route = 0; route < routeRows.size(); ++route)
    if (routeRows[route] != UINT32_MAX && routeRows[route] >= groupedRows)
      throw std::invalid_argument("BF16 MoE route row is out of range");

  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t dimension = 0; dimension < hiddenSize; ++dimension) {
      float value = decodeBf16(residual[size_t{row} * hiddenSize + dimension]);
      const size_t routeBase = size_t{row} * routesPerRow;
      for (uint32_t slot = 0; slot < routesPerRow; ++slot) {
        const size_t route = routeBase + slot;
        const uint32_t grouped = routeRows[route];
        if (grouped == UINT32_MAX)
          continue;
        value += routingWeights[route] *
                 decodeBf16(expertOutput[size_t{grouped} * hiddenSize + dimension]);
      }
      output[size_t{row} * hiddenSize + dimension] = encodeBf16(value);
    }
  }
}

} // namespace splash::ops::cpu_prototype
