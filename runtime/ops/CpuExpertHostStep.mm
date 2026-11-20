#include "ops/CpuExpertPrototype.hpp"

#include <bit>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace splash::ops::cpu_prototype {
namespace {

float decode(const uint16_t bits) noexcept {
  return std::bit_cast<float>(uint32_t{bits} << 16);
}

uint16_t encode(float value) noexcept {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  return static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

} // namespace

Bf16Matrix Bf16Matrix::fromRows(const QuantizedSegment &segment) {
  if (!segment.isFloat() || segment.formatId != GGUF_FMT_BF16 ||
      segment.layout != QuantizedSegment::Layout::Rows ||
      segment.plane0.storage() != metal::BufferStorage::Shared)
    throw std::invalid_argument("CPU expert requires Shared BF16 row weights");
  const uint64_t elements = uint64_t{segment.outputSize} * segment.inputSize;
  if (!segment.outputSize || !segment.inputSize ||
      elements > std::numeric_limits<size_t>::max() / sizeof(uint16_t) ||
      segment.plane0.sizeBytes() < elements * sizeof(uint16_t))
    throw std::invalid_argument("BF16 expert weight rows are truncated");
  const void *contents = segment.plane0.contents();
  if (!contents) throw std::invalid_argument("BF16 expert weights are not host visible");
  return {{static_cast<const std::byte *>(contents), static_cast<size_t>(elements * sizeof(uint16_t))},
          segment.outputSize, segment.inputSize};
}

void addBf16ExpertHostStep(metal::CommandGraph &graph,
                           const QuantizedSegment &gate,
                           const QuantizedSegment &up,
                           const QuantizedSegment &down,
                           metal::MetalBuffer input,
                           metal::MetalBuffer output,
                           uint32_t rows) {
  if (!rows || gate.inputSize != up.inputSize || gate.outputSize != up.outputSize ||
      down.inputSize != gate.outputSize)
    throw std::invalid_argument("BF16 expert host-step dimensions do not compose");
  const uint64_t inputElements = uint64_t{rows} * gate.inputSize;
  const uint64_t outputElements = uint64_t{rows} * down.outputSize;
  if (inputElements > std::numeric_limits<uint64_t>::max() / 2 ||
      outputElements > std::numeric_limits<uint64_t>::max() / 2 ||
      input.sizeBytes() < inputElements * 2 || output.sizeBytes() < outputElements * 2)
    throw std::invalid_argument("BF16 expert host-step activation buffer is undersized");
  graph.host([gate, up, down, input = std::move(input), output = std::move(output), rows] {
    if (input.storage() != metal::BufferStorage::Shared ||
        output.storage() != metal::BufferStorage::Shared || !input.contents() || !output.contents())
      throw std::runtime_error("BF16 expert host step requires Shared activation buffers");
    const Bf16Matrix gateMatrix = Bf16Matrix::fromRows(gate);
    const Bf16Matrix upMatrix = Bf16Matrix::fromRows(up);
    const Bf16Matrix downMatrix = Bf16Matrix::fromRows(down);
    const auto *source = static_cast<const uint16_t *>(input.contents());
    auto *destination = static_cast<uint16_t *>(output.contents());
    std::vector<float> values(gate.inputSize);
    for (uint32_t row = 0; row < rows; ++row) {
      const uint16_t *inputRow = source + size_t{row} * gate.inputSize;
      for (uint32_t i = 0; i < gate.inputSize; ++i) values[i] = decode(inputRow[i]);
      const std::vector<float> result = runBf16Expert(gateMatrix, upMatrix, downMatrix, values);
      uint16_t *outputRow = destination + size_t{row} * down.outputSize;
      for (uint32_t i = 0; i < down.outputSize; ++i) outputRow[i] = encode(result[i]);
    }
  });
}

} // namespace splash::ops::cpu_prototype
