#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "metal/CommandGraph.hpp"
#include "metal/abi/MoE.h"
#include "ops/Weights.hpp"

namespace splash::ops::cpu_prototype {

// Explicitly invoked CPU expert path. Weight bytes are
// row-major [output, input] BF16 as stored by GGUF/Safetensors/repack readers.
// This API never accepts a MetalBuffer: the caller must supply host-readable
// weight bytes and activations whose lifetime covers the call.
// Native GGUF Q4_0 rows: each 32-value block is fp16 d followed by 16
// packed bytes (low nibble = values 0..15, high nibble = values 16..31).
// The span must contain original row blocks, never Metal repacked planes.
struct Q4_0Matrix final {
  std::span<const std::byte> bytes;
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
  [[nodiscard]] static Q4_0Matrix fromRows(const QuantizedSegment &segment);
};

[[nodiscard]] std::vector<float> decodeQ4_0Row(Q4_0Matrix matrix, uint32_t row);

struct Bf16Matrix final {
  std::span<const std::byte> bytes;
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;

  // A zero-copy view over a real BF16 row-layout tensor segment. The segment
  // must remain alive and host-visible for the duration of CPU execution.
  [[nodiscard]] static Bf16Matrix fromRows(const QuantizedSegment &segment);
};

// Computes down(silu(gate(input)) * up(input)). Accumulates in float and
// returns float activations. Throws on malformed dimensions or byte lengths.
[[nodiscard]] std::vector<float> runBf16Expert(
    Bf16Matrix gate, Bf16Matrix up, Bf16Matrix down,
    std::span<const float> input);

// The three BF16 projections for one expert. Expert ids in tile descriptors
// index this host metadata directly, including any shared expert slices.
struct Bf16Expert final {
  Bf16Matrix gate;
  Bf16Matrix up;
  Bf16Matrix down;
};

// Native Q4_0 weights with BF16 grouped activations. Matrices retain their
// packed row bytes and are decoded a block at a time during FP32 dot products.
struct Q4_0Expert final {
  Q4_0Matrix gate;
  Q4_0Matrix up;
  Q4_0Matrix down;
};

// Builds expert-id-indexed matrices from routed slot-major row segments and
// one shared expert segment. slotTable maps expert id to routed slot; shared
// is appended at expert id slotTable.size(). Throws before CPU weight access
// unless all six segments are contiguous BF16 rows with compatible shapes.
[[nodiscard]] std::vector<Bf16Expert> makeBf16ExpertViews(
    const QuantizedSegment &routedGate, const QuantizedSegment &routedUp,
    const QuantizedSegment &routedDown, std::span<const uint32_t> slotTable,
    const QuantizedSegment &sharedGate, const QuantizedSegment &sharedUp,
    const QuantizedSegment &sharedDown);

// Same slot mapping and shape checks for six contiguous native Shared Q4_0 rows.
[[nodiscard]] std::vector<Q4_0Expert> makeQ4_0ExpertViews(
    const QuantizedSegment &routedGate, const QuantizedSegment &routedUp,
    const QuantizedSegment &routedDown, std::span<const uint32_t> slotTable,
    const QuantizedSegment &sharedGate, const QuantizedSegment &sharedUp,
    const QuantizedSegment &sharedDown);

// Runs every live grouped row through the expert named by its tile descriptor.
// Descriptors follow Metal grouping order; tile t occupies physical rows
// [t * tileRows, (t + 1) * tileRows), with only descriptor.rows rows live.
// groupedInput is BF16 row-major over physical grouped rows, groupedRoutes maps
// each physical grouped row to its flattened [input row, route slot] (or
// UINT32_MAX for padding), and routeRows is the reverse map. Output is FP32
// row-major over the same physical grouped rows; padding outputs are zero.
// All buffers are ordinary host spans, never MetalBuffer contents. This
// rejects malformed descriptors, route mappings, expert ids and extents.
[[nodiscard]] std::vector<float> runBf16GroupedExperts(
    std::span<const Bf16Expert> experts,
    std::span<const MoeTileDescriptor> tileDescriptors,
    std::span<const uint32_t> groupedRoutes,
    std::span<const uint32_t> routeRows,
    std::span<const uint16_t> groupedInput, uint32_t inputRows,
    uint32_t routesPerRow, uint32_t tileRows, uint32_t hiddenSize);

[[nodiscard]] std::vector<float> runQ4_0GroupedExperts(
    std::span<const Q4_0Expert> experts,
    std::span<const MoeTileDescriptor> tileDescriptors,
    std::span<const uint32_t> groupedRoutes,
    std::span<const uint32_t> routeRows,
    std::span<const uint16_t> groupedInput, uint32_t inputRows,
    uint32_t routesPerRow, uint32_t tileRows, uint32_t hiddenSize);

// Merges grouped BF16 expert rows into BF16 output rows. routeRows is indexed
// [rows, routesPerRow] and maps each route to a grouped expert-output row;
// UINT32_MAX skips a route, matching moe_combine. Routing weights are FP32,
// residual and output are BF16 bit patterns. The result is rounded to BF16
// once after adding residual and all weighted routes in slot order.
void mergeBf16Experts(std::span<const uint16_t> expertOutput,
                      std::span<const uint32_t> routeRows,
                      std::span<const float> routingWeights,
                      std::span<const uint16_t> residual,
                      std::span<uint16_t> output, uint32_t rows,
                      uint32_t routesPerRow, uint32_t hiddenSize);

// Appends a serialized host step that consumes and produces BF16 activation
// rows in Shared Metal buffers, using the three real BF16 tensor segments.
// Buffer contents are accessed only inside the host callback, after prior GPU
// work has completed. The callback throws if any buffer is not Shared or any
// segment is not a contiguous BF16 row tensor.
void addBf16ExpertHostStep(metal::CommandGraph &graph,
                           const QuantizedSegment &gate,
                           const QuantizedSegment &up,
                           const QuantizedSegment &down,
                           metal::MetalBuffer input,
                           metal::MetalBuffer output,
                           uint32_t rows);

} // namespace splash::ops::cpu_prototype
