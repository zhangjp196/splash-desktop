#pragma once

// Isolated host-side attention framework for the Inkling-2 operator wrappers.
//
// The fork's PagedAttention.hpp serves its four-lane prefill/verify kernels
// and cannot be replaced: it owns a different SplashChunkedPrefillParams ABI
// (with a layer field) and separate Prefill/Verify plan types. This header
// adds the upstream 1.7.2 host types the ported ops compile against, using
// the additive Inkling2AttentionABI.h for the kernel parameters. Existing
// fork families never include it, so their types and behavior are unchanged.

#include "metal/CommandGraph.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Inkling2AttentionABI.h"
#include "metal/abi/VerifyTree.h"
#include "ops/BufferExtent.hpp"
#include "ops/LaneBindings.hpp"
#include "ops/Linear.hpp"
#include "ops/PagedAttention.hpp"
#include "ops/PagedKv.hpp"
#include "ops/inkling2_LaneRows.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace splash::ops {

// Host alias of the additive Inkling-2 ABI's attention parameters.
using Inkling2AttentionParams = ::SplashAttentionParams;
static_assert(std::is_standard_layout_v<Inkling2AttentionParams>);
static_assert(std::is_trivially_copyable_v<Inkling2AttentionParams>);

// The Inkling-2 relative attention's sequence parameters are the upstream
// four-word command (the layer is a separate dispatch parameter).
struct Inkling2ChunkedPrefillParams final {
  uint32_t committed_tokens = 0;
  uint32_t chunk_tokens = 0;
  uint32_t chunk_stride = 0;
  uint32_t page_table_entries = 0;

  Inkling2ChunkedPrefillParams() = default;
  // The fork's five-word chunk has the same first four words and an extra kv
  // layer, which the relative attention's own kernels do not read. Accept it
  // so family sources that build the fork's kv::ChunkedPrefillParams work.
  Inkling2ChunkedPrefillParams(const ::SplashChunkedPrefillParams &params)
      : committed_tokens(params.committed_tokens), chunk_tokens(params.chunk_tokens),
        chunk_stride(params.chunk_stride), page_table_entries(params.page_table_entries) {}

  bool operator==(const Inkling2ChunkedPrefillParams &) const = default;
};
static_assert(std::is_standard_layout_v<Inkling2ChunkedPrefillParams>);

inline constexpr uint32_t kInkling2AttentionTileRows = SPLASH_ATTENTION_TILE_ROWS;
inline constexpr uint32_t kInkling2AttentionMaximumSplits = SPLASH_ATTENTION_MAXIMUM_SPLITS;
inline constexpr uint32_t kInkling2ChunkedPrefillMaximumRows = SPLASH_PREFILL_TOKEN_BUDGET;

[[nodiscard]] constexpr uint32_t inkling2AttentionTiles(uint32_t rows) noexcept {
  return (rows + kInkling2AttentionTileRows - 1) / kInkling2AttentionTileRows;
}
[[nodiscard]] constexpr uint32_t inkling2WindowAttentionPages(uint32_t windowTokens) noexcept {
  return (windowTokens + kInkling2AttentionTileRows - 1 + kv::kPageTokens - 1) / kv::kPageTokens + 1;
}
[[nodiscard]] constexpr uint32_t inkling2ChunkedPrefillRequiredPages(
    const Inkling2ChunkedPrefillParams &params) noexcept {
  return (params.committed_tokens + params.chunk_tokens + kv::kPageTokens - 1) / kv::kPageTokens;
}
[[nodiscard]] constexpr std::string_view
inkling2ChunkedPrefillValidationError(const Inkling2ChunkedPrefillParams &params) noexcept {
  if (!params.chunk_tokens || params.chunk_tokens > kInkling2ChunkedPrefillMaximumRows)
    return "chunk_tokens_out_of_range";
  if (uint64_t{params.committed_tokens} + params.chunk_tokens > kv::kMaximumPhysicalTokens)
    return "context_out_of_range";
  if (params.chunk_stride < params.chunk_tokens || params.chunk_stride > kInkling2ChunkedPrefillMaximumRows ||
      params.chunk_stride % kv::kPageTokens)
    return "chunk_stride_invalid";
  if (params.page_table_entries < inkling2ChunkedPrefillRequiredPages(params)) return "page_table_too_short";
  return {};
}

struct Inkling2AttentionWorkspace final {
  uint64_t partialsBytes = 0;
  uint64_t statisticsBytes = 0;

  // The fork's attention workspace holds the same two byte counts; convert
  // so family sources that use either name interoperate. A conversion
  // operator keeps this an aggregate (a constructor would not).
  operator ::splash::ops::AttentionWorkspace() const noexcept { return {partialsBytes, statisticsBytes}; }
};
inline constexpr uint32_t kInkling2AttentionSequenceSplits = 2 * kInkling2AttentionMaximumSplits;

// How one layer's Inkling-2 attention reads the pages. The fork's kv::Layout
// has no `reads` field; Inkling-2's relative attention always reads every
// page, which is what its ported kernels implement.
// The activation precision switch, as upstream names it (its Linear.hpp);
// defined here so the fork's Linear stays as is. Inkling2StepKind is the
// alias to ops::StepKind declared in inkling2_LaneRows.hpp (included above).
enum class Inkling2Int8Activations : uint8_t { Off, On };

struct Inkling2AttentionVariant final {
  bool rotated = true;
  uint32_t windowTokens = 0;
  bool operator==(const Inkling2AttentionVariant &) const = default;
};

struct Inkling2AttentionBlocks final {
  metal::MetalBuffer lists;
  metal::MetalBuffer counts;
  uint32_t stride = 0;
  uint32_t list = 0;
  metal::MetalBuffer locations{};
  metal::MetalBuffer locationPages{};
  metal::MetalBuffer rowMasks{};
};
struct Inkling2AttentionHistory final {
  SplashKvLayer kv{};
  metal::MetalBuffer pageTable;
  Inkling2AttentionBlocks blocks{};
};
struct Inkling2AttentionRows final {
  metal::MetalBuffer layout;
  uint32_t rows = 0;
  metal::MetalBuffer tree{};
};
enum class Inkling2AttentionOutput : uint8_t { Every, Last };
struct Inkling2AttentionOutputRows final {
  metal::MetalBuffer packed;
  metal::MetalBuffer hidden;
  Inkling2AttentionOutput output = Inkling2AttentionOutput::Every;
  LinearScratch scratch{};
  LinearInput input = LinearInput::Plain;
};

// Upstream 1.7.2's names for these types, used by the ported family sources.
// Aliases of the fork's own types: no ABI or behavior change.
using AttentionHistory = Inkling2AttentionHistory;
using AttentionRows = Inkling2AttentionRows;
using AttentionOutput = Inkling2AttentionOutput;
using AttentionOutputRows = Inkling2AttentionOutputRows;

// The plan shape upstream's RelativeAttentionPlan aliases as AttentionPlan.
class Inkling2AttentionPlan final {
public:
  struct Sequence final {
    uint32_t rows = 0;
    uint32_t historyTokens = 0;
    uint32_t firstTile = 0;
    uint32_t splits = 0;
    [[nodiscard]] uint32_t tiles() const noexcept { return inkling2AttentionTiles(rows); }
  };
  struct Dispatch final {
    std::array<uint32_t, SPLASH_MAXIMUM_BATCH_WIDTH> sequence{};
    uint32_t sequences = 0;
    uint32_t tiles = 0;
    uint32_t splits = 0;
    uint32_t slots = 0;
  };

  const kv::Layout layout;
  const uint32_t queryHeads;
  const bool sparse = false;
  const bool sinks = false;
  const uint32_t workspaceSlots;

  [[nodiscard]] std::span<const Sequence> sequences() const noexcept { return {sequences_.data(), sequenceCount_}; }
  [[nodiscard]] std::span<const Dispatch> dispatches() const noexcept { return {dispatches_.data(), dispatchCount_}; }
  [[nodiscard]] Inkling2AttentionWorkspace workspace() const noexcept;
  [[nodiscard]] const std::string &splitPipeline() const noexcept { return splitPipeline_; }
  [[nodiscard]] const std::string &reducePipeline() const noexcept { return reducePipeline_; }
  [[nodiscard]] metal::DispatchSize splitGroups(const Dispatch &dispatch) const noexcept {
    return {layout.kvHeads, dispatch.tiles, dispatch.splits};
  }
  [[nodiscard]] metal::DispatchSize reduceGroups(const Dispatch &dispatch) const noexcept {
    return {layout.kvHeads, kInkling2AttentionTileRows * (queryHeads / layout.kvHeads), dispatch.tiles};
  }

private:
  friend struct Inkling2AttentionPlanning;
  Inkling2AttentionPlan(kv::Layout layout, uint32_t queryHeads, uint32_t workspaceSlots)
      : layout(layout), queryHeads(queryHeads), workspaceSlots(workspaceSlots) {}
  std::array<Sequence, SPLASH_MAXIMUM_BATCH_WIDTH> sequences_{};
  uint32_t sequenceCount_ = 0;
  std::array<Dispatch, SPLASH_MAXIMUM_BATCH_WIDTH> dispatches_{};
  uint32_t dispatchCount_ = 0;
  std::string splitPipeline_;
  std::string reducePipeline_;
};

// The shared planning checks of the ported attention operators.
struct Inkling2AttentionPlanning final {
  [[nodiscard]] static std::string kernelFormat(kv::Layout layout) {
    const kv::KvStorage storage = layout.effectiveStorage();
    if (storage.k.bits == 4 || storage.v.bits == 4)
      throw std::invalid_argument("Inkling-2 relative attention has no packed int4 KV kernels");
    return layout.format == kv::Format::BFloat16 ? "bf16" : "q8";
  }
  [[nodiscard]] static Inkling2AttentionWorkspace workspace(uint64_t rows, uint32_t headDimension) {
    return {rows * headDimension * sizeof(float), rows * 2 * sizeof(float)};
  }
  [[nodiscard]] static uint64_t slotRows(uint32_t queryHeads) {
    return uint64_t{kInkling2AttentionTileRows} * queryHeads;
  }
  [[nodiscard]] static uint64_t readPages(uint64_t visibleTokens, uint32_t windowTokens) {
    const uint64_t seen = (visibleTokens + kv::kPageTokens - 1) / kv::kPageTokens;
    return windowTokens ? std::min<uint64_t>(seen, inkling2WindowAttentionPages(windowTokens)) : seen;
  }
  static uint32_t attentionSplits(uint32_t tiles, uint64_t pages, uint32_t windowTokens, uint32_t pagesPerSplit) {
    constexpr uint32_t kSharedSplits = 32;
    constexpr uint32_t kTilePagesPerSplit = 16;
    const uint32_t fewest = std::clamp(kSharedSplits / tiles, 1U, kSharedSplits);
    const uint32_t most = std::clamp(kInkling2AttentionSequenceSplits / tiles, fewest, kSharedSplits);
    const uint32_t splits = static_cast<uint32_t>(
        tiles == 1 ? std::clamp<uint64_t>((pages + kTilePagesPerSplit - 1) / kTilePagesPerSplit, kSharedSplits,
                                          kInkling2AttentionMaximumSplits)
                   : std::clamp<uint64_t>(pages / pagesPerSplit, fewest, most));
    return windowTokens ? std::min(splits, inkling2WindowAttentionPages(windowTokens)) : splits;
  }

  struct CommandRows final {
    std::array<uint32_t, SPLASH_MAXIMUM_BATCH_WIDTH> stageRow{};
    uint64_t bounds = 0;
    uint32_t mostRows = 0;
    uint64_t staged = 0;
    Inkling2ChunkedPrefillParams last{};
    uint32_t kvHeads = 0;

    [[nodiscard]] uint64_t rowBytes(uint64_t width, uint32_t reach) const noexcept {
      return (staged - kvHeads * uint64_t{last.chunk_stride} + (kvHeads - 1) * uint64_t{last.chunk_stride} + reach) *
             width * sizeof(uint16_t);
    }
  };
  static CommandRows commandRows(std::span<const Inkling2ChunkedPrefillParams> sequences, kv::Layout layout) {
    if (sequences.empty() || sequences.size() > SPLASH_MAXIMUM_BATCH_WIDTH)
      throw std::invalid_argument("invalid attention sequences");
    CommandRows result;
    for (size_t s = 0; s < sequences.size(); ++s) {
      const std::string_view error = inkling2ChunkedPrefillValidationError(sequences[s]);
      if (!error.empty()) throw std::invalid_argument(std::string(error));
      result.stageRow[s] = static_cast<uint32_t>(result.staged);
      result.bounds += sequences[s].chunk_tokens;
      result.mostRows = std::max(result.mostRows, sequences[s].chunk_tokens);
      result.staged += uint64_t{layout.kvHeads} * sequences[s].chunk_stride;
    }
    result.last = sequences.back();
    result.kvHeads = layout.kvHeads;
    if (result.staged > std::numeric_limits<uint32_t>::max())
      throw std::invalid_argument("attention staging exceeds the kernel ABI");
    return result;
  }

  static SplashAttentionSequence sequenceRows(const Inkling2ChunkedPrefillParams &chunk, const CommandRows &rows,
                                              size_t s, uint32_t first) {
    return {.committed_tokens = chunk.committed_tokens + first,
            .row_bound = chunk.chunk_tokens - first,
            .layout_row = first,
            .stage_row = rows.stageRow[s] + first,
            .chunk_stride = chunk.chunk_stride,
            .page_table_entries = chunk.page_table_entries,
            .split_count = 0,
            .first_tile = 0,
            .first_slot = 0,
            .block_stride = 0,
            .block_list = 0};
  }
  static void requireRows(const Inkling2AttentionRows &rows, const CommandRows &command, size_t sequences) {
    if (!rows.rows || rows.rows > command.bounds)
      throw std::invalid_argument("attention rows exceed their sequences' bounds");
    requireBytes(rows.layout, sequences * sizeof(SplashLaneRows), "attention layout");
    if (rows.tree) requireBytes(rows.tree, uint64_t{rows.rows} * sizeof(SplashTreeRow), "attention tree rows");
  }
  static void requirePageTable(const metal::MetalBuffer &table, const Inkling2ChunkedPrefillParams &chunk) {
    requireBytes(table, uint64_t{inkling2ChunkedPrefillRequiredPages(chunk)} * sizeof(SplashKvPage), "page table");
  }
  [[nodiscard]] static bool sameLayer(SplashKvLayer a, SplashKvLayer b) {
    return a.extent_pages == b.extent_pages && a.offset == b.offset;
  }
  static uint32_t groupDispatches(std::span<const Inkling2AttentionPlan::Sequence> sequences, uint32_t workspaceSlots,
                                  std::array<Inkling2AttentionPlan::Dispatch, SPLASH_MAXIMUM_BATCH_WIDTH> &dispatches,
                                  uint32_t splitSlots = 1) {
    uint32_t count = 0;
    for (uint32_t s = 0; s < sequences.size(); ++s) {
      const Inkling2AttentionPlan::Sequence &sequence = sequences[s];
      const uint32_t tiles = inkling2AttentionTiles(sequence.rows) - sequence.firstTile;
      const uint32_t slots = tiles * sequence.splits * splitSlots;
      if (slots > workspaceSlots)
        throw std::invalid_argument("the attention workspace holds no sequence's splits");
      if (!count || dispatches[count - 1].slots + slots > workspaceSlots) dispatches[count++] = {};
      Inkling2AttentionPlan::Dispatch *dispatch = &dispatches[count - 1];
      dispatch->sequence[dispatch->sequences++] = s;
      dispatch->tiles += tiles;
      dispatch->splits = std::max(dispatch->splits, sequence.splits);
      dispatch->slots += slots;
    }
    return count;
  }
  static void addKvPathCommit(metal::CommandGraph &graph, std::string (*kernel)(kv::Layout), kv::Layout layout,
                              uint32_t extentPages, std::span<const metal::MetalBuffer> pageTables,
                              std::span<const uint32_t> committed, uint32_t laneRows,
                              const metal::MetalBuffer &paths) {
    const uint32_t lanes = static_cast<uint32_t>(committed.size());
    if (!lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH || pageTables.size() != lanes || !extentPages || !laneRows ||
        laneRows > SPLASH_TARGET_VERIFY_ROWS)
      throw std::invalid_argument("invalid KV path commit");
    SplashKvPathCommitParams params{
        .extent_pages = extentPages, .regions = layout.attentionLayers, .committed_tokens = {}};
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const uint64_t pages = (uint64_t{committed[lane]} + laneRows + kv::kPageTokens - 1) / kv::kPageTokens;
      requireBytes(pageTables[lane], pages * sizeof(SplashKvPage), "page table");
      params.committed_tokens[lane] = committed[lane];
    }
    requireBytes(paths, uint64_t{lanes} * sizeof(uint32_t), "KV path commit paths");
    const auto tables = laneBindings(pageTables);
    std::vector<metal::MetalBuffer> bindings(tables.begin(), tables.end());
    bindings.push_back(paths);
    graph.add(kernel(layout), std::move(bindings), params,
              {uint64_t{layout.attentionLayers} * layout.kvHeads, lanes, 1}, {layout.headDimension, 1, 1});
  }
};

// The ported kernels bind eight per-lane buffers (metal/kernels/common/
// inkling2_lane_bindings.h) while the fork's host binds four lanes. Pad the
// sequence's bindings to eight; lanes past the fork's batch width never run.
inline std::array<metal::MetalBuffer, 8> inkling2LaneBindings(std::span<const metal::MetalBuffer> lanes) {
  if (lanes.empty() || lanes.size() > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("lane bindings must name 1 to every lane");
  std::array<metal::MetalBuffer, 8> result;
  for (size_t lane = 0; lane < result.size(); ++lane) result[lane] = lanes[lane < lanes.size() ? lane : 0];
  return result;
}

inline Inkling2AttentionWorkspace Inkling2AttentionPlan::workspace() const noexcept {
  uint64_t slots = 0;
  for (const Dispatch &dispatch : dispatches()) slots = std::max<uint64_t>(slots, dispatch.slots);
  return Inkling2AttentionPlanning::workspace(slots * Inkling2AttentionPlanning::slotRows(queryHeads),
                                              layout.headDimension);
}

} // namespace splash::ops
