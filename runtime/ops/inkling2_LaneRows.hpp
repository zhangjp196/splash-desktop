#pragma once

// Inkling-2 lane-row types, kept in a separate header so the fork's four-lane
// kernels and their host ops are unchanged. The values are the upstream
// 1.7.2 ones; the ABI structs they wrap are shared with the kernel port.

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/VerifyTree.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <span>

namespace splash::ops {

// Upstream 1.7.2's StepKind name, used by the ported Inkling-2 family sources:
// the work a plan computes, a prefill chunk's rows or a decode step's. Defined
// here (rather than aliasing Inkling2StepKind) to avoid a header cycle with
// inkling2_AttentionFramework.hpp, which includes this header.
enum class StepKind : uint8_t { Prefill, Decode };
// Upstream 1.7.2's internal name for the same enumeration, used by the
// ported attention framework and operators.
using Inkling2StepKind = StepKind;

// A lane's rows among a command's packed rows: `rows` rows from row `first`.
struct LaneRows {
  uint32_t first = 0;
  uint32_t rows = 0;
};

using TreeRow = SplashTreeRow;

static_assert(sizeof(LaneRows) == sizeof(SplashLaneRows));

// The least lane cap of a decode step of `rows` rows over `lanes` lanes that
// lets the narrow lane kernels hold them. The fork's verify rows and the
// draft's query rows are both eight, so the cap is always a verify block's
// when a lane holds more.
[[nodiscard]] constexpr uint32_t inkling2VerifyLaneCap(uint32_t rows, uint32_t lanes) noexcept {
  return rows <= uint64_t{SPLASH_DRAFT_QUERY_ROWS} * lanes ? SPLASH_DRAFT_QUERY_ROWS : SPLASH_TARGET_VERIFY_ROWS;
}

// A decode step's rows as the proposer lays them out for the kernels.
struct StepRows final {
  uint32_t lanes = 0;
  uint32_t rows = 0;
  uint32_t laneCap = 0;
};

// Construct the real DFlash proposal topology: each lane's proposals form
// a linear chain in its fixed verify slot. The GPU selector may use a prefix
// of these rows, but never changes their parent relationships.
inline void makeLinearVerifyLayout(std::span<TreeRow> tree, std::span<LaneRows> lanes,
                                  uint32_t verifyRows) {
  if (lanes.empty() || lanes.size() > SPLASH_MAXIMUM_BATCH_WIDTH ||
      verifyRows != lanes.size() * SPLASH_TARGET_VERIFY_ROWS ||
      tree.size() != verifyRows)
    throw std::invalid_argument("invalid linear verify layout buffers");
  for (uint32_t lane = 0; lane < lanes.size(); ++lane) {
    lanes[lane] = {lane * SPLASH_TARGET_VERIFY_ROWS, SPLASH_TARGET_VERIFY_ROWS};
    for (uint32_t row = 0; row < SPLASH_TARGET_VERIFY_ROWS; ++row)
      tree[lane * SPLASH_TARGET_VERIFY_ROWS + row] = {
          .ancestors = (uint32_t{1} << (row + 1)) - 1,
          .parent = static_cast<uint16_t>(row ? row - 1 : SPLASH_TREE_ROOT),
          .depth = static_cast<uint16_t>(row)};
  }
}

inline void requireStepRows(const StepRows &step) {
  if (!step.lanes || step.lanes > SPLASH_MAXIMUM_BATCH_WIDTH || !step.laneCap ||
      step.laneCap > SPLASH_TARGET_VERIFY_ROWS || step.rows < step.lanes ||
      step.rows > uint64_t{step.lanes} * step.laneCap)
    throw std::invalid_argument("invalid decode step rows");
}

// A command's rows as its layout on the device holds them.
struct LayoutRows final {
  uint32_t sequences = 0;
  uint32_t rows = 0;
  uint32_t bound = 0;
};

inline void requireLayoutRows(const LayoutRows &rows, uint32_t limit = std::numeric_limits<uint32_t>::max()) {
  if (!rows.sequences || rows.sequences > SPLASH_MAXIMUM_BATCH_WIDTH || !rows.bound || rows.bound > limit ||
      rows.rows < rows.sequences || rows.rows > uint64_t{rows.sequences} * rows.bound)
    throw std::invalid_argument("invalid layout rows");
}

// Upstream's kNarrowLaneRows name, for Inkling-2 family sources.
inline constexpr uint32_t kNarrowLaneRows = SPLASH_DRAFT_QUERY_ROWS;
[[nodiscard]] constexpr uint32_t verifyLaneCap(uint32_t rows, uint32_t lanes) noexcept {
  return inkling2VerifyLaneCap(rows, lanes);
}

} // namespace splash::ops
