#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace splash::model {

// `--recall`: draft from what earlier requests read and wrote. The host finds
// the longest suffix of the current token history that occurred earlier and
// offers the tokens that followed that earlier occurrence as copy candidates
// the target then verifies like any draft (runtime/model/RecallMatch.cpp).
struct RecallMatch final {
  // The matched suffix's length in tokens; 0 when the history holds no match.
  uint32_t length = 0;
  // Where the earlier occurrence starts in the history.
  uint32_t offset = 0;
  // The tokens that followed the earlier occurrence, up to the requested
  // maximum: the copy candidates in order.
  std::vector<uint32_t> continuation;
};

// The longest suffix of `history` of at most `maximumSuffix` and at least
// `minimum` tokens that occurs earlier, with up to `maximumContinuation`
// tokens that followed that occurrence. The earlier occurrence ends at or
// before the suffix starts, so the two do not overlap; among the longest
// matches the most recent occurrence wins.
[[nodiscard]] RecallMatch longestSuffixMatch(std::span<const uint32_t> history,
                                             uint32_t minimum,
                                             uint32_t maximumSuffix,
                                             uint32_t maximumContinuation);

} // namespace splash::model
