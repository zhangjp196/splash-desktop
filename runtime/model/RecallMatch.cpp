#include "model/RecallMatch.hpp"

#include <algorithm>

namespace splash::model {

RecallMatch longestSuffixMatch(std::span<const uint32_t> history,
                               uint32_t minimum, uint32_t maximumSuffix,
                               uint32_t maximumContinuation) {
  const uint32_t size = static_cast<uint32_t>(history.size());
  if (minimum == 0 || maximumSuffix < minimum || maximumContinuation == 0 ||
      size < 2 * minimum)
    return {};
  const uint32_t longest = std::min(maximumSuffix, size - minimum);
  for (uint32_t length = longest;; --length) {
    const uint32_t start = size - length;
    // An earlier occurrence must end at or before `start` (no overlap), and
    // one with tokens after it is the one a copy drafts from.
    if (start >= length) {
      for (uint32_t offset = start - length + 1; offset-- > 0;) {
        if (!std::equal(history.begin() + offset,
                        history.begin() + offset + length,
                        history.begin() + start))
          continue;
        const uint32_t follow =
            std::min(maximumContinuation, start - (offset + length));
        if (follow == 0)
          continue;
        RecallMatch match;
        match.length = length;
        match.offset = offset;
        match.continuation.assign(history.begin() + offset + length,
                                  history.begin() + offset + length + follow);
        return match;
      }
    }
    if (length == minimum)
      break;
  }
  return {};
}

} // namespace splash::model
