#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace splash::model {

// Bounded n-gram prompt lookup. `context` includes the current anchor as its
// final token. It finds the longest suffix that occurred earlier without
// overlapping the current suffix, then returns the tokens that followed it.
struct PromptLookupConfig final {
  uint32_t minNgram = 3;
  uint32_t maxNgram = 8;
  uint32_t maxCandidates = 4;
  uint32_t maxContextTokens = 4096;
};

// Run prompt history lookup only for ordinary, non-cutoff decoding. Keeping
// the gate callable based lets CPU tests prove disallowed paths never search.
template <typename Lookup>
[[nodiscard]] inline std::vector<uint32_t>
promptLookupIfAllowed(bool constrained, bool thinkingCutoff,
                       uint32_t maximumProposals, Lookup &&lookup) {
  if (constrained || thinkingCutoff || maximumProposals == 0) return {};
  return std::forward<Lookup>(lookup)();
}

[[nodiscard]] inline std::vector<uint32_t>
promptLookupProposals(std::span<const uint32_t> context,
                      const PromptLookupConfig &config = {}) {
  std::vector<uint32_t> result;
  if (!config.minNgram || config.minNgram > config.maxNgram ||
      !config.maxCandidates || config.maxContextTokens == 0 || context.empty())
    return result;

  const size_t begin = context.size() > config.maxContextTokens
                           ? context.size() - config.maxContextTokens
                           : 0;
  const size_t suffixEnd = context.size();
  const size_t available = suffixEnd - begin;
  const size_t maximumN = std::min<size_t>(config.maxNgram, available);
  if (maximumN < config.minNgram) return result;

  // Prefer the longest matching n-gram, and the most recent prior occurrence
  // for that n-gram. This deterministic choice avoids duplicate/redundant
  // searches and keeps each proposal sequence bounded to one continuation.
  for (size_t n = maximumN; n >= config.minNgram; --n) {
    if (suffixEnd < begin + 2 * n) {
      if (n == config.minNgram) break;
      continue;
    }

    const size_t suffixBegin = suffixEnd - n;
    size_t occurrence = suffixBegin - n;
    bool found = false;
    for (;;) {
      if (std::equal(context.begin() + occurrence,
                     context.begin() + occurrence + n,
                     context.begin() + suffixBegin)) {
        found = true;
        break;
      }
      if (occurrence == begin) break;
      --occurrence;
    }
    if (!found) {
      if (n == config.minNgram) break;
      continue;
    }

    const size_t continuation = occurrence + n;
    const size_t count = std::min<size_t>(
        config.maxCandidates, suffixEnd - continuation);
    result.insert(result.end(), context.begin() + continuation,
                  context.begin() + continuation + count);
    return result;
  }
  return result;
}

} // namespace splash::model
