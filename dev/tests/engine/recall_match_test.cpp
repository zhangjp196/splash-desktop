// `--recall`: the host's longest-suffix copy candidates
// (runtime/model/RecallMatch.hpp). CPU only: no GPU, no engine library.
#include "model/RecallMatch.hpp"

#include <cstdint>
#include <iostream>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char *what) {
  if (!condition) {
    std::cerr << "FAIL: " << what << "\n";
    ++failures;
  }
}

std::vector<std::uint32_t> history(std::initializer_list<std::uint32_t> tokens) {
  return {tokens};
}

} // namespace

int main() {
  using splash::model::longestSuffixMatch;

  // Nothing repeats.
  {
    const auto tokens = history({1, 2, 3, 4, 5, 6});
    const auto match = longestSuffixMatch(tokens, 2, 4, 4);
    expect(match.length == 0, "a history without a repeat has no match");
    expect(match.continuation.empty(), "no match carries no continuation");
  }

  // The longest suffix wins, and the tokens after the earlier occurrence are
  // the candidates.
  {
    const auto tokens = history({1, 2, 3, 7, 8, 1, 2, 3});
    const auto match = longestSuffixMatch(tokens, 2, 4, 4);
    expect(match.length == 3, "the longest repeated suffix wins");
    expect(match.offset == 0, "the earlier occurrence is reported");
    expect(match.continuation == std::vector<std::uint32_t>({7, 8}),
           "the tokens after the earlier occurrence are the candidates");
  }

  // An occurrence with no room after it is skipped for an earlier one.
  {
    const auto tokens = history({1, 2, 9, 1, 2, 1, 2});
    const auto match = longestSuffixMatch(tokens, 2, 2, 4);
    expect(match.length == 2 && match.offset == 0,
           "an adjacent occurrence is skipped");
    expect(match.continuation == std::vector<std::uint32_t>({9, 1, 2}),
           "the earlier occurrence's continuation is used");
  }

  // The continuation is capped.
  {
    const auto tokens = history({1, 2, 9, 8, 7, 6, 1, 2});
    const auto match = longestSuffixMatch(tokens, 2, 2, 2);
    expect(match.continuation == std::vector<std::uint32_t>({9, 8}),
           "the continuation keeps its cap");
  }

  // The matched suffix is capped.
  {
    const auto tokens = history({1, 2, 3, 4, 5, 1, 2, 3, 4, 5});
    const auto match = longestSuffixMatch(tokens, 2, 3, 8);
    expect(match.length == 3, "the longest matched suffix keeps its cap");
    expect(match.continuation == std::vector<std::uint32_t>({1, 2}),
           "the shorter match still carries its continuation");
  }

  // A history too short for the least match has none.
  {
    const auto tokens = history({1, 2, 1, 2});
    expect(longestSuffixMatch(tokens, 3, 4, 4).length == 0,
           "a history shorter than two least matches has none");
  }

  if (failures) {
    std::cerr << failures << " recall match checks failed\n";
    return 1;
  }
  std::cout << "recall match checks passed\n";
  return 0;
}
