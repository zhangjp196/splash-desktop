// Bounded prompt lookup proposals (runtime/model/PromptLookup.hpp).
#include "model/PromptLookup.hpp"

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

} // namespace

int main() {
  using splash::model::PromptLookupConfig;
  using splash::model::promptLookupIfAllowed;
  using splash::model::promptLookupProposals;

  expect((promptLookupProposals(std::vector<uint32_t>{1, 2, 3, 4, 1, 2, 3}) ==
          std::vector<uint32_t>{4, 1, 2, 3}),
         "the suffix includes the current anchor and returns its continuation");
  expect(promptLookupProposals(std::vector<uint32_t>{1, 2, 3}).empty(),
         "a suffix without an earlier occurrence has no proposal");
  expect((promptLookupProposals(std::vector<uint32_t>{7, 8, 9, 7, 8, 9, 6, 7, 8, 9}) ==
          std::vector<uint32_t>{6, 7, 8, 9}),
         "the longest recent match wins");

  PromptLookupConfig bounded;
  bounded.minNgram = 2;
  bounded.maxCandidates = 2;
  bounded.maxContextTokens = 6;
  expect((promptLookupProposals(std::vector<uint32_t>{1, 2, 3, 4, 1, 2, 3}, bounded) ==
          std::vector<uint32_t>{4, 1}),
         "context and proposal counts are bounded");

  PromptLookupConfig invalid;
  invalid.minNgram = 0;
  expect(promptLookupProposals(std::vector<uint32_t>{1, 2, 3, 1, 2, 3}, invalid).empty(),
         "invalid configuration safely returns no proposals");

  int lookupCalls = 0;
  const auto forbiddenLookup = [&] {
    ++lookupCalls;
    return std::vector<uint32_t>{99};
  };
  expect(promptLookupIfAllowed(true, false, 4, forbiddenLookup).empty(),
         "constrained decoding skips prompt lookup");
  expect(lookupCalls == 0, "constrained decoding does not call history lookup");
  expect(promptLookupIfAllowed(false, true, 4, forbiddenLookup).empty(),
         "thinking cutoff skips prompt lookup");
  expect(lookupCalls == 0, "thinking cutoff does not call history lookup");
  expect(promptLookupIfAllowed(false, false, 0, forbiddenLookup).empty(),
         "zero proposal budget skips prompt lookup");
  expect(lookupCalls == 0, "zero proposal budget does not call history lookup");
  const auto allowed = promptLookupIfAllowed(false, false, 4, [&] {
    ++lookupCalls;
    return promptLookupProposals(std::vector<uint32_t>{1, 2, 3, 4, 1, 2, 3});
  });
  expect(lookupCalls == 1, "ordinary decoding calls history lookup once");
  expect((allowed == std::vector<uint32_t>{4, 1, 2, 3}),
         "ordinary decoding preserves prompt lookup proposals");

  if (failures) {
    std::cerr << failures << " prompt lookup checks failed\n";
    return 1;
  }
  std::cout << "prompt lookup checks passed\n";
  return 0;
}
