#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace splash::model {

// `--loop-guard`: a thinking block that repeats a short token cycle is ended
// early. The last kLoopGuardMinimumRun thinking tokens must repeat with a lag
// of at most kLoopGuardMaximumPeriod tokens, and at least kLoopGuardRunIn
// tokens must precede the repeated run, so ordinary repetition inside a proof
// or a code block does not fire the guard. TensorFold's rule, unchanged.
inline constexpr uint32_t kLoopGuardMaximumPeriod = 8;
inline constexpr uint32_t kLoopGuardMinimumRun = 256;
inline constexpr uint32_t kLoopGuardRunIn = 64;

// The thinking tokens a guard needs before it can fire, and the window it
// compares within.
inline constexpr uint64_t kLoopGuardFirstRun =
    uint64_t{kLoopGuardRunIn} + kLoopGuardMinimumRun + 1;
inline constexpr std::size_t kLoopGuardWindow =
    kLoopGuardMinimumRun + kLoopGuardMaximumPeriod + 1;

// One request's thinking tokens as a guard sees them: the last
// kLoopGuardWindow of them, and how many were seen in all.
class LoopGuard final {
public:
  // Appends one thinking token, returning whether the tokens that end here
  // repeat a short cycle. The caller ends the thinking block when true.
  [[nodiscard]] bool push(uint32_t token) noexcept {
    ring_[seen_ % kLoopGuardWindow] = token;
    ++seen_;
    return fires();
  }

  [[nodiscard]] uint64_t seen() const noexcept { return seen_; }

private:
  [[nodiscard]] uint32_t at(uint64_t index) const noexcept {
    return ring_[index % kLoopGuardWindow];
  }

  [[nodiscard]] bool fires() const noexcept {
    if (seen_ < kLoopGuardFirstRun)
      return false;
    for (uint32_t period = 1; period <= kLoopGuardMaximumPeriod; ++period) {
      if (seen_ < kLoopGuardFirstRun + period - 1)
        break;
      bool matches = true;
      for (uint32_t k = 1; k <= kLoopGuardMinimumRun && matches; ++k) {
        if (at(seen_ - k) != at(seen_ - k - period))
          matches = false;
      }
      if (matches)
        return true;
    }
    return false;
  }

  std::array<uint32_t, kLoopGuardWindow> ring_{};
  uint64_t seen_ = 0;
};

} // namespace splash::model
