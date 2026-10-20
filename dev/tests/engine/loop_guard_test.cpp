// `--loop-guard`: the cycle detector that ends a repeated thinking block
// (runtime/model/LoopGuard.hpp). CPU only: no GPU, no engine library.
#include "model/LoopGuard.hpp"

#include <cstdint>
#include <iostream>

namespace {

int failures = 0;

void expect(bool condition, const char *what) {
  if (!condition) {
    std::cerr << "FAIL: " << what << "\n";
    ++failures;
  }
}

// A deterministic sequence with no short period.
std::uint32_t noise(std::uint64_t i) {
  std::uint64_t x = i * 0x9E3779B97F4A7C15ULL + 0x1234567ULL;
  x ^= x >> 33;
  x *= 0xFF51AFD7ED558CCDULL;
  x ^= x >> 33;
  return std::uint32_t(x % 100000) + 1000;
}

} // namespace

int main() {
  using splash::model::kLoopGuardFirstRun;
  using splash::model::kLoopGuardMinimumRun;
  using splash::model::LoopGuard;

  // A constant run fires as soon as the guard has its run-in, and not before.
  {
    LoopGuard guard;
    bool early = false;
    for (std::uint64_t i = 0; i + 1 < kLoopGuardFirstRun; ++i)
      early = guard.push(42) || early;
    const bool atRun = guard.push(42);
    expect(!early, "a constant run does not fire before its run-in");
    expect(atRun, "a constant run fires at its first full run");
  }

  // A period past the maximum never fires, however long it runs.
  {
    LoopGuard guard;
    bool fired = false;
    for (std::uint64_t i = 0; i < 2000; ++i)
      fired = guard.push(std::uint32_t(i % 9)) || fired;
    expect(!fired, "a period past the maximum does not fire");
  }

  // Noise does not fire.
  {
    LoopGuard guard;
    bool fired = false;
    for (std::uint64_t i = 0; i < 2000; ++i)
      fired = guard.push(noise(i)) || fired;
    expect(!fired, "noise does not fire");
  }

  // A short cycle after a long warm-in fires.
  {
    LoopGuard guard;
    bool fired = false;
    for (std::uint64_t i = 0; i < 400; ++i)
      fired = guard.push(noise(i)) || fired;
    constexpr std::uint32_t block[6] = {11, 22, 33, 44, 55, 66};
    for (std::uint64_t i = 0; i < kLoopGuardMinimumRun + 24 && !fired; ++i)
      fired = guard.push(block[i % 6]);
    expect(fired, "a short cycle after a warm-in fires");
  }

  if (failures) {
    std::cerr << failures << " loop guard checks failed\n";
    return 1;
  }
  std::cout << "loop guard checks passed\n";
  return 0;
}
