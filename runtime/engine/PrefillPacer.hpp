#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>

namespace splash::engine {

// The lowest duty a configuration may ask for: below this even a short prompt
// would take longer than a person waits, and the heat is already below what
// the fans answer.
inline constexpr double kMinimumPrefillDutyPercent = 10.0;

// How much of the wall time prefill may keep the GPU busy. Prefill is the one
// command that pins the GPU, and the Neural Engine beside it, for seconds at a
// time: it is what heats a Mac and what its fans answer. Decode is short and
// stays at full speed, so pacing only prefill lowers the average power without
// touching generation. A prompt takes about 100/duty as long to process, which
// is the price of the quiet.
//
// The pacer is a token bucket in GPU milliseconds: at `duty` percent it earns
// duty/100 milliseconds of GPU time per millisecond of wall time, spends what
// the commands that ran actually took, and refuses to start another prefill
// while the balance is empty. A pacer at 100 percent never refuses anything,
// which is the engine's behavior without pacing, so the pacer only ever holds
// a sustained stream back: a Mac that has been quiet, by decoding or by being
// idle, has a full cycle to spend on the next prompt.
class PrefillPacer final {
public:
  // The length of a duty cycle, and the most credit a command can start with:
  // a spelling of quiet longer than this does not accumulate into a burst, and
  // whatever duty a command runs at, the first one of a stream runs at full
  // speed.
  static constexpr double kCycleMilliseconds = 1000.0;

  // --prefill-duty PERCENT, the share of wall time prefill may run. Called from
  // the engine's configuration and again whenever a running engine's duty is
  // changed, so it also accepts a value that differs from the one before it.
  void configure(double percent) noexcept {
    duty_ = std::clamp(percent, 0.0, 100.0) / 100.0;
  }

  [[nodiscard]] double dutyPercent() const noexcept { return duty_ * 100.0; }

  // Whether pacing can hold a command back at all; a caller at 100 percent
  // skips the pacer entirely.
  [[nodiscard]] bool active() const noexcept { return duty_ < 1.0; }

  // macOS's thermal state scales the duty while pacing is on: a Mac already
  // dropping clocks for heat is fed less work, so it cools instead of being
  // pushed at the same rate. One keeps the configured duty, and zero stops
  // prefill until the Mac cools: the balance is cut to what the new duty can
  // hold, so a state of zero does not let one more command start.
  void setThermalScale(double scale) noexcept {
    thermalScale_ = std::clamp(scale, 0.0, 1.0);
    creditMilliseconds_ = std::min(creditMilliseconds_, cycleCredit());
  }

  [[nodiscard]] double thermalScale() const noexcept { return thermalScale_; }

  // Whether another prefill command may start now, refilling the balance by
  // the wall time that passed since the last call. A command that starts keeps
  // what one cycle of credit is worth, however long the quiet before it was,
  // and runs to its end at full speed.
  [[nodiscard]] bool allows(double nowMilliseconds) noexcept {
    if (!active() || !std::isfinite(nowMilliseconds))
      return true;
    refill(nowMilliseconds);
    if (creditMilliseconds_ > 0.0) {
      creditMilliseconds_ = std::min(creditMilliseconds_, cycleCredit());
      if (waitStartedMilliseconds_) {
        pacedWaitMilliseconds_ += nowMilliseconds - *waitStartedMilliseconds_;
        waitStartedMilliseconds_.reset();
      }
      return true;
    }
    if (!waitStartedMilliseconds_) {
      ++pacedWaits_;
      waitStartedMilliseconds_ = nowMilliseconds;
    }
    return false;
  }

  // A prefill command that ran, and what its GPU time cost. The command runs
  // to its end once started; the pacer only decides when the next one begins.
  void record(double nowMilliseconds, double gpuMilliseconds) noexcept {
    if (!active() || !std::isfinite(nowMilliseconds) ||
        !std::isfinite(gpuMilliseconds))
      return;
    refill(nowMilliseconds);
    creditMilliseconds_ -= std::max(0.0, gpuMilliseconds);
  }

  // Prefills the pacer held back, and the wall time they waited, since the
  // engine started.
  [[nodiscard]] uint64_t pacedWaits() const noexcept { return pacedWaits_; }
  [[nodiscard]] double pacedWaitMilliseconds() const noexcept {
    return pacedWaitMilliseconds_;
  }

private:
  [[nodiscard]] double effectiveDuty() const noexcept {
    return duty_ * thermalScale_;
  }

  [[nodiscard]] double cycleCredit() const noexcept {
    return effectiveDuty() * kCycleMilliseconds;
  }

  void refill(double nowMilliseconds) noexcept {
    if (!started_) {
      started_ = true;
      lastMilliseconds_ = nowMilliseconds;
      creditMilliseconds_ = cycleCredit();
      return;
    }
    // What the command in flight earned while it ran is kept uncapped, so the
    // duty is the duty whatever a single command costs; a spell of quiet is
    // bounded where a command starts, in allows().
    creditMilliseconds_ +=
        std::max(0.0, nowMilliseconds - lastMilliseconds_) * effectiveDuty();
    lastMilliseconds_ = nowMilliseconds;
  }

  double duty_ = 1.0;
  double thermalScale_ = 1.0;
  double creditMilliseconds_ = 0.0;
  double lastMilliseconds_ = 0.0;
  bool started_ = false;
  // When the current wait for credit began, while the pacer refuses.
  std::optional<double> waitStartedMilliseconds_;
  uint64_t pacedWaits_ = 0;
  double pacedWaitMilliseconds_ = 0.0;
};

} // namespace splash::engine
