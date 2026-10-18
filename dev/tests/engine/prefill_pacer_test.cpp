// The prefill pacer (engine::PrefillPacer): what a duty lets a prefill run
// for, the gaps it leaves between commands, what the Mac's thermal state does
// to them, and that a Mac at 100 percent never waits.

#include "TestChecks.hpp"
#include "engine/PrefillPacer.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

using namespace splash;
using namespace splash::engine;

namespace {

using splash::test::require;

// The share of the wall time a stream of commands kept the GPU busy, run the
// way tick() runs it: a command runs to its end once it starts, and the next
// one waits for the pacer. Measured from the moment the pacer first holds a
// command back, so the cycle a quiet pacer starts with is not counted and the
// two ends of the window sit at the same point in a cycle.
double observedDutyPercent(double duty, double commandMilliseconds,
                           int commands = 200) {
  PrefillPacer pacer;
  pacer.configure(duty);
  double now = 0.0;
  require(pacer.allows(now), "the first command runs at once");
  double busy = 0.0;
  std::optional<double> startBusy;
  double startNow = 0.0;
  for (int command = 0; command < commands; ++command) {
    busy += commandMilliseconds;
    now += commandMilliseconds;
    pacer.record(now, commandMilliseconds);
    bool waited = !pacer.allows(now);
    if (waited && !startBusy) {
      startBusy = busy;
      startNow = now;
    }
    while (waited) {
      now += 1.0;
      if (now > 1e9)
        throw std::runtime_error("the pacer never allowed another command");
      waited = !pacer.allows(now);
    }
  }
  return (busy - *startBusy) / (now - startNow) * 100.0;
}

// A Mac at 100 percent is the engine without a throttle: every command starts
// at once, however long the one before it kept the GPU.
void testFullDutyNeverWaits() {
  PrefillPacer pacer;
  pacer.configure(100.0);
  require(!pacer.active(), "a full duty is not paced");
  for (int command = 0; command < 10; ++command) {
    require(pacer.allows(1000.0 * command), "every command runs at once");
    pacer.record(1000.0 * command, 1600.0);
  }
  require(pacer.pacedWaits() == 0, "a full duty holds nothing back");
  require(pacer.dutyPercent() == 100.0, "the duty is reported");
}

// Pacing prefill is what makes a Mac quiet: over a long stream the share of
// the wall time a command keeps the GPU is the configured duty, whatever a
// single command costs.
void testSustainedPrefillRunsAtItsDuty() {
  for (const double duty : {10.0, 25.0, 60.0, 90.0}) {
    for (const double command : {1600.0, 250.0}) {
      const double observed = observedDutyPercent(duty, command);
      require(std::abs(observed - duty) < 1.0,
              "a duty of " + std::to_string(duty) + " percent with " +
                  std::to_string(command) + " ms commands ran at " +
                  std::to_string(observed) + " percent");
    }
  }
}

// A Mac that reports thermal pressure is fed less: the state scales the duty,
// so a state the engine cannot answer by freeing anything it can answer by
// starting less.
void testThermalScaleScalesTheDuty() {
  PrefillPacer pacer;
  pacer.configure(60.0);
  require(pacer.thermalScale() == 1.0, "a fresh pacer keeps the whole duty");
  require(std::abs(observedDutyPercent(60.0, 1600.0) - 60.0) < 1.0,
          "a configured duty runs at its duty");

  // Half the duty is half the wall time: 60 percent scaled by a fair state.
  pacer.setThermalScale(0.5);
  require(pacer.thermalScale() == 0.5, "the scale is reported");
  double now = 0.0;
  require(pacer.allows(now), "the first command runs at once");
  double busy = 0.0;
  for (int command = 0; command < 100; ++command) {
    busy += 1000.0;
    now += 1000.0;
    pacer.record(now, 1000.0);
    while (!pacer.allows(now))
      now += 1.0;
  }
  const double observed = busy / now * 100.0;
  require(std::abs(observed - 30.0) < 1.0,
          "half of 60 percent ran at " + std::to_string(observed));

  // Critical stops prefill until the Mac cools, and then it starts again: a
  // request waiting for scheduling does not consume its resource wait limit.
  pacer.setThermalScale(0.0);
  const double held = now;
  for (int step = 0; step < 100; ++step) {
    now += 1000.0;
    require(!pacer.allows(now), "critical holds prefill back");
  }
  require(now > held, "the state took time to pass");
  pacer.setThermalScale(1.0);
  require(pacer.allows(now + 5000.0), "prefill resumes once the Mac cools");
}

// A hold is counted once, however many commands it keeps waiting, and the wait
// itself is measured: the app shows both.
void testTheWaitIsCounted() {
  PrefillPacer pacer;
  pacer.configure(60.0);
  require(pacer.allows(0.0), "the first command runs at once");
  pacer.record(0.0, 1600.0);
  require(!pacer.allows(0.0), "the next command waits");
  require(pacer.pacedWaits() == 1, "the hold is counted when it begins");
  require(!pacer.allows(500.0), "the hold goes on");
  require(pacer.pacedWaits() == 1, "a hold that goes on is counted once");
  require(pacer.allows(2000.0), "the wait ends once the balance is back");
  require(pacer.pacedWaitMilliseconds() > 0.0, "the wait is measured");
  require(pacer.pacedWaitMilliseconds() <= 2000.0,
          "the wait is measured from when it began");
}

// A command whose timing is not a number must not poison the balance: a
// balance of not-a-number would hold every prefill back forever.
void testNonFiniteReadingsFailOpen() {
  PrefillPacer pacer;
  pacer.configure(60.0);
  require(pacer.allows(0.0), "the first command runs at once");
  pacer.record(0.0, std::nan(""));
  pacer.record(std::nan(""), 100.0);
  require(pacer.allows(1.0), "a command without a usable time spends nothing");
  require(pacer.allows(std::nan("")), "a time that is not a number allows");
}

// A duty outside the range is clamped, so a caller that forgot to check cannot
// stop prefill altogether or invent a duty above the whole wall time.
void testOutOfRangeDutiesAreClamped() {
  PrefillPacer pacer;
  pacer.configure(250.0);
  require(!pacer.active(), "above the range is a full duty");
  pacer.configure(-1.0);
  require(pacer.active(), "below the range paces");
  require(pacer.dutyPercent() == 0.0, "below the range is no duty at all");
  pacer.setThermalScale(4.0);
  require(pacer.thermalScale() == 1.0, "a scale above one is one");
  pacer.setThermalScale(-2.0);
  require(pacer.thermalScale() == 0.0, "a scale below zero is zero");
}

} // namespace

int main() {
  try {
    testFullDutyNeverWaits();
    testSustainedPrefillRunsAtItsDuty();
    testThermalScaleScalesTheDuty();
    testTheWaitIsCounted();
    testNonFiniteReadingsFailOpen();
    testOutOfRangeDutiesAreClamped();
    std::cout << "prefill pacer tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "prefill pacer tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
