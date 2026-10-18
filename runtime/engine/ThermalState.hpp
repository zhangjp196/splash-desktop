#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace splash::engine {

// The Mac's thermal state as macOS reports it (NSProcessInfo.thermalState).
// As the state rises, the CPU and GPU clocks drop, on some Macs already at
// fair, so prefill and decode slow down with nothing in the engine having
// changed. The engine reports it, and paces prefill by it when prefill is
// paced at all (prefillThermalScale): heat is not a resource the engine can
// free, but it is one it can stop feeding.
//
// The state lags what the Mac's own sensors say. A Mac can hold its fans at
// several thousand RPM, with its CPU clusters past 90 degrees, while macOS
// still reports nominal — so a quiet Mac cannot rely on this state alone,
// which is why the fans are read and reported beside it.
enum class ThermalState : uint8_t {
  Nominal,
  Fair,
  Serious,
  Critical,
};

// One spelling of the states for status JSON and the log.
[[nodiscard]] inline const char *thermalStateName(ThermalState state) noexcept {
  switch (state) {
  case ThermalState::Nominal:
    return "nominal";
  case ThermalState::Fair:
    return "fair";
  case ThermalState::Serious:
    return "serious";
  case ThermalState::Critical:
    return "critical";
  }
  return "critical";
}

// The state macOS reports now; a state this build does not know reads as
// critical.
[[nodiscard]] ThermalState queryThermalState() noexcept;

// The share of its configured prefill duty the engine keeps in this state,
// which the native bootstrap applies to the pacer as it samples: a Mac that is
// already dropping clocks is fed less work, so it cools instead of being
// pushed at the same rate. Critical stops prefill until the Mac cools, which a
// request waiting for it survives: a wait for scheduling does not consume its
// resource wait limit.
[[nodiscard]] inline double prefillThermalScale(ThermalState state) noexcept {
  switch (state) {
  case ThermalState::Nominal:
    return 1.0;
  case ThermalState::Fair:
    return 0.75;
  case ThermalState::Serious:
    return 0.5;
  case ThermalState::Critical:
    return 0.0;
  }
  return 0.0;
}

// Emits only transitions, as MemoryStatusReporter does: the first state when
// it is not nominal, then each change. macOS changes the state over seconds
// to minutes of load, so the lines stay few.
class ThermalStateReporter final {
public:
  [[nodiscard]] std::string update(ThermalState state) {
    const std::optional<ThermalState> previous = state_;
    state_ = state;
    if (previous == state || (!previous && state == ThermalState::Nominal))
      return {};
    std::string line = "Thermal state: ";
    if (previous)
      line = line + thermalStateName(*previous) + " → ";
    return line + thermalStateName(state);
  }
  // The last state given, nominal before the first.
  [[nodiscard]] ThermalState state() const noexcept {
    return state_.value_or(ThermalState::Nominal);
  }

private:
  std::optional<ThermalState> state_;
};

} // namespace splash::engine
