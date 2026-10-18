#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace splash::engine {

// One fan, as the system management controller reports it. Reading the SMC
// needs no privilege, while writing a fan's mode and target does: this engine
// reads, and never takes a Mac's cooling into its own hands.
struct FanReading final {
  double rotationsPerMinute = 0.0;
  double minimumRotationsPerMinute = 0.0;
  double maximumRotationsPerMinute = 0.0;
  double targetRotationsPerMinute = 0.0;
  // The mode the SMC reports, verbatim: 3 is the system's own control on some
  // Macs, and an M3 Max reports 0 while the system controls the fans, so the
  // status passes it on as it finds it rather than reading anything into it.
  uint8_t mode = 0;
};

// One temperature sensor. The key is the SMC's own name for it, e.g. "Tp05",
// which is the only identity a Mac reports for a sensor.
struct TemperatureReading final {
  std::string key;
  double celsius = 0.0;
};

// The fans and the temperatures of one sample. A Mac with no fans (a MacBook
// Air, a VM) reads as an empty list, and `available` false when the SMC could
// not be opened at all — a state that never fails the engine, since a Mac that
// cannot report its fans still serves.
struct FanState final {
  bool available = false;
  std::vector<FanReading> fans;
  std::vector<TemperatureReading> temperatures;
  // The hottest sensor of the sample and its key, empty when none was read.
  double hottestCelsius = 0.0;
  std::string hottestKey;
};

// Reads every fan and every temperature sensor of the probe list once. Keys
// the firmware does not have are skipped, so one sample costs a few dozen SMC
// round trips on any Mac and returns what that Mac actually reports.
[[nodiscard]] FanState readFanState();

// The sensor keys read, in the order they are tried. The CPU's performance and
// efficiency cores first, then the GPU, then the memory, ambient and fan-zone
// sensors the Macs that have them report.
[[nodiscard]] std::span<const std::string_view> temperatureKeys();

// Decodes one of the SMC's scalar encodings, the formats the fan and thermal
// keys use: "fpe2" (big-endian 14.2 fixed point, Intel's fans), "flt "
// (little-endian IEEE 754 single, what Apple silicon reports), "sp78"
// (big-endian signed 8.8), "ui8 " and "ui16". Nullopt for a format this build
// does not know or a short value, which a caller reports as no reading rather
// than as zero.
[[nodiscard]] std::optional<double>
decodeSmcValue(std::string_view type, std::span<const uint8_t> bytes);

// Keeps the latest sample for the status document and spells the line the log
// gets when the fans change speed, as ThermalStateReporter does for the
// thermal state: a fan spinning up is what a person hears, and nothing else in
// the log says why. Speeds are rounded to 50 RPM, so the log holds the ramp
// rather than every sample, and the first sample reports, because the speed a
// Mac idles at is worth knowing.
class FanStateReporter final {
public:
  // The line to log, or empty when nothing changed since the last sample.
  [[nodiscard]] std::string update(FanState sample);
  [[nodiscard]] const FanState &state() const noexcept { return state_; }

private:
  FanState state_;
  // The key of the last line: the rounded speeds, or nothing at all when the
  // SMC did not answer, which is a different state from a Mac with no fans.
  std::optional<std::vector<int>> reported_;
  bool started_ = false;
};

// A temperature outside this range is a key that holds something other than a
// temperature in degrees, or a sensor that failed: it is dropped rather than
// reported as the hottest reading of the Mac. The upper bound is below what an
// Apple silicon part reports before it throttles, so a key that holds a
// constant (Tf26 reads 115.56 on an M3 Max at idle and under load alike) is
// not mistaken for the hottest sensor of the machine.
inline constexpr double kMinimumPlausibleCelsius = 0.0;
inline constexpr double kMaximumPlausibleCelsius = 110.0;

} // namespace splash::engine
