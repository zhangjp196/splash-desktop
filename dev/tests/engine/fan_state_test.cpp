// The Mac's fans and temperatures (engine::FanState): the SMC's encodings,
// the probe list, the reporter the log gets its lines from, and a read of the
// machine this runs on. Reading the SMC needs no privilege and no fan is
// touched: the engine only ever reads.

#include "TestChecks.hpp"
#include "engine/FanState.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace splash;
using namespace splash::engine;

namespace {

using splash::test::require;

std::optional<double> decode(std::string_view type, std::initializer_list<uint8_t> bytes) {
  const std::vector<uint8_t> value(bytes);
  return decodeSmcValue(type, std::span<const uint8_t>(value));
}

// The encodings of the keys the fans and the temperatures answer on. The
// fixed-point formats are big-endian whatever the machine stores words as,
// because that is how the SMC hands them over; flo is the SoC's own
// little-endian float.
void testSmcEncodings() {
  require(decode("fpe2", {0x15, 0x40}) == std::optional<double>(1360.0),
          "fpe2 is 14.2 fixed point, big-endian");
  require(decode("fpe2", {0x00, 0x00}) == std::optional<double>(0.0), "fpe2 zero");
  require(decode("sp78", {0x40, 0x00}) == std::optional<double>(64.0),
          "sp78 is 8.8 fixed point, big-endian");
  require(decode("sp78", {0xC0, 0x00}) == std::optional<double>(-64.0),
          "sp78 is signed");
  require(decode("ui8 ", {0x02}) == std::optional<double>(2.0), "ui8 is one byte");
  require(decode("ui16", {0x10, 0x00}) == std::optional<double>(4096.0),
          "ui16 is two bytes, big-endian");

  const float celsius = 63.19987f;
  std::array<uint8_t, 4> flo{};
  std::memcpy(flo.data(), &celsius, sizeof(celsius));
  const std::optional<double> decoded = decodeSmcValue("flt ", flo);
  require(decoded && std::abs(*decoded - double(celsius)) < 1e-9,
          "flt is the SoC's own single-precision float");

  require(!decode("bad ", {0x00, 0x00}).has_value(), "an unknown format reads as nothing");
  require(!decode("fpe2", {0x15}).has_value(), "a short fpe2 reads as nothing");
  require(!decode("flt ", {0x00, 0x00, 0x00}).has_value(), "a short flt reads as nothing");
  require(!decode("ui8 ", {}).has_value(), "an empty ui8 reads as nothing");
}

// The sensors one sample reads, in a fixed order, so two samples of the same
// Mac name the same sensors.
void testProbeList() {
  const std::span<const std::string_view> keys = temperatureKeys();
  require(!keys.empty(), "the probe list is not empty");
  for (const std::string_view key : keys)
    require(key.size() == 4, "an SMC key is four characters: " + std::string(key));
}

FanState sample(double first, double second, double celsius) {
  FanState state;
  state.available = true;
  state.fans.push_back(FanReading{.rotationsPerMinute = first, .mode = 3});
  state.fans.push_back(FanReading{.rotationsPerMinute = second, .mode = 3});
  state.temperatures.push_back(TemperatureReading{.key = "Tp05", .celsius = celsius});
  state.hottestCelsius = celsius;
  state.hottestKey = "Tp05";
  return state;
}

// The log holds the fans spinning up, not every sample: a Mac that is loud is
// what a person hears, and the line says how loud and how hot.
void testReporterReportsChanges() {
  FanStateReporter reporter;
  const std::string first = reporter.update(sample(0.0, 0.0, 50.0));
  require(!first.empty(), "the first sample reports");
  require(first.starts_with("Fans:"), "the line names the fans: " + first);
  require(first.find("0 RPM") != std::string::npos, "the line holds the speed");
  require(first.find("hottest Tp05") != std::string::npos,
          "the line holds the hottest sensor: " + first);

  require(reporter.update(sample(0.0, 0.0, 50.0)).empty(),
          "an unchanged sample stays out of the log");
  require(reporter.update(sample(24.0, 20.0, 51.0)).empty(),
          "a drift within the rounding stays out of the log");

  const std::string ramp = reporter.update(sample(3186.0, 3437.0, 94.8));
  require(!ramp.empty(), "a ramp reports");
  require(ramp.find("3186 RPM") != std::string::npos, "the line holds the speed");
  require(ramp.find("94.8") != std::string::npos, "the line holds the heat: " + ramp);
  require(reporter.state().fans.size() == 2, "the reporter keeps the latest sample");

  require(reporter.update(sample(0.0, 0.0, 50.0)).find("0 RPM") != std::string::npos,
          "the fans spinning down report");
}

// A Mac with no fans, and one whose SMC did not answer, are both states the
// engine serves in.
void testWithoutFans() {
  FanStateReporter reporter;
  FanState fanless;
  fanless.available = true;
  require(reporter.update(fanless) == "Fans: none reported", "a fanless Mac says so");
  FanState unreachable;
  require(reporter.update(unreachable) == "Fans: the SMC could not be read",
          "an unreadable SMC says so");
  require(reporter.update(unreachable).empty(), "the same state stays out of the log");
}

// The machine this runs on, read the way the engine reads it. A Mac without
// fans, or a machine whose SMC does not answer, still passes: every reading is
// checked for being a reading, not for being a particular one.
void testTheMachineReadsCoherently() {
  const FanState state = readFanState();
  for (const FanReading &fan : state.fans) {
    require(fan.rotationsPerMinute >= 0.0, "a fan speed is not negative");
    require(fan.minimumRotationsPerMinute >= 0.0, "a minimum is not negative");
    require(fan.maximumRotationsPerMinute >= fan.minimumRotationsPerMinute,
            "a fan's maximum is at least its minimum");
  }
  for (const TemperatureReading &temperature : state.temperatures) {
    require(temperature.key.size() == 4, "a sensor key is four characters");
    require(temperature.celsius >= kMinimumPlausibleCelsius &&
                temperature.celsius <= kMaximumPlausibleCelsius,
            "a temperature is within what a Mac reports");
  }
  require(state.hottestKey.empty() == state.temperatures.empty(),
          "the hottest sensor is one of the sensors read");
  if (!state.hottestKey.empty()) {
    double hottest = 0.0;
    for (const TemperatureReading &temperature : state.temperatures)
      hottest = std::max(hottest, temperature.celsius);
    require(state.hottestCelsius == hottest, "the hottest is the hottest of them");
  }
}

} // namespace

int main() {
  try {
    testSmcEncodings();
    testProbeList();
    testReporterReportsChanges();
    testWithoutFans();
    testTheMachineReadsCoherently();
    std::cout << "fan state tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "fan state tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
