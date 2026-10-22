#include "engine/FanState.hpp"

#include <IOKit/IOKitLib.h>

#include <algorithm>
#include <array>
#include <cstring>

namespace splash::engine {
namespace {

// The AppleSMC user client, whose ABI has not changed since the Intel
// machines: a 4-character key, a version block, the key's info, and up to 32
// bytes of value. The same driver serves Apple silicon, where the fan and
// thermal keys are floats rather than the fixed-point values Intel reported.
constexpr uint32_t kKernelIndexSmc = 2;
constexpr uint8_t kCommandReadBytes = 5;
constexpr uint8_t kCommandReadKeyInfo = 9;
constexpr size_t kMaximumValueBytes = 32;

struct SmcVersionBlock final {
  char major;
  char minor;
  char build;
  char reserved[1];
  uint16_t release;
};

struct SmcPowerLimitBlock final {
  uint16_t version;
  uint16_t length;
  uint32_t cpuLimit;
  uint32_t gpuLimit;
  uint32_t memoryLimit;
};

struct SmcKeyInfoBlock final {
  uint32_t dataSize;
  uint32_t dataType;
  char attributes;
};

struct SmcParam final {
  uint32_t key;
  SmcVersionBlock version;
  SmcPowerLimitBlock limits;
  SmcKeyInfoBlock keyInfo;
  char result;
  char status;
  char command;
  uint32_t value32;
  uint8_t value[kMaximumValueBytes];
};

// The four characters of a key, most significant first: the driver reads the
// key as a big-endian word however the machine stores words.
[[nodiscard]] uint32_t packedKey(std::string_view key) noexcept {
  return (uint32_t(uint8_t(key[0])) << 24) | (uint32_t(uint8_t(key[1])) << 16) |
         (uint32_t(uint8_t(key[2])) << 8) | uint32_t(uint8_t(key[3]));
}

[[nodiscard]] std::string unpackedKey(uint32_t key) {
  std::string name(4, '\0');
  for (size_t index = 0; index < name.size(); ++index)
    name[index] = char((key >> (8 * (3 - index))) & 0xff);
  return name;
}

// One key's raw value and its SMC data type.
struct SmcValue final {
  std::array<uint8_t, kMaximumValueBytes> bytes{};
  size_t size = 0;
  std::string type;
};

// An open AppleSMC connection for the length of one sample. Reading needs no
// privilege: the SMC refuses writes, not reads.
class SmcConnection final {
public:
  SmcConnection() noexcept {
    const io_service_t service =
        IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AppleSMC"));
    if (!service)
      return;
    if (IOServiceOpen(service, mach_task_self(), 0, &connection_) != KERN_SUCCESS)
      connection_ = 0;
    IOObjectRelease(service);
  }

  SmcConnection(const SmcConnection &) = delete;
  SmcConnection &operator=(const SmcConnection &) = delete;

  ~SmcConnection() {
    if (connection_)
      IOServiceClose(connection_);
  }

  [[nodiscard]] bool valid() const noexcept { return connection_ != 0; }

  // The key's value, or nullopt for a key this Mac does not have.
  [[nodiscard]] std::optional<SmcValue> read(std::string_view key) const {
    if (key.size() != 4 || !valid())
      return std::nullopt;
    SmcParam input{};
    SmcParam info{};
    input.key = packedKey(key);
    input.command = char(kCommandReadKeyInfo);
    if (call(input, info) != KERN_SUCCESS || info.result != 0)
      return std::nullopt;
    if (info.keyInfo.dataSize == 0 || info.keyInfo.dataSize > kMaximumValueBytes)
      return std::nullopt;
    SmcParam value{};
    input = {};
    input.key = packedKey(key);
    input.keyInfo.dataSize = info.keyInfo.dataSize;
    input.command = char(kCommandReadBytes);
    if (call(input, value) != KERN_SUCCESS || value.result != 0)
      return std::nullopt;
    SmcValue read;
    read.size = info.keyInfo.dataSize;
    std::memcpy(read.bytes.data(), value.value, read.size);
    read.type = unpackedKey(info.keyInfo.dataType);
    return read;
  }

private:
  [[nodiscard]] kern_return_t call(const SmcParam &input, SmcParam &output) const {
    size_t outputSize = sizeof(output);
    return IOConnectCallStructMethod(connection_, kKernelIndexSmc, &input,
                                     sizeof(input), &output, &outputSize);
  }

  io_connect_t connection_ = 0;
};

// "F0Ac" through "F9Ac", and past nine fans the letter the driver itself uses.
void fanKey(char *key, size_t fan, std::string_view suffix) noexcept {
  key[0] = 'F';
  key[1] = char(fan < 10 ? '0' + fan : 'A' + (fan - 10));
  key[2] = suffix[0];
  key[3] = suffix[1];
  key[4] = '\0';
}

[[nodiscard]] std::optional<double> readScalar(const SmcConnection &smc,
                                               std::string_view key) {
  const std::optional<SmcValue> value = smc.read(key);
  if (!value)
    return std::nullopt;
  return decodeSmcValue(value->type, std::span<const uint8_t>(value->bytes.data(), value->size));
}

} // namespace

std::span<const std::string_view> temperatureKeys() {
  // One key per cluster plus the ambient, storage and IO fabric sensors: a
  // key a Mac does not have reads as nullopt, so the list costs a few
  // microseconds on a machine with two of them and covers the rest.
  static constexpr std::string_view keys[] = {
      // The performance and efficiency cores, which the fan curve follows.
      "Tp01", "Tp05", "Tp09", "Tp0D", "Tp0b", "Tp0f", "Tp0j", "Tp0n", "Tp0r",
      "Tp0v", "Tp0z", "Tp0B", "Tp0F", "Tp0J", "Tp0N", "Tp0R",
      // The GPU, and its own cluster on the Max and Ultra parts.
      "Tg0D", "Tg0d", "Tg0L", "Tg0l", "Tg1D", "Tg1d",
      // The efficiency clusters.
      "Te05", "Te09", "Te0L", "Te0S", "Te0H",
      // The memory controller and the ambient sensors.
      "Tm0P", "Ta0P", "Ta1P",
      // The storage: the NAND packages' own sensors.
      "Ts0P", "Ts1P",
      // The IO fabric. Tf26 is left out: on the Macs that report it it holds a
      // constant near 108 C rather than a temperature.
      "Tf0D", "Tf08",
  };
  return keys;
}

std::optional<double> decodeSmcValue(std::string_view type,
                                     std::span<const uint8_t> bytes) {
  const auto bigEndian16 = [&]() -> std::optional<uint16_t> {
    if (bytes.size() < 2)
      return std::nullopt;
    return uint16_t((uint16_t(bytes[0]) << 8) | bytes[1]);
  };
  if (type == "flt ") {
    if (bytes.size() < 4)
      return std::nullopt;
    // The driver hands these back the way the SoC stores them, little-endian
    // on both architectures.
    float value = 0.0F;
    std::memcpy(&value, bytes.data(), sizeof(value));
    return double(value);
  }
  if (type == "fpe2") {
    const std::optional<uint16_t> raw = bigEndian16();
    if (!raw)
      return std::nullopt;
    return double(*raw) / 4.0;
  }
  if (type == "sp78") {
    const std::optional<uint16_t> raw = bigEndian16();
    if (!raw)
      return std::nullopt;
    return double(int16_t(*raw)) / 256.0;
  }
  if (type == "ui8 ") {
    if (bytes.empty())
      return std::nullopt;
    return double(bytes[0]);
  }
  if (type == "ui16") {
    const std::optional<uint16_t> raw = bigEndian16();
    if (!raw)
      return std::nullopt;
    return double(*raw);
  }
  return std::nullopt;
}

FanState readFanState() {
  FanState state;
  const SmcConnection smc;
  if (!smc.valid())
    return state;
  state.available = true;

  // The fan count, and the keys each fan answers on. A machine whose firmware
  // does not report a count reports no fans, which is the truth on a fanless
  // Mac.
  const std::optional<double> count = readScalar(smc, "FNum");
  const size_t fanCount =
      count ? size_t(std::clamp(*count, 0.0, 32.0)) : 0;
  for (size_t fan = 0; fan < fanCount; ++fan) {
    char key[5];
    FanReading reading;
    fanKey(key, fan, "Ac");
    const std::optional<double> rotations = readScalar(smc, key);
    if (!rotations)
      continue;
    reading.rotationsPerMinute = *rotations;
    fanKey(key, fan, "Mn");
    if (const std::optional<double> minimum = readScalar(smc, key))
      reading.minimumRotationsPerMinute = *minimum;
    fanKey(key, fan, "Mx");
    if (const std::optional<double> maximum = readScalar(smc, key))
      reading.maximumRotationsPerMinute = *maximum;
    fanKey(key, fan, "Tg");
    if (const std::optional<double> target = readScalar(smc, key))
      reading.targetRotationsPerMinute = *target;
    fanKey(key, fan, "Md");
    if (const std::optional<double> mode = readScalar(smc, key))
      reading.mode = uint8_t(std::clamp(*mode, 0.0, 255.0));
    state.fans.push_back(reading);
  }

  for (const std::string_view key : temperatureKeys()) {
    const std::optional<double> celsius = readScalar(smc, key);
    if (!celsius || *celsius < kMinimumPlausibleCelsius ||
        *celsius > kMaximumPlausibleCelsius)
      continue;
    state.temperatures.push_back({std::string(key), *celsius});
    if (state.hottestKey.empty() || *celsius > state.hottestCelsius) {
      state.hottestCelsius = *celsius;
      state.hottestKey = std::string(key);
    }
  }
  return state;
}

} // namespace splash::engine
