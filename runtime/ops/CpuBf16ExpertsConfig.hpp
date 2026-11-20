#pragma once

#include <string_view>

namespace splash::ops {

// Only the explicit value "1" opts in. Missing, empty, and unsupported values
// preserve the Metal expert path.
[[nodiscard]] constexpr bool cpuBf16ExpertsEnabled(std::string_view value) noexcept {
  return value == "1";
}

} // namespace splash::ops
