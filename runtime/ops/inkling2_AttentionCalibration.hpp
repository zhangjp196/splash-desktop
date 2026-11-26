#pragma once

// The attention-only subset of upstream 1.7.2's KernelCalibration. The full
// table drives upstream's Linear/MoE kernel selection, which the fork does
// not use; only these fields are needed to plan Inkling-2's attention.
// Kept separate so fork plans and their calibration stay untouched.

#include "metal/DeviceCapabilities.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "ops/Linear.hpp"

#include <cstdint>

namespace splash::ops {

struct Inkling2AttentionCalibration final {
  uint32_t pagesPerSplit = 16;
  uint32_t int8QueriesFromPages = 16;
  uint32_t cores = 0;
  [[nodiscard]] static constexpr Inkling2AttentionCalibration of(const DeviceCapabilities &device) noexcept {
    return {16, 16, plannedGpuCores(device)};
  }
};

} // namespace splash::ops
