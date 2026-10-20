#pragma once

#include "metal/MetalBackend.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace splash::metal {

// GPU time of one dispatch committed as its own command while profiling.
struct DispatchTiming {
  std::string pipelineName;
  double gpuSeconds = 0.0;
};

// What one pipeline allows a launch to ask for: threads per threadgroup (the
// pipeline's own limit, which falls with its register count on M1 and M2),
// its thread execution width, and the device's threadgroup memory limit.
struct LaunchLimits {
  uint32_t maxThreadsPerThreadgroup = 0;
  uint32_t threadExecutionWidth = 0;
  uint64_t maxThreadgroupMemoryBytes = 0;
};

// What tests and benchmarks observe of a MetalBackend beyond its serving
// API. Only the instrumented build of MetalBackend.mm
// (SPLASH_BACKEND_INSTRUMENTATION) defines these; production binaries never
// link it.
class BackendInstrumentation final {
public:
  // Commands that passed the submission gate since construction.
  [[nodiscard]] static uint64_t submittedCommands(const MetalBackend &backend);
  [[nodiscard]] static size_t cachedPipelines(const MetalBackend &backend);
  // The launch limits of one pipeline, compiled here if it is not cached yet.
  [[nodiscard]] static LaunchLimits launchLimits(MetalBackend &backend,
                                                 std::string_view pipeline);
  // While profiling, submitCommandAsync() commits every dispatch as its own
  // command and waits for it, recording its GPU time, then invokes
  // completion inline and returns an already-completed ticket with the
  // summed timing; it refuses a command with event steps (EventStep).
  // takeDispatchProfile() reads and clears the timings.
  static void setDispatchProfiling(MetalBackend &backend, bool enabled);
  [[nodiscard]] static std::vector<DispatchTiming>
  takeDispatchProfile(MetalBackend &backend);
};

} // namespace splash::metal
