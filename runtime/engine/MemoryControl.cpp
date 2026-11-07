#include "engine/MemoryControl.hpp"

#include "StderrLine.hpp"

#include <optional>
#include <sstream>

namespace splash::engine {

std::string MemoryStatusReporter::update(const ResourceWaitSnapshot &wait,
                                         bool hostGrowthAllowed) {
  const unsigned state = (!hostGrowthAllowed ? 1u : 0u) |
                         (wait.memory ? 2u : 0u) |
                         (wait.suspended ? 4u : 0u) |
                         (wait.draining ? 8u : 0u) |
                         (wait.heldBehindRefusal ? 16u : 0u);
  if (state == state_)
    return {};
  state_ = state;
  if (!state)
    return "Memory: growth available; resource wait cleared";
  std::ostringstream out;
  out << "Memory: growth " << (hostGrowthAllowed ? "available" : "paused")
      << "; waiting=" << wait.memory << "; suspended=" << wait.suspended;
  if (wait.heldBehindRefusal)
    out << "; held=" << wait.heldBehindRefusal;
  if (wait.draining)
    out << "; waiting for resident requests to finish";
  return out.str();
}

bool MemoryControl::run(MemoryPressure pressure) {
  loop_.releaseIdleWeights();
  loop_.spillIdleCache();
  governor_.setPressure(pressure);
  const double now = loop_.monotonicMilliseconds();
  // Requests held back by a refusal wait for memory too, the refused one
  // included while a pass defers it.
  const ResourceWaitSnapshot wait = loop_.resourceWaitSnapshot();
  const bool requestWaiting = wait.memory || wait.suspended || wait.heldBehindRefusal;
  governor_.setRequestWaiting(requestWaiting);
  static_cast<void>(backend_.refreshMemoryStats());
  const MemoryGovernorSnapshot memory = governor_.evaluate();
  const std::string diagnostic = reporter_.update(wait, memory.hostGrowthAllowed);
  if (!diagnostic.empty())
    logLine(diagnostic);
  // An elastic expert store is the first cache a short host and a waiting
  // request take from, a chunk at a time down to its floor.
  constexpr uint64_t kRoutedChunkBytes = 1ULL << 30;
  if (loop_.streamedMemoryElastic() &&
      (requestWaiting || !memory.hostGrowthAllowed)) {
    const uint64_t held = loop_.streamedMemoryBytes();
    const uint64_t least = loop_.minimumStreamedMemoryBytes();
    if (held > least)
      static_cast<void>(loop_.setStreamedMemory(
          held - least > kRoutedChunkBytes ? held - kRoutedChunkBytes : least));
  }
  const std::optional<MemoryReclaimDirective> directive =
      policy_.update(memory, now, requestWaiting);
  // With no request waiting and room to grow, the store takes a chunk back.
  const auto grow = [&] {
    if (!loop_.streamedMemoryElastic() || requestWaiting ||
        !memory.hostGrowthAllowed ||
        memory.hostHeadroomBytes <= 2 * kRoutedChunkBytes)
      return;
    const uint64_t held = loop_.streamedMemoryBytes();
    const uint64_t most = loop_.maximumStreamedMemoryBytes();
    if (held < most)
      static_cast<void>(loop_.setStreamedMemory(
          most - held > kRoutedChunkBytes ? held + kRoutedChunkBytes : most));
  };
  if (!directive) {
    grow();
    static_cast<void>(backend_.refreshMemoryStats());
    return false;
  }
  const MemoryReclaimResult reclaim = loop_.reclaimMemory(*directive);
  policy_.reclaimed(*directive, reclaim);
  governor_.reclaimed(reclaim.outcome);
  grow();
  static_cast<void>(backend_.refreshMemoryStats());
  // What transfers held back continues at the next command-free point.
  return reclaim.outcome == ReclaimOutcome::Pending;
}

} // namespace splash::engine
