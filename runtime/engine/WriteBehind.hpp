#pragma once

#include "engine/Cache.hpp"

#include <cstdint>
#include <deque>
#include <optional>
#include <unordered_set>

namespace splash::engine {

struct WriteBehindSnapshot final {
  // Points waiting to fall due, or the one being written.
  uint32_t waiting = 0;
  // Points durable when they fell due or once their writes landed.
  uint64_t durable = 0;
  // Points their conversation went on from, or gone, before they fell due.
  uint64_t unneeded = 0;
  // Points the tier could not keep.
  uint64_t refused = 0;
};

// Keeps a persistent tier's newest restore points on disk while they stay
// in RAM. An ordinary point published or reused at least kMinimumTokens
// deep is made durable (Cache::persist) once it has waited
// kDelayMilliseconds, unless its conversation has gone on from it by then:
// a point that is newest only briefly, like a step of a tool loop, costs no
// write. The shallowest point of a branch is durable whatever follows it
// (Cache::superseded): a conversation that shares the prefix up to it and
// diverges there resumes from it. kMinimumTokens is set by the tier's quota, not by the write: a
// point costs its state (153 MiB for 27B, 64 MiB for 35B) plus the KV of
// its chain, and the state dominates at shallow depths, so the floor is
// where a point's KV stops being a rounding error beside its state. What
// the floor decides is only whether a conversation shallower than it is
// restorable at all: every conversation keeps its newest point on the
// floor's other side. 512 tokens is about 1.7 s of prefill to save for
// some 217 MiB of a quota that now clears 64 GiB, against never saving
// it. Points become durable one at a time, in the order they were
// published. Like every write of the tier these draw on its write budget,
// when one is set (model::DiskBudget): a point waits while the budget holds
// its state back, which is written before its chain.
class WriteBehind final {
public:
  static constexpr double kDelayMilliseconds = 10'000.0;
  static constexpr uint32_t kMinimumTokens = 512;

  explicit WriteBehind(Cache &cache);
  WriteBehind(const WriteBehind &) = delete;
  WriteBehind &operator=(const WriteBehind &) = delete;

  // A point published or reused at this block, `tokens` deep. Nothing in a
  // temporary tier.
  void published(uint64_t block, uint32_t tokens, double now);
  // Steps the points that are due. True when one became durable or was
  // given up; a write it starts wakes the engine when it lands.
  [[nodiscard]] bool run(double now);
  // At a clean stop: steps every waiting point, the newest first, due or
  // not, whatever the write budget (Cache::liftDiskWriteBudget). True once
  // none is left.
  [[nodiscard]] bool flush();
  // When run() next has something to do that no landing write will wake it
  // for: the next point falling due, or, for a step the write budget held
  // back, when the budget admits it again.
  [[nodiscard]] std::optional<double> nextWakeup() const;
  [[nodiscard]] WriteBehindSnapshot snapshot() const noexcept;

private:
  struct Point final {
    uint64_t block = 0;
    double due = 0.0;
  };
  // Counts a step that settled its point; false while it is in progress.
  bool settle(PersistStatus status);

  Cache &cache_;
  std::deque<Point> waiting_;
  std::unordered_set<uint64_t> queued_;
  // The front point is due and waits for its writes, or the tier's, to land,
  // or for the write budget.
  bool stalled_ = false;
  WriteBehindSnapshot counters_;
};

} // namespace splash::engine
