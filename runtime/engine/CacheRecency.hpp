#pragma once

#include <cstdint>
#include <limits>
#include <tuple>

namespace splash::engine {

// Shared monotonic access order for KV blocks and composite states.
class CacheRecency final {
public:
  [[nodiscard]] uint64_t next() noexcept {
    if (value_ != std::numeric_limits<uint64_t>::max())
      ++value_;
    return value_;
  }
  // Continues after a value an earlier process handed out, for what a
  // persistent tier takes back.
  void continueAfter(uint64_t value) noexcept {
    if (value > value_)
      value_ = value;
  }

private:
  uint64_t value_ = 0;
};

// Where one candidate stands in its cache's eviction order: a protected tier
// first (objects a reclaim takes after every ordinary one), then the policy's
// rank (fewer touches for Lfu, more bytes for Largest), then the use order
// the policy keeps, then the id for a total order.
struct CacheEvictionCandidate final {
  uint64_t id = 0;
  uint64_t lastUsed = 0;
  uint64_t rank = 0;
  uint8_t tier = 0;
};

// Whether left leaves before right in a reclaim's order.
[[nodiscard]] inline bool evictsBefore(const CacheEvictionCandidate &left,
                                       const CacheEvictionCandidate &right) noexcept {
  return std::tie(left.tier, left.rank, left.lastUsed, left.id) <
         std::tie(right.tier, right.rank, right.lastUsed, right.id);
}

} // namespace splash::engine
