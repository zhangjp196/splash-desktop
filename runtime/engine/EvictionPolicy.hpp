#pragma once

#include <cstdint>
#include <limits>

namespace splash::engine {

// How a reclaim picks the next cached object to give up.
//
// Lru keeps the order of use: the object no request has touched for longest
// goes first. Fifo keeps the order objects entered the cache in, whatever
// touches them afterwards. Lfu orders by the number of touches, fewest
// first, and then by use among equals: a prefix reused many times outlives
// one a single prompt filled. Largest orders by the bytes the object holds,
// most first and then by use: an eviction gives back memory sooner, at the
// cost of the cache's hit rate.
enum class EvictionPolicy : uint8_t { Lru, Fifo, Lfu, Largest };

// What a cache's owner asks of its eviction, from the model's settings.
//
// protectHits puts an object that many touches into a protected tier: no
// object of the ordinary tier is evicted while one of the protected tier
// remains, whatever the policy's order says. Zero disables the tier, so
// that a default cache evicts exactly as it did before the setting existed.
struct EvictionSettings final {
  EvictionPolicy policy = EvictionPolicy::Lru;
  uint8_t protectHits = 0;

  bool operator==(const EvictionSettings &) const = default;
};

// The policy's name in the engine's and the server's vocabulary.
[[nodiscard]] constexpr const char *evictionPolicyName(EvictionPolicy policy) noexcept {
  switch (policy) {
  case EvictionPolicy::Lru: return "lru";
  case EvictionPolicy::Fifo: return "fifo";
  case EvictionPolicy::Lfu: return "lfu";
  case EvictionPolicy::Largest: return "largest";
  }
  return "lru";
}

// The rank a policy orders its candidates by, beside their use: zero for the
// policies whose order is the use itself, the touch count for Lfu, and the
// complement of the bytes for Largest so that a larger object sorts first.
[[nodiscard]] constexpr uint64_t evictionRank(EvictionPolicy policy, uint64_t bytes,
                                              uint32_t hits) noexcept {
  switch (policy) {
  case EvictionPolicy::Lfu: return hits;
  case EvictionPolicy::Largest:
    return std::numeric_limits<uint64_t>::max() - bytes;
  case EvictionPolicy::Lru:
  case EvictionPolicy::Fifo: return 0;
  }
  return 0;
}

// The tier an object with this many touches waits in: protected, after every
// object of the ordinary tier, once protectHits is nonzero and reached.
[[nodiscard]] constexpr uint8_t evictionTier(uint8_t protectHits, uint32_t hits) noexcept {
  return protectHits && hits >= protectHits ? 1 : 0;
}

} // namespace splash::engine
