#pragma once

// One ledger of the memory the engine holds, by the account that holds it,
// in the fixed order a reclaim frees it. The accounts are what /status
// reports under memory_governor.accounts, so an operator sees the same order
// the engine frees in: what the model rebuilds first, then what can be
// reused, then what a request is using, and last what a start built.

#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <span>
#include <string_view>
#include <vector>

namespace splash::engine {

enum class MemoryLedgerOrder : uint8_t {
  // Cache the model can rebuild: a streamed table's row cache, the expert
  // store's streamed arena, the KV block cache, encoded image rows.
  Rebuildable,
  // Memory that can be reused by another request but is not rebuilt from
  // nothing: pooled state buffers and cached composite states.
  Reusable,
  // What requests in service use: the KV pool and lane state.
  InUse,
  // What a start built and never frees while the engine runs: weights,
  // scratch, pipelines and runtime reserves.
  Floor,
};

[[nodiscard]] inline constexpr std::string_view
memoryLedgerOrderName(MemoryLedgerOrder order) noexcept {
  switch (order) {
  case MemoryLedgerOrder::Rebuildable: return "rebuildable";
  case MemoryLedgerOrder::Reusable: return "reusable";
  case MemoryLedgerOrder::InUse: return "in_use";
  case MemoryLedgerOrder::Floor: return "floor";
  }
  return "floor";
}

// Why an allocation is made. Floor is what a start builds and a weights
// restore while requests wait; InService is a request's own growth; Start is
// a request starting; Growth is cache and pools growing for later; Optional
// is a request's growth that could wait.
enum class ClaimClass : uint8_t { Floor, InService, Start, Growth, Optional };
inline constexpr size_t kClaimClasses = 5;
inline constexpr std::string_view kClaimClassNames[] = {"floor", "in_service", "start",
                                                        "growth", "optional"};
static_assert(std::size(kClaimClassNames) == kClaimClasses);

inline constexpr std::string_view claimClassName(ClaimClass cls) noexcept {
  return kClaimClassNames[static_cast<size_t>(cls)];
}

// Who memory goes to first, the larger first.
struct ClaimRank final {
  uint8_t priority = 0;          // 2 - RequestPriority
  bool decoding = false;         // at equal priority a decoding lane outranks a prefill
  uint64_t completedTokens = 0;  // then more completed work
  uint64_t earliness = 0;        // then earlier submission
  auto operator<=>(const ClaimRank &) const = default;
};

struct ClaimSpec final {
  ClaimClass cls = ClaimClass::InService;
  uint64_t requestId = 0;
  ClaimRank rank;
};

// Claims take ids from 1 up, never again.
struct ClaimId final {
  uint64_t value = 0;
  explicit operator bool() const noexcept { return value != 0; }
  friend bool operator==(ClaimId, ClaimId) = default;
};

// What the ledger made of each class: the claims it granted and their bytes,
// those still held, and those refused.
struct ClaimClassSnapshot final {
  uint64_t granted = 0;
  uint64_t grantedBytes = 0;
  uint64_t refused = 0;
  uint64_t held = 0;
  uint64_t heldBytes = 0;
};

class ClaimLedger final {
public:
  ClaimId add(const ClaimSpec &spec, uint64_t bytes) {
    const ClaimId id{++next_};
    active_.push_back({id, spec, bytes});
    ClaimClassSnapshot &counts = classes_[static_cast<size_t>(spec.cls)];
    ++counts.granted;
    counts.grantedBytes += bytes;
    ++counts.held;
    counts.heldBytes += bytes;
    return id;
  }

  void release(ClaimId id) {
    for (size_t index = 0; index < active_.size(); ++index) {
      if (active_[index].id == id) {
        ClaimClassSnapshot &counts =
            classes_[static_cast<size_t>(active_[index].spec.cls)];
        --counts.held;
        counts.heldBytes -= active_[index].bytes;
        active_.erase(active_.begin() + static_cast<ptrdiff_t>(index));
        return;
      }
    }
  }

  void refused(ClaimClass cls) {
    ++classes_[static_cast<size_t>(cls)].refused;
  }

  [[nodiscard]] std::array<ClaimClassSnapshot, kClaimClasses>
  classes() const noexcept {
    return classes_;
  }

private:
  struct Active final {
    ClaimId id;
    ClaimSpec spec;
    uint64_t bytes = 0;
  };
  uint64_t next_ = 0;
  std::vector<Active> active_;
  std::array<ClaimClassSnapshot, kClaimClasses> classes_{};
};

struct MemoryLedgerAccount final {
  std::string_view name;
  MemoryLedgerOrder order = MemoryLedgerOrder::Floor;
  // What the account may not go below while requests run, and what it holds
  // now.
  uint64_t floorBytes = 0;
  uint64_t currentBytes = 0;
};

// Built in the release order, so entries() is the order a reclaim frees in:
// rebuildable first, floor last.
class MemoryLedger final {
public:
  void add(std::string_view name, MemoryLedgerOrder order, uint64_t floorBytes,
           uint64_t currentBytes) {
    accounts_.push_back({name, order, floorBytes, currentBytes});
  }
  [[nodiscard]] std::span<const MemoryLedgerAccount> accounts() const noexcept {
    return accounts_;
  }
  [[nodiscard]] uint64_t totalBytes() const noexcept {
    uint64_t total = 0;
    for (const MemoryLedgerAccount &account : accounts_) total += account.currentBytes;
    return total;
  }

private:
  std::vector<MemoryLedgerAccount> accounts_;
};

} // namespace splash::engine
