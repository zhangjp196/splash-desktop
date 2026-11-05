#pragma once

#include "engine/CacheRecency.hpp"

#include <cstdint>
#include <optional>
#include <set>
#include <tuple>

namespace splash::engine {

// Members in eviction order: the protected tier, the policy's rank, the use
// order the policy keeps, then the id. A member allocates its tree node once,
// when it is published, and keeps it while unlinked, so linking and unlinking
// under memory pressure never allocate. Unlinking an unlinked member does
// nothing.
class RecencyOrder final {
public:
  using Key = std::tuple<uint64_t, uint64_t, uint64_t, uint64_t>;

  class Node final {
  private:
    friend class RecencyOrder;
    [[nodiscard]] bool linked() const noexcept { return owner_ != nullptr; }

    std::set<Key>::node_type handle_;
    Key key_{};
    RecencyOrder *owner_ = nullptr;
  };

  // The only operation that allocates; link() sets the key.
  [[nodiscard]] static Node allocate() {
    std::set<Key> scratch;
    Node node;
    node.handle_ = scratch.extract(scratch.emplace(0, 0, 0, 0).first);
    return node;
  }

  void link(Node &node, uint64_t tier, uint64_t rank, uint64_t lastUsed,
            uint64_t id) noexcept {
    if (node.linked() || node.handle_.empty())
      std::terminate();
    node.handle_.value() = {tier, rank, lastUsed, id};
    if (!order_.insert(std::move(node.handle_)).inserted)
      std::terminate();
    node.key_ = {tier, rank, lastUsed, id};
    node.owner_ = this;
  }

  static void unlink(Node &node) noexcept {
    if (!node.linked())
      return;
    node.handle_ = node.owner_->order_.extract(node.key_);
    if (node.handle_.empty())
      std::terminate();
    node.owner_ = nullptr;
  }

  [[nodiscard]] std::optional<CacheEvictionCandidate> oldest() const noexcept {
    if (order_.empty())
      return std::nullopt;
    return candidate(*order_.begin());
  }

  // The member after `after` in the order, for a scan that skips members.
  [[nodiscard]] std::optional<CacheEvictionCandidate>
  next(const CacheEvictionCandidate &after) const noexcept {
    auto position =
        order_.upper_bound({after.tier, after.rank, after.lastUsed, after.id});
    if (position == order_.end())
      return std::nullopt;
    return candidate(*position);
  }

  [[nodiscard]] std::optional<CacheEvictionCandidate> newest() const noexcept {
    if (order_.empty())
      return std::nullopt;
    return candidate(*order_.rbegin());
  }

  [[nodiscard]] size_t size() const noexcept { return order_.size(); }

private:
  [[nodiscard]] static CacheEvictionCandidate candidate(const Key &key) noexcept {
    return CacheEvictionCandidate{std::get<3>(key), std::get<2>(key),
                                  std::get<1>(key),
                                  static_cast<uint8_t>(std::get<0>(key))};
  }

  std::set<Key> order_;
};

} // namespace splash::engine
