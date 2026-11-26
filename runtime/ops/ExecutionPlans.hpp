#pragma once

#include "ops/CpuBf16ExpertsConfig.hpp"
#include "ops/DraftAttention.hpp"
#include "ops/Linear.hpp"
#include "ops/MoE.hpp"
#include "ops/PagedAttention.hpp"
#include "ops/inkling2_AttentionCalibration.hpp"

#include <span>

namespace splash::ops {

// The device's plans of every operator a model runs. One runtime owns this
// object and production models borrow it; it never changes after creation.
class ExecutionPlans final {
public:
  explicit ExecutionPlans(const DeviceCapabilities &device);
  [[nodiscard]] const Linear &linear() const noexcept { return linear_; }
  // Upstream 1.7.2's decode-step linear, used by the ported Inkling-2 family.
  // The fork's Linear runs both phases from one object, so this is the same
  // one; no existing family's plan selection changes.
  [[nodiscard]] const Linear &decodeLinear() const noexcept { return linear_; }
  // The device's attention calibration, for Inkling-2's relative attention
  // plans. The fork's existing plans keep their device policy.
  [[nodiscard]] const Inkling2AttentionCalibration &calibration() const noexcept { return calibration_; }

  [[nodiscard]] PrefillAttentionPlan prefillAttention(
      uint32_t rows, uint32_t queryHeads, kv::Layout layout,
      AttentionVariant variant = {}) const;
  [[nodiscard]] VerifyAttentionPlan verifyAttention(
      uint32_t lanes, uint32_t queryHeads, kv::Layout layout,
      std::span<const uint32_t> historyTokens, AttentionVariant variant = {}) const;
  [[nodiscard]] DraftAttentionPlan draftAttention(
      DraftAttentionShape shape, uint32_t lanes) const;
  [[nodiscard]] MoePlan moePrefill(MoeShape shape, uint32_t rows) const;
  [[nodiscard]] MoePlan moeDecode(MoeShape shape, uint32_t lanes) const;

  // Bounds cover every row count up to the requested maximum, not just that
  // one. Decode arenas use a per-lane stride of
  // max_B ceil(requiredBytes(B)/B), independently for each scratch field.
  [[nodiscard]] AttentionWorkspace prefillAttentionWorkspace(
      uint32_t maximumRows, uint32_t queryHeads, kv::Layout layout) const;
  [[nodiscard]] AttentionWorkspace verifyAttentionWorkspacePerLane(
      uint32_t queryHeads, kv::Layout layout) const;
  [[nodiscard]] DraftAttentionWorkspace draftAttentionWorkspacePerLane(
      DraftAttentionShape shape) const;
  [[nodiscard]] MoeWorkspace moePrefillWorkspace(
      MoeShape shape, uint32_t maximumRows) const;
  [[nodiscard]] MoeWorkspace moeDecodeWorkspacePerLane(MoeShape shape) const;
  // This scratch is one whole-command buffer, not a per-lane arena field.
  [[nodiscard]] uint64_t gateUpWorkspace(ProjectionShape shape) const;

private:
  // The device's configuration of a MoE plan of `rows` rows in `phase`.
  [[nodiscard]] MoeConfig moeConfig(MoeShape shape, uint32_t rows, MoePhase phase) const;

  Linear linear_;
  Inkling2AttentionCalibration calibration_;
  GpuFamilyClass family_;
  uint32_t gpuCores_;
  bool cpuBf16Experts_;
};

} // namespace splash::ops
