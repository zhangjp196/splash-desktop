#include "ops/MoE.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Gguf.h"
#include "metal/abi/MoE.h"
#include "ops/BufferExtent.hpp"
#include "ops/CpuExpertPrototype.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>

#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace splash::ops {
namespace {

// A GGUF tensor of output x input weights, and the planes of its GGUF_FMT_*
// format; what reads a float tensor checks its fp32 values.
void requireSegment(const QuantizedSegment &segment, uint32_t output, uint32_t input, bool floatWeights,
                    const std::string &name) {
  if (segment.isFloat() != floatWeights || (!floatWeights && segment.formatId >= GGUF_FMT_COUNT) ||
      segment.outputSize != output || segment.inputSize != input)
    throw std::invalid_argument(name + " does not match the plan");
  if (!floatWeights) requireSegmentPlanes(segment, name);
}

// Every weight the plan's dispatches read. The router runs through
// addGgufFloat, which checks its weights.
void validate(const BlockMoeWeights &weights, MoeShape shape) {
  const uint32_t hidden = shape.hiddenSize, intermediate = shape.expertIntermediateSize;
  requireSegment(weights.router, shape.experts, hidden, true, "MoE router");
  if (shape.sharedGate) {
    requireSegment(weights.sharedScalarGate, 1, hidden, true, "MoE shared scalar gate");
    // The select kernel reads the scalar gate's fp32 weights.
    if (weights.sharedScalarGate.formatId != QuantizedSegment::kFloat32)
      throw std::invalid_argument("the MoE shared scalar gate is not F32");
    requireBytes(weights.sharedScalarGate.plane0, uint64_t{hidden} * sizeof(float), "MoE shared scalar gate weight");
  }
  if (shape.selectionBias)
    requireBytes(weights.selectionBias, uint64_t{shape.experts} * sizeof(float), "MoE selection bias");
  requireBytes(weights.expertScales, uint64_t{3} * shape.experts * sizeof(float), "MoE expert scales");
  if (!weights.slots) throw std::invalid_argument("MoE routed experts have no slots");
  if (shape.rotated() != static_cast<bool>(weights.rotation) ||
      shape.rotated() != static_cast<bool>(weights.intermediateRotation))
    throw std::invalid_argument("a MoE block's stored rotation does not match its shape");
  if (shape.rotated() &&
      (hidden % weights.rotation.block || intermediate % weights.intermediateRotation.block ||
       weights.rotation.block != shape.rotationBlock || weights.intermediateRotation.block != shape.rotationBlock))
    throw std::invalid_argument("a rotated MoE block takes whole rotation blocks");
  requireBytes(weights.slotTable, uint64_t{shape.experts} * sizeof(uint32_t), "MoE slot table");
  for (const auto &[projection, name, output, input] :
       {std::tuple{&weights.gate, "gate", intermediate, hidden}, std::tuple{&weights.up, "up", intermediate, hidden},
        std::tuple{&weights.down, "down", hidden, intermediate}}) {
    requireSegment(projection->routed, weights.slots * output, input, false, std::string("MoE expert ") + name);
    // A family with no shared expert (Inkling 2, sharedSlices=0) lays out no
    // shared segment; requiring it with zero rows rejects the block.
    if (shape.sharedSlices)
      requireSegment(projection->shared, shape.sharedSlices * output, input, false, std::string("MoE shared ") + name);
  }
}

MoeWorkspace workspaceFor(MoeShape shape, uint32_t rows, uint32_t tileRows, MoeGgufTile ggufTile) {
  if (!shape.valid())
    throw std::invalid_argument("invalid MoE workspace shape");
  const uint64_t routes = uint64_t{rows} * shape.routesPerToken();
  const uint32_t tiles = moeMaximumTiles(rows, shape, tileRows);
  const uint64_t groupedRows = uint64_t{tiles} * tileRows;
  const uint32_t widest = std::max(shape.hiddenSize, shape.expertIntermediateSize);
  // The router's fp32 scores, a row of routerWidth() expert slots per row,
  // live in the grouped input until the gather overwrites them. Register
  // plans also hold the down pass's Table16 tiles there.
  const uint64_t scoreBytes = uint64_t{rows} * shape.routerWidth() * sizeof(float);
  const bool table16 = ggufTile == MoeGgufTile::Register;
  const uint64_t sumsBytes = table16 ? tableSumsBytes(LinearInput::Table16, widest, groupedRows) : 0;
  return {routes * sizeof(uint32_t), routes * sizeof(float),
          uint64_t{tiles} * sizeof(MoeTileDescriptor), sizeof(MoeTileCount),
          groupedRows * sizeof(uint32_t), routes * sizeof(uint32_t),
          std::max(tableBytes(table16 ? widest : shape.hiddenSize, groupedRows), scoreBytes),
          groupedRows * shape.expertIntermediateSize * sizeof(uint16_t),
          groupedRows * widest * sizeof(uint16_t), sumsBytes,
          shape.rotated() ? uint64_t{rows} * shape.hiddenSize * sizeof(uint16_t) : 0};
}

// The shape's compiled routing (MoeShape::compiledRouting).
const CompiledMoeRouting &compiledRouting(MoeShape shape) {
  if (const CompiledMoeRouting *compiled = shape.compiledRouting()) return *compiled;
  throw std::invalid_argument("no MoE kernel routes the shape's router and shared slices");
}

// A routing kernel's instance for the shape's router width (past
// SPLASH_MOE_EXPERT_SLOTS experts, its wide one) and its compiled routing's
// `suffix`.
std::string routingKernel(std::string name, MoeShape shape, std::string_view CompiledMoeRouting::*suffix) {
  static_assert(SPLASH_MOE_WIDE_EXPERT_SLOTS == 512, "the wide routing kernels are the _e512 instances");
  if (shape.routerWidth() == SPLASH_MOE_WIDE_EXPERT_SLOTS) name += "_e512";
  name += compiledRouting(shape).*suffix;
  return name;
}

// Whether a plan runs gate and up in one pass
// (moe_expert_gguf_m<rows>_gate_up[_mixed]): on the staged tile, whose simdgroups
// each decode their own pass's formats. The register tile runs two passes.
// Shared-expert staged plans also use two passes: the fork's shared fused
// variant can leave a command scheduled under Metal validation, while the
// independent passes are bitwise-equivalent and complete on every tested GPU.
bool oneGateUpPass(const MoePlan &plan) noexcept {
  return plan.configuration().ggufTile == MoeGgufTile::Staged && plan.shape().sharedSlices == 0;
}

// The expert passes over grouped tiles, each on the grid of the live
// tiles that the grouping wrote (MoeTileCount), within the column tiles of
// `group` and maximumTiles(): gate and up in one pass into
// expertIntermediate (oneGateUpPass), or gate into expertOutput (the down
// pass overwrites it after the up pass consumed it) and up with silu(gate)
// into expertIntermediate; then down into expertOutput. The gate/up passes
// read the grouped rows the gather writes into groupedInput, as Table16 tiles
// on register plans, whose down pass reads the tiles a prepare dispatch
// writes there.
void addExperts(metal::CommandGraph &graph, const MoeScratch &scratch, const BlockMoeWeights &weights,
                const MoePlan &plan, const MoeGroupParams &group, bool rotated) {
  const MoeShape shape = plan.shape();
  const uint32_t tiles = plan.maximumTiles();
  const bool table16 = plan.configuration().ggufTile == MoeGgufTile::Register;
  if (table16)
    for (const BlockExpertProjection *projection : {&weights.gate, &weights.up, &weights.down})
      for (const QuantizedSegment *s : {&projection->routed, &projection->shared})
        if (!quant_register_tensor(s->formatId, s->inputSize))
          throw std::invalid_argument(std::string("the register expert tile does not take ") + s->name() +
                                      " weights of " + std::to_string(s->inputSize) + " inputs");
  const uint32_t threads = table16 ? GGUF_REGISTER_THREADS : GGUF_STAGED_THREADS;
  const std::string kernel = table16 ? "moe_expert_gguf_sg" : "moe_expert_gguf_m" + std::to_string(plan.tileRows());
  // The passes' instances for the shape's shared slices.
  const std::string slices(compiledRouting(shape).slicesSuffix);
  const std::string activation(compiledRouting(shape).activationSuffix);
  const metal::IndirectGrid gateUpGrid{scratch.tileCount, offsetof(MoeTileCount, gate_up_grid)};
  const metal::IndirectGrid downGrid{scratch.tileCount, offsetof(MoeTileCount, down_grid)};
  // A family with no shared expert (Inkling 2, sharedSlices=0) has empty
  // shared planes. The s0 kernel variants never read them, but Metal rejects
  // an empty binding. Use a live scratch buffer as the placeholder: a
  // streamed expert's routed plane is released between commands and would be
  // reported as released memory once a real request follows warmup.
  const metal::MetalBuffer &noShared = scratch.expertOutput;
  const auto sharedPlane0 = [&](const BlockExpertProjection &p) {
    return shape.sharedSlices ? p.shared.plane0 : noShared;
  };
  const auto sharedPlane1 = [&](const BlockExpertProjection &p) {
    return shape.sharedSlices ? p.shared.plane1Slot() : noShared;
  };
  const auto sharedMeta = [&](const BlockExpertProjection &p) {
    return shape.sharedSlices ? p.shared.meta : noShared;
  };
  const auto params = [&](const BlockExpertProjection &projection, uint32_t n, uint32_t k, uint32_t index) {
    return MoeGgufExpertParams{k, n, shape.experts, projection.routed.formatId, projection.shared.formatId, index};
  };
  const auto pass = [&](const BlockExpertProjection &projection, bool up,
                        const metal::MetalBuffer &passInput,
                        const metal::MetalBuffer &output, uint32_t n, uint32_t k,
                        uint32_t columns, const metal::IndirectGrid &grid, uint32_t index) {
    std::vector<metal::MetalBuffer> bindings{passInput};
    if (table16) bindings.push_back(scratch.groupedSums);
    bindings.insert(bindings.end(),
                    {scratch.tileDescriptors, scratch.tileCount,
                     projection.routed.plane0, projection.routed.plane1Slot(),
                     projection.routed.meta, sharedPlane0(projection),
                     sharedPlane1(projection), sharedMeta(projection), output,
                     scratch.expertOutput, weights.slotTable, weights.expertScales});
    graph.add(kernel + (up ? "_g" : "_a") + slices, std::move(bindings), params(projection, n, k, index),
              {columns, tiles, 1},
              grid, {threads, 1, 1});
  };
  const uint32_t hidden = shape.hiddenSize;
  const uint32_t intermediate = shape.expertIntermediateSize;
  if (oneGateUpPass(plan)) {
    const BlockExpertProjection &gate = weights.gate, &up = weights.up;
    // One pair table where gate and up share their formats, one each otherwise.
    const bool mixed = gate.routed.formatId != up.routed.formatId || gate.shared.formatId != up.shared.formatId;
    graph.add(kernel + (mixed ? "_gate_up_mixed" : "_gate_up") + slices + activation,
              {scratch.groupedInput, scratch.tileDescriptors, scratch.tileCount,
               gate.routed.plane0, gate.routed.plane1Slot(), gate.routed.meta,
               sharedPlane0(gate), sharedPlane1(gate), sharedMeta(gate),
               up.routed.plane0, up.routed.plane1Slot(), up.routed.meta,
               sharedPlane0(up), sharedPlane1(up), sharedMeta(up),
               scratch.expertIntermediate, weights.slotTable, weights.expertScales},
              MoeGgufGateUpParams{hidden, intermediate, shape.experts, gate.routed.formatId, gate.shared.formatId,
                                  up.routed.formatId, up.shared.formatId},
              {group.gate_up_columns, tiles, 1}, gateUpGrid, {threads, 1, 1});
  } else {
    if (!activation.empty())
      throw std::invalid_argument("a softcapped MoE activation needs the staged gate/up tile");
    pass(weights.gate, false, scratch.groupedInput, scratch.expertOutput,
         intermediate, hidden, group.gate_up_columns, gateUpGrid, 0);
    pass(weights.up, true, scratch.groupedInput, scratch.expertIntermediate,
         intermediate, hidden, group.gate_up_columns, gateUpGrid, 1);
  }
  if (rotated) {
    // The routed tiles' intermediate, H (D e), which the down pass's routed
    // weights multiply: the shared expert's tiles follow the routed ones and
    // keep theirs.
    const uint32_t block = weights.intermediateRotation.block;
    graph.add("moe_rotate_intermediate",
              {scratch.expertIntermediate, scratch.tileDescriptors, weights.intermediateRotation.signs},
              GgufRotationParams{intermediate, block, rotationScale(block), plan.tileRows()},
              {intermediate / block, tiles, plan.tileRows()},
              metal::IndirectGrid{scratch.tileCount, offsetof(MoeTileCount, routed_grid)}, {block / 4, 1, 1});
  }
  if (table16)
    graph.add("moe_prepare_table16",
              {scratch.expertIntermediate, scratch.tileCount,
               scratch.groupedInput, scratch.groupedSums},
              intermediate, {tiles, (intermediate + 255) / 256, 1});
  pass(weights.down, false,
       table16 ? scratch.groupedInput : scratch.expertIntermediate,
       scratch.expertOutput, hidden, intermediate, group.down_columns, downGrid, 2);
}

} // namespace

// Plans run 8-row tiles in decode and 8- or 32-row tiles in prefill; the
// register tile runs 8-row tiles.
MoePlan::MoePlan(MoeShape shape, uint32_t rows, MoeConfig config,
                 MoePhase phase)
    : shape_(shape), rows_(rows), phase_(phase), config_(config) {
  if (phase == MoePhase::Decode && config.expertTile != MoeExpertTile::M8)
    throw std::invalid_argument("invalid MoE expert tile configuration");
  if (config.ggufTile == MoeGgufTile::Register && config.expertTile != MoeExpertTile::M8)
    throw std::invalid_argument("the register expert tile takes 8-row tiles");
  workspace_ = workspaceFor(shape, rows, tileRows(), config.ggufTile);
  maximumTiles_ = moeMaximumTiles(rows, shape, tileRows());
}

namespace {

// A step's grouping into tiles, which its expert passes read: the gate and up
// passes' column tiles of a staged GGUF plan that runs them in one pass are
// its staged columns.
MoeGroupParams groupParams(const MoePlan &plan) {
  const MoeShape shape = plan.shape();
  const bool oneGateUp = oneGateUpPass(plan);
  return {plan.rows(), shape.expertsPerToken, plan.tileRows(), shape.experts,
          shape.expertIntermediateSize / (oneGateUp ? GGUF_STAGED_COLUMNS : GGUF_TILE_COLUMNS),
          shape.hiddenSize / GGUF_TILE_COLUMNS, SPLASH_TARGET_VERIFY_ROWS,
          shape.rotated() ? shape.expertIntermediateSize / shape.rotationBlock : 0};
}

} // namespace

void MoE::add(metal::CommandGraph &graph, const MoeBuffers &buffers,
              const BlockMoeWeights &weights, const MoePlan &plan) {
  const MoeShape shape = plan.shape();
  bool cpuGrouped = false;
  bool cpuQ4_0 = false;
  bool cpuStreamedQ4_0 = false;
  auto originalQ4_0Sink = std::shared_ptr<MoeOriginalQ4_0Sink>{};
  std::vector<cpu_prototype::Bf16Expert> bf16Experts;
  std::vector<cpu_prototype::Q4_0Expert> q4Experts;
  // Streamed CPU Q4_0 is selected before graph construction, so unsupported
  // layouts always take the complete Metal route.
  if (plan.configuration().cpuBf16Experts && weights.stream &&
      weights.stream->supportsCpuQ4_0(weights.streamBlock) && !shape.rotated() &&
      shape.sharedSlices == 1 && !buffers.verifyRows && plan.configuration().ggufTile == MoeGgufTile::Staged) {
    try {
      validate(weights, shape);
      const std::array<const QuantizedSegment *, 3> shared{
          &weights.gate.shared, &weights.up.shared, &weights.down.shared};
      for (const QuantizedSegment *segment : shared)
        if (segment->formatId != GGUF_FMT_Q40 || segment->layout != QuantizedSegment::Layout::Rows ||
            segment->inputSize % 32 || segment->plane0.storage() != metal::BufferStorage::Shared ||
            !segment->plane0.contents())
          throw std::invalid_argument("CPU streamed Q4_0 requires host-visible native shared rows");
      const uint64_t activationBytes = uint64_t{plan.rows()} * shape.hiddenSize * sizeof(uint16_t);
      requireBytes(buffers.input, activationBytes, "MoE input");
      requireBytes(buffers.residual, activationBytes, "MoE residual");
      requireBytes(buffers.output, activationBytes, "MoE output");
      for (const MoeScratchField &field : kMoeScratchFields) {
        const metal::MetalBuffer &buffer = buffers.scratch.*field.buffer;
        requireBytes(buffer, plan.workspace().*field.bytes, field.name);
        if (buffer.storage() != metal::BufferStorage::Shared || !buffer.contents())
          throw std::invalid_argument("CPU streamed Q4_0 requires host-visible scratch");
      }
      cpuStreamedQ4_0 = true;
      originalQ4_0Sink = std::make_shared<MoeOriginalQ4_0Sink>();
    } catch (const std::exception &) {
      cpuStreamedQ4_0 = false;
      originalQ4_0Sink.reset();
    }
  }
  // Decide before emitting any graph work. A failure here leaves the graph
  // untouched, so the established Metal route remains a complete fallback.
  if (!cpuStreamedQ4_0 && plan.configuration().cpuBf16Experts && !weights.stream && !shape.rotated() &&
      shape.sharedSlices == 1 && !buffers.verifyRows &&
      plan.configuration().ggufTile == MoeGgufTile::Staged) {
    try {
      validate(weights, shape);
      const uint64_t hiddenRows = uint64_t{plan.rows()} * shape.hiddenSize * sizeof(uint16_t);
      requireBytes(buffers.input, hiddenRows, "MoE input");
      requireBytes(buffers.residual, hiddenRows, "MoE residual");
      requireBytes(buffers.output, hiddenRows, "MoE output");
      const MoeWorkspace &required = plan.workspace();
      for (const MoeScratchField &field : kMoeScratchFields)
        requireBytes(buffers.scratch.*field.buffer, required.*field.bytes, field.name);
      const MoeScratch &scratch = buffers.scratch;
      for (const metal::MetalBuffer *buffer : {&buffers.input, &buffers.residual, &buffers.output,
                                               &scratch.tileDescriptors, &scratch.tileCount,
                                               &scratch.groupedRoutes, &scratch.routeRows,
                                               &scratch.groupedInput, &scratch.expertOutput,
                                               &scratch.routingWeights})
        if (buffer->storage() != metal::BufferStorage::Shared || !buffer->contents())
          throw std::invalid_argument("CPU MoE requires host-visible Shared buffers");
      if (weights.slotTable.storage() != metal::BufferStorage::Shared || !weights.slotTable.contents())
        throw std::invalid_argument("CPU MoE requires a host-visible Shared slot table");
      const std::span<const uint32_t> slots{
          static_cast<const uint32_t *>(weights.slotTable.contents()), shape.experts};
      const std::array<const QuantizedSegment *, 6> segments{
          &weights.gate.routed, &weights.up.routed, &weights.down.routed,
          &weights.gate.shared, &weights.up.shared, &weights.down.shared};
      const bool allBf16 = std::all_of(segments.begin(), segments.end(), [](const auto *s) {
        return s->isFloat() && s->layout == QuantizedSegment::Layout::Rows &&
               s->formatId == GGUF_FMT_BF16;
      });
      const bool allQ4_0 = std::all_of(segments.begin(), segments.end(), [](const auto *s) {
        return !s->isFloat() && s->layout == QuantizedSegment::Layout::Rows &&
               s->formatId == GGUF_FMT_Q40 && s->inputSize % 32 == 0;
      });
      if (allBf16) {
        bf16Experts = cpu_prototype::makeBf16ExpertViews(
            weights.gate.routed, weights.up.routed, weights.down.routed, slots,
            weights.gate.shared, weights.up.shared, weights.down.shared);
        cpuGrouped = true;
      } else if (allQ4_0) {
        q4Experts = cpu_prototype::makeQ4_0ExpertViews(
            weights.gate.routed, weights.up.routed, weights.down.routed, slots,
            weights.gate.shared, weights.up.shared, weights.down.shared);
        cpuGrouped = cpuQ4_0 = true;
      }
    } catch (const std::exception &) {
      bf16Experts.clear();
      q4Experts.clear();
    }
  }
  addRouting(graph, buffers, weights, plan);
  if (weights.stream) {
    const MoeLoadPart part{&buffers, &plan, cpuStreamedQ4_0 ? originalQ4_0Sink : nullptr};
    weights.stream->addLoad(graph, {&part, 1}, weights.streamBlock);
  }
  if (cpuStreamedQ4_0) {
    const MoeScratch &scratch = buffers.scratch;
    const uint32_t rows = plan.rows(), tileRows = plan.tileRows(), tiles = plan.maximumTiles();
    graph.add("moe_gather_rows",
              {buffers.input, scratch.groupedRoutes, scratch.tileDescriptors, scratch.tileCount, scratch.groupedInput},
              MoeGatherParams{tileRows, shape.hiddenSize, shape.routesPerToken()},
              {tiles, shape.hiddenSize / 256, 1});
    graph.host([sink = std::move(originalQ4_0Sink), weightsKeepalive = weights, scratch,
                residual = buffers.residual, output = buffers.output, rows,
                routes = shape.routesPerToken(), tileRows, hidden = shape.hiddenSize,
                maximum = plan.maximumTiles(), experts = shape.experts,
                sharedGate = weights.gate.shared, sharedUp = weights.up.shared,
                sharedDown = weights.down.shared] {
      (void)weightsKeepalive;
      const auto entries = sink->snapshot();
      std::vector<cpu_prototype::Q4_0Expert> cpuExperts(size_t{experts} + 1);
      std::vector<bool> present(experts, false);
      for (const MoeOriginalQ4_0Expert &entry : entries) {
        if (!entry.supported || entry.expert >= experts)
          throw std::runtime_error("unsupported streamed Q4_0 expert reached CPU execution");
        const std::array<uint32_t, 3> outputs{sharedGate.outputSize, sharedUp.outputSize, sharedDown.outputSize};
        const std::array<uint32_t, 3> inputs{sharedGate.inputSize, sharedUp.inputSize, sharedDown.inputSize};
        std::array<cpu_prototype::Q4_0Matrix, 3> matrices;
        for (size_t z = 0; z < 3; ++z) {
          const uint64_t expected = uint64_t{outputs[z]} * ((inputs[z] + 31) / 32) * 18;
          if (entry.projections[z].size() != expected)
            throw std::runtime_error("streamed Q4_0 expert bytes have an invalid extent");
          matrices[z] = {std::span<const std::byte>{
                             reinterpret_cast<const std::byte *>(entry.projections[z].data()),
                             entry.projections[z].size()}, outputs[z], inputs[z]};
        }
        cpuExperts[entry.expert] = {matrices[0], matrices[1], matrices[2]};
        present[entry.expert] = true;
      }
      const auto sharedMatrix = [](const QuantizedSegment &segment) {
        return cpu_prototype::Q4_0Matrix::fromRows(segment);
      };
      cpuExperts[experts] = {sharedMatrix(sharedGate), sharedMatrix(sharedUp), sharedMatrix(sharedDown)};
      const auto *count = static_cast<const MoeTileCount *>(scratch.tileCount.contents());
      if (!count || count->tiles > maximum)
        throw std::runtime_error("invalid CPU streamed MoE tile count");
      const size_t groupedRows = size_t{count->tiles} * tileRows;
      const size_t routeCount = size_t{rows} * routes;
      const auto *descriptors = static_cast<const MoeTileDescriptor *>(scratch.tileDescriptors.contents());
      const auto *groupedRoutes = static_cast<const uint32_t *>(scratch.groupedRoutes.contents());
      const auto *routeRows = static_cast<const uint32_t *>(scratch.routeRows.contents());
      const auto *groupedInput = static_cast<const uint16_t *>(scratch.groupedInput.contents());
      const auto *routingWeights = static_cast<const float *>(scratch.routingWeights.contents());
      auto *expertOutput = static_cast<uint16_t *>(scratch.expertOutput.contents());
      if (!descriptors || !groupedRoutes || !routeRows || !groupedInput || !routingWeights || !expertOutput ||
          !residual.contents() || !output.contents())
        throw std::runtime_error("CPU streamed MoE buffers are not host visible");
      for (uint32_t t = 0; t < count->tiles; ++t) {
        const uint32_t expert = descriptors[t].expert;
        if (expert < experts && !present[expert])
          throw std::runtime_error("CPU streamed Q4_0 sink missed a routed expert");
      }
      const size_t inputCount = groupedRows * hidden;
      const auto values = cpu_prototype::runQ4_0GroupedExperts(
          cpuExperts, {descriptors, count->tiles}, {groupedRoutes, groupedRows}, {routeRows, routeCount},
          {groupedInput, inputCount}, rows, routes, tileRows, hidden);
      for (size_t i = 0; i < values.size(); ++i) {
        const uint32_t bits = std::bit_cast<uint32_t>(values[i]);
        expertOutput[i] = static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
      }
      cpu_prototype::mergeBf16Experts(
          {expertOutput, inputCount}, {routeRows, routeCount}, {routingWeights, routeCount},
          {static_cast<const uint16_t *>(residual.contents()), size_t{rows} * hidden},
          {static_cast<uint16_t *>(output.contents()), size_t{rows} * hidden}, rows, routes, hidden);
    });
    return;
  }
  if (!cpuGrouped) {
    addRouted(graph, buffers, weights, plan);
    return;
  }

  const MoeScratch &scratch = buffers.scratch;
  const uint32_t rows = plan.rows(), tileRows = plan.tileRows(), tiles = plan.maximumTiles();
  const MoeGatherParams gather{tileRows, shape.hiddenSize, shape.routesPerToken()};
  graph.add("moe_gather_rows",
            {buffers.input, scratch.groupedRoutes, scratch.tileDescriptors,
             scratch.tileCount, scratch.groupedInput},
            gather, {tiles, shape.hiddenSize / 256, 1});
  graph.host([bf16Experts = std::move(bf16Experts), q4Experts = std::move(q4Experts),
              cpuQ4_0, weightKeepalive = weights, scratch, residual = buffers.residual,
              output = buffers.output, rows, routes = shape.routesPerToken(), tileRows,
              hidden = shape.hiddenSize, maximum = plan.maximumTiles()] {
    (void)weightKeepalive;
    const auto *count = static_cast<const MoeTileCount *>(scratch.tileCount.contents());
    if (!count || count->tiles > maximum)
      throw std::runtime_error("invalid CPU MoE tile count");
    const size_t groupedRows = size_t{count->tiles} * tileRows;
    const size_t routeCount = size_t{rows} * routes;
    const size_t inputCount = groupedRows * hidden;
    const auto *descriptors = static_cast<const MoeTileDescriptor *>(scratch.tileDescriptors.contents());
    const auto *groupedRoutes = static_cast<const uint32_t *>(scratch.groupedRoutes.contents());
    const auto *routeRows = static_cast<const uint32_t *>(scratch.routeRows.contents());
    const auto *groupedInput = static_cast<const uint16_t *>(scratch.groupedInput.contents());
    const auto *weights = static_cast<const float *>(scratch.routingWeights.contents());
    auto *expertOutput = static_cast<uint16_t *>(scratch.expertOutput.contents());
    if (!descriptors || !groupedRoutes || !routeRows || !groupedInput || !weights || !expertOutput ||
        !residual.contents() || !output.contents())
      throw std::runtime_error("CPU MoE buffers are not host visible");
    const auto values = cpuQ4_0
        ? cpu_prototype::runQ4_0GroupedExperts(
              q4Experts, {descriptors, count->tiles}, {groupedRoutes, groupedRows},
              {routeRows, routeCount}, {groupedInput, inputCount}, rows, routes, tileRows, hidden)
        : cpu_prototype::runBf16GroupedExperts(
              bf16Experts, {descriptors, count->tiles}, {groupedRoutes, groupedRows},
              {routeRows, routeCount}, {groupedInput, inputCount}, rows, routes, tileRows, hidden);
    for (size_t i = 0; i < values.size(); ++i) {
      const uint32_t bits = std::bit_cast<uint32_t>(values[i]);
      expertOutput[i] = static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
    }
    cpu_prototype::mergeBf16Experts(
        {expertOutput, inputCount}, {routeRows, routeCount}, {weights, routeCount},
        {static_cast<const uint16_t *>(residual.contents()), size_t{rows} * hidden},
        {static_cast<uint16_t *>(output.contents()), size_t{rows} * hidden}, rows, routes, hidden);
  });
}

void MoE::addRouting(metal::CommandGraph &graph, const MoeBuffers &buffers,
                     const BlockMoeWeights &weights, const MoePlan &plan) {
  const MoeShape shape = plan.shape();
  const uint32_t rows = plan.rows();
  validate(weights, shape);
  const MoeWorkspace &required = plan.workspace();
  const uint64_t hiddenRows = uint64_t{rows} * shape.hiddenSize * sizeof(uint16_t);
  requireBytes(buffers.input, hiddenRows, "MoE input");
  requireBytes(buffers.residual, hiddenRows, "MoE residual");
  requireBytes(buffers.output, hiddenRows, "MoE output");
  const MoeScratch &scratch = buffers.scratch;
  for (const MoeScratchField &field : kMoeScratchFields)
    requireBytes(scratch.*field.buffer, required.*field.bytes, field.name);
  addRoutes(graph, buffers.input, weights.router, weights.sharedScalarGate, weights.selectionBias, scratch.groupedInput,
            scratch.selectedExperts, scratch.routingWeights, rows, shape, plan.configuration().ggufRouterTile);
  // The expert passes launch the live tiles alone, on grids the grouping
  // writes with the tile count. Over every tile a step could fill, a 35B
  // decode step of one lane launched 65 tiles' threadgroups for a median of
  // 30 live ones; the rest return at once yet cost a 40-core M5 Max 0.8-4.1%
  // of a decode cycle, a 12-core M6 0.1-1.0% and a 40-core M3 Max up to 4.8%.
  std::vector<metal::MetalBuffer> grouping{scratch.selectedExperts, scratch.tileDescriptors, scratch.tileCount,
                                           scratch.groupedRoutes, scratch.routeRows};
  if (buffers.verifyRows) {
    // A verify step's rows are its lanes' blocks.
    if (plan.phase() != MoePhase::Decode)
      throw std::invalid_argument("only a verify step's MoE routes its verify rows");
    requireBytes(buffers.verifyRows, uint64_t{rows / SPLASH_TARGET_VERIFY_ROWS} * sizeof(uint32_t),
                 "MoE verify rows");
    grouping.push_back(buffers.verifyRows);
  }
  graph.add(routingKernel(buffers.verifyRows ? "moe_group_routes_verified" : "moe_group_routes", shape,
                          &CompiledMoeRouting::slicesSuffix),
            std::move(grouping), groupParams(plan), {1, 1, 1}, {shape.routerWidth(), 1, 1});
}

void MoE::addRouted(metal::CommandGraph &graph, const MoeBuffers &buffers,
                    const BlockMoeWeights &weights, const MoePlan &plan,
                    const MoeCombineConvolution *convolution) {
  addRouted(graph, buffers, weights, plan);
  if (convolution) {
    auto convBuffers = convolution->buffers;
    convBuffers.input = buffers.output;
    ShortConvolution::addConvolution(graph, convBuffers, convolution->shape, ShortConvOutput::RoundedSum,
                                     convolution->stateIn, {}, convolution->state, convolution->layout,
                                     convolution->rows, {}, convolution->tree, convolution->tape);
  }
}

void MoE::addRouted(metal::CommandGraph &graph, const MoeBuffers &buffers,
                    const BlockMoeWeights &weights, const MoePlan &plan) {
  const MoeShape shape = plan.shape();
  const uint32_t rows = plan.rows();
  const uint32_t tileRows = plan.tileRows();
  const uint32_t tiles = plan.maximumTiles();
  const MoeScratch &scratch = buffers.scratch;
  const MoeGatherParams gather{tileRows, shape.hiddenSize, shape.routesPerToken()};
  if (shape.rotated()) {
    // H (D x) of the block's input once, whose rows the gather of every routed
    // route reads (BlockMoeWeights::rotation).
    const uint32_t block = shape.rotationBlock;
    graph.add("gguf_rotate", {buffers.input, weights.rotation.signs, scratch.rotatedInput},
              GgufRotationParams{shape.hiddenSize, block, rotationScale(block), 0},
              {shape.hiddenSize / block, rows, 1}, {block / 4, 1, 1});
  }
  if (plan.configuration().ggufTile == MoeGgufTile::Register) {
    if (shape.rotated())
      graph.add("moe_gather_table16_rotated",
                {buffers.input, scratch.rotatedInput, scratch.groupedRoutes, scratch.tileCount,
                 scratch.groupedInput, scratch.groupedSums},
                gather, {tiles, shape.hiddenSize / 256, 1});
    else
      graph.add("moe_gather_table16",
                {buffers.input, scratch.groupedRoutes, scratch.tileCount,
                 scratch.groupedInput, scratch.groupedSums},
                gather, {tiles, shape.hiddenSize / 256, 1});
  } else {
    if (shape.rotated())
      graph.add("moe_gather_rows_rotated",
                {buffers.input, scratch.rotatedInput, scratch.groupedRoutes, scratch.tileDescriptors,
                 scratch.tileCount, scratch.groupedInput},
                gather, {tiles, shape.hiddenSize / 256, 1});
    else
      graph.add("moe_gather_rows",
                {buffers.input, scratch.groupedRoutes, scratch.tileDescriptors,
                 scratch.tileCount, scratch.groupedInput},
                gather, {tiles, shape.hiddenSize / 256, 1});
  }
  addExperts(graph, scratch, weights, plan, groupParams(plan), shape.rotated());
  graph.add("moe_combine",
            {scratch.expertOutput, scratch.routeRows, scratch.routingWeights,
             buffers.residual, buffers.output},
            MoeCombineParams{rows, shape.hiddenSize, shape.routesPerToken()},
            {rows, shape.hiddenSize / 256, 1});
}

void MoE::addRoutes(metal::CommandGraph &graph, const metal::MetalBuffer &input, const QuantizedSegment &router,
                    const QuantizedSegment &sharedScalarGate, const metal::MetalBuffer &selectionBias,
                    const metal::MetalBuffer &scores, const metal::MetalBuffer &selected,
                    const metal::MetalBuffer &routingWeights, uint32_t rows, MoeShape shape, FloatTile routerTile) {
  // The router's fp32 scores, in the format its checkpoint holds it
  // (BlockMoeWeights), in rows of routerWidth() expert slots, as the select
  // kernel of that width reads, which weights gated shared slices by the
  // sigmoid of the input's product with the scalar gate.
  const uint32_t width = shape.routerWidth();
  addGgufFloat(graph, input, router, scores, rows, width, 0, FloatOutput::Float32, routerTile);
  const MoeRouteParams route{rows, shape.hiddenSize, shape.experts, shape.expertsPerToken};
  const std::string select = routingKernel("moe_route_select_f32", shape, &CompiledMoeRouting::selectSuffix);
  if (shape.selectionBias) {
    graph.add(select, {scores, selectionBias, selected, routingWeights},
              MoeBiasedRouteParams{route, shape.routedScale}, {rows, 1, 1}, {width, 1, 1});
    return;
  }
  std::vector<metal::MetalBuffer> bindings{scores};
  if (shape.sharedGate) bindings.insert(bindings.end(), {input, sharedScalarGate.plane0});
  bindings.insert(bindings.end(), {selected, routingWeights});
  graph.add(select, std::move(bindings), route, {rows, 1, 1}, {width, 1, 1});
}

uint32_t MoE::expertLoadGroups(const MoeExpertLoadParams &params) noexcept {
  uint64_t groups = 0;
  for (const MoeExpertLoadProjection &projection : params.projections)
    groups = std::max(groups, (uint64_t{projection.rows} * (projection.columns / 32) + MOE_EXPERT_LOAD_THREADS - 1) /
                                  MOE_EXPERT_LOAD_THREADS);
  return static_cast<uint32_t>(groups);
}

void MoE::addExpertLoad(metal::CommandGraph &graph, const metal::MetalBuffer &staging,
                        const metal::MetalBuffer &arena, const metal::MetalBuffer &slots,
                        const metal::MetalBuffer &sources, const MoeExpertLoadParams &params, uint32_t jobs,
                        const metal::IndirectGrid &grid) {
  graph.add("moe_expert_load", {staging, arena, slots, sources}, params, {expertLoadGroups(params), jobs, 3}, grid,
            {MOE_EXPERT_LOAD_THREADS, 1, 1});
}

MoePlan MoE::prefillPlan(MoeShape shape, uint32_t rows, MoeConfig config) {
  if (!rows || rows > SPLASH_PREFILL_TOKEN_BUDGET)
    throw std::invalid_argument("invalid MoE prefill rows");
  return MoePlan(shape, rows, config, MoePhase::Prefill);
}

MoePlan MoE::decodePlan(MoeShape shape, uint32_t lanes, MoeConfig config) {
  if (!lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid MoE decode batch width");
  return MoePlan(shape, lanes * SPLASH_TARGET_VERIFY_ROWS, config, MoePhase::Decode);
}

} // namespace splash::ops
