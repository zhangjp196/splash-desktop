#pragma once

#include "ane/Handoff.hpp"
#include "ane/Program.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "ops/Linear.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace splash::ops {

// One projection of a prefill chunk split by row between the GPU and the
// Neural Engine: the GPU runs the chunk's leading rows with the projection's
// prefill kernel, the ANE its trailing ones as one W8A8 program, each the
// projection's whole output width. The split is by rows, not by columns,
// because a projection's GPU kernel loses much of its rate at half the
// columns while the ANE loses its at half the rows' width; measured on an M3
// Max, an 8192x5120 projection at 2048 rows takes 7.34 ms for 1024 rows on
// the GPU and 6.89 ms on the ANE, so half each overlaps to about 7.3 ms
// against 14.7 ms on the GPU alone. A column split of the same projection
// measured 1.0-1.1x, which is why this split is by row.
//
// The program, memory and weight staging are the FFN split's (ops/AneFfn.cpp)
// carried over: the block Hadamard rotation, int8 weights staged one layer
// ahead and the shared-event handoff. A projection's MIL stops at the
// matmul, its weight row scales and the tokens' scales; there is no silu, no
// intermediate rotation and no down projection. A projection of fused
// segments is not taken yet: `unsupported` says so.
//
// Any failure of the ANE's work stops the split for the life of the process:
// finish() then reports the command's results unusable and the caller runs the
// chunk again on the GPU alone.
class AneProjection final {
public:
  // The same function set as the FFN split's: one every kProgramStep rows from
  // kMinimumRows to the prefill budget.
  static constexpr uint32_t kMinimumRows = 128, kProgramStep = 128, kMaximumRows = SPLASH_PREFILL_TOKEN_BUDGET;
  static_assert(kMaximumRows >= kMinimumRows && (kMaximumRows - kMinimumRows) % kProgramStep == 0,
                "the functions cover the chunks the split takes");

  // Why the split does not take `layers`, or null when it does: one unrotated
  // quantized projection of one segment and one shape, on a hidden size the
  // rotate and the program take.
  [[nodiscard]] static const char *unsupported(std::span<const Projection *const> layers);
  // The ANE's share of `rows` rows: the share's rows, rounded up to the
  // program's step and clamped to leave the GPU its least rows too.
  [[nodiscard]] static uint32_t aneRowsOf(uint32_t rows, double share) noexcept;
  // Whether a chunk of `rows` rows can be split at all: two least parts.
  [[nodiscard]] static bool splittable(uint32_t rows) noexcept { return rows >= 2 * kMinimumRows; }
  // The Metal memory of the split of `layers` at `share`.
  [[nodiscard]] static uint64_t plannedBytes(std::span<const Projection *const> layers, double share);

  // The split of `layers`, every layer of one shape, with the ANE taking
  // `share` of each chunk's rows. The program is compiled and loaded here, and
  // the wait for the ANE's service ends with ane::Interrupted once
  // `interrupted` returns true.
  AneProjection(metal::MetalBackend &backend, std::span<const Projection *const> layers, double share,
                std::function<bool()> interrupted);
  AneProjection(const AneProjection &) = delete;
  AneProjection &operator=(const AneProjection &) = delete;

  // The rows of a chunk of `rows` rows the ANE takes, and the GPU.
  [[nodiscard]] uint32_t aneRows(uint32_t rows) const noexcept { return aneRowsOf(rows, share_); }
  [[nodiscard]] uint32_t gpuRows(uint32_t rows) const noexcept { return rows - aneRows(rows); }
  [[nodiscard]] uint32_t outputSize() const noexcept { return shape_.output; }
  // The columns the ANE computes (the projection's leading quantized
  // segments) and those it leaves to the caller (a float tail, padding).
  [[nodiscard]] uint32_t columns() const noexcept { return shape_.columns; }
  [[nodiscard]] uint32_t tailColumns() const noexcept { return shape_.output - shape_.columns; }
  // A scratch of kMaximumRows rows of the tail's columns, which a caller
  // computes the tail into and copies after the ANE's rows.
  [[nodiscard]] metal::MetalBuffer tail() const noexcept { return memory_.tail; }
  [[nodiscard]] double share() const noexcept { return share_; }
  // What Metal allocated for the split, which plannedBytes() bounds.
  [[nodiscard]] uint64_t allocatedBytes() const noexcept { return allocatedBytes_; }

  // The least rows of a chunk the split takes, from 2 * kMinimumRows to
  // kMaximumRows.
  [[nodiscard]] uint32_t minimumRows() const noexcept { return minimumRows_; }
  void setMinimumRows(uint32_t rows);
  // Whether the split takes a chunk of `rows` rows.
  [[nodiscard]] bool splits(uint32_t rows) const;
  // Where the split stopped, and why.
  [[nodiscard]] bool retired() const { return handoff_.retired(); }
  [[nodiscard]] std::string reason() const { return handoff_.reason(); }

  // Starts encoding a command; false once the split stopped.
  bool begin();
  // Layer `layer`'s ANE part of a chunk of `rows` rows: rotates and packs its
  // trailing `aneRows(rows)` rows of `input` (bf16 rows of hiddenSize values),
  // stages the weights of layer `layer` and of the next one, and signals the
  // ANE to evaluate. The caller then encodes the GPU's leading rows into
  // `output` and calls join().
  void beginLayer(metal::CommandGraph &graph, uint32_t layer, metal::MetalBuffer input, uint32_t rows);
  // Layer `layer`'s ANE part of a chunk of `rows` rows: evaluates its trailing
  // rows through `gpuPart` (the GPU's leading `gpuRows(rows)` rows into the
  // same `output`) and writes the ANE's rows into `output` (bf16 rows of
  // outputSize values) after the GPU part.
  void add(metal::CommandGraph &graph, uint32_t layer, metal::MetalBuffer input, metal::MetalBuffer output,
           uint32_t rows, const std::function<void()> &gpuPart, metal::MetalBuffer residual = {});
  // Waits for the evaluation beginLayer() signaled and writes its rows into
  // `output`.
  void join(metal::CommandGraph &graph, metal::MetalBuffer output, uint32_t rows, metal::MetalBuffer residual = {});
  // Queues the ANE's evaluations of the command encoded since begin(), for a
  // command another split's commit submits (the FFN split's, when both run).
  void prepare();
  // Queues the evaluations and submits the command
  // (MetalBackend::submitCommandAsync).
  [[nodiscard]] metal::CommandTicket commit(const metal::CommandGraph &graph, metal::CommandCompletion completion);
  // After the command committed last has completed: whether its outputs hold
  // the split's rows. False when its ANE work failed, which stops the split and
  // logs why, once.
  [[nodiscard]] bool finish();

  // The commands the split served, their evaluations, and the Neural Engine's
  // milliseconds over them (ane::Handoff::finish).
  struct Served final {
    uint64_t commands = 0, evaluations = 0;
    double milliseconds = 0.0;
  };
  [[nodiscard]] const Served &served() const noexcept { return served_; }

private:
  // The split's width: hidden, the output stride, the columns the ANE takes
  // (the leading quantized segments), the layers, the segment of hidden
  // inputs one matmul multiplies, and those segments' sizes.
  struct Shape final {
    uint32_t hidden = 0, output = 0, columns = 0, layers = 0, segment = 0;
    std::vector<uint32_t> segments;
  };
  // A staging set of the ANE's int8 weights: one surface per segment of the
  // hidden inputs, and the output rows' scales.
  struct Weights final {
    std::vector<ane::Surface> rows;
    ane::Surface scale;
  };
  // Everything the split allocates.
  struct Memory final {
    metal::MetalBuffer signs, rowScales, rotated, tail, status;
    std::vector<ane::Surface> inputs;
    ane::Surface tokenScale, partial;
    std::array<Weights, 2> sets;
  };
  // A projection segment's weight planes as the ane_ffn kernels bind them.
  struct Planes final {
    Planes(const QuantizedSegment &segment, uint32_t inputSize);
    std::array<metal::MetalBuffer, 3> buffers;
    uint32_t groups = 0, format = 0;
  };
  // An input of the ANE program's functions.
  struct Input final {
    std::string name;
    std::array<const ane::Surface *, 2> surfaces;
    uint32_t rows = 0, width = 0;
  };
  // The program's functions, by set, and their rows.
  struct Evaluation final {
    uint32_t rows = 0;
    std::vector<ane::Program::Binding> bindings;
  };
  // An evaluation of weight set `set`: it starts once the event reaches
  // `ready`, and raises it to `done`.
  struct Job final {
    uint32_t evaluation, set;
    uint64_t ready, done;
  };

  [[nodiscard]] static Shape shapeOf(std::span<const Projection *const> layers, double share);
  template <class MakeBuffer, class MakeSurface>
  [[nodiscard]] static Memory allocate(const Shape &shape, MakeBuffer &&buffer, MakeSurface &&surface);
  [[nodiscard]] static std::vector<Input> programInputs(const Shape &shape, const Memory &memory);
  [[nodiscard]] static std::string program(const Shape &shape, std::span<const Input> inputs,
                                           const ane::Surface &output, std::span<const uint32_t> rows);
  [[nodiscard]] static std::string function(const Shape &shape, std::span<const Input> inputs,
                                            const ane::Surface &output, uint32_t rows);
  [[nodiscard]] metal::MetalBuffer rowScales(uint32_t layer) const;
  void addWeights(metal::CommandGraph &graph, uint32_t layer, uint32_t set) const;
  void queue(const Job &job);
  void warnStopped();

  metal::MetalBackend &backend_;
  Shape shape_;
  double share_ = 0.5;
  std::vector<std::vector<Planes>> planes_;
  Memory memory_;
  std::unique_ptr<ane::Program> program_;
  std::vector<Evaluation> evaluations_;
  ane::Handoff handoff_;
  std::vector<Job> jobs_;
  Served served_;
  uint32_t minimumRows_ = 2 * kMinimumRows;
  uint32_t nextLayer_ = 0;
  uint32_t committedEvaluations_ = 0;
  // The evaluation beginLayer() signaled, its weight set and rows.
  uint64_t pendingReady_ = 0;
  uint32_t pendingSet_ = 0, pendingAneRows_ = 0, pendingRows_ = 0;
  bool pending_ = false;
  bool prepared_ = false;
  bool unfinished_ = false;
  bool warned_ = false;
  uint64_t allocatedBytes_ = 0;
};

} // namespace splash::ops
