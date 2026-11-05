#include "ops/AneProjection.hpp"

#include "AwakeClock.hpp"
#include "Checked.hpp"
#include "StderrLine.hpp"
#include "ane/Program.hpp"
#include "metal/abi/AneFfn.h"
#include "ops/BufferExtent.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::ops {
namespace {

using Element = ane::Surface::Element;

// The same 2560-channel maximum segment as the FFN split (ops/AneFfn.cpp):
// the largest whole number of rotation blocks at most it that divides the
// hidden size.
constexpr uint32_t kMaximumSegment = 2560;
static_assert(kMaximumSegment % ANE_FFN_INPUT_BLOCK == 0 && kMaximumSegment % ANE_FFN_INTERMEDIATE_BLOCK == 0 &&
                  kMaximumSegment % ANE_FFN_TILE == 0,
              "segments hold whole rotation blocks and packing tiles");
constexpr uint32_t kRotateGroup = ANE_FFN_INPUT_BLOCK * (ANE_FFN_ROTATE_THREADS / 32);
static_assert(ANE_FFN_WEIGHT_ROWS && kMaximumSegment % ANE_FFN_WEIGHT_ROWS == 0,
              "the weight kernels' threadgroups take whole rows");
// How long the ANE may take to compile and load the program (the FFN split's
// bound).
constexpr auto kProgramLimit = std::chrono::seconds(120);

uint32_t segmentFor(uint32_t hidden) {
  for (uint32_t segment = kMaximumSegment; segment >= ANE_FFN_INTERMEDIATE_BLOCK;
       segment -= ANE_FFN_INTERMEDIATE_BLOCK)
    if (hidden % segment == 0) return segment;
  return 0;
}

// The name of the program's function of `rows` rows.
std::string functionName(uint32_t rows) { return "projection" + std::to_string(rows); }
std::string dimensions(uint64_t rows, uint64_t width) {
  return "[1, 1, " + std::to_string(rows) + ", " + std::to_string(width) + "]";
}
std::string tensor(const char *type, uint64_t rows, uint64_t width) {
  return std::string("tensor<") + type + ", " + dimensions(rows, width) + ">";
}
std::string fp16(double value) {
  char text[32];
  std::snprintf(text, sizeof text, "fp16(%a)", value);
  return text;
}

std::vector<uint32_t> functionRows() {
  std::vector<uint32_t> rows;
  for (uint32_t row = AneProjection::kMinimumRows; row <= AneProjection::kMaximumRows;
       row += AneProjection::kProgramStep)
    rows.push_back(row);
  return rows;
}

} // namespace

AneProjection::Planes::Planes(const QuantizedSegment &segment, uint32_t inputSize) {
  requireSegmentPlanes(segment, "ANE projection");
  groups = inputSize / 32;
  format = segment.formatId;
  buffers = {segment.plane0, segment.plane1Slot(), segment.meta};
}

namespace {
// The leading segments of `segments` that are quantized, which the ANE takes.
size_t prefixSegments(std::span<const QuantizedSegment> segments) {
  size_t count = 0;
  for (const QuantizedSegment &segment : segments) {
    if (segment.isFloat()) break;
    ++count;
  }
  return count;
}
} // namespace

const char *AneProjection::unsupported(std::span<const Projection *const> layers) {
  if (layers.empty()) return "ANE projection split has no layers";
  const uint32_t hidden = layers.front()->inputSize, output = layers.front()->outputSize;
  if (!segmentFor(hidden)) return "ANE projection split needs a hidden size of whole rotation blocks";
  if (hidden % kRotateGroup || hidden / kRotateGroup > ANE_FFN_ROTATE_BLOCKS)
    return "ANE projection split needs a hidden size ane_ffn_rotate takes";
  if (!output) return "ANE projection split needs an output width";
  const auto &first = layers.front()->blocks().segments;
  uint32_t covered = 0, columns = 0;
  for (const QuantizedSegment &segment : first) {
    if (segment.columnOffset != covered) return "ANE projection split segments do not tile the projection";
    if (segment.isFloat()) break;
    covered += segment.outputSize;
    columns += segment.outputSize;
  }
  // The ANE takes the leading quantized segments; a float segment and the
  // padding after it (Qwen's GDN input is 16640 wide, 16384 quantized, 96
  // float and 160 of padding) are the caller's tail.
  if (!columns || columns > output) return "ANE projection split needs a quantized prefix";
  for (const Projection *layer : layers) {
    if (layer->inputSize != hidden || layer->outputSize != output)
      return "ANE projection split layers differ in shape";
    if (layer->rotation) return "ANE projection split needs an unrotated projection";
    const auto &segments = layer->blocks().segments;
    if (segments.size() != first.size()) return "ANE projection split layers differ in segments";
    for (size_t s = 0; s < segments.size(); ++s) {
      if (segments[s].outputSize != first[s].outputSize || segments[s].columnOffset != first[s].columnOffset ||
          segments[s].inputSize != hidden)
        return "ANE projection split layers differ in segments";
      if (s < prefixSegments(first) && segments[s].isFloat())
        return "ANE projection split needs quantized weights";
    }
  }
  return nullptr;
}

uint32_t AneProjection::aneRowsOf(uint32_t rows, double share) noexcept {
  const uint32_t most = ((rows - kMinimumRows) / kProgramStep) * kProgramStep;
  uint32_t ane = static_cast<uint32_t>(std::llround(double(rows) * share));
  ane = std::clamp(ane, kMinimumRows, rows - kMinimumRows);
  ane = ((ane + kProgramStep - 1) / kProgramStep) * kProgramStep;
  return std::min(ane, most);
}

AneProjection::Shape AneProjection::shapeOf(std::span<const Projection *const> layers, double share) {
  if (const char *reason = unsupported(layers)) throw std::invalid_argument(reason);
  if (!(share > 0.0 && share < 1.0)) throw std::invalid_argument("ANE projection split needs a share in (0, 1)");
  Shape shape;
  shape.hidden = layers.front()->inputSize;
  shape.output = layers.front()->outputSize;
  shape.layers = static_cast<uint32_t>(layers.size());
  shape.segment = segmentFor(shape.hidden);
  const auto &segments = layers.front()->blocks().segments;
  const size_t prefix = prefixSegments(segments);
  for (size_t s = 0; s < prefix; ++s) {
    shape.segments.push_back(segments[s].outputSize);
    shape.columns += segments[s].outputSize;
  }
  return shape;
}

template <class MakeBuffer, class MakeSurface>
AneProjection::Memory AneProjection::allocate(const Shape &shape, MakeBuffer &&buffer, MakeSurface &&surface) {
  const uint32_t inputSegments = shape.hidden / shape.segment;
  Memory memory;
  memory.signs = buffer(ANE_FFN_INTERMEDIATE_BLOCK * sizeof(float), "ane projection signs");
  memory.rowScales = buffer(uint64_t{shape.layers} * shape.columns * sizeof(_Float16), "ane projection row scales");
  memory.rotated = buffer(uint64_t{kMaximumRows} * shape.hidden * sizeof(_Float16), "ane projection rotated input");
  if (shape.columns < shape.output)
    memory.tail = buffer(uint64_t{kMaximumRows} * (shape.output - shape.columns) * sizeof(_Float16),
                         "ane projection tail");
  memory.status = buffer(sizeof(uint32_t), "ane projection status");
  for (uint32_t k = 0; k < inputSegments; ++k)
    memory.inputs.push_back(surface(shape.segment, kMaximumRows, Element::Int8));
  memory.tokenScale = surface(1, kMaximumRows, Element::Float16);
  memory.partial = surface(shape.columns, kMaximumRows, Element::Float16);
  for (Weights &set : memory.sets) {
    for (uint32_t k = 0; k < inputSegments; ++k)
      set.rows.push_back(surface(shape.columns, shape.segment, Element::Int8));
    set.scale = surface(shape.columns, 1, Element::Float16);
  }
  return memory;
}

uint64_t AneProjection::plannedBytes(std::span<const Projection *const> layers, double share) {
  uint64_t bytes = 0;
  static_cast<void>(allocate(
      shapeOf(layers, share),
      [&](uint64_t size, const char *) {
        bytes += alignUp(size);
        return metal::MetalBuffer{};
      },
      [&](uint32_t rows, uint32_t width, Element element) {
        bytes += ane::Surface::bytes(rows, width, element);
        return ane::Surface{};
      }));
  return bytes;
}

// x<k>: the rotated input rows of segment k, channel-major in int8, and tx
// their per-token scales; w<k>: the projection's output rows over segment k,
// and sw their per-row scales.
std::vector<AneProjection::Input> AneProjection::programInputs(const Shape &shape, const Memory &memory) {
  const auto both = [&](const ane::Surface &surface) { return std::array{&surface, &surface}; };
  std::vector<Input> inputs;
  for (uint32_t k = 0; k < memory.inputs.size(); ++k)
    inputs.push_back({"x" + std::to_string(k), both(memory.inputs[k]), shape.segment, 0});
  inputs.push_back({"tx", both(memory.tokenScale), 1, 0});
  for (uint32_t k = 0; k < memory.inputs.size(); ++k)
    inputs.push_back(
        {"w" + std::to_string(k), {&memory.sets[0].rows[k], &memory.sets[1].rows[k]}, shape.columns, shape.segment});
  inputs.push_back({"sw", {&memory.sets[0].scale, &memory.sets[1].scale}, shape.columns, 1});
  return inputs;
}

std::string AneProjection::function(const Shape &shape, std::span<const Input> inputs, const ane::Surface &output,
                                    uint32_t rows) {
  const uint32_t inputSegments = shape.hidden / shape.segment, columns = shape.columns;
  const std::string unit = fp16(1.0 / ANE_FFN_INT8_UNIT);
  std::string parameters, body;
  const auto line = [&](const std::string &text) { body += "        " + text + ";\n"; };
  const auto f16 = [&](const std::string &name, uint64_t height, uint64_t width, const std::string &expression) {
    line(tensor("fp16", height, width) + " " + name + " = " + expression);
  };
  for (const Input &input : inputs) {
    const uint32_t width = input.width ? input.width : rows;
    parameters +=
        (parameters.empty() ? "" : ", ") + input.surfaces[0]->bufferType(input.rows, width) + " " + input.name;
    line(tensor(ane::Surface::milType(input.surfaces[0]->element), input.rows, width) + " " + input.name +
         "_t = tensor_buffer_to_tensor<ios17>(input = " + input.name + ")");
  }
  const auto matmul = [&](const std::string &name, const std::string &weights, const std::string &values,
                          uint64_t height, uint64_t width) {
    f16(weights + "_d", height, width, "dequantize(input = " + weights + "_t, scale = " + unit + ")");
    line(tensor("fp16", height, rows) + " " + name +
         " = matmul(transpose_x = bool(false), transpose_y = bool(false), x = " + weights + "_d, y = " + values + ")");
  };
  const auto sum = [&](const std::string &prefix, size_t terms, uint64_t height) {
    std::string total = prefix + "0";
    for (size_t term = 1; term < terms; ++term) {
      const std::string next = prefix + "_sum" + std::to_string(term);
      f16(next, height, rows, "add(x = " + total + ", y = " + prefix + std::to_string(term) + ")");
      total = next;
    }
    return total;
  };

  for (uint32_t k = 0; k < inputSegments; ++k)
    f16("x" + std::to_string(k) + "_d", shape.segment, rows,
        "dequantize(input = x" + std::to_string(k) + "_t, scale = " + unit + ")");
  for (uint32_t k = 0; k < inputSegments; ++k)
    matmul("m" + std::to_string(k), "w" + std::to_string(k), "x" + std::to_string(k) + "_d", columns, shape.segment);
  f16("scaled", columns, rows, "mul(x = " + sum("m", inputSegments, columns) + ", y = sw_t)");
  f16("tokens", columns, rows, "mul(x = scaled, y = tx_t)");
  line(output.bufferType(columns, rows) +
       " y = tensor_to_tensor_buffer<ios17>(input = tokens, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), "
       "strides = tensor<int64, [4]>(" +
       output.strides(columns) + "))");
  return "    func " + functionName(rows) + "<ios18>(" + parameters + ") {\n" + body + "    } -> (y);\n";
}

std::string AneProjection::program(const Shape &shape, std::span<const Input> inputs, const ane::Surface &output,
                                   std::span<const uint32_t> rows) {
  std::string text = "program(1.3)\n{\n";
  for (const uint32_t count : rows) text += function(shape, inputs, output, count);
  return text + "}\n";
}

AneProjection::AneProjection(metal::MetalBackend &backend, std::span<const Projection *const> layers, double share,
                             std::function<bool()> interrupted)
    : backend_(backend), shape_(shapeOf(layers, share)), share_(share), handoff_(backend, std::chrono::seconds(2)) {
  memory_ = allocate(
      shape_,
      [&](uint64_t bytes, const char *label) {
        metal::MetalBuffer buffer = backend_.allocateBuffer(bytes, metal::BufferStorage::Shared, label);
        allocatedBytes_ += buffer.allocatedBytes();
        return buffer;
      },
      [&](uint32_t rows, uint32_t width, Element element) {
        ane::Surface surface = ane::Surface::create(backend_, rows, width, element);
        allocatedBytes_ += surface.buffer.allocatedBytes();
        return surface;
      });
  for (const Projection *layer : layers) {
    std::vector<Planes> segments;
    const auto &all = layer->blocks().segments;
    const size_t prefix = prefixSegments(all);
    for (size_t s = 0; s < prefix; ++s) segments.emplace_back(all[s], shape_.hidden);
    planes_.push_back(std::move(segments));
  }
  // The signs of the block Hadamard rotation, as the FFN split's.
  std::mt19937 generator(20260930);
  std::array<float, ANE_FFN_INTERMEDIATE_BLOCK> signs{};
  for (float &sign : signs) sign = (generator() & 1) ? -1.0f : 1.0f;
  std::memcpy(memory_.signs.contents(), signs.data(), sizeof signs);
  *static_cast<uint32_t *>(memory_.status.contents()) = 0;

  // Each output row's int8 scale over the hidden inputs.
  {
    metal::CommandGraph graph;
    for (uint32_t layer = 0; layer < shape_.layers; ++layer) {
      uint32_t offset = 0;
      for (size_t s = 0; s < shape_.segments.size(); ++s) {
        const uint32_t rows = shape_.segments[s];
        const metal::MetalBuffer scales =
            backend_.view(rowScales(layer), uint64_t{offset} * sizeof(_Float16), uint64_t{rows} * sizeof(_Float16));
        graph.add("ane_ffn_row_scale_gguf_inputs",
                  {planes_[layer][s].buffers[0], planes_[layer][s].buffers[1], planes_[layer][s].buffers[2], scales,
                   memory_.signs},
                  AneFfnWeightParams{planes_[layer][s].groups, 0, 0, shape_.hidden, 0, 0, planes_[layer][s].format},
                  {rows / ANE_FFN_WEIGHT_ROWS, 1, 1}, {ANE_FFN_WEIGHT_THREADS, 1, 1});
        offset += rows;
      }
    }
    static_cast<void>(backend_.submitCommandAsync(graph.dispatches()).wait());
  }

  const std::vector<Input> inputs = programInputs(shape_, memory_);
  program_ = std::make_unique<ane::Program>(program(shape_, inputs, memory_.partial, functionRows()),
                                            std::span<const uint8_t>{},
                                            ane::Program::Limits{kProgramLimit, std::move(interrupted)});
  for (const uint32_t rows : functionRows()) {
    Evaluation &evaluation = evaluations_.emplace_back();
    evaluation.rows = rows;
    const uint32_t procedure = program_->procedure(functionName(rows));
    for (uint32_t set = 0; set < 2; ++set) {
      std::vector<ane::Surface> surfaces;
      for (const std::string &name : program_->inputs(procedure)) {
        const auto input = std::ranges::find(inputs, name, &Input::name);
        if (input == inputs.end()) throw std::logic_error("ANE projection program has an unknown input " + name);
        surfaces.push_back(*input->surfaces[set]);
      }
      evaluation.bindings.push_back(program_->bind(procedure, surfaces, memory_.partial));
    }
  }
}

metal::MetalBuffer AneProjection::rowScales(uint32_t layer) const {
  return backend_.view(memory_.rowScales, uint64_t{layer} * shape_.columns * sizeof(_Float16),
                       uint64_t{shape_.columns} * sizeof(_Float16));
}

void AneProjection::setMinimumRows(uint32_t rows) {
  if (rows < 2 * kMinimumRows || rows > kMaximumRows)
    throw std::invalid_argument("ANE projection split has no least chunk of " + std::to_string(rows) + " rows");
  minimumRows_ = rows;
}

bool AneProjection::splits(uint32_t rows) const {
  return rows >= minimumRows_ && rows <= kMaximumRows && !handoff_.retired();
}

// Layer `layer`'s int8 weights and row scales into staging set `set`.
void AneProjection::addWeights(metal::CommandGraph &graph, uint32_t layer, uint32_t set) const {
  const std::vector<Planes> &source = planes_.at(layer);
  const Weights &target = memory_.sets[set];
  const uint32_t scaleStride = target.scale.strideBytes / uint32_t{sizeof(_Float16)};
  const metal::MetalBuffer scales = rowScales(layer);
  uint32_t offset = 0;
  for (size_t s = 0; s < shape_.segments.size(); ++s) {
    const uint32_t rows = shape_.segments[s];
    const metal::MetalBuffer segmentScales =
        backend_.view(scales, uint64_t{offset} * sizeof(_Float16), uint64_t{rows} * sizeof(_Float16));
    for (uint32_t k = 0; k < target.rows.size(); ++k) {
      // The segment's rows at their offsets of the ANE's full-width surfaces.
      metal::MetalBuffer output = target.rows[k].buffer;
      output = backend_.view(output, uint64_t{offset} * target.rows[k].strideBytes,
                             uint64_t{rows} * target.rows[k].strideBytes);
      metal::MetalBuffer scale = target.scale.buffer;
      scale = backend_.view(scale, uint64_t{offset} * target.scale.strideBytes,
                            uint64_t{rows} * target.scale.strideBytes);
      graph.add("ane_ffn_weights_gguf_inputs",
                {source[s].buffers[0], source[s].buffers[1], source[s].buffers[2], segmentScales, output, scale,
                 memory_.signs},
                AneFfnWeightParams{source[s].groups, 0, k * shape_.segment, shape_.segment,
                                   target.rows[k].strideBytes, scaleStride, source[s].format},
                {rows / ANE_FFN_WEIGHT_ROWS, shape_.segment / ANE_FFN_INPUT_BLOCK, 1},
                {ANE_FFN_WEIGHT_THREADS, 1, 1});
    }
    offset += rows;
  }
}

bool AneProjection::begin() {
  if (std::exchange(unfinished_, false)) {
    handoff_.cancel("a command it split did not finish");
    warnStopped();
  }
  jobs_.clear();
  nextLayer_ = 0;
  pending_ = false;
  prepared_ = false;
  return !handoff_.retired();
}

void AneProjection::beginLayer(metal::CommandGraph &graph, uint32_t layer, metal::MetalBuffer input, uint32_t rows) {
  if (rows < minimumRows_ || rows > kMaximumRows)
    throw std::invalid_argument("ANE projection split does not take a chunk of these rows");
  if (pending_) throw std::logic_error("ANE projection split begins a layer before joining the last");
  if (layer != nextLayer_ || layer >= shape_.layers)
    throw std::logic_error("ANE projection split begins layer " + std::to_string(layer) + ", not its next layer " +
                           std::to_string(nextLayer_) + " (begin() starts each command at layer 0)");
  const uint32_t aneRows = this->aneRows(rows), gpuRows = rows - aneRows;
  requireBytes(input, uint64_t{rows} * shape_.hidden * sizeof(uint16_t), "ANE projection input");
  const uint32_t set = layer & 1, tiles = (aneRows + ANE_FFN_TILE - 1) / ANE_FFN_TILE;
  const auto found =
      std::ranges::find_if(evaluations_, [&](const Evaluation &evaluation) { return evaluation.rows >= aneRows; });
  if (found == evaluations_.end()) throw std::invalid_argument("ANE projection has no function of the chunk's rows");
  const metal::MetalBuffer aneInput =
      backend_.view(input, uint64_t{gpuRows} * shape_.hidden * sizeof(uint16_t),
                    uint64_t{aneRows} * shape_.hidden * sizeof(uint16_t));
  if (!layer) addWeights(graph, 0, 0);
  graph.add("ane_ffn_rotate", {aneInput, memory_.signs, memory_.rotated, memory_.tokenScale.buffer},
            AneFfnRotateParams{shape_.hidden}, {aneRows, 1, 1}, {ANE_FFN_ROTATE_THREADS, 1, 1});
  for (uint32_t k = 0; k < memory_.inputs.size(); ++k)
    graph.add("ane_ffn_pack", {memory_.rotated, memory_.inputs[k].buffer},
              AneFfnPackParams{shape_.hidden, k * shape_.segment, memory_.inputs[k].strideBytes},
              {tiles, shape_.segment / ANE_FFN_TILE, 1}, {ANE_FFN_TILE, ANE_FFN_TILE_ROWS, 1});
  pendingReady_ = handoff_.next();
  pendingSet_ = set;
  pendingAneRows_ = aneRows;
  pendingRows_ = rows;
  pending_ = true;
  graph.signal(handoff_.event(), pendingReady_);
  if (layer + 1 < shape_.layers) addWeights(graph, layer + 1, set ^ 1);
  const auto index = static_cast<uint32_t>(found - evaluations_.begin());
  jobs_.push_back({index, set, pendingReady_, 0});
}

void AneProjection::add(metal::CommandGraph &graph, uint32_t layer, metal::MetalBuffer input, metal::MetalBuffer output,
                        uint32_t rows, const std::function<void()> &gpuPart, metal::MetalBuffer residual) {
  beginLayer(graph, layer, input, rows);
  gpuPart();
  join(graph, output, rows, residual);
}

void AneProjection::join(metal::CommandGraph &graph, metal::MetalBuffer output, uint32_t rows,
                         metal::MetalBuffer residual) {
  if (!pending_ || pendingRows_ != rows)
    throw std::logic_error("ANE projection split joins without the layer it began");
  requireBytes(output, uint64_t{rows} * shape_.output * sizeof(uint16_t), "ANE projection output");
  const uint32_t tiles = (pendingAneRows_ + ANE_FFN_TILE - 1) / ANE_FFN_TILE;
  const uint64_t done = handoff_.next();
  graph.wait(handoff_.event(), done);
  const AneProjectionParams params{rows - pendingAneRows_, 0, shape_.output,
                                   memory_.partial.strideBytes / uint32_t{sizeof(_Float16)}, pendingAneRows_,
                                   shape_.columns};
  if (residual) {
    requireBytes(residual, uint64_t{rows} * shape_.output * sizeof(uint16_t), "ANE projection residual");
    graph.add("ane_projection_out_residual", {output, memory_.partial.buffer, residual, memory_.status}, params,
              {tiles, shape_.columns / ANE_FFN_TILE, 1}, {ANE_FFN_TILE, ANE_FFN_TILE_ROWS, 1});
  } else {
    graph.add("ane_projection_out", {output, memory_.partial.buffer, memory_.status}, params,
              {tiles, shape_.columns / ANE_FFN_TILE, 1}, {ANE_FFN_TILE, ANE_FFN_TILE_ROWS, 1});
  }
  jobs_.back().done = done;
  pending_ = false;
  ++nextLayer_;
}

void AneProjection::prepare() {
  if (std::exchange(prepared_, true)) return;
  if (pending_) throw std::logic_error("ANE projection split commits a layer it did not join");
  const std::vector<Job> jobs = std::exchange(jobs_, {});
  for (const Job &job : jobs) queue(job);
  unfinished_ = !jobs.empty();
  committedEvaluations_ = static_cast<uint32_t>(jobs.size());
}

metal::CommandTicket AneProjection::commit(const metal::CommandGraph &graph, metal::CommandCompletion completion) {
  prepare();
  const auto cancel = [&](const std::string &reason) {
    if (!std::exchange(unfinished_, false)) return;
    handoff_.cancel(reason);
    warnStopped();
  };
  try {
    return backend_.submitCommandAsync(graph.command(), std::move(completion));
  } catch (const std::exception &error) {
    cancel(std::string("its Metal command was not submitted: ") + error.what());
    throw;
  } catch (...) {
    cancel("its Metal command was not submitted");
    throw;
  }
}

bool AneProjection::finish() {
  prepared_ = false;
  unfinished_ = false;
  std::optional<AwakeClock::duration> ran = handoff_.finish();
  if (std::exchange(*static_cast<uint32_t *>(memory_.status.contents()), 0u) && ran) {
    handoff_.retire("the Neural Engine's output was not finite");
    ran.reset();
  }
  if (!ran) {
    warnStopped();
    return false;
  }
  ++served_.commands;
  served_.evaluations += committedEvaluations_;
  served_.milliseconds += std::chrono::duration<double, std::milli>(*ran).count();
  return true;
}

void AneProjection::queue(const Job &job) {
  const ane::Program::Binding &binding = evaluations_.at(job.evaluation).bindings.at(job.set);
  handoff_.queue(job.ready, job.done,
                 [&](const metal::SharedEvent &event, uint64_t wait, uint64_t signal, ane::Handoff::Report report) {
                   program_->enqueue(binding, event, wait, signal, std::move(report));
                 });
}

void AneProjection::warnStopped() {
  if (std::exchange(warned_, true)) return;
  logWarning("Neural Engine projection split stopped (", handoff_.reason(),
             "); the GPU runs the prefill projection alone until the engine restarts.");
}

} // namespace splash::ops
