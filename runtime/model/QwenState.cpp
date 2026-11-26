#include "model/QwenState.hpp"
#include "model/GdnStateCodec.hpp"
#include "metal/abi/GdnStateCodec.h"

#include <cstring>
#include <bit>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::model {
namespace {

using metal::MetalBuffer;

void *writableContents(const MetalBuffer &buffer, const char *name) {
  void *contents = buffer.contents();
  if (!contents) {
    throw std::logic_error(std::string(name) + " is not CPU-visible");
  }
  return contents;
}

// The buffer of a cell of no bytes is empty (QwenGdnCell): nothing to copy
// or clear.
void copyExact(const MetalBuffer &destination, const MetalBuffer &source,
               const char *name) {
  if (destination.sizeBytes() != source.sizeBytes()) {
    throw std::logic_error(std::string(name) + " shape mismatch");
  }
  if (destination)
    std::memcpy(writableContents(destination, name),
                writableContents(source, name), destination.sizeBytes());
}

void clear(const MetalBuffer &buffer, const char *name) {
  if (buffer)
    std::memset(writableContents(buffer, name), 0, buffer.sizeBytes());
}

uint16_t floatToBFloat16(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu))
    return static_cast<uint16_t>((bits >> 16) | 0x0040u);
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

float bfloat16ToFloat(uint16_t value) {
  return std::bit_cast<float>(uint32_t{value} << 16);
}

float loadRecurrent(const std::byte *base, uint64_t index, ops::GdnStateFormat format) {
  if (format == ops::GdnStateFormat::Float32) {
    float value;
    std::memcpy(&value, base + index * sizeof(float), sizeof(value));
    return value;
  }
  uint16_t value;
  std::memcpy(&value, base + index * sizeof(value), sizeof(value));
  return bfloat16ToFloat(value);
}

void storeRecurrent(std::byte *base, uint64_t index, ops::GdnStateFormat format, float value) {
  if (format == ops::GdnStateFormat::Float32) {
    std::memcpy(base + index * sizeof(value), &value, sizeof(value));
    return;
  }
  const uint16_t encoded = floatToBFloat16(value);
  std::memcpy(base + index * sizeof(encoded), &encoded, sizeof(encoded));
}

void encodeCachedGdn(std::span<std::byte> cached, std::span<const std::byte> running,
                     const GdnStateLayout &layout) {
  if (cached.size() != layout.cachedCellBytes() || running.size() != layout.cellBytes())
    throw std::logic_error("cached GDN codec shape mismatch");
  std::fill(cached.begin(), cached.end(), std::byte{0});
  const uint64_t conv = layout.convolutionBytes();
  if (conv) std::memcpy(cached.data(), running.data(), conv);
  const uint64_t rows = uint64_t{layout.recurrentGroups} * layout.recurrentRows;
  const uint64_t columns = layout.recurrentColumns;
  const uint64_t blocks = (columns + 31) / 32;
  const uint64_t cachedStride = layout.cachedRecurrentLayerBytes();
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    const std::byte *source = running.data() + conv + uint64_t{layer} * layout.recurrentLayerBytes();
    std::byte *destination = cached.data() + conv + uint64_t{layer} * cachedStride;
    auto *codes = reinterpret_cast<int8_t *>(destination);
    std::byte *scales = destination + rows * columns;
    for (uint64_t row = 0; row < rows; ++row) {
      for (uint64_t block = 0; block < blocks; ++block) {
        const uint64_t begin = block * 32;
        const uint64_t count = std::min<uint64_t>(32, columns - begin);
        float maximum = 0.0f;
        for (uint64_t i = 0; i < count; ++i) {
          const float value = loadRecurrent(source, row * columns + begin + i, layout.format);
          if (std::isnan(value)) { maximum = value; break; }
          maximum = std::max(maximum, std::abs(value));
        }
        uint16_t encodedScale = floatToBFloat16(maximum == 0.0f ? 1.0f : maximum / 127.0f);
        if (maximum > 0.0f && encodedScale == 0) encodedScale = 1;
        std::memcpy(scales + (row * blocks + block) * sizeof(encodedScale),
                    &encodedScale, sizeof(encodedScale));
        const float scale = bfloat16ToFloat(encodedScale);
        for (uint64_t i = 0; i < count; ++i) {
          const float value = loadRecurrent(source, row * columns + begin + i, layout.format);
          const float normalized = value / scale;
          const long rounded = std::isfinite(normalized) ? std::lrint(normalized) : 0;
          codes[row * columns + begin + i] = static_cast<int8_t>(std::clamp(rounded, -127L, 127L));
        }
      }
    }
  }
  const uint64_t auxOffset = conv + layout.cachedRecurrentBytes();
  if (layout.auxiliaryBytes)
    std::memcpy(cached.data() + auxOffset, running.data() + conv + layout.recurrentBytes(),
                layout.auxiliaryBytes);
}

void decodeCachedGdn(std::span<std::byte> running, std::span<const std::byte> cached,
                     const GdnStateLayout &layout) {
  if (cached.size() != layout.cachedCellBytes() || running.size() != layout.cellBytes())
    throw std::logic_error("cached GDN codec shape mismatch");
  std::fill(running.begin(), running.end(), std::byte{0});
  const uint64_t conv = layout.convolutionBytes();
  if (conv) std::memcpy(running.data(), cached.data(), conv);
  const uint64_t rows = uint64_t{layout.recurrentGroups} * layout.recurrentRows;
  const uint64_t columns = layout.recurrentColumns;
  const uint64_t blocks = (columns + 31) / 32;
  const uint64_t cachedStride = layout.cachedRecurrentLayerBytes();
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    const std::byte *source = cached.data() + conv + uint64_t{layer} * cachedStride;
    const auto *codes = reinterpret_cast<const int8_t *>(source);
    const std::byte *scales = source + rows * columns;
    std::byte *destination = running.data() + conv + uint64_t{layer} * layout.recurrentLayerBytes();
    for (uint64_t row = 0; row < rows; ++row)
      for (uint64_t column = 0; column < columns; ++column) {
        uint16_t scaleBits;
        std::memcpy(&scaleBits, scales + (row * blocks + column / 32) * sizeof(scaleBits),
                    sizeof(scaleBits));
        const float scale = bfloat16ToFloat(scaleBits);
        storeRecurrent(destination, row * columns + column, layout.format,
                       static_cast<float>(codes[row * columns + column]) * scale);
      }
  }
  if (layout.auxiliaryBytes)
    std::memcpy(running.data() + conv + layout.recurrentBytes(),
                cached.data() + conv + layout.cachedRecurrentBytes(), layout.auxiliaryBytes);
}

// Disk reads land in the active runtime cell. Expand from the end toward the
// beginning so writes never destroy encoded codes or scales that remain unread.
void dispatchGdnCacheCodec(metal::MetalBackend &backend, const MetalBuffer &running,
                           const MetalBuffer &cached, const GdnStateLayout &layout,
                           bool quantize) {
  if (!layout.layers || !layout.recurrentGroups || !layout.recurrentRows ||
      !layout.recurrentColumns)
    return;
  if (layout.recurrentColumns % SPLASH_GDN_CACHE_BLOCK)
    throw std::invalid_argument("Metal GDN cache codec requires columns in 32-value blocks");
  const uint32_t rows = layout.recurrentGroups * layout.recurrentRows;
  const uint32_t blocks = rows * (layout.recurrentColumns / SPLASH_GDN_CACHE_BLOCK);
  constexpr uint32_t threads = 128;
  const GdnStateCodecParams params{
      layout.recurrentLayerBytes(), layout.cachedRecurrentLayerBytes(),
      uint64_t{rows} * layout.recurrentColumns, rows, layout.recurrentColumns};
  metal::ComputeDispatch dispatch;
  const bool bfloat = layout.format == ops::GdnStateFormat::BFloat16;
  dispatch.pipelineName = quantize
      ? (bfloat ? "gdn_cache_quantize_bf16_state" : "gdn_cache_quantize")
      : (bfloat ? "gdn_cache_dequantize_bf16_state" : "gdn_cache_dequantize");
  dispatch.buffers = quantize ? std::vector<metal::BufferBinding>{{0, running}, {1, cached}}
                              : std::vector<metal::BufferBinding>{{0, cached}, {1, running}};
  dispatch.bytes = {{2, &params, sizeof(params)}};
  dispatch.threadgroups = {(blocks + threads - 1) / threads, layout.layers, 1};
  dispatch.threadsPerThreadgroup = {threads, 1, 1};
  static_cast<void>(backend.submit(dispatch));
}

void encodeCachedGdnMetal(metal::MetalBackend &backend, const MetalBuffer &cached,
                          const MetalBuffer &running, const GdnStateLayout &layout) {
  if (cached.sizeBytes() != layout.cachedCellBytes() || running.sizeBytes() != layout.cellBytes())
    throw std::logic_error("cached GDN codec shape mismatch");
  if (layout.layers && layout.recurrentColumns % SPLASH_GDN_CACHE_BLOCK) {
    encodeCachedGdn(
        {cached ? static_cast<std::byte *>(writableContents(cached, "cached GDN state")) : nullptr,
         cached.sizeBytes()},
        {running ? static_cast<const std::byte *>(writableContents(running, "running GDN state")) : nullptr,
         running.sizeBytes()}, layout);
    return;
  }
  if (!cached.sizeBytes()) return;
  auto *destination = static_cast<std::byte *>(writableContents(cached, "cached GDN state"));
  const auto *source = static_cast<const std::byte *>(writableContents(running, "running GDN state"));
  std::fill(destination, destination + cached.sizeBytes(), std::byte{0});
  const uint64_t conv = layout.convolutionBytes();
  if (conv) std::memcpy(destination, source, conv);
  if (layout.recurrentBytes())
    dispatchGdnCacheCodec(backend,
        backend.view(running, conv, layout.recurrentBytes()),
        backend.view(cached, conv, layout.cachedRecurrentBytes()), layout, true);
  if (layout.auxiliaryBytes)
    std::memcpy(destination + conv + layout.cachedRecurrentBytes(),
                source + conv + layout.recurrentBytes(), layout.auxiliaryBytes);
}

void decodeCachedGdnMetal(metal::MetalBackend &backend, const MetalBuffer &running,
                          const MetalBuffer &cached, const GdnStateLayout &layout) {
  if (cached.sizeBytes() != layout.cachedCellBytes() || running.sizeBytes() != layout.cellBytes())
    throw std::logic_error("cached GDN codec shape mismatch");
  if (layout.layers && layout.recurrentColumns % SPLASH_GDN_CACHE_BLOCK) {
    decodeCachedGdn(
        {running ? static_cast<std::byte *>(writableContents(running, "restored GDN state")) : nullptr,
         running.sizeBytes()},
        {cached ? static_cast<const std::byte *>(writableContents(cached, "cached GDN state")) : nullptr,
         cached.sizeBytes()}, layout);
    return;
  }
  if (!running.sizeBytes()) return;
  auto *destination = static_cast<std::byte *>(writableContents(running, "restored GDN state"));
  const auto *source = static_cast<const std::byte *>(writableContents(cached, "cached GDN state"));
  std::fill(destination, destination + running.sizeBytes(), std::byte{0});
  const uint64_t conv = layout.convolutionBytes();
  if (conv) std::memcpy(destination, source, conv);
  if (layout.recurrentBytes())
    dispatchGdnCacheCodec(backend,
        backend.view(running, conv, layout.recurrentBytes()),
        backend.view(cached, conv, layout.cachedRecurrentBytes()), layout, false);
  if (layout.auxiliaryBytes)
    std::memcpy(destination + conv + layout.recurrentBytes(),
                source + conv + layout.cachedRecurrentBytes(), layout.auxiliaryBytes);
}

void decodeCachedGdnInPlace(std::span<std::byte> cell, const GdnStateLayout &layout) {
  if (cell.size() != layout.cellBytes())
    throw std::logic_error("in-place cached GDN codec shape mismatch");
  const uint64_t conv = layout.convolutionBytes();
  const uint64_t rows = uint64_t{layout.recurrentGroups} * layout.recurrentRows;
  const uint64_t columns = layout.recurrentColumns;
  const uint64_t blocks = (columns + 31) / 32;
  const uint64_t cachedStride = layout.cachedRecurrentLayerBytes();
  std::vector<std::byte> auxiliary(layout.auxiliaryBytes);
  std::vector<uint16_t> scales(rows * blocks * layout.layers);
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    const std::byte *source = cell.data() + conv + uint64_t{layer} * cachedStride;
    std::memcpy(scales.data() + uint64_t{layer} * rows * blocks,
                source + rows * columns, rows * blocks * sizeof(uint16_t));
  }
  if (layout.auxiliaryBytes)
    std::memcpy(auxiliary.data(), cell.data() + conv + layout.cachedRecurrentBytes(),
                layout.auxiliaryBytes);
  for (uint32_t layer = layout.layers; layer > 0;) {
    --layer;
    const std::byte *source = cell.data() + conv + uint64_t{layer} * cachedStride;
    std::byte *destination = cell.data() + conv + uint64_t{layer} * layout.recurrentLayerBytes();
    for (uint64_t row = rows; row > 0;) {
      --row;
      for (uint64_t block = blocks; block > 0;) {
        --block;
        const uint64_t begin = block * 32;
        const uint64_t count = std::min<uint64_t>(32, columns - begin);
        const uint16_t scaleBits = scales[uint64_t{layer} * rows * blocks + row * blocks + block];
        const float scale = bfloat16ToFloat(scaleBits);
        std::array<int8_t, 32> codes{};
        std::memcpy(codes.data(), source + row * columns + begin, count);
        for (uint64_t i = 0; i < count; ++i)
          storeRecurrent(destination, row * columns + begin + i, layout.format,
                         static_cast<float>(codes[i]) * scale);
      }
    }
    const uint64_t used = rows * columns * ops::elementBytes(layout.format);
    if (layout.recurrentLayerBytes() > used)
      std::fill(destination + used,
                destination + layout.recurrentLayerBytes(), std::byte{0});
  }
  if (layout.auxiliaryBytes)
    std::memcpy(cell.data() + conv + layout.recurrentBytes(), auxiliary.data(),
                layout.auxiliaryBytes);
}

// What a state file slot holds of a state: its GDN cell, unless it has no
// bytes, then its context window.
std::vector<std::span<std::byte>> stateSpans(const GdnParityBuffers &gdn,
                                             const MetalBuffer &window) {
  std::vector<std::span<std::byte>> spans;
  for (const MetalBuffer *buffer : {&gdn.stateBase, &window})
    if (*buffer)
      spans.emplace_back(static_cast<std::byte *>(writableContents(*buffer, "state IO")),
                         buffer->sizeBytes());
  return spans;
}

class FileOffload final : public StateOffload {
public:
  FileOffload(std::shared_ptr<SlotFile::Operation> operation,
              std::shared_ptr<const CompositeState> state,
              std::shared_ptr<StateStaging> staging)
      : operation_(std::move(operation)), state_(std::move(state)),
        staging_(std::move(staging)) {}
  // The staging copy is the write's source until the worker has stopped.
  ~FileOffload() override { operation_->drain(); }
  bool ready() const noexcept override { return operation_->ready(); }
  bool finish() override { return operation_->wait(); }
  const std::shared_ptr<const CompositeState> &state() const noexcept override {
    return state_;
  }
private:
  std::shared_ptr<SlotFile::Operation> operation_;
  std::shared_ptr<const CompositeState> state_;
  std::shared_ptr<StateStaging> staging_;
};

class FileRestore final : public StateRestore {
public:
  FileRestore(std::shared_ptr<SlotFile::Operation> operation,
              std::function<void()> committed,
              std::function<std::shared_ptr<const CompositeState>()> snapshot,
              std::function<void()> decode = {})
      : operation_(std::move(operation)), committed_(std::move(committed)),
        snapshot_(std::move(snapshot)), decode_(std::move(decode)) {}
  ~FileRestore() override { operation_->drain(); }
  bool ready() const noexcept override { return operation_->ready(); }
  void cancel() noexcept override { operation_->cancel(); }
  bool finish() override {
    if (!operation_->wait()) return false;
    if (decode_) decode_();
    committed_();
    finished_ = true;
    return true;
  }
  std::shared_ptr<const CompositeState> snapshot() override {
    return finished_ ? snapshot_() : nullptr;
  }
private:
  std::shared_ptr<SlotFile::Operation> operation_;
  std::function<void()> committed_;
  std::function<std::shared_ptr<const CompositeState>()> snapshot_;
  std::function<void()> decode_;
  bool finished_ = false;
};

class CachedCellLease final {
public:
  CachedCellLease(std::shared_ptr<QwenBufferPool> pool,
                  std::shared_ptr<QwenGdnCell> cell)
      : pool_(std::move(pool)), cell_(std::move(cell)) {}
  ~CachedCellLease() { release(); }
  const std::shared_ptr<QwenGdnCell> &cell() const noexcept { return cell_; }
  void release() noexcept {
    if (!cell_) return;
    if (pool_) pool_->giveCachedCell(std::move(cell_));
    else cell_.reset();
  }
private:
  std::shared_ptr<QwenBufferPool> pool_;
  std::shared_ptr<QwenGdnCell> cell_;
};

} // namespace

void encodeCachedGdnState(std::span<std::byte> cached,
                          std::span<const std::byte> running,
                          const GdnStateLayout &layout) {
  encodeCachedGdn(cached, running, layout);
}

void decodeCachedGdnState(std::span<std::byte> running,
                          std::span<const std::byte> cached,
                          const GdnStateLayout &layout) {
  decodeCachedGdn(running, cached, layout);
}

void decodeCachedGdnStateInPlace(std::span<std::byte> cell,
                                 const GdnStateLayout &layout) {
  decodeCachedGdnInPlace(cell, layout);
}

GdnParityBuffers gdnCellBuffers(metal::MetalBackend &backend, metal::MetalBuffer base,
                                  const GdnStateLayout &layout, bool cached) {
  const uint64_t bytes = cached ? layout.cachedCellBytes() : layout.cellBytes();
  if (base.sizeBytes() < bytes) throw std::invalid_argument("a GDN cell is larger than its buffer");
  const uint64_t recurrentStride = cached ? layout.cachedRecurrentLayerBytes()
                                          : layout.recurrentLayerBytes();
  const uint64_t recurrentBytes = cached ? layout.cachedRecurrentBytes()
                                         : layout.recurrentBytes();
  GdnParityBuffers cell;
  cell.stateBase = base.sizeBytes() == bytes
      ? std::move(base)
      : (bytes ? backend.view(base, 0, bytes) : metal::MetalBuffer{});
  cell.convolutionLayers.resize(layout.layers);
  cell.recurrentLayers.resize(layout.layers);
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    cell.convolutionLayers[layer] = backend.view(cell.stateBase, uint64_t{layer} * layout.convolutionLayerBytes(),
                                                 layout.convolutionLayerBytes());
    cell.recurrentLayers[layer] =
        backend.view(cell.stateBase, layout.convolutionBytes() + uint64_t{layer} * recurrentStride,
                     recurrentStride);
  }
  if (layout.auxiliaryBytes)
    cell.auxiliary = backend.view(cell.stateBase, layout.convolutionBytes() + recurrentBytes,
                                  layout.auxiliaryBytes);
  return cell;
}

QwenGdnCell::QwenGdnCell(metal::MetalBackend &backend,
                         std::shared_ptr<StateAllocationTracker> tracker,
                         GdnStateLayout layout, std::string_view label, bool cached)
    : tracker_(std::move(tracker)) {
  if (!tracker_)
    throw std::invalid_argument("Qwen state allocation tracker is empty");
  if (!layout.valid())
    throw std::invalid_argument("Qwen GDN state layout is invalid");
  // Metal allocates no buffer of no bytes, and such a cell needs none.
  const uint64_t cellBytes = cached ? layout.cachedCellBytes() : layout.cellBytes();
  if (!cellBytes)
    return;
  const uint64_t before = backend.memoryStats().allocatedBytes;
  buffers_ = gdnCellBuffers(backend, backend.allocateBuffer(cellBytes, metal::BufferStorage::Shared, label),
                            layout, cached);
  actualAllocatedBytes_ =
      metal::allocationDelta(before, backend.memoryStats().allocatedBytes);
  if (actualAllocatedBytes_ < cellBytes) {
    throw std::logic_error("Qwen GDN allocation is below declared bytes");
  }
  tracker_->bytes.fetch_add(actualAllocatedBytes_, std::memory_order_relaxed);
}

QwenGdnCell::~QwenGdnCell() {
  tracker_->bytes.fetch_sub(actualAllocatedBytes_, std::memory_order_relaxed);
}

QwenStateReservation::QwenStateReservation(std::shared_ptr<QwenBufferPool> pool,
                                           QwenCachedBuffers buffers,
                                           std::shared_ptr<QwenGdnCell> captureCell)
    : pool_(std::move(pool)), buffers_(std::move(buffers)),
      captureCell_(std::move(captureCell)) {
  if (!pool_ || !buffers_.gdn || !buffers_.window)
    throw std::invalid_argument("state reservation buffers are empty");
}

QwenStateReservation::~QwenStateReservation() {
  if (captureCell_)
    pool_->giveCell(std::move(captureCell_));
  if (buffers_.gdn)
    pool_->give(std::move(buffers_));
}

const QwenCachedBuffers &QwenStateReservation::buffers() const {
  if (!buffers_.gdn)
    throw std::logic_error("a sealed state reservation holds no buffers");
  return buffers_;
}

const GdnParityBuffers &QwenStateReservation::captureBuffers() const {
  if (!buffers_.gdn)
    throw std::logic_error("a sealed state reservation holds no buffers");
  return captureCell_ ? captureCell_->buffers() : buffers_.gdn->buffers();
}

QwenCompositeState::QwenCompositeState(std::shared_ptr<QwenBufferPool> pool,
                                       QwenCachedBuffers buffers,
                                       CompositeStateLayout layout,
                                       QwenLogicalLengths lengths,
                                       std::shared_ptr<SlotFile> file,
                                       std::shared_ptr<StateStaging> staging)
    : pool_(std::move(pool)), buffers_(std::move(buffers)), layout_(layout),
      lengths_(lengths), file_(std::move(file)), staging_(std::move(staging)) {
  if (!pool_ || !buffers_.gdn || !buffers_.window) {
    throw std::invalid_argument("composite state buffers are empty");
  }
}

QwenCompositeState::QwenCompositeState(CompositeStateLayout layout,
    QwenLogicalLengths lengths, std::shared_ptr<SlotFile> file,
    std::shared_ptr<SlotFile::Slot> disk)
    : layout_(layout), lengths_(lengths), file_(std::move(file)), disk_(std::move(disk)) {}

std::unique_ptr<StateOffload>
QwenCompositeState::offload(std::function<void()> completion) const {
  // The caller asked the write budget; the write needs what a copy in use
  // does.
  if (!canOffload(true)) return {};
  return write(file_, staging_,
               stateSpans(buffers_.gdn->buffers(), buffers_.window->buffer()),
               layout_, lengths_, std::move(completion));
}

std::unique_ptr<StateOffload> QwenCompositeState::write(
    const std::shared_ptr<SlotFile> &file, const std::shared_ptr<StateStaging> &staging,
    const std::vector<std::span<std::byte>> &spans, CompositeStateLayout layout,
    QwenLogicalLengths lengths, std::function<void()> completion) {
  auto disk = file->acquire();
  if (!disk) return {};
  auto result = std::shared_ptr<const QwenCompositeState>(
      new QwenCompositeState(layout, lengths, file, disk));
  const std::span<std::byte> staged = staging->bytes();
  std::byte *cursor = staged.data();
  for (auto span : spans) {
    if (span.data() == staged.data()) continue;
    cursor = std::copy(span.begin(), span.end(), cursor);
  }
  std::shared_ptr<SlotFile::Operation> operation;
  try {
    operation = file->write(
        std::move(disk), {std::span<const std::byte>(staged)},
        std::move(completion));
    // Callers check that the file takes writes, and only a failed write of
    // its own closes it: none is in flight beside this one.
    if (!operation)
      throw std::logic_error("the state file closed with no state write in flight");
    return std::make_unique<FileOffload>(operation, std::move(result), staging);
  } catch (...) {
    if (operation)
      operation->drain();
    throw;
  }
}

std::unique_ptr<StateOffload>
QwenCompositeState::persist(std::function<void()> completion) const {
  // As for offload().
  if (!canOffload(true))
    return {};
  auto disk = file_->acquire();
  if (!disk)
    return {};
  auto copy = std::shared_ptr<const CompositeState>(
      new QwenCompositeState(layout_, lengths_, file_, disk));
  const auto spans = stateSpans(buffers_.gdn->buffers(), buffers_.window->buffer());
  auto operation = file_->write(std::move(disk), {spans.begin(), spans.end()},
                                std::move(completion));
  // As for write(): nothing stops the file between the check and the write.
  if (!operation)
    throw std::logic_error("the state file closed with no state write in flight");
  return std::make_unique<FileOffload>(operation, std::move(copy), nullptr);
}

void QwenCompositeState::label(std::vector<std::byte> label) const {
  if (disk_)
    file_->label(disk_, std::move(label));
}

QwenCompositeState::~QwenCompositeState() {
  if (pool_)
    pool_->give(std::move(buffers_));
}

QwenStateStorage::QwenStateStorage(metal::MetalBackend &backend,
                                   metal::AllocationAdmission admitAllocation,
                                   CompositeStateLayout layout,
                                   std::shared_ptr<SlotFile> file)
    : backend_(backend), admitAllocation_(std::move(admitAllocation)),
      layout_(layout),
      allocations_(std::make_shared<StateAllocationTracker>()),
      pool_(std::make_shared<QwenBufferPool>()) {
  if (!admitAllocation_)
    throw std::invalid_argument("Qwen state allocation admission is required");
  if (!layout_.valid())
    throw std::invalid_argument("Qwen composite state layout is invalid");
  if (layout_.target.cachedCellBytes() > layout_.target.cellBytes())
    throw std::invalid_argument("cached GDN state exceeds its running cell");
  if (file) {
    file_ = std::move(file);
    staging_ = std::make_shared<StateStaging>();
    staging_->buffer = backend_.allocateBuffer(
        layout_.cachedBytes(), metal::BufferStorage::Shared, "qwen-state-staging");
  }
  if (!layout_.target.cellBytes())
    emptyCell_ = std::shared_ptr<QwenGdnCell>(
        new QwenGdnCell(backend_, allocations_, layout_.target, "qwen-state-empty-gdn"));
}

QwenStateStorage::~QwenStateStorage() {
  pool_->open = false;
  pool_->cells.clear();
  pool_->cachedCells.clear();
  pool_->rings.clear();
  pool_->windows.clear();
}

const QwenLaneMetadata &QwenStateStorage::metadata(uint32_t index) const {
  return lane(index).metadata;
}

const GdnParityBuffers &QwenStateStorage::current(uint32_t index) const {
  const Lane &assigned = lane(index);
  requireAssigned(assigned);
  return assigned.cells.gdn[assigned.metadata.activeParity]->buffers();
}

const GdnParityBuffers &QwenStateStorage::next(uint32_t index) const {
  const Lane &assigned = lane(index);
  requireAssigned(assigned);
  return assigned.cells.gdn[assigned.metadata.activeParity ^ 1]->buffers();
}

const std::vector<DFlashDraftRingLayer> &
QwenStateStorage::draft(uint32_t index) const {
  const Lane &assigned = lane(index);
  requireAssigned(assigned);
  return assigned.cells.draft->layers();
}

const MetalBuffer &QwenStateStorage::window(uint32_t index) const {
  const Lane &assigned = lane(index);
  requireAssigned(assigned);
  return assigned.cells.window->buffer();
}

metal::AllocationResult
QwenStateStorage::tryActivateLane(uint32_t index, uint64_t requestId, uint64_t extraBytes,
                                  const std::function<void()> &allocateExtra) {
  if (!requestId)
    throw std::invalid_argument("request id must be non-zero");
  Lane &current = lane(index);
  if (current.metadata.assigned()) {
    throw std::logic_error("Qwen lane is already assigned");
  }
  Buffers buffers;
  if (auto admission = acquire(CompositeStateLayout::kLaneGdnCells, true,
                               "qwen-state-lane-" + std::to_string(index),
                               buffers, extraBytes, allocateExtra);
      !admission)
    return admission;
  current.cells = std::move(buffers);
  current.metadata = {requestId, 0, {}, 0};
  return {};
}

void QwenStateStorage::releaseLane(uint32_t index, uint64_t requestId) {
  Lane &current = lane(index);
  requireAssigned(current);
  if (!requestId || current.metadata.requestId != requestId) {
    throw std::logic_error("Qwen lane owner mismatch");
  }
  // Parity one first, so the next activation pops parity zero first and a
  // reactivated lane gets its previous buffers back in the same order.
  for (uint32_t parity = current.cells.gdn.size(); parity > 0;) {
    --parity;
    pool_->giveCell(std::move(current.cells.gdn[parity]));
  }
  pool_->rings.push_back(std::move(current.cells.draft));
  pool_->windows.push_back(std::move(current.cells.window));
  current.metadata = {};
}

void QwenStateStorage::clearForColdStart(uint32_t index) {
  const Lane &target = lane(index);
  requireAssigned(target);
  if (target.metadata.lengths != QwenLogicalLengths{})
    throw std::logic_error("a cold start begins at logical length zero");
  clear(current(index).stateBase, "cold-start GDN state");
}

void QwenStateStorage::clearDraftRings(uint32_t index) {
  const Lane &target = lane(index);
  requireAssigned(target);
  for (const DFlashDraftRingLayer &layer : target.cells.draft->layers()) {
    clear(layer.keys, "draft ring keys");
    clear(layer.values, "draft ring values");
  }
}

uint64_t QwenStateStorage::releaseOneIdle(bool keepLane) noexcept {
  const uint64_t before = backend_.memoryStats().allocatedBytes;
  const uint32_t kept = keepLane ? 1 : 0;
  if (pool_->cells.size() > kept * CompositeStateLayout::kLaneGdnCells)
    pool_->cells.pop_back();
  else if (!pool_->cachedCells.empty())
    pool_->cachedCells.pop_back();
  else if (idleRings() > kept)
    pool_->rings.pop_back();
  else if (idleWindows() > kept)
    pool_->windows.pop_back();
  else
    return 0;
  const uint64_t after = backend_.memoryStats().allocatedBytes;
  return before >= after ? before - after : 0;
}

uint32_t QwenStateStorage::idleCells() const noexcept {
  return static_cast<uint32_t>(pool_->cells.size() + pool_->cachedCells.size());
}

uint32_t QwenStateStorage::idleRings() const noexcept {
  return static_cast<uint32_t>(pool_->rings.size());
}

uint32_t QwenStateStorage::idleWindows() const noexcept {
  return static_cast<uint32_t>(pool_->windows.size());
}

uint64_t QwenStateStorage::missingBytes(uint32_t cells, bool rings, bool cached) const noexcept {
  const auto &pool = cached ? pool_->cachedCells : pool_->cells;
  const uint64_t missing = cells - std::min<uint64_t>(pool.size(), cells);
  const uint64_t cellBytes = cached ? layout_.target.cachedCellBytes() : layout_.target.cellBytes();
  return missing * cellBytes +
         (rings && pool_->rings.empty() ? layout_.draft.ringBytes() : 0) +
         (pool_->windows.empty() ? layout_.draft.windowBytes() : 0);
}

uint32_t QwenStateStorage::statesToActivate() const noexcept {
  if (!idleRings())
    return std::numeric_limits<uint32_t>::max();
  // A lane takes no cell when its cells have no bytes.
  const uint32_t laneCells = emptyCell_ ? 0 : CompositeStateLayout::kLaneGdnCells;
  const uint32_t cells = laneCells - std::min<uint32_t>(
      static_cast<uint32_t>(pool_->cells.size()), laneCells);
  const uint32_t windows = idleWindows() ? 0 : 1;
  return std::max(cells, windows);
}

void QwenStateStorage::updateLengths(uint32_t index,
                                     QwenLogicalLengths lengths) {
  validateLengths(lengths, false);
  Lane &current = lane(index);
  requireAssigned(current);
  current.metadata.lengths = lengths;
}

void QwenStateStorage::swapParity(uint32_t index) {
  Lane &current = lane(index);
  requireCommitted(current);
  current.metadata.activeParity ^= 1;
}

void QwenStateStorage::swapDecodeParity(uint32_t index, uint32_t retainedRows) {
  Lane &current = lane(index);
  requireAssigned(current);
  if (!retainedRows || retainedRows > ExecutionLimits::targetVerifyRows)
    throw std::invalid_argument("a decode step retains one to all of its rows");
  current.metadata.activeParity ^= 1;
  current.metadata.pendingRows = retainedRows;
}

std::shared_ptr<const QwenCompositeState>
QwenStateStorage::snapshot(uint32_t index) {
  return snapshot(index, lane(index).metadata.lengths);
}

std::shared_ptr<const QwenCompositeState>
QwenStateStorage::snapshot(uint32_t index, QwenLogicalLengths lengths) {
  // Before anything is allocated; copyState() and seal() check again.
  requireCommitted(lane(index));
  validateLengths(lengths, true);
  metal::AllocationFailure refusal;
  const std::shared_ptr<QwenStateReservation> reservation = reserve(refusal);
  if (!reservation)
    return nullptr;
  copyState(index, *reservation);
  return seal(*reservation, lengths);
}

std::shared_ptr<QwenStateReservation>
QwenStateStorage::reserve(metal::AllocationFailure &refusal) {
  Buffers buffers;
  refusal = acquire(1, false, "qwen-state-cache", buffers, 0, {}, true).failure;
  if (refusal != metal::AllocationFailure::None)
    return nullptr;
  return std::shared_ptr<QwenStateReservation>(new QwenStateReservation(
      pool_, {std::move(buffers.gdn[0]), std::move(buffers.window)}));
}

std::shared_ptr<QwenStateReservation>
QwenStateStorage::reserveCapture(metal::AllocationFailure &refusal) {
  if (emptyCell_) return reserve(refusal);
  Buffers buffers;
  std::shared_ptr<QwenGdnCell> captureCell;
  const uint64_t captureBytes = pool_->cells.empty() ? layout_.target.cellBytes() : 0;
  const auto allocateCapture = [&] {
    if (captureBytes)
      captureCell = std::shared_ptr<QwenGdnCell>(new QwenGdnCell(
          backend_, allocations_, layout_.target, "qwen-state-capture-gdn"));
  };
  refusal = acquire(1, false, "qwen-state-capture-cache", buffers,
                    captureBytes, allocateCapture, true).failure;
  if (refusal != metal::AllocationFailure::None) return nullptr;
  if (!captureCell) {
    captureCell = std::move(pool_->cells.back());
    pool_->cells.pop_back();
  }
  return std::shared_ptr<QwenStateReservation>(new QwenStateReservation(
      pool_, {std::move(buffers.gdn[0]), std::move(buffers.window)},
      std::move(captureCell)));
}

void QwenStateStorage::copyState(uint32_t index, QwenStateReservation &reservation) const {
  requireCommitted(lane(index));
  const MetalBuffer &cachedCell = reservation.buffers().gdn->buffers().stateBase;
  const MetalBuffer &runningCell = current(index).stateBase;
  encodeCachedGdnMetal(backend_, cachedCell, runningCell, layout_.target);
  copyWindow(index, reservation);
}

void QwenStateStorage::copyCapturedState(QwenStateReservation &reservation) const {
  if (reservation.pool_ != pool_ || !reservation.captureCell_)
    throw std::invalid_argument("state reservation has no raw capture cell");
  encodeCachedGdnMetal(backend_, reservation.buffers().gdn->buffers().stateBase,
                       reservation.captureCell_->buffers().stateBase, layout_.target);
}

void QwenStateStorage::copyWindow(uint32_t index, QwenStateReservation &reservation) const {
  copyExact(reservation.buffers().window->buffer(), window(index), "cached context window");
}

std::shared_ptr<const QwenCompositeState>
QwenStateStorage::seal(QwenStateReservation &reservation, QwenLogicalLengths lengths) const {
  if (reservation.pool_ != pool_)
    throw std::invalid_argument("a state reservation of another storage");
  validateLengths(lengths, true);
  // A reservation sealed before holds no buffers, which the state refuses.
  if (reservation.captureCell_)
    pool_->giveCell(std::move(reservation.captureCell_));
  return std::shared_ptr<const QwenCompositeState>(new QwenCompositeState(
      pool_, std::exchange(reservation.buffers_, {}), layout_, lengths, file_, staging_));
}

std::unique_ptr<StateOffload>
QwenStateStorage::snapshotToDisk(uint32_t index, std::function<void()> completion) {
  const Lane &source = lane(index);
  requireCommitted(source);
  validateLengths(source.metadata.lengths, true);
  // The caller asked the write budget; the write needs what a copy in use
  // does.
  if (!canSnapshotToDisk(true))
    return {};
  std::span<std::byte> staged = staging_->bytes();
  const uint64_t cellBytes = layout_.target.cachedCellBytes();
  const MetalBuffer &runningCell = current(index).stateBase;
  MetalBuffer stagedCell = cellBytes ? backend_.view(staging_->buffer, 0, cellBytes) : MetalBuffer{};
  encodeCachedGdnMetal(backend_, stagedCell, runningCell, layout_.target);
  const uint64_t windowBytes = layout_.draft.windowBytes();
  if (windowBytes)
    std::memcpy(staged.data() + cellBytes, writableContents(window(index), "cached context window"), windowBytes);
  return QwenCompositeState::write(file_, staging_, {staged}, layout_,
                                   source.metadata.lengths, std::move(completion));
}

metal::AllocationResult
QwenStateStorage::acquire(uint32_t cells, bool rings, std::string_view label,
                          Buffers &buffers, uint64_t extraBytes,
                          const std::function<void()> &allocateExtra, bool cached) {
  // Cells of no bytes are all the empty cell, which neither an admission nor
  // the pool sees.
  if (emptyCell_) {
    std::fill_n(buffers.gdn.begin(), cells, emptyCell_);
    cells = 0;
  }
  auto &cellPool = cached ? pool_->cachedCells : pool_->cells;
  const uint32_t pooledCells =
      std::min(cells, static_cast<uint32_t>(cellPool.size()));
  const bool pooledRings = !pool_->rings.empty();
  const bool pooledWindow = !pool_->windows.empty();
  Buffers fresh;
  if (const uint64_t bytes = missingBytes(cells, rings, cached) + extraBytes) {
    const auto admission = admitAllocation_(bytes, [&] {
      if (allocateExtra)
        allocateExtra();
      for (uint32_t cell = pooledCells; cell < cells; ++cell) {
        fresh.gdn[cell] = std::shared_ptr<QwenGdnCell>(new QwenGdnCell(
            backend_, allocations_, layout_.target,
            std::string(label) + "-gdn-" + std::to_string(cell), cached));
      }
      if (rings && !pooledRings) {
        fresh.draft = std::shared_ptr<DFlashDraftRing>(new DFlashDraftRing(
            backend_, allocations_, layout_.draft,
            std::string(label) + "-draft"));
      }
      if (!pooledWindow) {
        fresh.window = std::shared_ptr<DFlashContextWindow>(new DFlashContextWindow(
            backend_, allocations_, layout_.draft, std::string(label) + "-window"));
      }
    });
    // What a driver's refusal left of the attempt goes with `fresh`.
    if (!admission)
      return admission;
  } else {
    // A plan of no bytes (a target without GDN state, and no draft) still
    // holds the objects its lanes and reservations name: their buffers are
    // empty, which is a complete state.
    if (rings && !pooledRings)
      fresh.draft = std::shared_ptr<DFlashDraftRing>(new DFlashDraftRing(
          backend_, allocations_, layout_.draft, std::string(label) + "-draft"));
    if (!pooledWindow)
      fresh.window = std::shared_ptr<DFlashContextWindow>(new DFlashContextWindow(
          backend_, allocations_, layout_.draft, std::string(label) + "-window"));
  }
  for (uint32_t cell = 0; cell < pooledCells; ++cell) {
    buffers.gdn[cell] = std::move(cellPool.back());
    cellPool.pop_back();
  }
  for (uint32_t cell = pooledCells; cell < cells; ++cell)
    buffers.gdn[cell] = std::move(fresh.gdn[cell]);
  const auto take = [](auto &pool, auto &allocated, bool pooled) {
    if (!pooled)
      return std::move(allocated);
    auto taken = std::move(pool.back());
    pool.pop_back();
    return taken;
  };
  if (rings)
    buffers.draft = take(pool_->rings, fresh.draft, pooledRings);
  buffers.window = take(pool_->windows, fresh.window, pooledWindow);
  return {};
}

std::shared_ptr<QwenGdnCell> QwenStateStorage::acquireCachedCellForRestore() {
  if (emptyCell_) return emptyCell_;
  if (!pool_->cachedCells.empty()) {
    auto cell = std::move(pool_->cachedCells.back());
    pool_->cachedCells.pop_back();
    return cell;
  }
  std::shared_ptr<QwenGdnCell> cell;
  const uint64_t bytes = layout_.target.cachedCellBytes();
  const metal::AllocationResult admission = admitAllocation_(bytes, [&] {
    cell = std::shared_ptr<QwenGdnCell>(new QwenGdnCell(
        backend_, allocations_, layout_.target, "qwen-state-restore-gdn", true));
  });
  if (!admission) return {};
  return cell;
}

void QwenStateStorage::restore(uint32_t index, const QwenCompositeState &state,
                               bool restoreDraftState) {
  const MetalBuffer &runningCell = current(index).stateBase;
  const MetalBuffer &cachedCell = state.buffers_.gdn->buffers().stateBase;
  decodeCachedGdnMetal(backend_, runningCell, cachedCell, layout_.target);
  if (restoreDraftState)
    copyExact(window(index), state.buffers_.window->buffer(), "restored context window");
  restoreLengths(index, state.lengths_, restoreDraftState);
}

void QwenStateStorage::restoreLengths(uint32_t index, QwenLogicalLengths lengths,
                                     bool restoreDraftState) {
  Lane &destination = lane(index);
  if (!restoreDraftState) {
    lengths.draftBase = lengths.targetTokens;
    lengths.draftLength = 0;
    lengths.windowEnd = lengths.targetTokens;
  }
  destination.metadata.lengths = lengths;
}

std::unique_ptr<StateRestore> QwenStateStorage::beginRestore(
    uint32_t index, const CompositeState &state, bool restoreDraftState,
    std::function<void()> completion, std::function<void()> committed) {
  const auto *typed = dynamic_cast<const QwenCompositeState *>(&state);
  if (!typed || typed->layout_ != layout_)
    throw std::invalid_argument("incompatible Qwen composite state");
  validateLengths(typed->lengths_, true);
  requireCommitted(lane(index));
  if (!typed->disk_) {
    restore(index, *typed, restoreDraftState);
    committed();
    return {};
  }
  // The read takes the window along whether or not the restore uses it:
  // the snapshot that promotes the state copies it from the lane.
  auto cellLease = std::make_shared<CachedCellLease>(pool_, acquireCachedCellForRestore());
  const MetalBuffer &encoded = cellLease->cell()
      ? cellLease->cell()->buffers().stateBase
      : current(index).stateBase;
  auto spans = stateSpans(gdnCellBuffers(backend_, encoded, layout_.target, true), window(index));
  auto commit = [this, index, lengths = typed->lengths_, restoreDraftState,
                 committed = std::move(committed)] {
    restoreLengths(index, lengths, restoreDraftState);
    committed();
  };
  auto decode = [this, index, cellLease] {
    if (cellLease->cell()) {
      decodeCachedGdnMetal(backend_, current(index).stateBase,
                           cellLease->cell()->buffers().stateBase, layout_.target);
      cellLease->release();
      return;
    }
    const MetalBuffer &running = current(index).stateBase;
    if (running)
      decodeCachedGdnInPlace(
          {static_cast<std::byte *>(writableContents(running, "restored GDN state")),
           running.sizeBytes()}, layout_.target);
  };
  auto operation = typed->file_->read(typed->disk_, std::move(spans), std::move(completion));
  try {
    return std::make_unique<FileRestore>(operation, std::move(commit),
        [this, index, lengths = typed->lengths_] { return snapshot(index, lengths); }, std::move(decode));
  } catch (...) {
    operation->drain();
    throw;
  }
}

std::shared_ptr<const CompositeState> QwenStateStorage::adopt(const SlotRecord &record,
                                                              uint32_t tokens) {
  if (!file_)
    throw std::logic_error("Qwen states are taken back only from a disk tier");
  const bool hasDraft = layout_.draft.layers != 0;
  const uint32_t window = hasDraft ? std::min(tokens, ExecutionLimits::draftContextTokens) : 0;
  const QwenLogicalLengths lengths = hasDraft
      ? QwenLogicalLengths{tokens, tokens - window, window, tokens}
      : QwenLogicalLengths{tokens, tokens, 0, tokens};
  validateLengths(lengths, true);
  if (record.payloadBytes != layout_.cachedBytes())
    throw std::invalid_argument("a recorded Qwen state does not match the state layout");
  return std::shared_ptr<const CompositeState>(
      new QwenCompositeState(layout_, lengths, file_, file_->adopt(record)));
}

QwenStateStorage::Lane &QwenStateStorage::lane(uint32_t index) {
  if (index >= lanes_.size()) {
    throw std::out_of_range("invalid Qwen lane");
  }
  return lanes_[index];
}

const QwenStateStorage::Lane &QwenStateStorage::lane(uint32_t index) const {
  if (index >= lanes_.size()) {
    throw std::out_of_range("invalid Qwen lane");
  }
  return lanes_[index];
}

void QwenStateStorage::validateLengths(const QwenLogicalLengths &lengths,
                                       bool cacheSnapshot) const {
  const bool hasDraft = layout_.draft.layers != 0;
  if (lengths.draftLength > ExecutionLimits::draftContextTokens ||
      lengths.draftEnd() > lengths.targetTokens ||
      lengths.windowEnd > lengths.draftEnd()) {
    throw std::invalid_argument("invalid draft ring metadata");
  }
  if (cacheSnapshot &&
      (!lengths.targetTokens ||
       (hasDraft && !lengths.hasCompleteDraftWindow(ExecutionLimits::draftContextTokens)) ||
       lengths.targetTokens % kv::kPageTokens)) {
    throw std::invalid_argument(
        "composite snapshot requires equal page-aligned committed lengths "
        "(targetTokens=" + std::to_string(lengths.targetTokens) +
        ", draftBase=" + std::to_string(lengths.draftBase) +
        ", draftLength=" + std::to_string(lengths.draftLength) +
        ", draftEnd=" + std::to_string(lengths.draftEnd()) +
        ", windowEnd=" + std::to_string(lengths.windowEnd) +
        ", draftCapacity=" +
        std::to_string(ExecutionLimits::draftContextTokens) +
        ", pageTokens=" + std::to_string(kv::kPageTokens) +
        ", hasDraft=" + (hasDraft ? "true" : "false") + ")");
  }
  // States are published in prefill, which keeps the window with the rings.
  // Without a draft there is no window to keep.
  if (cacheSnapshot && hasDraft && !lengths.hasCurrentContextWindow())
    throw std::logic_error("a composite snapshot needs the context window of its draft rings");
}

void QwenStateStorage::requireAssigned(const Lane &current) {
  if (!current.metadata.assigned()) {
    throw std::logic_error("Qwen lane is not assigned");
  }
}

void QwenStateStorage::requireCommitted(const Lane &current) {
  requireAssigned(current);
  if (current.metadata.pendingRows)
    throw std::logic_error("Qwen lane state has decoded rows pending");
}

} // namespace splash::model
