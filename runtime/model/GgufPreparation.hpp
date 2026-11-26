#pragma once

#include "metal/MetalBackend.hpp"
#include "model/GgufImage.hpp"
#include "model/WeightSource.hpp"

#include <cstdint>
#include <optional>

namespace splash::model {

// The Metal staging of the rows of one repack step, whatever the tensor,
// layer or expert count.
inline constexpr uint64_t kGgufRepackStagingBytes = 32 << 20;

// Writes every byte of a planned image into its buffer: the header and
// descriptors, the copied rows and the planes the GPU repacks into it, each
// read from the source its rows name (a GGUF's, or an MLX checkpoint's
// tensors).
void writeGgufImage(metal::MetalBackend &backend, const metal::MetalBuffer &image, const gguf::Image &plan);

// Rows [first, first + count) of a repack (its sources' rows in order), whole
// and back to back into `to`, count * ggufRowBytes(format, columns) bytes:
// the native rows the GPU repacks into those rows' planes. A streamed
// expert's rows are read with it (model/ExpertStore.hpp), from any thread.
void readRepackRows(const gguf::Repack &repack, uint64_t first, uint64_t count, uint8_t *to);

// Reads original GGUF Q4_0 rows without passing through the Metal repacker.
// Returns false unless every requested row is a direct, ordered GGUF Q4_0
// source row with the exact native row width. `to` receives count rows.
[[nodiscard]] bool readOriginalQ4_0Rows(const gguf::Repack &repack, uint64_t first,
                                        uint64_t count, uint8_t *to);
// Metadata-only eligibility check for a row range accepted by the reader above.
[[nodiscard]] bool supportsOriginalQ4_0Rows(const gguf::Repack &repack, uint64_t first,
                                            uint64_t count) noexcept;

// Where rows [first, first + count) of a repack are in a file, when one of
// its sources stores them as they are and in order, so that they are read
// with the file's bytes alone: the file and the offset of the first row from
// its start. None when they are built (a quantized safetensors tensor's,
// NVFP4 rows, a quantized float tensor) or permuted, or span two sources.
struct StoredRows final {
  const WeightSource *file = nullptr;
  uint64_t offset = 0;
};
[[nodiscard]] std::optional<StoredRows> storedRepackRows(const gguf::Repack &repack, uint64_t first, uint64_t count);

// Whether every value of a 16-bit float tensor at `offset` (`values` of
// `type`, ggml::kBF16 or ggml::kF16) is exactly representable as MXFP8, so
// the rows can be built at half the bytes. Results are cached by file and
// offset: a start plans its target several times from the same source.
[[nodiscard]] bool exactlyMxfp8(const WeightSource &file, uint64_t offset, uint64_t values, uint32_t type);

} // namespace splash::model
