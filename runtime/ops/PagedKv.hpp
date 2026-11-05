#pragma once

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/PagedAttention.h"
#include "metal/MetalBackend.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace splash::kv {

// Selected once for a runtime and its entire page pool. Weight storage is
// independent of the KV format; requests never change it while serving.
// Int4 packs two elements per byte, low nibble first, and keeps INT8's one
// fp32 scale per (KV head, token), so a page's scale geometry is unchanged.
enum class Format : uint32_t { Int8 = 1, BFloat16 = 2, Int4 = 3 };

[[nodiscard]] constexpr bool validFormat(Format format) noexcept {
  return format == Format::Int8 || format == Format::BFloat16 || format == Format::Int4;
}

// One element's bit width, and whether its scale is fp32 rather than carried
// by the element: the quantized formats' scales are fp32, one per (KV head,
// token).
struct ElementFormat final {
  uint32_t bits = 8;
  bool floating = false;

  bool operator==(const ElementFormat &) const = default;
};

// A layout's keys and values, each in its own element format. They share a
// page, but not a size: Int4 packs two value elements per byte, one key
// element.
struct KvStorage final {
  ElementFormat k = ElementFormat{8, false};
  ElementFormat v = ElementFormat{8, false};

  bool operator==(const KvStorage &) const = default;
};

[[nodiscard]] constexpr ElementFormat elementFormat(Format format) noexcept {
  switch (format) {
  case Format::Int8: return ElementFormat{8, false};
  case Format::BFloat16: return ElementFormat{16, true};
  case Format::Int4: return ElementFormat{4, false};
  }
  return ElementFormat{0, false};
}

// Whether a format's elements keep one fp32 scale per (KV head, token).
[[nodiscard]] constexpr bool quantizedFormat(Format format) noexcept {
  return format != Format::BFloat16;
}

// How many elements of a format one byte of its data region holds.
[[nodiscard]] constexpr uint32_t elementsPerByte(Format format) noexcept {
  return format == Format::Int4 ? 2u : 1u;
}

[[nodiscard]] constexpr bool validElementFormat(ElementFormat format) noexcept {
  return format.floating ? format.bits == 16 : format.bits == 8 || format.bits == 4;
}

// The storages the layouts implement: one element format for both tensors,
// or INT8 keys with packed 4-bit values (k8v4). A 4-bit key with 8-bit
// values is not a page any kernel or reader knows.
[[nodiscard]] constexpr bool validKvStorage(const KvStorage &storage) noexcept {
  if (!validElementFormat(storage.k) || !validElementFormat(storage.v))
    return false;
  return storage.k == storage.v ||
         (storage.k.bits == 8 && !storage.k.floating && storage.v.bits == 4 &&
          !storage.v.floating);
}

// A single-format layout's storage: the format's element for both keys and
// values. A two-element layout (k8v4) names its own.
[[nodiscard]] constexpr KvStorage storageOf(Format format) noexcept {
  const ElementFormat element = elementFormat(format);
  return {element, element};
}

// The one Format a storage holds for both keys and values: BF16, INT4, or
// INT8 for the two-element k8v4 storage, whose keys are INT8.
[[nodiscard]] constexpr Format singleFormat(const KvStorage &storage) noexcept {
  if (storage.k.bits == 16 && storage.k.floating && storage.v.bits == 16 &&
      storage.v.floating)
    return Format::BFloat16;
  if (storage.k.bits == 4 && !storage.k.floating && storage.v.bits == 4 &&
      !storage.v.floating)
    return Format::Int4;
  return Format::Int8;
}

[[nodiscard]] constexpr bool quantizedFormat(ElementFormat format) noexcept {
  return !format.floating;
}

[[nodiscard]] constexpr uint64_t dataBytes(uint64_t elements, ElementFormat format) noexcept {
  return elements * format.bits / 8;
}

[[nodiscard]] constexpr std::string_view formatName(Format format) noexcept {
  switch (format) {
  case Format::Int8: return "int8";
  case Format::BFloat16: return "bf16";
  case Format::Int4: return "int4";
  }
  return "invalid";
}

[[nodiscard]] constexpr std::string_view storageFormatName(Format format, bool valuesTokenMajor) noexcept {
  switch (format) {
  case Format::Int8:
    return valuesTokenMajor ? "q8s8_f32_scale_per_token_head_k_token_major_v_token_major"
                            : "q8s8_f32_scale_per_token_head_k_token_major_v_dimension_major";
  case Format::BFloat16:
    return valuesTokenMajor ? "bf16_k_token_major_v_token_major" : "bf16_k_token_major_v_dimension_major";
  case Format::Int4:
    return valuesTokenMajor ? "q4s4_f32_scale_per_token_head_k_token_major_v_token_major"
                            : "q4s4_f32_scale_per_token_head_k_token_major_v_dimension_major";
  }
  return "invalid";
}

// What a pool page holds. A Full page holds its K/V and, with a block index
// (Layout::indexDimension), its blocks' index; an Index page holds only the
// index, its K/V living in the page's tier copy, which the request's reads
// of spilled KV serve (model::SpilledKvReads); a Slim page, of a layout with
// a sliding window, holds the K/V of the layers no window bounds and the
// index, for a page no row's window reaches any more. A page's form is its
// extent's: an extent holds the first Layout::regions(form) of a Full
// extent's K/V regions, in their order (Layout::region), and then the index
// plane, so that a layer's region sits at one offset in every extent that
// holds it. A layout that cannot spill (Layout::spillable) has no Index
// pages, and one whose pages cannot give up a window's layers
// (Layout::slimmable) no Slim pages. The values are a persistent cache's
// (KvSlotLabel).
enum class PageForm : uint8_t { Full, Index, Slim };

// A page whose content goes to another page.
struct PageCopy final {
  uint32_t from = 0;
  uint32_t to = 0;

  bool operator==(const PageCopy &) const = default;
};

// The memory of the engine's page pool: extents of extentPages() pages,
// pageCount() a whole number of them, each of one form (PageForm) from its
// allocation to its release. Only KvPool allocates and releases them.
// Implementations provide Metal storage or deterministic test storage.
class ExtentStorage {
public:
  virtual ~ExtentStorage() = default;
  [[nodiscard]] virtual uint32_t pageCount() const noexcept = 0;
  // What a page of the form takes in its extent.
  [[nodiscard]] virtual uint64_t bytesPerPage(PageForm form) const noexcept = 0;
  [[nodiscard]] virtual uint32_t extentPages() const noexcept = 0;
  // Allocates extent `extent`, which is not allocated, for pages of the
  // form: granted, or the refusal's cause with nothing allocated.
  // std::logic_error for an allocated extent, std::invalid_argument for
  // Index pages of storage whose pages cannot spill.
  [[nodiscard]] virtual metal::AllocationResult allocateExtent(uint32_t extent, PageForm form) = 0;
  // Releases allocated extent `extent` at once. std::logic_error, changing
  // nothing, for an unallocated extent or while a command is in flight (a
  // command reaches extents through its page tables without retaining them).
  virtual void releaseExtent(uint32_t extent) = 0;
  // Copies each page's content onto its destination, both in allocated
  // extents: what both pages hold, all of it between pages of one form and
  // the index between a Full page and an Index page. How the pool moves the
  // pages of an extent it empties, and pages to the other form.
  // std::logic_error, copying nothing, for a page of an unallocated extent
  // or while a command is in flight (it reaches both pages through its
  // tables and may still write the source).
  virtual void copyPages(std::span<const PageCopy> copies) = 0;
  // Zeroes each page's content, in an allocated extent: a page one request
  // holds alone and that no command reaches yet (KvPool::clearPage).
  // std::logic_error, clearing nothing, for a page of an unallocated extent.
  virtual void clearPages(std::span<const uint32_t> pages) = 0;
};

// Shared cache format and execution limits; model dimensions live in Layout.
inline constexpr uint32_t kPageTokens = SPLASH_TARGET_KV_BLOCK_TOKENS;
inline constexpr uint32_t kMaximumLogicalTokens =
    SPLASH_MAXIMUM_CONTEXT_TOKENS;
inline constexpr uint32_t kMaximumPhysicalTokens =
    SPLASH_MAXIMUM_PHYSICAL_KV_TOKENS;
inline constexpr int32_t kQuantizedMinimum = -127;
inline constexpr int32_t kQuantizedMaximum = 127;
// Every tensor region of an extent starts on this boundary. The attention
// kernels were tuned on it, and regions aligned to less cost Apple10's 35B
// verify kernel a fixed ~25 µs per dispatch.
inline constexpr uint64_t kExtentRegionAlignmentBytes = 64 * 1024;
// The size a pool aims its extents at (Layout::extentPagesFor stays within
// half and one and a half times it): each extent costs the serving loop an
// allocation and a release, and an extent returns memory only once all of
// its pages are free.
inline constexpr uint64_t kAllocationExtentTargetBytes = 128ull * 1024 * 1024;

namespace detail {

[[nodiscard]] constexpr uint64_t gcd(uint64_t left, uint64_t right) noexcept {
  while (right) {
    const uint64_t remainder = left % right;
    left = right;
    right = remainder;
  }
  return left;
}

[[nodiscard]] constexpr uint64_t lcm(uint64_t left, uint64_t right) noexcept {
  return left && right ? left / gcd(left, right) * right : 0;
}

// Pages whose bytes of one tensor fill whole region alignment units.
[[nodiscard]] constexpr uint64_t pagesForAlignedRegion(
    uint64_t bytesPerPage) noexcept {
  return bytesPerPage
             ? kExtentRegionAlignmentBytes /
                   gcd(kExtentRegionAlignmentBytes, bytesPerPage)
             : 0;
}

} // namespace detail

// Physical KV geometry: Page32, either BF16 or per-(token, head) symmetric INT8.
// Layer and head counts vary by target.
struct Layout final {
  uint32_t attentionLayers = 0;
  uint32_t kvHeads = 0;
  uint32_t headDimension = 0;
  Format format = Format::Int8;
  // A block index: each page also holds, per layer, the pooled keys of its
  // blocks of SPLASH_ATTENTION_BLOCK_TOKENS tokens, this many values each
  // (bf16 [block][dimension]), in an index region of its extent's index plane
  // (PageStorage::indexOffset). Only a complete block has a pooled key in a
  // page: whoever owns the sequence keeps the keys of its open block.
  uint32_t indexDimension = 0;
  // Whether a page holds each head's values token-major like its keys
  // ([token][dimension]) rather than dimension-major ([dimension][token]).
  // The page loop of dense attention reads whole pages, and its PV runs
  // faster on dimension-major values; a sparse attention
  // (ops::AttentionBlocks) gathers blocks of tokens, whose values are
  // contiguous only token-major.
  bool valuesTokenMajor = false;
  // A sliding window: a row at position p of a windowed layer reads the keys
  // of positions p - windowTokens to p alone, the other layers' every key up
  // to p. windowedLayers holds a bit for each windowed attention layer, bit l
  // for the l-th; both are zero for a layout without a window.
  uint32_t windowTokens = 0;
  uint64_t windowedLayers = 0;
  // The element format of the layout's keys and values. Int8 and BFloat16
  // hold one element format for both; Int4 and k8v4 hold two, as the values
  // pack two elements per byte while the keys keep one element per byte. Left
  // at its default, it is the layout's single format; a two-element layout
  // sets it and keeps Format::Int8 (its keys') as its `format`.
  KvStorage storage = {{8, false}, {8, false}};

  // The element formats the geometry uses: the explicit `storage` of a
  // two-element layout, or the single format's for every other layout, whose
  // `storage` is left at its default. A Layout built positionally from its
  // format therefore keeps that format's elements.
  [[nodiscard]] constexpr KvStorage effectiveStorage() const noexcept {
    return storage == KvStorage{} ? storageOf(format) : storage;
  }

  [[nodiscard]] constexpr bool valid() const noexcept {
    return attentionLayers && kvHeads && headDimension && validKvStorage(effectiveStorage()) &&
           format == singleFormat(effectiveStorage()) &&
           !windowTokens == !windowedLayers &&
           (attentionLayers >= 64 || !(windowedLayers >> attentionLayers));
  }
  [[nodiscard]] constexpr bool windowed(uint32_t layer) const noexcept {
    return layer < 64 && (windowedLayers >> layer & 1);
  }
  // Where a layer's K/V region sits among a page's: the layers no window
  // bounds first, then the windowed ones, each in layer order, so that what
  // a row reads of a page every window has passed lies in its first
  // regions. Without a window a layer's region is the layer itself.
  [[nodiscard]] constexpr uint32_t region(uint32_t layer) const noexcept {
    const uint64_t below = layer < 64 ? windowedLayers & ((uint64_t{1} << layer) - 1) : windowedLayers;
    const auto windowedBelow = static_cast<uint32_t>(std::popcount(below));
    const auto unwindowed = attentionLayers - static_cast<uint32_t>(std::popcount(windowedLayers));
    return windowed(layer) ? unwindowed + windowedBelow : layer - windowedBelow;
  }
  // The K/V regions a page of the form holds, the first ones of a Full
  // page's: every layer's, those of the layers no window bounds (Slim), or
  // none (Index).
  [[nodiscard]] constexpr uint32_t regions(PageForm form) const noexcept {
    switch (form) {
    case PageForm::Full: return attentionLayers;
    case PageForm::Slim: return attentionLayers - static_cast<uint32_t>(std::popcount(windowedLayers));
    case PageForm::Index: return 0;
    }
    return 0;
  }
  // The pages before a row's own that hold keys a windowed layer's row
  // reads: a row at or past a page boundary reads none before them.
  [[nodiscard]] constexpr uint32_t windowPages() const noexcept {
    return (windowTokens + kPageTokens - 1) / kPageTokens;
  }
  [[nodiscard]] constexpr uint32_t elementsPerScaleK() const noexcept {
    return quantizedFormat(effectiveStorage().k) ? headDimension : 0;
  }
  [[nodiscard]] constexpr uint32_t elementsPerScaleV() const noexcept {
    return quantizedFormat(effectiveStorage().v) ? headDimension : 0;
  }
  [[nodiscard]] constexpr uint64_t elementsPerLayerPage() const noexcept {
    return uint64_t{kPageTokens} * kvHeads * headDimension;
  }
  [[nodiscard]] constexpr uint64_t scalesPerTensorLayerPageK() const noexcept {
    return quantizedFormat(effectiveStorage().k) ? uint64_t{kPageTokens} * kvHeads : 0;
  }
  [[nodiscard]] constexpr uint64_t scalesPerTensorLayerPageV() const noexcept {
    return quantizedFormat(effectiveStorage().v) ? uint64_t{kPageTokens} * kvHeads : 0;
  }
  // Keys and values share a page, but not a size: Int4 packs two value
  // elements per byte, one key element.
  [[nodiscard]] constexpr uint64_t dataBytesPerLayerPageK() const noexcept {
    return dataBytes(elementsPerLayerPage(), effectiveStorage().k);
  }
  [[nodiscard]] constexpr uint64_t dataBytesPerLayerPageV() const noexcept {
    return dataBytes(elementsPerLayerPage(), effectiveStorage().v);
  }
  [[nodiscard]] constexpr uint64_t scaleBytesPerLayerPageK() const noexcept {
    return scalesPerTensorLayerPageK() * sizeof(float);
  }
  [[nodiscard]] constexpr uint64_t scaleBytesPerLayerPageV() const noexcept {
    return scalesPerTensorLayerPageV() * sizeof(float);
  }
  // The K/V bytes of one layer's page; the index bytes are apart.
  [[nodiscard]] constexpr uint64_t bytesPerLayerPage() const noexcept {
    return dataBytesPerLayerPageK() + scaleBytesPerLayerPageK() +
           dataBytesPerLayerPageV() + scaleBytesPerLayerPageV();
  }
  // The data and scale bytes of one tensor of one layer's page, as the
  // kernels' and the tests' single-tensor queries keep using them.
  [[nodiscard]] constexpr uint32_t elementsPerScale() const noexcept {
    return elementsPerScaleK();
  }
  [[nodiscard]] constexpr uint64_t scalesPerTensorLayerPage() const noexcept {
    return scalesPerTensorLayerPageK();
  }
  [[nodiscard]] constexpr uint64_t dataBytesPerLayerPage() const noexcept {
    return dataBytesPerLayerPageK();
  }
  [[nodiscard]] constexpr uint64_t scaleBytesPerLayerPage() const noexcept {
    return scaleBytesPerLayerPageK();
  }
  [[nodiscard]] constexpr uint64_t indexBytesPerLayerPage() const noexcept {
    return uint64_t{kPageTokens} / SPLASH_ATTENTION_BLOCK_TOKENS * indexDimension * 2;
  }
  // A page's K/V and its index, in every layer.
  [[nodiscard]] constexpr uint64_t kvBytesPerModelPage() const noexcept {
    return uint64_t{attentionLayers} * bytesPerLayerPage();
  }
  [[nodiscard]] constexpr uint64_t indexBytesPerModelPage() const noexcept {
    return uint64_t{attentionLayers} * indexBytesPerLayerPage();
  }
  // What a page of the form holds (PageForm).
  [[nodiscard]] constexpr uint64_t bytesPerModelPage(PageForm form = PageForm::Full) const noexcept {
    return uint64_t{regions(form)} * bytesPerLayerPage() + indexBytesPerModelPage();
  }
  // Whether a page's K/V can leave memory while its index stays (PageForm::
  // Index): the layout has a block index, which selects the blocks a row
  // reads; its values are token-major, so that a block's tokens are
  // contiguous in every tensor; and a page holds whole location places
  // (SPLASH_ATTENTION_LOCATION_BLOCK_BITS), so that a block read back sits at
  // its own place in a page of the hot block cache.
  [[nodiscard]] constexpr bool spillable() const noexcept {
    return indexDimension && valuesTokenMajor &&
           kPageTokens == SPLASH_ATTENTION_BLOCK_TOKENS << SPLASH_ATTENTION_LOCATION_BLOCK_BITS;
  }
  // Whether a page can give up its windowed layers' K/V (PageForm::Slim)
  // once no row's window reaches it: the layout has a window, and its tier
  // slots hold a page's spans in order (kv::SlotFormat version 0), of which
  // a Slim page's are the first; the block records of a spillable layout's
  // slots hold every layer's.
  [[nodiscard]] constexpr bool slimmable() const noexcept { return windowedLayers && !spillable(); }

  // An extent holds a whole number of these pages, so that every tensor
  // region starts 64 KiB-aligned. The INT8 scales are the tightest
  // constraint: 4 heads require 128 pages and 2 heads require 256. BF16
  // needs only 1 or 2 pages, or 32 with a 128-wide index. This is
  // allocation geometry only; prefix matching remains Page32 in both cases.
  [[nodiscard]] constexpr uint32_t extentAlignmentPages() const noexcept {
    // Every tensor region starts on a kExtentRegionAlignmentBytes boundary:
    // the page count has to make each region's bytes a whole number of
    // alignment units, and BF16 has no scales to align.
    uint64_t pages = 0;
    if (dataBytesPerLayerPageK())
      pages = detail::pagesForAlignedRegion(dataBytesPerLayerPageK());
    if (scaleBytesPerLayerPageK())
      pages = detail::lcm(pages, detail::pagesForAlignedRegion(scaleBytesPerLayerPageK()));
    if (dataBytesPerLayerPageV())
      pages = detail::lcm(pages, detail::pagesForAlignedRegion(dataBytesPerLayerPageV()));
    if (scaleBytesPerLayerPageV())
      pages = detail::lcm(pages, detail::pagesForAlignedRegion(scaleBytesPerLayerPageV()));
    if (indexDimension)
      pages = detail::lcm(pages, detail::pagesForAlignedRegion(indexBytesPerLayerPage()));
    return static_cast<uint32_t>(pages);
  }

  // Extents hold whole alignment units, between half and one and a half
  // times the allocation target; a unit larger than that is an extent on its
  // own. These are the smallest and the largest sizes.
  [[nodiscard]] constexpr uint32_t minimumExtentPages() const noexcept {
    const uint64_t unit = uint64_t{extentAlignmentPages()} * bytesPerModelPage();
    if (!unit)
      return 0;
    const uint64_t units =
        std::max<uint64_t>(1, (kAllocationExtentTargetBytes / 2 + unit - 1) / unit);
    return static_cast<uint32_t>(units * extentAlignmentPages());
  }
  [[nodiscard]] constexpr uint32_t maximumExtentPages() const noexcept {
    const uint64_t unit = uint64_t{extentAlignmentPages()} * bytesPerModelPage();
    if (!unit)
      return 0;
    const uint64_t units = kAllocationExtentTargetBytes * 3 / 2 / unit;
    return std::max(static_cast<uint32_t>(units * extentAlignmentPages()),
                    minimumExtentPages());
  }

  // The extent size of a pool that holds `pages` pages: all of a pool's
  // extents hold the same number of pages, because kernels find a layer's
  // region from that number. Of the sizes above that fit the pool, the one
  // that leaves the fewest pages over, and the one nearest the target on a
  // tie. Zero when the pool holds fewer pages than the smallest size.
  [[nodiscard]] constexpr uint32_t extentPagesFor(uint64_t pages) const noexcept {
    const uint32_t step = extentAlignmentPages();
    const uint64_t smallest = minimumExtentPages();
    if (!smallest || pages < smallest)
      return 0;
    const uint64_t largest = std::min<uint64_t>(maximumExtentPages(), pages);
    uint64_t best = 0, bestLeft = 0, bestDistance = 0;
    for (uint64_t extent = smallest; extent <= largest; extent += step) {
      const uint64_t bytes = extent * bytesPerModelPage();
      const uint64_t distance = bytes > kAllocationExtentTargetBytes
                                    ? bytes - kAllocationExtentTargetBytes
                                    : kAllocationExtentTargetBytes - bytes;
      if (!best || pages % extent < bestLeft ||
          (pages % extent == bestLeft && distance < bestDistance)) {
        best = extent;
        bestLeft = pages % extent;
        bestDistance = distance;
      }
    }
    return static_cast<uint32_t>(best);
  }

  bool operator==(const Layout &other) const noexcept {
    return attentionLayers == other.attentionLayers && kvHeads == other.kvHeads &&
           headDimension == other.headDimension && format == other.format &&
           indexDimension == other.indexDimension &&
           valuesTokenMajor == other.valuesTokenMajor &&
           windowTokens == other.windowTokens && windowedLayers == other.windowedLayers &&
           effectiveStorage() == other.effectiveStorage();
  }
};

// The layout's name: k8v4 for the two-element formats, the single format's
// name otherwise. The storage name (storageFormatName) spells out the scales
// and value order.
[[nodiscard]] constexpr std::string_view formatName(const Layout &layout) noexcept {
  const KvStorage storage = layout.effectiveStorage();
  if (storage.k.bits == 8 && !storage.k.floating && storage.v.bits == 4 &&
      !storage.v.floating)
    return "k8v4";
  return formatName(layout.format);
}

// The layout's element formats, as a name: q8s8 or q4s4 with the value
// order, as the single format's storage name.
[[nodiscard]] constexpr std::string_view storageFormatName(const Layout &layout) noexcept {
  const KvStorage storage = layout.effectiveStorage();
  if (storage.k.bits == 8 && !storage.k.floating && storage.v.bits == 4 && !storage.v.floating)
    return layout.valuesTokenMajor
               ? "q8s4_f32_scale_per_token_head_k_token_major_v_token_major"
               : "q8s4_f32_scale_per_token_head_k_token_major_v_dimension_major";
  return storageFormatName(layout.format, layout.valuesTokenMajor);
}

} // namespace splash::kv
