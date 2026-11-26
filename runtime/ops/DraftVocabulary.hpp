#pragma once

// The vocabulary block a DFlash2 draft proposes from while a step's text
// stays in it, ported from Splash 1.7.2 (ops/DraftSelector.hpp). A family
// whose vocabulary keeps no such block leaves both bounds zero: then the
// draft proposes from the whole vocabulary.
//
// This is the data view only, as QwenTargetDimensions needs it: the columns
// of a vocabulary head a draft's selector reads and the tokens they name.
// The draft selector's kernels live with each target's own draft path.

#include <array>
#include <cstdint>

namespace splash::ops {

struct DraftVocabulary final {
  uint32_t leading = 0;
  uint32_t trailing = 0;

  // The columns of its logits rows in a vocabulary of `vocabulary` tokens.
  [[nodiscard]] constexpr uint32_t columns(uint32_t vocabulary) const noexcept {
    return leading + (vocabulary - trailing);
  }
  [[nodiscard]] constexpr bool holds(uint32_t token) const noexcept { return token < leading || token >= trailing; }
  // The token of logits column `column`, as a draft's selector reads it.
  [[nodiscard]] constexpr uint32_t token(uint32_t column) const noexcept {
    return column < leading ? column : column - leading + trailing;
  }
  // The rows of a vocabulary head of `vocabulary` rows that write its logits
  // columns, in column order. A RowRange carries a first row and a count.
  struct RowRange final {
    uint32_t first = 0;
    uint32_t count = 0;
  };
  [[nodiscard]] constexpr std::array<RowRange, 2> rows(uint32_t vocabulary) const noexcept {
    return {{{0, leading}, {trailing, vocabulary - trailing}}};
  }
  bool operator==(const DraftVocabulary &) const = default;
};

} // namespace splash::ops
