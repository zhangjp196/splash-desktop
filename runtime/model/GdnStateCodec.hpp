#pragma once

#include "model/StateLayout.hpp"

#include <cstddef>
#include <span>

namespace splash::model {

// Encodes the recurrent layers for cache storage, retaining convolution and
// auxiliary state in their runtime representation.
void encodeCachedGdnState(std::span<std::byte> cached,
                          std::span<const std::byte> running,
                          const GdnStateLayout &layout);

// Restores a cache cell to the configured runtime GDN representation.
void decodeCachedGdnState(std::span<std::byte> running,
                          std::span<const std::byte> cached,
                          const GdnStateLayout &layout);

// Expands an encoded payload in the prefix of a full runtime cell in place.
void decodeCachedGdnStateInPlace(std::span<std::byte> cell,
                                 const GdnStateLayout &layout);

} // namespace splash::model
