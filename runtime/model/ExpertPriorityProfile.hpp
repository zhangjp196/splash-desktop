#pragma once

#include <cstdint>
#include <filesystem>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace splash::model {

using ExpertPriorityHints = std::unordered_map<uint32_t, std::vector<uint32_t>>;

// A missing path disables profiles. Invalid or mismatched documents are
// ignored as a whole and reported to stderr.
[[nodiscard]] ExpertPriorityHints loadExpertPriorityProfile(const std::filesystem::path &path,
                                                           std::string_view modelIdentity,
                                                           std::string_view weightManifestHash,
                                                           uint32_t topK = 8);

} // namespace splash::model
