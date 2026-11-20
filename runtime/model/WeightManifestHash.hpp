#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace splash::model {

struct WeightManifestEntry {
  std::string logicalName;
  std::filesystem::path path;
};

// Hashes a path-independent, sorted manifest and the complete contents of each file.
// Throws std::invalid_argument for an empty manifest, empty/duplicate names, and
// std::runtime_error when a file cannot be read or changes while being hashed.
std::string sha256WeightManifest(const std::vector<WeightManifestEntry> &entries);

} // namespace splash::model
