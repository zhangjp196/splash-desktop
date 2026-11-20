#include "WeightManifestHash.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace splash::model {
namespace {
constexpr size_t kBufferSize = 1024 * 1024;

void update(CC_SHA256_CTX &context, const void *data, size_t size) {
  if (size != 0 && CC_SHA256_Update(&context, data, static_cast<CC_LONG>(size)) != 1)
    throw std::runtime_error("SHA-256 update failed");
}

void updateU64(CC_SHA256_CTX &context, uint64_t value) {
  std::array<uint8_t, 8> encoded{};
  for (size_t i = 0; i < encoded.size(); ++i)
    encoded[encoded.size() - 1 - i] = static_cast<uint8_t>(value >> (i * 8));
  update(context, encoded.data(), encoded.size());
}

std::string hexDigest(const unsigned char *digest) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result(CC_SHA256_DIGEST_LENGTH * 2, '0');
  for (size_t i = 0; i < CC_SHA256_DIGEST_LENGTH; ++i) {
    result[i * 2] = digits[digest[i] >> 4];
    result[i * 2 + 1] = digits[digest[i] & 0x0f];
  }
  return result;
}
} // namespace

std::string sha256WeightManifest(const std::vector<WeightManifestEntry> &entries) {
  if (entries.empty()) throw std::invalid_argument("weight manifest is empty");
  if (entries.size() > std::numeric_limits<uint64_t>::max())
    throw std::invalid_argument("weight manifest has too many files");

  std::vector<const WeightManifestEntry *> sorted;
  sorted.reserve(entries.size());
  for (const auto &entry : entries) {
    if (entry.logicalName.empty()) throw std::invalid_argument("weight manifest logical name is empty");
    sorted.push_back(&entry);
  }
  std::sort(sorted.begin(), sorted.end(), [](const auto *left, const auto *right) {
    return left->logicalName < right->logicalName;
  });
  for (size_t i = 1; i < sorted.size(); ++i)
    if (sorted[i - 1]->logicalName == sorted[i]->logicalName)
      throw std::invalid_argument("duplicate weight manifest logical name: " + sorted[i]->logicalName);

  CC_SHA256_CTX context;
  if (CC_SHA256_Init(&context) != 1) throw std::runtime_error("SHA-256 initialization failed");
  updateU64(context, static_cast<uint64_t>(sorted.size()));
  std::array<unsigned char, kBufferSize> buffer{};
  for (const auto *entry : sorted) {
    const auto &name = entry->logicalName;
    updateU64(context, static_cast<uint64_t>(name.size()));
    update(context, name.data(), name.size());

    std::ifstream file(entry->path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot read weight file: " + entry->path.string());
    file.seekg(0, std::ios::end);
    const auto end = file.tellg();
    if (end < 0) throw std::runtime_error("cannot determine weight file size: " + entry->path.string());
    const auto size = static_cast<uint64_t>(end);
    updateU64(context, size);
    file.seekg(0, std::ios::beg);
    if (!file) throw std::runtime_error("cannot seek weight file: " + entry->path.string());

    uint64_t remaining = size;
    while (remaining != 0) {
      const auto chunk = static_cast<std::streamsize>(std::min<uint64_t>(remaining, buffer.size()));
      file.read(reinterpret_cast<char *>(buffer.data()), chunk);
      const auto read = file.gcount();
      if (read != chunk) throw std::runtime_error("weight file changed or became unreadable: " + entry->path.string());
      update(context, buffer.data(), static_cast<size_t>(read));
      remaining -= static_cast<uint64_t>(read);
    }
    char extra;
    if (file.read(&extra, 1)) throw std::runtime_error("weight file changed while hashing: " + entry->path.string());
    if (!file.eof()) throw std::runtime_error("failed reading weight file: " + entry->path.string());
  }

  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  if (CC_SHA256_Final(digest.data(), &context) != 1) throw std::runtime_error("SHA-256 finalization failed");
  return hexDigest(digest.data());
}

} // namespace splash::model
