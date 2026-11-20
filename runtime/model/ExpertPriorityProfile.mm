#import <Foundation/Foundation.h>
#import <CoreFoundation/CoreFoundation.h>

#include "model/ExpertPriorityProfile.hpp"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <map>
#include <set>

namespace splash::model {
namespace {

bool unsignedInteger(id value, uint32_t &result) {
  if (![value isKindOfClass:[NSNumber class]] || CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID()) return false;
  const char *type = [(NSNumber *)value objCType];
  if (type[0] == 'd' || type[0] == 'f') return false;
  const long long number = [(NSNumber *)value longLongValue];
  if (number < 0 || static_cast<unsigned long long>(number) > std::numeric_limits<uint32_t>::max()) return false;
  result = static_cast<uint32_t>(number);
  return true;
}

void warning(const char *message) { std::fprintf(stderr, "expert priority profile: %s; ignored\n", message); }

} // namespace

ExpertPriorityHints loadExpertPriorityProfile(const std::filesystem::path &path, std::string_view modelIdentity,
                                              std::string_view weightManifestHash, uint32_t topK) {
  if (path.empty()) return {};
  if (modelIdentity.empty() || weightManifestHash.empty() || topK == 0 || topK > 64) {
    warning("invalid identity or top-k configuration");
    return {};
  }
  @autoreleasepool {
    NSError *error = nil;
    NSData *data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path.string().c_str()] options:0 error:&error];
    if (!data) { warning("cannot read profile path"); return {}; }
    id root = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    if (error || ![root isKindOfClass:[NSDictionary class]]) { warning("invalid JSON object"); return {}; }
    NSDictionary *document = root;
    uint32_t version = 0;
    if (!unsignedInteger(document[@"schema_version"], version) || version != 3) { warning("unsupported schema"); return {}; }
    NSString *fingerprint = document[@"weight_manifest_hash"];
    if (![fingerprint isKindOfClass:[NSString class]] ||
        ![fingerprint isEqualToString:[NSString stringWithUTF8String:std::string(weightManifestHash).c_str()]]) {
      warning("weight manifest hash mismatch"); return {};
    }
    NSString *identity = document[@"model_identity"];
    if (![identity isKindOfClass:[NSString class]] || ![identity isEqualToString:[NSString stringWithUTF8String:std::string(modelIdentity).c_str()]]) {
      warning("model identity mismatch"); return {};
    }
    NSArray *groups = document[@"groups"];
    if (![groups isKindOfClass:[NSArray class]]) { warning("groups must be an array"); return {}; }
    std::map<uint32_t, std::map<uint32_t, uint64_t>> counts;
    for (id item in groups) {
      if (![item isKindOfClass:[NSDictionary class]]) { warning("invalid group"); return {}; }
      NSDictionary *group = item;
      uint32_t block = 0;
      if (!unsignedInteger(group[@"block"], block) || ![group[@"phase"] isKindOfClass:[NSString class]] ||
          !([group[@"phase"] isEqualToString:@"prefill"] || [group[@"phase"] isEqualToString:@"decode"])) {
        warning("invalid block or phase"); return {};
      }
      NSArray *experts = group[@"experts"];
      if (![experts isKindOfClass:[NSArray class]]) { warning("experts must be an array"); return {}; }
      std::set<uint32_t> seen;
      for (id entry in experts) {
        if (![entry isKindOfClass:[NSDictionary class]]) { warning("invalid expert entry"); return {}; }
        NSDictionary *record = entry;
        uint32_t expert = 0, loads = 0;
        if (!unsignedInteger(record[@"expert"], expert) || !unsignedInteger(record[@"loads"], loads) || loads == 0 || !seen.insert(expert).second) {
          warning("invalid or duplicate expert entry"); return {};
        }
        counts[block][expert] += loads;
      }
    }
    ExpertPriorityHints result;
    for (const auto &[block, frequencies] : counts) {
      std::vector<std::pair<uint32_t, uint64_t>> ranked(frequencies.begin(), frequencies.end());
      std::sort(ranked.begin(), ranked.end(), [](const auto &a, const auto &b) { return a.second != b.second ? a.second > b.second : a.first < b.first; });
      auto &hints = result[block];
      for (size_t i = 0; i < std::min<size_t>(topK, ranked.size()); ++i) hints.push_back(ranked[i].first);
    }
    return result;
  }
}

} // namespace splash::model
