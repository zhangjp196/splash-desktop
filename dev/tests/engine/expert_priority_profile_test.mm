#import <Foundation/Foundation.h>

#include "model/ExpertPriorityProfile.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

int main() {
  @autoreleasepool {
    const auto path = std::filesystem::temp_directory_path() /
                      ("expert-priority-profile-" + std::to_string(getpid()) + ".json");
    assert(splash::model::loadExpertPriorityProfile({}, "model", "hash-a").empty());
    {
      std::ofstream file(path);
      file << R"({"schema_version":3,"model_identity":"model","weight_manifest_hash":"hash-a","groups":[{"block":3,"phase":"prefill","experts":[{"expert":9,"loads":2},{"expert":4,"loads":2}]},{"block":3,"phase":"decode","experts":[{"expert":9,"loads":3}]}]})";
    }
    auto hints = splash::model::loadExpertPriorityProfile(path, "model", "hash-a", 1);
    assert(hints.size() == 1 && hints.at(3).size() == 1 && hints.at(3)[0] == 9);
    assert(splash::model::loadExpertPriorityProfile(path, "model", "hash-b").empty());
    assert(splash::model::loadExpertPriorityProfile(path, "other", "hash-a").empty());
    {
      std::ofstream file(path);
      file << R"({"schema_version":1,"model_identity":"model","groups":[]})";
    }
    assert(splash::model::loadExpertPriorityProfile(path, "model", "hash-a").empty());
    {
      std::ofstream file(path);
      file << R"({"schema_version":3,"model_identity":"model","weight_manifest_hash":"hash-a","groups":[{"block":0,"phase":"decode","experts":[{"expert":1,"loads":0}]}]})";
    }
    assert(splash::model::loadExpertPriorityProfile(path, "model", "hash-a").empty());
    std::filesystem::remove(path);
  }
}
