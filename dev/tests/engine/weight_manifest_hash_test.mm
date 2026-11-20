#include "TestChecks.hpp"
#include "TestFiles.hpp"
#include "model/WeightManifestHash.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace splash::model;
using splash::test::require;
namespace {
void rejects(const auto &call, const std::string &message, const std::string &failure) {
  try {
    call();
  } catch (const std::exception &error) {
    require(std::string(error.what()).find(message) != std::string::npos, failure + ": wrong error");
    return;
  }
  throw std::runtime_error(failure);
}
}

int main() {
  try {
    const splash::test::TemporaryDirectory directory("splash-weight-manifest");
    const auto root = directory.path();
    const auto first = root / "first.bin";
    const auto second = root / "second.bin";
    splash::test::writeFile(first, "abc");
    splash::test::writeFile(second, "de");

    const std::vector<WeightManifestEntry> manifest{{"weights/a", first}, {"weights/b", second}};
    const auto baseline = sha256WeightManifest(manifest);
    require(baseline == sha256WeightManifest({{"weights/b", second}, {"weights/a", first}}),
            "manifest order changed hash");

    const auto elsewhere = root / "elsewhere";
    std::filesystem::create_directory(elsewhere);
    splash::test::writeFile(elsewhere / "one", "abc");
    splash::test::writeFile(elsewhere / "two", "de");
    require(baseline == sha256WeightManifest({{"weights/a", elsewhere / "one"},
                                               {"weights/b", elsewhere / "two"}}),
            "filesystem paths affected hash");

    splash::test::writeFile(first, "abC");
    require(baseline != sha256WeightManifest(manifest), "content byte change did not change hash");
    splash::test::writeFile(first, "ab");
    splash::test::writeFile(second, "cde");
    require(baseline != sha256WeightManifest(manifest), "file boundary change did not change hash");
    splash::test::writeFile(first, "abc");
    splash::test::writeFile(second, "de");
    require(baseline != sha256WeightManifest({{"weights/x", first}, {"weights/b", second}}),
            "logical name change did not change hash");

    rejects([] { (void)sha256WeightManifest({}); }, "weight manifest is empty", "empty manifest accepted");
    rejects([&] { (void)sha256WeightManifest({{"same", first}, {"same", second}}); },
            "duplicate weight manifest logical name", "duplicate name accepted");
    rejects([&] { (void)sha256WeightManifest({{"missing", root / "missing"}}); },
            "cannot read weight file", "unreadable file accepted");
    std::cout << "weight manifest hash: streaming, identity, boundaries and errors PASS\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
