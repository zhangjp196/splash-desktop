#pragma once

// A Metal kernel's buffer discipline, checked from a test: the same inputs
// produce the same output bytes on every run, inputs a kernel does not
// declare in-place stay unchanged, guard bytes around every buffer stay
// intact, and an input and an output that share memory are declared in-place.
// The contract is deliberately about memory and determinism; numerical
// agreement stays with each kernel's own reference test.

#include "metal/MetalBackend.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <vector>

namespace splash::test {

struct ContractBuffer final {
  splash::metal::MetalBuffer base;
  splash::metal::MetalBuffer view;
  uint64_t bytes = 0;
  bool input = false;
  bool output = false;
  // The kernel may write this input in place; its bytes need not survive.
  bool inPlace = false;
};

class KernelContract final {
public:
  static constexpr uint8_t kPoison = 0xA5;

  explicit KernelContract(splash::metal::MetalBackend &backend,
                          uint64_t guardBytes = 512)
      : backend_(backend), guard_(guardBytes) {}

  // A Shared buffer of `bytes`, with `guard_` poison bytes on each side; the
  // returned view is the middle. Callers fill the view before check().
  [[nodiscard]] ContractBuffer make(uint64_t bytes, bool input, bool output,
                                    bool inPlace = false) {
    ContractBuffer result;
    result.bytes = bytes;
    result.input = input;
    result.output = output;
    result.inPlace = inPlace;
    // Addressed: check() compares buffers' GPU addresses to find input and
    // output aliasing, which only an addressed buffer reports.
    result.base = backend_.allocateBuffer(
        bytes + 2 * guard_, splash::metal::BufferStorage::Shared,
        "kernel contract", splash::metal::BufferReach::Addressed);
    result.view = backend_.view(result.base, guard_, bytes);
    std::memset(result.base.contents(), kPoison, bytes + 2 * guard_);
    return result;
  }

  static void fill(const ContractBuffer &buffer, uint8_t value) {
    std::memset(buffer.view.contents(), value, buffer.bytes);
  }

  // Runs `encode(buffers)` `runs` times, restoring every input (in-place
  // ones too) before each run, and returns what broke; empty means the
  // contract held. `encode` submits and waits, as the test's other kernel
  // calls do.
  template <typename Encode>
  [[nodiscard]] std::vector<std::string>
  check(std::span<const ContractBuffer> buffers, Encode &&encode,
        uint32_t runs = 5) {
    std::vector<std::string> failures;
    if (runs < 2)
      runs = 2;
    for (const ContractBuffer &buffer : buffers) {
      if (buffer.view.contents() == nullptr) {
        failures.emplace_back("a contract buffer is not Shared");
        return failures;
      }
    }
    std::vector<std::vector<uint8_t>> inputs(buffers.size());
    std::vector<std::vector<uint8_t>> guards(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
      if (buffers[i].input)
        inputs[i] = bytesOf(buffers[i].view);
      guards[i] = guardBytes(buffers[i]);
    }
    // A buffer that is both an input and an output must say it writes in
    // place; one that only shares another buffer's memory is checked below.
    for (const ContractBuffer &buffer : buffers) {
      if (buffer.input && buffer.output && !buffer.inPlace)
        failures.emplace_back("input/output overlap without in-place");
    }
    for (size_t i = 0; i < buffers.size(); ++i) {
      for (size_t j = i + 1; j < buffers.size(); ++j) {
        const ContractBuffer &a = buffers[i];
        const ContractBuffer &b = buffers[j];
        if (!a.output && !b.output)
          continue;
        if (a.base.gpuAddress() != b.base.gpuAddress())
          continue;
        const int64_t offsetA = int64_t(a.view.gpuAddress() - a.base.gpuAddress());
        const int64_t offsetB = int64_t(b.view.gpuAddress() - b.base.gpuAddress());
        const bool overlap = offsetA < offsetB + int64_t(b.bytes) &&
                             offsetB < offsetA + int64_t(a.bytes);
        if (overlap && !(a.inPlace && b.inPlace))
          failures.emplace_back("input/output overlap without in-place");
      }
    }
    std::vector<std::vector<uint8_t>> firstOutputs(buffers.size());
    bool haveFirst = false;
    for (uint32_t run = 0; run < runs; ++run) {
      for (size_t i = 0; i < buffers.size(); ++i)
        if (buffers[i].input)
          write(buffers[i].view, inputs[i]);
      encode(buffers);
      for (size_t i = 0; i < buffers.size(); ++i) {
        const ContractBuffer &buffer = buffers[i];
        if (buffer.input && !buffer.inPlace &&
            bytesOf(buffer.view) != inputs[i])
          failures.emplace_back("input changed");
        if (buffer.output) {
          std::vector<uint8_t> now = bytesOf(buffer.view);
          if (!haveFirst)
            firstOutputs[i] = std::move(now);
          else if (now != firstOutputs[i])
            failures.emplace_back("output differs between runs");
        }
        if (guardBytes(buffer) != guards[i])
          failures.emplace_back("guard byte changed");
      }
      haveFirst = true;
    }
    std::sort(failures.begin(), failures.end());
    failures.erase(std::unique(failures.begin(), failures.end()),
                   failures.end());
    return failures;
  }

private:
  static std::vector<uint8_t> bytesOf(const splash::metal::MetalBuffer &buffer) {
    const auto *data = static_cast<const uint8_t *>(buffer.contents());
    return {data, data + buffer.sizeBytes()};
  }

  static void write(const splash::metal::MetalBuffer &buffer,
                    const std::vector<uint8_t> &bytes) {
    std::memcpy(buffer.contents(), bytes.data(), bytes.size());
  }

  std::vector<uint8_t> guardBytes(const ContractBuffer &buffer) const {
    const auto *data = static_cast<const uint8_t *>(buffer.base.contents());
    const uint64_t total = buffer.bytes + 2 * guard_;
    std::vector<uint8_t> guards;
    guards.insert(guards.end(), data, data + guard_);
    guards.insert(guards.end(), data + guard_ + buffer.bytes, data + total);
    return guards;
  }

  splash::metal::MetalBackend &backend_;
  uint64_t guard_;
};

} // namespace splash::test
