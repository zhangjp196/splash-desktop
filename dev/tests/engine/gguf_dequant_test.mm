// The staged GGUF tile's dequantization (dequant32, kernels/shared/gguf_linear.metal, which the test kernels of
// gguf_dequant_test.metal expose before the matmul): every weight of every format is the half rounding of its
// GGML fp32 value, the CPU reference that gguf-reference pins to llama.cpp's golden hashes.
#include "GgufFormatReference.hpp"
#include "KernelContract.hpp"
#include "TestBuffers.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Gguf.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <vector>

using namespace gguf_reference;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::test::ContractBuffer;
using splash::test::KernelContract;

namespace {

ContractBuffer upload(KernelContract &contract,
                      const std::vector<uint8_t> &bytes) {
  ContractBuffer buffer = contract.make(bytes.size(), true, false);
  std::memcpy(buffer.view.contents(), bytes.data(), bytes.size());
  return buffer;
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc != 2) {
      std::cerr << "usage: gguf-dequant <gguf-dequant.metallib>\n";
      return 2;
    }
    // One thread per row and group of 32 weights, 32 threads per threadgroup (gguf_dequant_test.metal).
    constexpr uint32_t N = 256, K = 1024, kGroup = 32, kThreads = 32;
    std::mt19937 rng(42);
    int failures = 0;
    try {
      MetalBackend backend(argv[1]);
      for (int fi = 0; fi < FMT_COUNT; ++fi) {
        const Fmt f = Fmt(fi);
        std::vector<float> values;
        const Packed planes = repack(f, makeNative(f, N, K, rng), N, K, &values);
        KernelContract contract(backend);
        const ContractBuffer w0 = upload(contract, planes.w0);
        const ContractBuffer meta = upload(contract, planes.meta);
        const ContractBuffer w1 =
            kQuantFormats[f].plane1_bytes ? upload(contract, planes.w1) : meta;
        const ContractBuffer output = contract.make(uint64_t{N} * K * 2, false, true);
        const std::array<ContractBuffer, 4> buffers{w0, w1, meta, output};
        const auto contractFailures = contract.check(
            std::span<const ContractBuffer>(buffers),
            [&](std::span<const ContractBuffer> bound) {
              CommandGraph graph;
              graph.add(std::string("gguf_test_dequant_") + fmtName(f),
                        {bound[0].view, bound[1].view, bound[2].view, bound[3].view},
                        GgufDecodeParams{K, 1, N, 0, 0},
                        {N * (K / kGroup) / kThreads, 1, 1}, {kThreads, 1, 1});
              static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
            });
        for (const std::string &why : contractFailures)
          std::cerr << fmtName(f) << ": contract: " << why << '\n';
        failures += !contractFailures.empty();
        const auto *got = static_cast<const uint16_t *>(output.view.contents());
        size_t differ = 0;
        for (size_t i = 0; i < values.size(); ++i) differ += got[i] != f2h(values[i]);
        std::cout << fmtName(f) << ": " << differ << " of " << values.size()
                  << " weights differ from FP16(GGML fp32 dequantization) " << (differ ? "FAIL" : "ok") << '\n';
        failures += differ != 0;
      }
    } catch (const std::exception &e) {
      std::cerr << "gguf-dequant: FAIL: " << e.what() << '\n';
      return 1;
    }
    return failures ? 1 : 0;
  }
}
