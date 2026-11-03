# The mixer's projections on the Neural Engine: spike notes

Where the work stands, and what the next session starts with.

## Why (measured)

- The prefill FFN's split runs the ANE and the GPU 50/50 and takes the FFN's
  layer from 92.5 ms to 47.0 ms (**1.96×**) on an M3 Max with Qwen3.8-27B.
- The ANE's own rate on that shape is **≥12.4 TFLOPS** (37.1 TFLOP of
  evaluation over 2985 ms of `ane_ffn.ane_ms` in one 2048-row chunk), against
  the GPU's **11.8–12.1 TFLOPS** measured by `gguf-projection-benchmark` on the
  QKV, O, gate/up and down shapes across iq3s, q4k, iq4xs and bf16.
- The mixer GEMMs (QKV 8192×5120, O 5120×6144, 64 layers) take **1.62 s** of a
  6.04 s 2075-row prefill. Split like the FFN's, they would take about 0.85 s:
  **~13% of the prefill**.
- Everything else in that prefill: FFN 3.08 s, draft context 0.08 s, the rest
  mixers, attention, norm, KV writes and the split's pack/join.

So the upside is real and bounded, and the GPU side has no room: its kernels
sit at ~84% of the part's arithmetic peak, whatever the shape or format.

## Where the first attempt stopped

`dev/benchmarks/ane_projection_benchmark.mm` builds a single projection's MIL
by hand (int8 activations times int8 weights over the segments of K, summed
and scaled). The service **compiles** it — `ANECCompile` accepts the program
once the tensors are 4-D `[1, 1, H, W]` — and then **fails the evaluation**,
with no `NSError` at all (`Program::evaluationError()` is empty). It fails the
same at every stage: one matmul only, 8192×2560×128, with and without a weights
blob, with the event raised before and after the request, and with the inputs
bound in the function's declared order.

A failure the service reports without a reason points at the request it was
given, not at the arithmetic. The production path (`ops/AneFfn`) makes such
requests all day, so the next step reuses its program and request shape
instead of hand-writing a second one.

## Next session

1. Move the weight fixtures from `dev/tests/engine/ane_ffn_test.mm`
   (`blockModel`, `withLayers`, the `Projection` builder) into a shared
   header, and give them the mixer's geometry: `kHidden = 5120`,
   `kIntermediate = 8192`.
2. Build `ops::AneFfn` over three such layers at share 0.5 and evaluate one
   2048-row layer, timing it as `ane_ffn`'s test does.
3. Record ms and TFLOPS against the GPU's 12 TFLOPS on the same shape.
   **Stop if the ANE's rate is under 8 TFLOPS** or if a successful evaluation
   is not reached within the session's first hour: both mean this path does
   not pay for the weeks the split would take.
4. Only then design the split itself (one layer's QKV, GPU on the other half
   of its channels, the join between the ANE's output and attention's input).

## Constraints worth remembering

- The service runs one program's work at a time on one queue: a compile costs
  10–60 s cold and is cached under the source's hash.
- `--ane-share 0.65` already fails to compile for the FFN's program on an M3
  Max: the engine's 512-channel units are the split's granularity, and the
  ANE's compiler is the limit, not the memory plan.
- The production split stops itself when its evaluations take as long as the
  GPU alone's layers; a mixer split needs the same guard.
