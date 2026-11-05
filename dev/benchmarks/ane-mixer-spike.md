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

## The split's program on the mixer's shape

`benchmark-ane-mixer` (`dev/benchmarks/ane_mixer_probe.mm`) runs three dense
FFN layers of the mixer's geometry through `ops::AneFfn` and the GPU's dense
path, 2048 rows, IQ3_S, the median of five commands:

| ANE units | GPU alone | split | ANE's rate |
| ---: | ---: | ---: | ---: |
| 4 of 16 | 131.0 ms | 99.2 ms (1.32×) | 12.05 TFLOPS |
| **8 of 16** | **129.3 ms** | **81.0 ms (1.60×)** | **12.41 TFLOPS** |
| 12 of 16 | 129.7 ms | 106.7 ms (1.21×) | 12.45 TFLOPS |

The rate is flat at about 12 TFLOPS; the split's optimum is the 50/50 share the
FFN's calibration also finds. The probe now warms up and takes medians, and the
ANE's own milliseconds come from `AneFfn::served()` rather than a bound.

## A single projection's program

`dev/benchmarks/ane_projection_benchmark.mm` builds a projection's MIL by hand:
int8 activations times int8 weights over the segments of K, dequantized by
`1 / ANE_FFN_INT8_UNIT`, summed, scaled by the weights' row scales and the
tokens' scales. It failed every evaluation until it bound its surfaces by
`Program::inputs(procedure)` — the **compiled** function's input order, which
is not the MIL's declaration order. `ops::AneFfn` already binds that way; the
hand-written program did not.

With the binding fixed, at 8192×5120×2048 rows and 12.6 ms an evaluation
(`ane-projection-benchmark build/splash.metallib 8192 5120 2048 3`):

- **13.63 TFLOPS** on the ANE, against the GPU's ~12 on the same shape;
- **1.31%** from the CPU's int8 reference (the FFN split's error, 1.9–2.3%,
  is the same order).

The spike's stop line was 8 TFLOPS, so the projection path is worth building.
It is also simpler than the FFN's: a projection's split needs no silu, no
rotation of intermediate rows and no cross-part sum.

## The split, built and measured

`ops::AneProjection` (and `ane-projection-split`, its test) splits one
projection by **row**: the GPU runs a chunk's leading rows with the
projection's prefill kernel, the ANE its trailing rows as one W8A8 program,
each the projection's whole output width. A column split was measured first
and rejected: at 50% it took 1.0–1.1x, because the GPU's leading half loses
much of its rate at half the columns while the ANE loses its at half the
rows' width. Rows keep both widths whole.

- **Rows**: `AneProjection::aneRows(rows)` takes the share's rows, rounded up
  to the 128-row function step and clamped to leave the GPU its least rows.
  The GPU writes rows `[0, gpuRows)` into the same bf16 output; the ANE's
  `ane_projection_out` kernel writes `[gpuRows, rows)`.
- **Program**: the projection MIL from `ane_projection_benchmark.mm`, without
  silu, intermediate rotation or a cross-part sum. Bound by the compiled
  input names (the benchmark's bug was binding by declaration order).
- **Weights**: `ane_ffn_weights_gguf_inputs` stages the **whole** projection
  per layer, double-buffered, one layer ahead; a row split needs every output
  column the ANE's rows read. A fused projection (the 27B's QKV is
  `attn-q`/`attn-k`/`attn-v`) stages each segment into its rows' offset of the
  ANE's full-width surfaces; the test exercises a two-segment projection
  (1.58% error).
- **Residual**: an attention output projection's rows need the layer's input
  added, which `ane_projection_out_residual` does (1.51% error in the test).
- **Ordering**: the FFN split's shared-event handoff; the GPU part is encoded
  between the ANE's `ready` signal and its `done` wait.

Measured on an M3 Max, IQ3_S, 8192×5120, 2048 rows, two layers, medians of
three:

| share | rows GPU / ANE | GPU alone | split | speedup |
| ---: | ---: | ---: | ---: | ---: |
| 0.35 | 1280 / 768 | 29.21 ms | 38.37 ms | 0.76× |
| 0.50 | 1024 / 1024 | 30.18 ms | 30.78 ms | 0.98× |
| **0.65** | **640 / 1408** | **29-35 ms** | **24.98 ms** | **1.2-1.4×** |

The ANE's rows are 1.58% from the GPU's whole projection, fused segments and
a residual alike. The 27B's mixer widths are 14336 (full-attention QKV) and
16640 (GDN input), so the runtime needs one program per width. The gain is smaller
than the FFN's ~2×, and the reason is visible in the parts: the ANE half
costs 21.4 ms against the GPU half's 9.5 ms at the 0.65 share, because the
ANE stages the **full** 8192-row int8 weights every layer, rotates and packs
the input, and converts the output, against a projection that takes only
~15 ms on the GPU alone. The FFN's layer is 92 ms, so the same overheads are
a few percent of it. The GPU-alone baseline itself moves 29-35 ms run to run
(thermal), so the ratio is 1.2-1.4× rather than a single number.

So the mixer split as built pays about 16% on a QKV-shaped projection, not
the ~50% the spike hoped. The projected end-to-end gain is ~4–5% of a 2K
prefill, not 13%.

Things that could change that, in order of expected value:

1. Stage the ANE's weights once per layer for the whole prompt and reuse
   them across chunks (they do not change), rather than once per command.
2. Split a **pair** of projections per layer (QKV and O) through one program
   so the rotate/pack and the staging amortize over twice the work.
3. Fuse the ANE's output conversion into attention's QKV preparation.
4. A fused QKV projection (Q/K/V segments) needs per-segment staging and a
   concatenated MIL output; `unsupported` refuses it today.

## Constraints worth remembering

- The service runs one program's work at a time on one queue: a compile costs
  10–60 s cold and is cached under the source's hash.
- `--ane-share 0.65` already fails to compile for the FFN's program on an M3
  Max: the engine's 512-channel units are the split's granularity, and the
  ANE's compiler is the limit, not the memory plan.
- The production split stops itself when its evaluations take as long as the
  GPU alone's layers; a mixer split needs the same guard.
- Bind program inputs by their compiled names, never by declaration order.

## On the real 27B

Qwen3.8-27B-GSQ-RCO-IQ3_S (GGUF), M3 Max 40-core, a 3641-token completion with
`max_tokens 1`, the FFN split at share 0.53 and minimum 903 rows in both
runs, context 256K:

| `--ane-mixer` | wall | prefill tokens/s |
| --- | ---: | ---: |
| off | 11.01, 11.36, 11.19, 11.36 s | 336.0, 331.2, 328.1, 326.5 |
| **on** | **10.88, 10.90, 10.80, 10.79 s** | **341.3, 339.5, 339.5, 339.4** |

The on runs held 340+ t/s immediately after four off runs had heated the
machine and the off rate had fallen to 326 t/s: part of the gain is the GPU
doing less work and throttling less. The startup log confirms the split ran:
`Neural Engine mixer split at share 0.65 over 16 projections, 229 MiB.` The
first start compiles the 14336x5120 program (15-25 s); later starts load it.

A long prompt shows more. 23,801 (off) and 26,601 (on) tokens, FFN split on
both, the on run taken after the off run had heated the machine:

| run | tokens | wall | prefill tokens/s |
| --- | ---: | ---: | ---: |
| off | 23,801 | 87.9, 97.5 s | 271.6, 257.6 |
| on (QKV+O) | 23,801 | 82.3, 89.0 s | 296.8, 283.0 |
| on (QKV+O, hot) | 26,601 | 91.0, 94.7 s | 293.9, 287.9 |

The on rate is 4-12% above the off rate, and the last on run followed the off
runs on a hot machine, so the gain is not thermal ordering. The FFN split
calibrated to 0.53/903 rows on the on starts and 0.56/1281 on the off one;
the lower share is not an advantage to the on side.

So the wired QKV+O split is worth about **3-5% of a 3.6K prefill and 8-12% of
a 24-27K one**, mostly because the ANE takes work off a throttling GPU.

## The runtime, wired

A start builds up to three `ops::AneProjection`s at startup over a Qwen3.8
target's full-attention QKV (14336x5120, 16 layers), its output projections
(5120x6144, 64 layers, with the residual), and its GDN input projections
(16640x5120, 48 layers), on by default; `--ane-mixer off` (the server's
`--disable-ane-mixer`) leaves them on the GPU. Each reserves its `plannedBytes` beside
the FFN split's and runs in `QwenTarget::addPrefillMixer`; the GPU takes the
chunk's leading rows and the ANE its trailing ones. `Runtime::prefill`
begins and finishes the splits, and a failure reruns the chunk on the GPU
alone. The startup log names each: QKV takes 229 MiB, the output split 118
MiB. The GDN split takes the projection's leading quantized columns (16384:
10240@0 and 6144@10240) on the ANE and leaves the 96 float alpha/beta columns
(at 16384) and the 160 of padding to the caller: `QwenTarget::addPrefillMixer`
computes the float tail for the ANE's rows on the GPU (padded to 128 rows for
the linear tile) and copies its 96 columns into the packed buffer with
`RowCopy`. It verifies, and a greedy 32-token completion with all three splits
was identical to the GPU's.

It is still **not served while the FFN split runs**: the ANE service is one
queue, and with the FFN split active the three mixer splits measured slower
than QKV and the output split alone (23.8K tokens: 284.7 and 261.5 t/s against
296.8 and 283.0), so `RuntimeResources` skips the GDN split whenever the FFN
split is set up and logs `GDN mixer split off: the FFN split already uses the
Neural Engine`. With the FFN split off, all three build and pass verify; their
rate was not better than QKV+O alone (232.6/223.1 t/s against 236.6/223.6).
Each split is verified at startup against the GPU's whole projection of the
same 256 rows: a split whose ANE rows differ by more than 5% (the int8 path
leaves about 1.5%) is dropped, as the FFN split's own check does. The FFN
start also falls back to a fixed 0.5 share when its calibration's timed
programs fail to compile (0.8 does on this M3 Max), so enabling the mixer
splits cannot cost the FFN split on a fresh calibration.

## Next session

1. Profile `ane_ffn_weights_gguf_inputs` alone on the QKV shape: how much of
   the 21.4 ms ANE half is staging rather than evaluation.
2. If staging dominates, keep each layer's ANE weights resident for a prompt
   (the surfaces are per set, so this needs one set per layer or a ring that
   is rewritten once per prompt) and measure again.
3. Only then wire `ops::AneProjection` into `QwenTarget::addPrefillMixer` for
   the full-attention QKV, with a calibration of the share and the FFN
   split's breaker.
