# Prefill on M3 Max and above

This note fixes the floor for prefill work at Apple9 (M3 Max) and records what
each engine is actually doing there. It is a measurement record, not a ranking
against other engines. The 27B and 35B figures below are the two target shapes
(`Qwen3.8-27B`, `Qwen3.6-35B-A3B`) on a 40-core M3 Max, macOS 26.4.

## Why INT8 is not the baseline

M3 Max is `MTLGPUFamilyApple9`: `mpp::tensor_ops::matmul2d` runs its portable
shader path, with no M5 neural accelerators and no int8 tensor path. An int8
GPU prefill kernel would be an M5-only fast path, not a floor. The GPU dense
and MoE prefill kernels therefore stay bf16 with fp32 accumulation, and the
exactness contract stays the one `gguf-projection` and `gguf-moe` check. The
existing dense FFN split already runs int8 on the Neural Engine (W8A8); this
note rules out a second, GPU int8 path, not the calibrated ANE one.

## Measured limits

`attention-sweep` (one prefill attention layer, 2048-row chunk, Page32 int8 KV,
median of 5):

| shape | history 0 | history 32768 |
| --- | ---: | ---: |
| 27B (24 Q / 4 KV, d=256) | 5.760 ms | 158.642 ms |
| 35B-A3B (16 Q / 2 KV, d=256) | 4.048 ms | 106.451 ms |

The per-layer work is affine in history for a fixed 2048-row chunk, so the
whole-prompt attention time is the sum over chunk starts. With 16 full-attention
layers (27B, every fourth layer of 64) and 10 (35B, every fourth of 40):

| prompt | 27B attention | 35B attention |
| --- | ---: | ---: |
| 2K | ~0.09 s | ~0.04 s |
| 32K | ~19.8 s | ~8.3 s |
| 128K | ~76 s | ~30 s |

`gguf-projection-benchmark` (Q4_K, 17408x5120, 2048 rows, one projection):
30.68 ms, i.e. ~11.9 TFLOPS. The attention kernels measure ~10.7 TFLOPS at
32K history, so attention is not an outlier: the whole bf16 prefill sits at the
same tensor-throughput ceiling.

`gguf-moe-benchmark` (35B shape, Q4_K/Q5_K experts, Q8_0 shared, 2048 rows,
40 MoE layers):

| expert tile | one layer | 40 layers | share of a 2K chunk |
| --- | ---: | ---: | ---: |
| register (Apple9 decode tile) | 14.0 ms | 560 ms | 50% |
| staged 32-row prefill tile | 11.7 ms | 469 ms | 42% |

The 2K chunk at 1,814 tok/s is 1.129 s, so MoE is ~47% of it. The staged tile's
16% gain is ~8% of the whole 35B prefill; the register tile is kept for decode
only because its 8-row form matches the dense decode path.

A 256-row prefill tile (eight simdgroups) was measured and rejected: one
Q4_K 2048x17408x5120 projection took 74.9 ms against 30.7 ms with the 128-row
tile on the same M3 Max. The 128-row tile lets two 128-thread threadgroups share
a core; 256 threads with one weight stage do not. Keep `GGUF_PREFILL_SIMDGROUPS`
at 4.

A 64-row staged expert tile was also measured and rejected: 14.16 ms per 35B
MoE layer against 11.29 ms for the 32-row tile. The larger accumulator loses
more than the halved expert-weight stream saves.

## Real-model profile (27B, 2048-token chunk)

`decode-profile` on the installed `mlx-community/Qwen3.8-27B-4bit` (fp32 state,
int8 KV, ANE split as the start calibrates it) attributes the fused prefill
command (8951 ms) across its 697 dispatches:

| pipeline group | GPU ms | share |
| --- | ---: | ---: |
| GGUF prefill projections (`gguf_prefill_af4g64_a/_r/_g`) | 8689 | **96.8%** |
| GDN scan | 145 | 1.6% |
| prefill attention (split + qkv + gate + reduce + store) | 107 | 1.2% |
| norms and the rest | 14 | 0.2% |

The projections move 110.6 TFLOP in 8.69 s, about 12.7 TFLOPS on the GPU
portion. The fused command and the sum of its dispatches differ by 0.35%, so
dispatch boundaries are not the cost. The Neural Engine is not a dispatch in
this table: on this M3 Max the start calibrates a split at **share 0.53**, and
one 2048-row FFN layer takes **49.5 ms against 91.7 ms on the GPU alone**
(calibrated in 48 s, `dense.split`). The ANE runs 53% of the FFN's channels
while the GPU runs the rest and the other projections, so the 96.8% above is
the GPU's share of the critical path, not the whole compute. GDN scan and
attention are still noise at 2K; chunking the scan or widening the attention
tile would each buy at most ~1%.

### The calibrated ANE share is not the end-to-end optimum

The calibration times one split FFN layer with the rest of the layer stood in
for; it does not see the GDN and attention projections the GPU still runs.
End-to-end on the real 27B, 2048-token prefill (`model-runtime-oracle`):

| `--ane-ffn-share` | prefill 2048 wall |
| ---: | ---: |
| 0 (GPU alone) | 11.32 s |
| 0.45 | 8.38 s |
| **0.53 (calibrated)** | **7.47–8.27 s** |
| **0.60** | **7.14–7.57 s** |
| 0.65 | ANE compiler refused the program |
| 0.70 | ANE compiler refused the program |

0.60 measured **0–10% faster than the calibrated 0.53** across repeated
2048-token prefills (7.14–7.57 s against 7.47–8.27 s) and **27–37% faster than
the GPU alone**. The spread is thermal and clock variation; this is not a
controlled fan/clock run, and one 0.53 repeat stalled in the ANE compiler. Treat
the share as a tuning lever to measure on the machine, not a guaranteed gain.
The calibration's per-layer model does not see the GPU's other work, so a share
a unit or two above the layer-time optimum can win end to end. Until the engine
probes shares on the real model, the flag is the lever:

```sh
splash serve --model ... --ane-ffn-share 0.60
```

The cost is KV context: the split's buffers come out of the attention cache,
so check the start's logged context against 0.53 before adopting it.

## C1, verified end to end

`dev/tests/smoke_real.py --persistent-cache` restarts a server in a fresh cache
directory and resumes the conversation from disk. On this machine:

- 100-item prompt: **4896/4909** prompt tokens reused after the restart
- 700-item prompt (`--persistent-cache-items 700`): **32800/32809** prompt
  tokens reused after the restart (the C1 acceptance: at least 32768 reused,
  no lost-state misses)

```sh
.venv/bin/python dev/tests/smoke_real.py \
  --model mlx-community/Qwen3.8-27B-4bit --model-root install/models/.selections/... \
  --persistent-cache --persistent-cache-items 700 --max-cache-disk 8G
```

The second run is a 32.8K prompt, i.e. the agent-sized case: first turn cold,
second turn served from the persistent tier.

## Derived composition

The 27B 2K row is now measured (above); the rest are estimated from the
attention sweep and model geometry.

| scenario | GEMM (FFN / projections) | attention | GDN scan |
| --- | ---: | ---: | ---: |
| 27B 2K | **96.8%** | 1.2% | 1.6% |
| 35B 2K | ~96% (47% of it MoE) | ~4% | small |
| 27B 32K | ~88% | ~11% | ~1% |
| 35B 32K | ~66% | ~33% | small |

Consequences for exact (no new quantization) work:

- 27B is a dense-GEMM problem and is already at ~12.7 TFLOPS on a 40-core M3
  Max. There is no 20% kernel win hiding in attention, GDN or dispatch overhead;
  256-row and 64-row tiles both measured slower.
- 35B is split between MoE and attention at 32K. The staged prefill tile is the
  first real kernel win; after that, MoE tile shape and attention are separate
  small levers.
- The cache wins by two orders of magnitude: 32K cold is ~2.5 min (27B) and
  ~20-25 s (35B) on this Mac, against ~0.3 s for a cached replay.

## Commands

```sh
build/engine-tests/attention-sweep build/splash.metallib
build/engine-tests/gguf-projection-benchmark build/splash.metallib q4k 17408 5120 none 3 prefill=2048
make benchmark-gguf-moe GGUF_MOE_ARGS='10'
```

`gguf-moe-benchmark` now prints prefill on both expert tiles: the device's plan
and the other tile, with the tile rows `moeGgufPrefillTile` gives it.
