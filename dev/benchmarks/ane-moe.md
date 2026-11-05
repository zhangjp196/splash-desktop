# The Neural Engine in MoE targets: what can be optional

The prefill FFN split (`ops::AneFfn`) is dense-only: `model::aneFfnLayers()`
returns layers for `Qwen3_8Weights` alone, so every MoE family (Qwen3.6-35B-A3B,
Qwen3.5-122B-A10B, Qwen4-Exp, Flint) leaves the Neural Engine idle. This note
asks whether a MoE target can opt in, and answers it with the split's own
program on this Mac (40-core M3 Max, macOS 27.2) rather than with theory.

## The parts a MoE target has

| part | shape (35B) | in the MoE kernel | ANE candidate |
| --- | --- | --- | --- |
| routed experts, top 8 of 256 | 8 × 512 wide, weights stream or swap per chunk | yes, grouped tiles | no |
| shared expert | 512 wide (1024 on 122B, 2 × 512 on Flint) | yes, one more route per row | only for the dense shape |
| leading dense layers | none on Qwen; Flint's first 2, 6144 wide | no, dense path already | yes |
| attention mixers | 9216/12544 packed width | no | the mixer spike's target |

Routed experts are not an ANE problem: a 2048-row chunk routes about 64 rows to
each expert, the weights change every chunk, and the ANE runs one program's work
at a time with static shapes. The rest are fixed-shape GEMMs.

## Measured on the split's own program

`benchmark-ane-mixer` (`dev/benchmarks/ane_mixer_probe.mm`) runs three dense
layers through `ops::AneFfn` and through the GPU's dense path, 2048 rows,
IQ3_S weights:

| geometry | GPU alone | split | ANE's part |
| --- | ---: | ---: | ---: |
| 2048 × 6144, 6 of 12 units | 48.19 ms (9.62 TFLOPS) | 38.75 ms (1.24×) | ≥6.0 TFLOPS |
| 2048 × 1024, 1 of 2 units | 23.91 ms (3.23 TFLOPS) | 19.73 ms (1.21×) | ≥2.0 TFLOPS |

The 2048-hidden geometry compiles now that the program's segment is the largest
whole rotation block that divides the hidden size (2048 here, 2560 on the 27B),
instead of the fixed 2560. The ANE's rate collapses with the projection's width:
it is wide-GEMM hardware, and a 512-channel expert is far below the shape it
won 53% of on Qwen3.8-27B.

## What that means per option

- **Flint's two leading dense layers** are the only clean opt-in: they already
  use the dense path the split takes, and 1.24× on them saves about 13 ms of a
  2K prefill — around 1%, because 2 of 49 layers are dense. It needs an
  opt-in flag and a layer-prefix check, no kernel work.
- **The shared expert** should stay in the grouped kernel. On the 35B one MoE
  layer is 14.0 ms for its 9 routes, so the shared 512-wide route is about
  1.56 ms of it; a 1024-wide dense split layer costs 6.58 ms measured. Pulling
  the shared expert out to a dense split is therefore 2-4× slower than leaving
  it where it is, and the split's event order would make the ANE's evaluation
  block each layer's join instead of hiding under the routed GEMMs. A real
  version needs a 0-shared-slice MoE kernel, and still loses.
- **The mixers** are where an ANE split can move the needle on MoE targets too:
  QKV/O are 1.62 s of a 6.04 s 2K prefill on the 27B (`ane-mixer-spike.md`).
  That split is now built (`ops::AneProjection`, a row split) and measured at
  **1.2-1.4×** on an 8192-wide projection, so its end-to-end gain is ~4-6%,
  not the 13% the spike first projected; the ANE's full-width weight staging
  and the rotate/pack/output overhead eat the rest. It is still the open
  thread, not the MoE block.

So a `--ane-moe` flag can exist, but only as "the dense FFNs a MoE target still
has", which on today's families means Flint's two layers. It cannot put the
routed experts or the shared expert on the Neural Engine at a profit.
