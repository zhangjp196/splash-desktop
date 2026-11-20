# Validate

```sh
make check
make install test-real test-http-real MODEL=mlx-community/Qwen3.8-27B-4bit
```

`make check` needs no model weights, but it runs the Metal tests, so it needs a
Mac that Splash supports: it runs the source checks, `check-native-cpu`,
`check-native-metal`, `check-native-build` and the Python checks. `make
check-native-cpu` builds production and runs native CPU tests without a GPU;
`make check-native-metal` requires a supported Metal device and runs the kernel
tests under shader validation, and the Linear pipeline resource check without
it. They include loading small synthetic MLX, GGUF and vision sources and the
GGUF kernels on synthetic tensors. `make check-native-build` builds every
native test, benchmark and tool and runs none. Hosted CI runs the CPU checks,
`check-native-build`, the sanitizers and the Python tests. `make
test-sanitizers-metal` runs the Metal tests of the components whose host steps
share their state with the engine thread and the SSD readers (the hot block
cache of spilled KV and the history stream) under ThreadSanitizer, built with
it in `build/metal-tsan`. `make architecture-check`, one of the source checks,
also fails when a workflow under `.github/workflows` names a Hugging Face token
(`HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` or a `secrets.HF_*` secret): CI installs
public models alone and never receives one.

The native tests are in `dev/tests/engine`; each Python test is in the
directory of what it tests:
`dev/tests/server`, `dev/tests/install` (launcher, client configuration and
model installation) or `dev/tests/tools` (`dev/tools`, `dev/benchmarks`, the
Makefiles and the real-model harnesses).

The real-model targets run against an installation in this checkout's
`install/models`: run `make install` with the same options first, or in the
same command, as above. They take `MODEL` exactly as `splash serve --model`
does, and `REVISION`, `DRAFT_MODEL` and `LANGUAGE_ONLY=1` as its `--revision`,
`--draft-model` and `--language-only`:

| Target | Runs |
| --- | --- |
| `verify-models` | the installer's restarts without the Hub, `verify --full`, and the record of the weight images (`dev/tools/installer_restarts.py`, `weight-digests` with `WEIGHT_DIGESTS_ARGS` added, such as `WEIGHT_DIGESTS_ARGS="--expert-memory 7G --ple-memory 2G"` to stream the routed experts and the PLE table, [Release check](#release-check)) |
| `test-real` | vision parity with the family's fixture in `dev/tests/fixtures/vision-parity/` when the installation serves vision, and the native model runtime oracle, each of its three runs with `ORACLE_ARGS` added (`ORACLE_ARGS="--kv-format bf16"`, `"--expert-memory 16G"` or `"--max-context 1010000"`); for a block-sparse target their spill checks take the default prompts, two staging buffers, and 32K tokens with the default window |
| `test-http-real` | the HTTP frontend on an isolated server (`dev/tests/smoke_real.py`), whose greedy comparisons (a penalized greedy request repeats once its prompt is cached, `min_p` 1 answers as greedy) run where `/status` reports `verify_rows` `all`, which on any MoE target takes `HTTP_SMOKE_ARGS="--verify-rows all"`; where `/status` reports `reasoning_switch` `false` (Flint), it gives each request whose answer it checks room for the model's reasoning and leaves out the thinking budget's check; with `HTTP_SMOKE_ARGS="--persistent-cache --max-cache-disk 8G"`, instead of that smoke, the [persistent cache](server.md#persistent-cache) across a restart in a fresh cache directory: a clean stop, a restart that takes the restore point back, and a next turn that restores the prompt from disk; `release-check` runs both |
| `test-agent-real` | the five official clients through `splash serve` (`dev/tests/agent_real.py`), in `AGENT_SCENARIO` `complete` (the default) or `smoke` |
| `test-release-real` | the HTTP smoke and all five clients on one `splash serve` |
| `test-performance-real` | the native decode, partial-prefix and short-prompt benchmark, or with `BASELINE` its ABBA comparison with that build, in which both builds' `weight-digests` take `WEIGHT_DIGESTS_ARGS` (`dev/benchmarks/backend_regression.py`) |
| `test-pressure-real` | the installation on this checkout's own server under `PRESSURE_ARGS` (default `--max-memory 18G`), which must make it stream its routed experts, with the [serving benchmark](#serving-benchmark)'s three churn clients for `PRESSURE_MINUTES` (default 5), checked on its own by [`check.py run`](#serving-benchmark) on G4, G8, G9, Store, Churn (its load window and every client served) and First token; results in `build/release/<model>/pressure` |
| `release-check` | one model on this Mac ([Release check](#release-check)) |

`test-agent-real` and `test-release-real` first run the tests
`dev/tests/install/test_clients.py` has of the installed OpenCode, Codex and Pi
among the selected clients, which need no model.

`test-agent-real` runs Hermes in a profile of its own in the developer's Hermes
root, `splash-test-<id>`, which moves into the run's folder under
`build/release` when Hermes finishes.

Each phase's record keeps what its requests reused of the cache (`reuse`).
Every request after a phase's first resends the conversation, so a phase that
completed two or more requests and reused no cached prompt token fails. Two
phases are exempt: the cancellation phase (`cancel`), and the reference wave
that compacts the conversation, whose summary request and the request after it
share only the system prompt and tools. Replay points of unfinished requests
evicted during a phase only print a warning.

`benchmark-backend` and `benchmark-decode-profile` take `MODEL` the same way.
The models they are run with, one per family and source format:

| Family | MLX | GGUF |
| --- | --- | --- |
| Qwen3.8-27B | `mlx-community/Qwen3.8-27B-4bit` | `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M` |
| Qwen3.6-35B-A3B | `mlx-community/Qwen3.6-35B-A3B-4bit` | `unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M` |

Both source formats load into the block layout of the GGUF projection and MoE
kernels, an MLX target in its MLX formats, a GGUF target in its GGUF ones.

On a 24 GB Mac, `test-agent-real` stops a client's workflow at macOS's warning
memory pressure, which the smaller GGUF variants such a Mac uses can reach under
an agent's load; `SPLASH_TEST_PRESSURE_STOP=4` stops only at critical pressure,
to observe how the engine sheds its cache. The runtime oracle in `test-real` has
no production memory guard: the weights must fit in what macOS has available
above its reserve. When they do not, it measures again every second for up to
30 s before it stops, as a start waits for memory macOS refuses it: macOS can
return the memory of a process that has just exited a moment later (on a 36 GB
M5 Max, a server engine's memory came back within a second of its exit).
Once it has loaded, its checks can neither wait for memory nor yield it, so the
governor a start builds admits all the runtime then allocates, its arenas and
the lanes, KV and cached states of the requests the oracle runs together, as
memory a request in service needs: only the memory plan's budget and macOS's
critical pressure refuse it, and the run stops at critical pressure, naming
what it asked for. Production grants that only to its start, a request it
serves alone and the KV of the requests it serves; a start beside a resident
request, and the cache, are held to a combined reserve of 7% of physical
memory, including the 1 GiB margin above the host spare reserve, and macOS's
warning. A Neural Engine split leaves the plan room for what the
checks hold at once (every lane's state, six cached states and the oracle's KV
pages), as a start's split leaves `--max-context`, so the oracle's split may
take fewer of the Neural Engine's units than a server's does. With only desktop
applications open, a 24 GB Mac runs it for those variants.

The engine's `--kv-spill always`, which `splash serve` never passes, keeps
the KV of every request of a block-sparse target on the SSD cache
([KV on the SSD](server.md#kv-on-the-ssd)), so that a run of `build/splash
serve-native` with it compares spilled requests' outputs with those of a run
without it, which must match bit for bit.

Compare performance on the same idle Mac with the same model and workload.
Keep generated reports, profiles, local paths and experiment notes out of the
source tree and commits.

## Correctness coverage and run evidence

This matrix separates what the repository can exercise from what a dated run
record proves. A test, model list, or configured CI runner establishes coverage
support/configuration only. A hardware entry below counts as run evidence only
where the repository records the device, date, exercised combination, and
result. “Not recorded” means this checkout contains no such evidence for that
cell; do not infer a pass from a test target or workflow configuration.

| Dimension | Test support / configured coverage | Recorded real-device evidence | Status / gap |
| --- | --- | --- | --- |
| Hardware generation | The manual `release-hardware` workflow configures self-hosted M3 Max, M4 Pro/Max, and M5 Pro/Max runners; it schedules the Metal suite and release models. This is configuration, not a run receipt. | [KV-format validation](../dev/benchmarks/kv-formats.md#validation-on-2026-09-21) records M3 Max and M5 Pro runs; [combined serving validation](../dev/benchmarks/kv-formats.md#combined-serving-validation-with-pr-92) records M5 Pro 20 and M5 Pro 16. | M3 and M5 Pro have dated records for the listed work. M4 and other configured runner variants: not recorded here. |
| Model family / weight source | Release check lists Qwen3.8 27B and Qwen3.6 35B, each MLX and GGUF; GGUF 27B is language-only. `test-real` runs the configured model runtime oracle. | The 2026-09-21 record covers 27B/35B on M3 Max and M5 Pro; combined serving records 27B/35B in MLX/GGUF combinations on M5 Pro 20. | Evidence applies only to those recorded revisions, devices, and combinations; the workflow's current pinned revisions are configuration until artifacts/logs are retained and linked. |
| Quantization / weight format | Tests and release commands cover MLX 4-bit and GGUF UD-Q4_K_M. Unit/reference suites cover supported GGUF decoding formats; those do not establish end-to-end model quality. | The combined serving record covers named 27B/35B model builds; [upstream loading validation](../dev/benchmarks/upstream-loading.md) records particular GGUF reference checks. | No general real-device matrix across every weight quantization is recorded. K8V4 specifically still lacks a real-model logit comparison, as noted in its dated record. |
| KV format | Oracle and HTTP smoke accept `int8`, `int4`, `bf16`; tools also accept `k8v4`. `ORACLE_ARGS` selects KV format. | The 2026-09-21 record reports both 27B/35B models in INT8/BF16 on M3 Max and M5 Pro; the combined record reports four 27B/35B INT8/BF16 combinations on M5 Pro. The 2026-11-05 K8V4 record is explicitly kernel/geometry only, with no installed model. | INT4 and K8V4 end-to-end real-model correctness: not recorded in these release notes. K8V4 kernel coverage must not be described as model-runtime validation. |
| KV / recurrent state | Runtime oracle checks restore, prefill partitioning, preemption, batching, and state-related cycle behavior; CPU and kernel references cover individual layouts/operations. | The 2026-09-21 oracle runs record state restore and the listed runtime behaviors. The state-oracle investigation documents a focused recurrence/cycle check and an older whole-network BF16 cosine failure (0.984408 vs 0.99 on M5). | Record the exact oracle scope; do not imply the focused checks prove a universal whole-network state bound. |
| Context length / paths | Oracle supports `--max-context`; default oracle includes partitioned prefill and request paths. Spill checks exercise block-sparse targets; HTTP smoke and persistent-cache options cover serving/cache paths. | The 2026-09-21 record reports native 256K runs for 35B in both formats on both devices and 27B in both formats on both devices, including exact-prefix replay. Combined serving records HTTP feature paths; the K8V4 record covers kernel histories, not model generation. | Evidence is specific to the recorded synthetic prompt/replay and configurations. Other model/format/device/context combinations: not recorded. |
| Release workflow | `.github/workflows/ci.yml` defines the manually dispatched hardware matrix, model revisions, and artifact upload path. `release-check` compares against a retained baseline and invokes `test-real` and HTTP checks. | A workflow definition or successful build is not evidence by itself. The dated benchmark notes above are the repository's run records for their explicitly named cases. | For a release claim, retain and link the run artifact or report for each claimed matrix cell. |

### Minimum run registration

For each real-device correctness result, register at least:

- Run date/time and outcome, with the report/log or artifact link (including the
  workflow run URL when applicable).
- Repository commit and model revision; model family/size and source/weight
  format, including quantization.
- Device identifier and chip generation, RAM/GPU configuration, macOS version,
  and whether the run was on AC when known.
- Runtime path and test command/target, including relevant arguments and
  environment overrides (for example `ORACLE_ARGS`, KV/state format, maximum
  context, and language-only mode).
- Scope actually exercised and result: pass/fail, skipped/not applicable
  reason, and any known limitation. Keep separate rows for distinct formats,
  models, or devices rather than broadening one result to an unrun cell.

Update this matrix only from retained evidence: after a run, add or link its
record under the matching cell and state the exact combinations it covers. If a
workflow was configured but no artifact or result can be located, keep that cell
“not recorded / needs registration.” Do not convert configuration, source-code
support, or a kernel-only check into a real-model pass.

## Release check

A release is checked once per source identity, and then on each Apple GPU
family (an Apple9 M3 and an Apple10 M5) against a retained baseline build,
`BASELINE`: a checkout whose `build/` holds `splash`, `splash.metallib`,
`engine-tests/backend-benchmark` and, for a build that loads the weights into
memory (1.2.0 and later), `engine-tests/weight-digests`. `release-check` fails
without it. The baseline must load the model: it is the previous release's
build when that loads the model. From a clean checkout:

```sh
make check test-sanitizers                      # once, model-free, on one Mac
make check-native-metal                         # on the other Mac
make install release-check MODEL=mlx-community/Qwen3.8-27B-4bit REVISION=<commit> BASELINE=../splash-baseline
make install release-check MODEL=unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M LANGUAGE_ONLY=1 REVISION=<commit> BASELINE=...
make install release-check MODEL=mlx-community/Qwen3.6-35B-A3B-4bit REVISION=<commit> BASELINE=...
make install release-check MODEL=unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M REVISION=<commit> BASELINE=...
make install test-pressure-real MODEL=unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M REVISION=<commit> PRESSURE_ARGS="--max-memory 18G"
make install verify-models MODEL=mlx-community/Qwen3.6-35B-A3B-4bit
make test-agent-real MODEL=mlx-community/Qwen3.6-35B-A3B-4bit REVISION=<commit> AGENT_SCENARIO=smoke AGENT_CLIENTS=...
```

Pin each upstream model to one commit, the same on both Macs. Without a Hub
token for a private draft repository, set `DRAFT_MODEL` to a local copy of the
draft's checkpoint ([Drafts](models.md#drafts)). The Metal suite depends on the GPU
family, so it runs once on each Mac: within `make check` on the first, alone on
the other. Per model, `release-check`:

- runs the runtime oracle and, when the installation serves vision, vision
  parity (`test-real`);
- restarts the installer offline, with the Hub unreachable
  (`HF_ENDPOINT=http://127.0.0.1:9`) and with an empty `HF_HUB_CACHE`: each
  restart must start the same assembly within 10 seconds, name the Hub's
  reason on one line when it asked the Hub ([Revisions](server.md#revisions)), and
  download nothing. It then hashes the
  sources and records in `weights.json` the component, size and SHA-256 of
  every weight image the installation loads (`verify-models`);
- runs the HTTP smoke, which for a text-only installation checks the 400s
  instead of images (`test-http-real`), on the default verify rows: on a MoE
  target it leaves out the greedy comparisons, which run only with
  `HTTP_SMOKE_ARGS="--verify-rows all"`, and says so;
- stops and restarts a server with a [persistent cache](server.md#persistent-cache),
  and requires the conversation's next turn to restore its prompt from disk
  (`test-http-real` with `HTTP_SMOKE_ARGS="--persistent-cache --max-cache-disk 8G"`);
- compares this build with `BASELINE`, which must have another build
  identity, in ABBA order (`test-performance-real`), every round after the
  first to report a [Neural Engine share](internals.md#neural-engine-prefill) running that
  share when its build takes one: output tokens and acceptance must be
  identical (`EXPECT_OUTPUT_CHANGE=1` allows changed outputs with acceptance
  within 0.02; the benchmark verifies every row (`--verify-rows all`), so a
  MoE target's outputs repeat and its times measure the backend, not the
  verify budget of [Drafts](models.md#drafts)), and so must the bytes of the weight
  images both load (`dev/benchmarks/weights.py`); decode and prefill GPU time,
  with the cold prefills of short prompts (480 to 2016 tokens, one chunk each)
  when both
  builds' benchmarks run them, may regress by at most the larger of 2% and
  twice the run's own ABBA spread, a spread above 5% fails as inconclusive,
  and this build's
  memory plan must hold, without the Neural Engine split, at least the
  context the baseline's holds without it. `ANE_FFN_SHARE=SHARE` runs
  every round at that share instead, in each build whose benchmark takes one
  (0: the GPU alone); a baseline whose benchmark takes none must report none.

Where the Mac's memory cannot hold a model's weight images, or its largest
Metal buffer cannot hold one of them, as for Qwen3.8-Flash-Next,
`WEIGHT_DIGESTS_ARGS` streams its routed experts and PLE table in the loads
whose images are recorded and compared ([streamed experts](server.md#streamed-experts),
[PLE table streaming](server.md#ple-table-streaming)): it gives `weight-digests` the
`--expert-memory SIZE` and `--ple-memory SIZE` that `ORACLE_ARGS` gives the
runtime oracle, each SIZE a whole number of bytes or of `K`, `M` or `G`, such
as `7G`, in `verify-models` and in both builds of the comparison with
`BASELINE`, whose `weight-digests` must take them. The layers' images then
leave out their routed experts, and a PLE table that streams has no image
(`ple.bin`): the engine reads those bytes from the source files, which
`verify --full` hashes, and the digests do not cover them. Any size a store
takes records the same images. Where the images and stores of a load exceed
the engine's Metal memory budget, `weight-digests` refuses it before loading
any image and names the options.

Results go to `build/release/<owner>--<repo>[--VARIANT]/`. The weight images
do not depend on the GPU, so each model's `weights.json`, recorded with the
same `WEIGHT_DIGESTS_ARGS`, must be identical on the two Macs. The unpinned
`verify-models`, run after the pinned ones while the default branch still
names the pinned commit, resolves the branch online, and its unreachable-Hub
restart must fall back with the Hub's reason. The agent
clients depend on neither the model's format nor the GPU: run the smoke
scenario once per Mac, with the five clients split between the Macs, and
`AGENT_SCENARIO=complete` for one model when the client integration changed.
Expect about 1.5 hours on an M5 Pro and 2.5 hours on an M3 Max, most of it in
the three 27B comparisons. A laptop can cap its GPU power during a long
comparison and so make it inconclusive; rerun `make test-performance-real` for
that model alone once the Mac has cooled.

`.venv/bin/python dev/tools/kernel_identity.py BASELINE/build build` compares
the AIR of each kernel the decode path may run in the two builds'
`splash.metallib` (`--match` for others) and fails if any differs. The helpers
a kernel's module defines are compared in the order they appear there, not by
name: C++ names them for their template values, which a kernel template that
gains a value renames without changing their code. So are the numbers of its
lambdas' anonymous classes and its classes' names in the alias metadata, which
the same change renames.

Two pairs of constants in `runtime/engine/Protocol.hpp` and `server/protocol.py`
version what the server and the engine exchange. When the native wire layout
changed since the last release, bump `kProtocolVersion` and `PROTOCOL_VERSION`
together; each side refuses frames of another version. When a field of the
status document the engine writes was removed or changed meaning, bump
`kStatusSchemaVersion` and `STATUS_SCHEMA_VERSION` together; the server refuses
a status document of another schema. An added field keeps the version, as
`weights` did in 1.2.1 and `ane_ffn` in 1.3.0. Builds between releases share a
version while its layout changes.

## Local benchmarks

From a source checkout with the model installed, use the native benchmark for
prefill, decode and batch measurements:

```sh
make test-performance-real MODEL=mlx-community/Qwen3.8-27B-4bit
make test-performance-real MODEL=mlx-community/Qwen3.8-27B-4bit BASELINE=/path/to/baseline
```

The first characterizes this build: the decode widths B1-B4, a 14,096-token
partial-prefix request and the cold prefills of short prompts, three samples
each, in
`build/release/<owner>--<repo>[--VARIANT]/backend-benchmark.json`;
`make benchmark-backend MODEL=...` adds the 2K to 128K contexts the memory
plan holds. The second compares this build with a retained checkout's in ABBA
order, as the release check does ([Release check](#release-check)), and writes
`backend-regression.json` there. Neither is a comparison with another engine
or a test of agent task quality.

A change of arithmetic that moves outputs (`EXPECT_OUTPUT_CHANGE=1`) is
measured by its next-token distributions instead. The benchmark's `logits`
scenario dumps the full-vocabulary logits after 16 prefixes of each of the six
transcript requests' 256-token greedy answers, the score-only requests cold and
`--logits-width` at once so that they prefill beside each other, and
`dev/benchmarks/logit_compare.py` compares a dump with one whose prefixes it
scored (`--logits-from`): top-1 agreement, mean and p99 KL and the next-token
NLL change, failing past `--min-top1`, `--max-mean-kl` and `--max-p99-kl`. A
dump is three files, `sequences.txt`, `records.txt` and `logits.f32` (fp32
rows, little-endian as numpy's tofile writes them), so that another
engine's logits after the same prefixes compare the same way:

```sh
build/engine-tests/backend-benchmark build/splash.metallib "$MODEL_ROOT" --scenario logits --logits /tmp/base
build/engine-tests/backend-benchmark build/splash.metallib "$MODEL_ROOT" --scenario logits --logits /tmp/b4 --logits-from /tmp/base --logits-width 4
.venv/bin/python -m dev.benchmarks.logit_compare /tmp/base /tmp/b4
```

The second line also measures this build's noise floor: the same prefixes
prefilled four at a time.

A change of the weight images' layout (`EXPECT_IMAGE_CHANGE=1`) makes
`backend_regression` report the images it changed instead of failing on them.
`decode-profile --dispatch-list FILE` writes the profiled B1 cycles'
dispatches in order with their GPU time.

To compare the model's numerics with another engine on the same weights,
`make teacher-logits MODEL=... TEACHER_LOGITS_ARGS='TOKENS OUTPUT --window END:ROWS ...'`
writes the installed model's teacher-forced next-token logits over windows of
a file of uint32 token ids (`dev/benchmarks/teacher_logits.mm`): for each
window, the fp32 logits of the prefill's final position and of ROWS decode
positions after it, each computed by a verify row forced to the true next
token, in `OUTPUT-END.f32`, and prints how well they predict the true ids. It
also takes `--kv-format`, `--state-format`, `--expert-memory` and
`--ple-memory`, and, with `--captures yes`, writes the output rows of the
target layers the model's draft reads (its `target_layer_ids`) at the
positions a window's prefill computed, in one command, to
`OUTPUT-END.captures.f32` (fp32, a row per position of each layer's hidden
values in the draft's layer order), which a reference's residuals after those
layers compare with (`dev/tools/flint_reference.py` writes Flint's,
`residuals.npy`): a draft of the layers to compare, such as a test draft whose
`fc` reads them, makes them its captures.

`benchmark-backend` lists the contexts the memory plan cannot hold in its
report. Its cache checks reuse each context's cached prefix, so they need that
memory free: when other programs leave too little, the engine evicts cached
prefixes and the checks fail, naming what each lookup found.

## Serving benchmark

`dev/benchmarks/serving` measures a server under concurrent requests and
compares builds of different trees, such as a release candidate with the last
release, on one Mac. Each run starts the tree's own server, so every tree is
measured through the same OpenAI API whatever its native wire and status
schema, and the client times what a user sees.

| Scenario | Requests |
| --- | --- |
| `churn` | three clients, each sending 1500-token answers to prompts that grow every round (about 16K tokens, then more; `CHURN_BASE`/`CHURN_STEP` words), every fourth round resending its conversation, the first requests in a fixed arrival order (`--order 2,0,1` or `0,1,2`, 0.6 s apart), for `--minutes`; `/status` every 2 s |
| `ttft` | for each N of `--n`, on a fresh server: N conversations cache an ~8K-token prefix one at a time, then send it with distinct ~300-token suffixes at the same instant, `--rounds` times, each round recording its `/status` counters' changes and the largest `memory_governor.claims` its samples showed; `/status` every 0.5 s |
| `serve1` | one request at a time on an idle server, cold prompts of about `--sizes` tokens (300, 2K and 14K): time to first token and decode milliseconds per token; the `/status` each request waited idle for, and one after the last |
| `warm` | per cached depth of `--sizes`, a fresh cold prefix, then 2 discarded and 8 measured ~300-token suffixes after it: the server's prompt time of each (`--trial` picks the texts, the same for every tree); `/status` as `serve1` |

Every request timestamps each content chunk of its stream; its gaps are the
intervals between consecutive chunks after the first. Each request may take
`--request-seconds` from its send (by default twice the longest request
measured on any Mac). A request the client gets no whole answer to is
unanswered, the server's failure: one the server keeps streaming past that
time, though its keepalives hold the connection open, or one whose connection
fails or closes before the stream's `data: [DONE]`. It ends `serve1` and
`warm`, ends `ttft` as one of its warm-ups and fails its round otherwise; in
`churn` its client records it, with its chunk times and why, and stops.

```sh
.venv/bin/python -m dev.benchmarks.serving.run --tree TREE --machine m6 \
  --model ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF:Q2_0 --revision <commit> \
  --draft-model DRAFT --scenario churn --order 2,0,1 --minutes 15 --label LABEL --out RESULTS
```

`run.py` installs the model with the tree's own installer, offline, or takes
`--model-root`; refuses a port that already answers; checks that `/status`
names the server it started (`instance.pid`) and the build of the tree's
sources (`identity.cache.build_id` against the tree's own
`dev/tools/build_identity.py check` with its Makefile's constants); runs the
scenario; and stops the server with SIGINT, or, when it outlasts that, SIGTERM
after 60 s and SIGKILL after 30 s more. Beside the client's
`LABEL.events.jsonl` and `LABEL.status.jsonl` it writes `LABEL.out`, whose last
line is `DONE` only once everything completed, `LABEL.meta.json` (binary
SHA-256, build id, model, effective scenario configuration, client Python and
host OS versions, and Git commit and dirty state for both the measured tree
and the benchmark tools when available; unavailable Git values are null.
The `commit` field is the caller's `--commit`, which can differ from the
observed `source_state`. Git state describes the checkout at run time, not the
compiler or source used to produce an existing binary. It also records
spawn-to-ready seconds and whether the binary had
started on that Mac before, host memory and pressure, process snapshots at
the start, middle and end of the scenario while the server serves, the pids
the run started, each server's stop, the signals it took and its exit code,
and for a run that did not complete why and whose failure
that was: the server's, which exited, was not ready in time, failed a
request or, once ready, did not let the scenario end, a client's, which
raised an exception of its own, or `run.py`'s refusal to measure) and on the
M5 Pro
`LABEL.idle.jsonl`, its idle time and sleep assertions every 10 s. It refuses any output directory that resolves,
through whatever symlinks, into `~/.cache/huggingface` or an `install/models`
directory.

`check.py` judges runs:

- `run RESULTS LABEL [--minutes N] [--streamed] [--build BUILD_ID]` checks one
  run on its own and exits 1 on any failure; `test-pressure-real` runs it on
  its churn run.
- `compare A1 B1 B2 A2`, each run as `RESULTS/LABEL`, compares two builds on
  four runs in ABBA order on one Mac: each run must pass its own check on a
  quiet machine, the two A runs and the two B runs must each be one build,
  and every per-request metric measured by any run must be present in all four
  runs and pass the release check's speed rule (`dev/benchmarks/abba.py`). All
  four runs must use the same scenario and the same recorded scenario
  configuration; the listening port is ignored because it does not affect the
  workload. This prevents an absent metric or a changed workload dimension
  (such as TTFT concurrency or serve1 size) from silently dropping out of the
  comparison.

A run measured nothing and is to be repeated when it did not end with `DONE`
for a reason other than the server's, served a build other than `--build`,
or, with `--streamed`, did not start with its routed experts streamed. Only a
timed comparison needs a quiet machine: `compare` also fails a run that saw a
`serious` or `critical` thermal sample, found the GPU still busy after its wait
for idle (`--gpu-idle-percent`), on the M5 Pro was touched within a 10 s
sampling period or saw a sleep assertion held by a process it did not start
other than `powerd`, or had a process it did not start among the top five by
CPU at 24 % or more. `run`, whose gates judge freezes and correctness, which
these do not excuse, notes them.

| Gate | Rule |
| --- | --- |
| G4 | the client gaps between token events (chunks less than half the shortest decode command apart are one event's; an unanswered request counts its open gap, up to when the client stopped waiting): the largest at most 10 s, the p99 at most 5 s; every gap past 2.5 s is listed |
| G8 | no engine restart, failed request (an HTTP error but `context_length_exceeded`, an error event in a stream the server had begun, a request of a failed TTFT round, or one `/status` counts in `requests.failed`), unanswered request, Metal failure or loop tick of 5 s or more, no exit, late start or failed request that ended the run, and every server stopped by SIGINT alone with exit 0, the engine it started gone within SIGINT's grace too |
| G9 | the memory accounts sum to `charged_bytes` in every sample, no claim waits 30 s, nothing is owed at the end |
| Store | a regrown expert unit taken back within 60 s of its regrow is counted; a second such take back within 60 s of the one before is an oscillation and fails |
| Churn | the run lasted its load window (`--minutes`) and every client completed a request |
| First token | no answered request (G8 judges unanswered ones) waited for its first token more than twice the prefill time, at the run's own rate (the tokens its answered requests prefilled over the time from the first send to the last answer), of the work the scheduler may put ahead of it: the requests in flight at its send, 24,576 rows of later arrivals (three prefill commands of at most 8,192 rows, the most that may pass over it in a row) and its own prompt, each less its cached tokens; a starvation guard, not a speed check |

G8 applies to every scenario, the others to churn runs. The check also counts
the client gaps past the decode gap target of 500 ms by cause, each taking the
first of these that the `/status` samples from 2 s before it to 2 s after it
show: a claim waiting for memory, the experts' size changing, a compaction, a
replan, a history-stream wait, a committed prefill command; else it is
unattributed.

Copy a tree to another Mac by content (`rsync --checksum`): `git archive`
leaves `dev/tests` and `dev/benchmarks` out (`.gitattributes`), and a copy
that keeps the files' old times leaves make running stale binaries. For a
gate, delete the test binary before building it and record the commit, an
empty `git status --porcelain`, the binary's SHA-256 and its test count.

### Streamed expert routes

For an opt-in diagnostic run, set `SPLASH_EXPERT_ROUTE_TRACE` to a new file
path in the engine's environment. Each JSONL record contains version `v`,
zero-based `block`, `phase`, `rows`, and the sorted unique `experts` selected
by one streamed block load. Multiple grouped steps contribute their union;
repeated tiles and load rounds do not add records. File order is load order,
including work from concurrent requests. There are no request or token IDs,
route weights, prompts, or timestamps. A route can still reveal information
about the workload; protect and delete the file accordingly.

Only blocks that use the streamed load path appear: fully resident experts,
read-ahead predictions, and per-token frequencies are not represented. A
record describes routing even if a later load or command fails. The trace is
not a cache hit log. Use the same model and workload when comparing profiles.

The file is created exclusively with mode 0600 (subject to umask), never
appended to or overwritten. Give each process/store a separate path; if two
stores use the same path only the first can trace. Creation or write errors
are reported to stderr and disable tracing without failing inference. A failed
write can leave a partial last line, which offline readers must reject. Tracing
flushes each load under the store lock and can affect performance; leave the
variable unset for normal operation and performance measurements.

Analyze a completed trace offline, without loading a model:

```sh
python3 -m dev.benchmarks.expert_route_profile /path/to/routes.jsonl > profile.json
```

New trace records use version 2 and include `model_identity`, `weight_manifest_hash`,
and `identity_status`. The fingerprint hashes the complete contents of every
selected target source file, with stable logical names: every part of a split
GGUF and every Safetensors shard is included. Weight files are read for hashing
only when route tracing or `SPLASH_EXPERT_PRIORITY_PROFILE` is explicitly set.
If a source is unsupported, missing, or unreadable, the record says
`identity_status: "unknown"`, `priority_eligible: false`, and
`weight_manifest_hash: "unknown"`; profiles from that trace cannot supply
priority hints. The offline report preserves the eligibility flag. Offline processing rejects a trace
that mixes fingerprints, known and unknown identities, or model identities.
Legacy v1 traces remain analyzable but produce an unknown fingerprint.

The JSON report uses schema version 3, groups streamed loads by block and phase,
sums rows, and ranks experts by the number of load records containing each
expert. The offline `load_priority_hints(path, current_identity,
current_weight_manifest_hash, opt_in=True, top_k=8)` helper exposes hints only
when explicitly enabled and both the descriptor identity and complete weight
fingerprint exactly match. Unknown fingerprints, mixed-source traces, malformed
profiles, and either identity mismatch reject the profile as a whole. The C++
runtime applies the same checks through `SPLASH_EXPERT_PRIORITY_PROFILE`;
startup passes at most eight ranked experts per block to `ExpertStore` only
when the entire profile validates. Route tracing and profile loading remain
disabled by default; without either explicit setting, startup does not read
weight contents for hashing. These are set
membership counts, not token counts or routing weights. For locality, it also
counts intersections with the preceding record of the same block, provided that
record has the same phase. Records of other blocks may intervene; a different
phase on the same block breaks the pair. `reuse_fraction` divides the shared
expert selections by all current selections in those pairs, and is null when
that denominator is zero. This is route-set overlap, not a cache hit rate.
Concurrent requests are interleaved in file order, and incomplete or failed
loads can appear; a trace can also stop early after a write error. The tool
rejects malformed, unsupported, or unterminated records rather than reporting
a partial profile. An empty file is a valid empty profile, not proof that a
workload ran or tracing succeeded.

The trace lacks cache capacities and state, size classes, eviction and staging
transitions, read-ahead results and completed-load outcomes. It cannot strictly
replay the existing ARC behavior or establish actual hits, I/O, latency, or
performance. The tool therefore makes no cache-policy comparison. Profiles
are comparable only with compatible model/block identities and representative
workloads; the trace itself cannot verify either condition.

For how one running server scales under concurrent chat requests, with the
fans and hot sensors it reports, connect the benchmark to that server:

```sh
make benchmark-concurrency MODEL=mlx-community/Qwen3.8-27B-4bit \
  BASE_URL=http://127.0.0.1:8123 \
  CONCURRENCY_ARGS='--requests 1,2,4,8 --samples 3 --max-tokens 128'
```

It never starts a server, so run `splash serve` first. Each burst has its own
nonce, so the requests do not share a cached prefix; the report separates
aggregate output rate, time to first token, per-request decode rate, peak fan
RPM and hottest sensor. On a laptop the bursts heat the machine within seconds,
so compare bursts of the same width across cool starts, and read the fan and
temperature peaks beside the rate. For a cold prefill, pass
`CONCURRENCY_ARGS='--requests 1 --samples 1 --max-tokens 16 --prompt-chars 128000'`
and read the time to first token; it is the prefill time for that prompt.
