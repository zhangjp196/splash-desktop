# Open work

[Back to Splash](../README.md)

What the 1.9.3 code-organization and performance pass left open, best first.
Each item names where it lives and how to see it, so the next session starts
from the evidence rather than from this list.

## 1. The block indexer's pool race

`runtime/metal/kernels/shared/block_indexer.metal` (`block_indexer_pool`, from
line 106): the pooled keys are not rotated to their block start, and the
rollback case loses the keys a rejected row completed.

The failure is specific to the validation environment the native gate runs
in — `MTL_SHADER_VALIDATION=1` (dev/native.mk:581):

```bash
# passes
./build/engine-tests/qwen4-kernels build/engine-tests/production-and-test.metallib

# ten failures
MTL_SHADER_VALIDATION=1 ./build/engine-tests/qwen4-kernels \
    build/engine-tests/production-and-test.metallib
```

A result that depends on the validation layer is a synchronization defect
rather than kernel arithmetic: most likely a threadgroup barrier missing
between the writes a row's block complete and the rotation the next step
reads, or a read of memory a later dispatch in the same command writes. The
kernels arrived with Qwen3.8-Flash-Next (d6003397) and YaRN (40470bbd), which
narrow the search.

## 2. The automatic disk tier in the memory-plan tests

`dev/tests/engine/runtime_resources_test.mm:1105` reports "the automatic tier
did not start beside its staging, or the staging cost no context". It fails in
the default run too, unlike the block indexer, so it is a real outcome of this
machine's memory plan rather than a race:

```bash
./build/engine-tests/runtime-resources build/engine-tests/production-and-test.metallib
```

The test builds its budget from `start().minimumRequired`, so a machine whose
plan puts the automatic tier below its threshold would fail here honestly.
Decide first whether the test's budget or the tier's threshold is wrong.

## 3. The OpenAI dialect's paired completions

`server/dialects/openai.py:145-234`: `complete` and `complete_choices`, and
`text_completion` and `text_completion_choices`, are near copies, and
`complete_choices` builds a full response (usage, metrics, timings,
compaction) for every choice before keeping only the first. One
single-choice builder shared by both would remove the copy and the waste.

## 4. Engine work that was measured and left

- **Spilled-KV decode's per-layer host steps** (runtime/model/KvBlockCache.cpp:
  359, runtime/model/QwenTarget.cpp:448): each attention layer splits the
  command into its own buffer, ~96 extra buffers and 48 runner wakeups a
  cycle at full width, plus a per-layer directory lookup. `KvReadStats`
  (KvBlockCache.cpp:398-401) already records the resolve/pickup/resume
  split: measure before touching it, because the block list is produced by
  each layer's indexer just before its attention.
- **Building the next cycle's graph while the GPU runs** (Engine.cpp:324-340):
  everything after a command completes — scheduling, admission, graph build,
  encode, submit — is serial ahead of the GPU. Hiding it needs either two
  arena lanes or parameters bound after the build.

## 5. What only a model machine can check

- **The decode graph reuse's bitwise gate**: `--verify-rows all` against the
  CPU, and the engine's own output comparisons, on a real target. The CPU
  tests exercise the reuse path's bookkeeping but not a model's plans.
- **The logprob kernel against a real vocabulary**: the metal test compares
  `decode_logprob_mass` with the CPU scan at 1,003 tokens; a served request
  at 152K confirms the float denominator's error where it matters.
- **The full `make test-engine-metal` gate** once item 1 is fixed; the groups
  after `qwen4-kernels` (attention plans, linear plans, KV page tier, model
  loading, qwen state, q8, gdn, dflash, draft selector, target sampling,
  vision, handoff) have not all run since the pass.
- **Decode and prefill rates before and after**, on the same prompts and
  concurrency, in the manner of [Performance](performance.md).

## Not planned

The serving layer's `protocol.py` and `runtime.py`, and the runtime's
`Runtime::Impl`, stay whole. Splitting them would move 1,500-line files into
two 700-line files with no measured gain, and `Runtime::Impl` is the hottest
code in the tree; the seam at `server.py` was the one that mattered.
