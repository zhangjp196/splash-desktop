# Speculation roadmap: optional drafts and wider model coverage

Status: proposal. This document describes a change to the engine's contract,
not behavior of the current release. Today every model is served with a
DFlash2 draft; the sections below say how speculation becomes an optional,
source-agnostic capability without weakening the checked installation or the
served output. A Chinese edition, with the full engineering plan, is at
[speculation.zh.md](speculation.zh.md).

## Summary

Speculative decoding is currently a structural prerequisite, not a feature:

- the installer refuses a family with no released draft (`families.FAMILIES`,
  `install/upstream.py::_resolve_draft`, `install/local.py::_draft`);
- the assembly record always names a draft source (`assembly_record`), and
  `inspectModelRoot` refuses a model root without `draft/`;
- `ModelDescriptor::valid()` requires a draft whose hidden size, vocabulary,
  mask token and target layers agree with the target
  (`ModelDescriptor.mm`);
- `loadModel` always plans and loads a `DraftCheckpointLoader`;
- `CompositeStateLayout` always reserves draft rings and a context window
  (`StateLayout.hpp`), and the memory audit requires draft weight bytes
  (`RuntimeResources.mm::requireLoadedModel`, `MemoryAudit.cpp`);
- `Runtime::decodeAsync` always encodes the draft graph and the eight-row
  target verify (`Runtime.cpp`).

The proposal is to make that chain a *plan* with a first-class `none` value:

1. A model installation records a **speculation plan**: provider, format,
   source, proposal width, state kind and fallback policy.
2. The engine validates the plan, loads only what the plan names, and
   dispatches decode through a `Speculator` interface with a real
   no-speculation implementation.
3. Providers cover DFlash2 (safetensors or GGUF, including the `z-lab`
   mirrors), built-in MTP, external MTP and DSpark.
4. A missing, incompatible or unreachable draft degrades to `none` with a
   warning and a visible status field, unless the installation asked for a
   required draft.

The same change is the foundation for a wider model range: once a target no
longer needs a matching draft to load, a family only needs target kernels, a
tokenizer profile and a chat format to be servable.

## Goals

- `--model` / `--model-dir` serve a supported GGUF or MLX target with no
  draft at all.
- A draft may be a DFlash2 safetensors checkpoint, a DFlash2 GGUF, a
  built-in MTP head, an external MTP head or a DSpark bundle.
- Draft resolution follows one order, is recorded in the installation, and
  never silently substitutes a different model.
- Every start validates the target and the plan before weights are read.
- The served output of a no-draft model is the target's own greedy/sampled
  output; quality is unchanged, only speed differs.
- A draft-less installation and a draft installation of the same target can
  coexist and do not share state, cache namespaces or memory plans.
- The status API says which provider ran, from where, and why a requested
  provider did not.

## Non-goals

- A generic kernel generator. A new architecture still needs kernel shapes,
  a tokenizer profile and a chat format; this document widens the *draft*
  dimension, not the operator set.
- Training, converting or quantizing drafts at install time beyond the
  formats the engine already reads.
- Loading remote Python or model-side code. Drafts stay data.
- Distributed or multi-Mac serving.

## Terms

- **Speculation plan**: the record of how a model's decode cycles may propose
  tokens. It is part of the immutable installation record, not a runtime
  preference.
- **Provider**: the algorithm. `none`, `dflash2`, `mtp`, `dspark`.
- **Source**: where the provider's weights come from. `builtin` (inside the
  target checkpoint), `external` (a draft repository or directory), or
  `none`.
- **Format**: the container of the source weights. `safetensors`, `gguf`, or
  `none`.
- **Alignment**: the properties a draft must share with its target (hidden
  size, vocabulary, mask token, target layer ids, reduction, architecture
  family, tokenizer hash).
- **Proposal width**: the number of rows a cycle may propose, including the
  anchor row. DFlash2 uses eight; `none` uses one; MTP and DSpark are
  provider-defined.
- **Fallback**: what installation and loading do when the requested provider
  is unavailable or incompatible. `none` means serve without speculation.

## Design

### One plan, two sources

```text
target checkpoint ──► speculation plan ──► none    (plain autoregressive decode)
                                   └────► draft   (propose → verify → accept)
```

The target is the only required weight source. A plan with
`provider: none` is complete: no draft directory, no draft pins, no draft
state, no draft bytes in the memory budget.

A plan with a draft is a pair of independent sources, as a target and a draft
are today. The plan makes the pair explicit so the installer can assemble,
verify, pin and garbage-collect them as one installation.

### Record schema

`model.json` gains one optional object. Installations without it keep the
current meaning: a DFlash2 safetensors draft, as recorded by `sources.draft`.

```json
{
  "version": 2,
  "model": "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M",
  "family": "Qwen3.8-27B",
  "target_format": "gguf",
  "vision_format": "none",
  "speculation": {
    "version": 1,
    "provider": "dflash2",
    "format": "gguf",
    "source": {
      "kind": "repository",
      "repo": "z-lab/Qwen3.8-27B-DFlash2-GGUF",
      "revision": "<commit>",
      "files": {
        "draft/Qwen3.8-27B-DFlash2-Q8_0.gguf": "<digest>"
      }
    },
    "proposal_rows": 8,
    "state": "draft-rings",
    "fallback": "none"
  },
  "sources": {"target": {"repo": "...", "revision": "..."}}
}
```

Rules:

- `provider` is one of `none`, `dflash2`, `mtp`, `dspark`. An unknown value is
  refused before any weight read, naming the engine build that would read it.
- `format` is `none`, `safetensors` or `gguf` for DFlash2; `gguf` or
  `safetensors` for MTP according to the target; `external` for DSpark's own
  bundle.
- `source.kind` is `none`, `builtin` or `external`. Only `external` carries a
  repository or path.
- `proposal_rows` is 1 for `none`, the block's rows for DFlash2, and the
  provider's declared width otherwise.
- `state` is `none`, `draft-rings`, `mtp-cache` or `dspark-state`.
- `fallback` is `none` (degrade to no draft) or `required` (refuse to start).
  It is recorded from the user's request; the reason a fallback happened is
  runtime state, not installation identity.
- `version` is 2 when `speculation` is present. The reader accepts version 1
  and synthesizes the legacy DFlash2 plan, so old assemblies keep loading.

The record remains the cache identity: changing a provider, format, source
revision or proposal width yields another assembly and another persistent
cache namespace. The persistent-cache canonical line gains
`speculation <provider> <format> <proposal_rows> <state>`; a path with no
draft writes the literal `speculation none none 1 none`, so its caches never
collide with a draft's.

### Providers

| Provider | Proposes | Weights | State | Verify width | Status |
| --- | --- | --- | --- | --- | --- |
| `none` | anchor only | none | none | 1 | first |
| `dflash2` | one block of `proposal_rows` in a single draft forward | BF16 safetensors or GGUF | draft rings + context window | 1–8 | existing |
| `mtp` | `k` tokens, autoregressively through the target's own head(s) | built-in nextn layers or an external head | target KV for the head's layers | 1–k | second |
| `dspark` | parallel backbone + sequential Markov head, confidence-scheduled | external bundle | provider state | 1–n | experimental |

`none` is not a degraded DFlash2; it is a separate decode path that runs the
target once per token. It is the reference path every other provider is
tested against.

### Resolution order

One function, shared by `splash serve`, `--model-dir` and
`prepare_local_model`, decides the plan:

1. `--speculation none|dflash2|mtp|dspark` (new; `--no-draft` is an alias for
   `none`).
2. `--draft-model REPO|PATH` (existing) and its format probe.
3. A declaration in the target's own configuration: an MLX `speculation`
   object, or the GGUF key `splash.speculation`.
4. The family registry's default provider (today a DFlash2 repository).
5. `none`.

Explicit input wins, and a conflicting explicit pair is an error rather than a
guess: `--speculation none --draft-model X` is refused, and so is
`--speculation dflash2` with a path whose config is an MTP head.

`--draft-format auto|safetensors|gguf` resolves the container when a
repository holds both, such as a `z-lab` mirror beside its GGUF conversion.
`auto` prefers the format the target already uses (GGUF for a GGUF target,
safetensors otherwise), then the smaller complete checkpoint, and never mixes
files of two formats in one installation.

`--require-draft` sets `fallback: required`; the default is `none`.

### Validation and fallback

Validation happens twice, with the same plan:

- **Plan check**, before any weight download (`model-check`): the target's
  configuration is inspected as today; the draft's configuration or GGUF
  header is read and held to its provider's rules; alignment is checked
  symbolically (hidden size, vocabulary, mask token, layer ids, reduction,
  architecture family, tokenizer digest when the draft ships one).
- **Load check**, when every tensor's name, type and shape is known: the
  loader refuses a plan whose tensors do not match its configuration.

Fallback is allowed only when the plan says `fallback: none`, and only for
reasons that do not change the target:

- the draft repository or path is unreachable and no cached snapshot is
  pinned;
- the draft's format is not one this build loads;
- the draft's configuration or tensors are incompatible with the target;
- the engine build has no implementation of the provider (an older engine
  reading a newer installation).

Each fallback prints one warning naming the provider, the source and the
reason, and sets `/status` fields `speculation.provider = "none"`,
`speculation.requested`, `speculation.fallback_reason`. The installation
record is not rewritten: a later start with the source available uses it
again.

A fallback never substitutes a different family's draft, a different
revision's draft, or a quantized variant of another format silently. If the
requested draft is present but incompatible, the warning says which property
differs, as the installer's current `draft configuration is incompatible
with ...` errors do.

### Installation layout

A no-draft assembly links no `draft/` directory. The rest of the layout is
unchanged:

```text
config.json, target/, tokenizer/, vision/     target and its metadata
draft/                                        only when the plan has a source
model.json                                    the plan
```

`assembly.verify` gains the plan: `speculation.provider != none` requires
`draft/config.json` (or the GGUF's sidecar/config metadata) and at least one
weight file; `provider == none` requires none. The exact-key check on the
record is replaced by a schema check that accepts version 1 and version 2, so
an older assembly remains verifiable and a newer one is refused with its
version.

Garbage collection, pinning and selection links need no new concept: a
no-draft installation pins only the target's snapshot, and a draft
installation pins both, as today.

### Runtime interface

The decode loop stops naming DFlash2 and asks the loaded model for a
`Speculator`:

```cpp
class Speculator {
public:
  virtual ~Speculator() = default;
  virtual SpeculationKind kind() const noexcept = 0;
  // Rows the cycle may use, anchor included: 1 for none, 8 for DFlash2.
  virtual uint32_t proposalRows() const noexcept = 0;
  // Writes ProposedTokens/ProposalProbs/VerifyConfidence for the lanes.
  virtual void addPropose(CommandGraph &graph, ...) = 0;
  // Narrows each lane's verify rows, if the provider has an acceptance model.
  virtual void addVerifyRows(CommandGraph &graph, ...) = 0;
  // Commits the provider's own state after acceptance (rings, KV, Markov).
  virtual void addCommit(CommandGraph &graph, ...) = 0;
  virtual DraftStateLayout stateLayout() const noexcept = 0;
  virtual uint64_t weightsBytes() const noexcept = 0;
};
```

`DFlash2Speculator` wraps the existing `DFlashDraft` and preserves every
kernel call. `NoneSpeculator` has `proposalRows() == 1`, writes the anchor as
row 0, forces `VerifyRows = 1`, and makes `addCommit` a no-op. `MtpSpeculator`
and `DSparkSpeculator` add their own state and commit rules behind the same
interface.

The decode command becomes:

```text
propose(rows) → target verify(rows) → accept → target commit → speculator commit
```

For `none` the target verify runs one row per lane (the anchor's successor),
which is plain autoregressive decoding. The target kernels keep their
eight-row physical shape; only the selector and acceptance narrow the
committed rows, exactly as a short output budget does today.

`Runtime::decodeAsync` keeps one physical graph per call; it no longer
unconditionally calls `encodeDraftBatchGraph` or `encodeDraftStateCommitBatch`.
`finalizeDecode` treats `accepted` and `verifyRows` the same way, with
`verifyRows == 1` and `accepted == 0` for `none`.

### Scheduler, state and memory

- `DraftContextPlan` becomes optional. Admission with no draft skips draft
  window planning, draft restore and draft snapshots; a cached state holds
  only the target's cells.
- `CompositeStateLayout::valid()` accepts an all-zero draft layout;
  `laneBytes()` and `cachedBytes()` then count zero draft bytes. A
  `DraftStateLayout` with `provider == none` is the zero value.
- `configureDraftStatePlan` returns an empty plan for `none`, and the junction
  gain check (`kMinimumJunctionGain`) uses zero draft context tokens rather
  than the DFlash window.
- `VerifyBudget` is disabled when `proposalRows() == 1`; the cycles verify the
  anchor and nothing else, so there is no acceptance model to fit and no
  stochastic output to explain.
- `requireLoadedModel` requires draft bytes only when the plan has a draft.
  `EngineMemoryBreakdown::draftWeightsBytes` may be zero; the plan's
  `draft_weight_bytes_required` refusal disappears for `none`.
- `MemoryAudit` treats a zero draft category as complete when the plan is
  `none`, not as a missing measurement.
- Prefill captures target hidden rows only for providers that read them
  (DFlash2, MTP). `none` writes no capture rows and allocates no draft context
  window.
- Persistent and SSD caches use the plan-aware namespace described above. A
  cached state written with a draft is never restored into a no-draft model.

### Status and API

`/status` and the startup summary report:

```json
"speculation": {
  "provider": "dflash2",
  "format": "gguf",
  "proposal_rows": 8,
  "state": "draft-rings",
  "draft_weights_bytes": 123456789,
  "fallback_reason": null
}
```

`/v1/models` adds `speculation` beside `vision` and
`input_modalities`. The server's startup line says which provider runs and,
for `none`, that decode is autoregressive. Existing fields
(`drafted_tokens`, `accepted_draft_tokens`, `draft_acceptance_rate`) remain;
they read zero for `none`.

## Provider details

### DFlash2, safetensors (current)

Unchanged: `config.json` plus BF16 safetensors (or the shards its index
names), optional `mask_embedding.safetensors`; `DraftCheckpointLoader` plans
the images, quantizes projections to af4g64, and `DFlashDraft` runs them. The
plan's `format` is `safetensors`, `source.kind` is `external`, and the
existing alignment rules apply.

One mechanical change: `readDraftLayout` becomes
`readDraftLayout(provider, draft, target)` so the mask-token and target-layer
rules are selected by the planned provider rather than by a global "a draft
must be DFlash2" check.

### DFlash2, GGUF

A DFlash2 checkpoint may be published as GGUF, as the `z-lab` and
`neuralforgequantum` conversions are. Loading it reuses the target side's GGUF
machinery rather than the safetensors one:

- the draft's configuration is read from the GGUF's metadata when the header
  carries it (`dflash2.*` keys mirroring `config.json`), or from a
  `config.json` beside the file; a header without either is refused at plan
  check;
- tensor names are mapped from the GGUF's `draft.*`/`blk.*` naming to the
  layout's projections, and the planner checks every expected tensor exactly
  as `DraftCheckpointLoader` does;
- weights load as their GGUF types. The first release supports BF16, F16,
  Q8_0 and Q4_K for draft projections and BF16/F32 norms, then widens to the
  formats the target loader already reads. A type outside that set is a
  fallback reason, not a crash;
- a split draft GGUF follows the target's `NNNNN-of-MMMMM` rules and installs
  as `draft/` parts, so the loader reads it as one file.

Why GGUF matters: it decouples the draft from the BF16 safetensors release,
lets a 24 GB Mac pair a quantized target with a quantized draft, and makes
the `z-lab` mirrors usable without conversion.

### `z-lab` drafts

`z-lab/<MODEL>-DFlash2` is a DFlash2 release like any other; no special
runtime path is needed. The installer treats the namespace as a *preferred
mirror*, not a hard-coded provider:

- the family registry may name more than one candidate repository, in order
  (`incoai/<MODEL>-DFlash2`, then `z-lab/<MODEL>-DFlash2`);
- an explicit `--draft-model z-lab/...` always wins and is recorded verbatim;
- a `-GGUF` suffix selects the GGUF format when `--draft-format` is `auto`;
- repository names never decide compatibility: the checkpoint's own config and
  tensors do, as today (`families.py` states this rule).

This keeps one code path for every publisher and avoids a vendor allowlist.

### Built-in MTP

Some targets ship multi-token-prediction heads in the same checkpoint
(`nextn_predict_layers` in GGUF, `model.mtp.*`/`model.layers.N.mtp.*` in MLX,
Model Optimizer's `mtp` modules). Today they are deliberately excluded: the
GGUF layer count subtracts them (`gguf.py`) and the Model Optimizer loader
does not read them.

The built-in MTP provider:

- reads `num_nextn_predict_layers` (GGUF `nextn_predict_layers`) and the head's
  tensor set at plan check;
- requires `provider: mtp`, `source.kind: builtin`, `format` equal to the
  target's, and `alignment` that records the target's layer count, hidden size
  and head count;
- loads the head beside the target, from the same shards/GGUF, with no extra
  download;
- proposes `k` tokens through the head, verifies them with the target, and
  commits the head's own KV beside the target's;
- is refused, with fallback, when the target has no complete head or when the
  head's shapes are not ones this build's kernels can run.

The head is not a DFlash2 draft: it has no draft rings, no block selector and
no `dflash_config`. It gets its own `MtpLayout`, state kind `mtp-cache`, and
`proposal_rows` from the head's declared `k` (clamped to the verify shapes).

### External MTP

A separate repository or directory may hold only the MTP head. It is an
external source with the same `MtpLayout`, plus an alignment record naming the
base model it was trained for:

```json
"alignment": {
  "family": "Qwen3.8-27B",
  "hidden_size": 5120,
  "vocabulary_size": 151936,
  "target_layers": 64,
  "base_model_digest": "<sha256 of the target's config or GGUF header>"
}
```

The head must state an alignment that matches the target; a head with a
missing or different alignment is incompatible and falls back. There is no
"closest match" heuristic.

### DSpark

DSpark couples a parallel draft backbone with a lightweight sequential
(Markov) head and schedules verification by confidence and engine throughput
([DSpark](https://arxiv.org/abs/2607.05147)). It is an inference-time module
over unmodified target weights, which fits this design well:

- `provider: dspark`, `format: external`, `state: dspark-state`;
- the bundle declares the backbone, the Markov head, the tokenizer digest and
  its alignment;
- `addPropose` runs the parallel backbone, then the sequential head for the
  intra-block dependencies; `addVerifyRows` maps its confidence scores onto
  the existing `VerifyRows` machinery, whose acceptance model already
  estimates prefix survival;
- `addCommit` stores the Markov head's state beside the target's.

DSpark stays behind an experimental flag (`--speculation dspark`) until its
acceptance and output match the reference implementation on the release
prompts. A bundle whose ABI version this build does not know is refused.

## Wider model coverage

Removing the draft prerequisite is necessary but not sufficient. The current
model range is five families and four layout types (`kFamilies`), each with
compiled kernel shapes and a tokenizer profile. Expansion should be tiered:

- **Tier A — tuned**: the existing families, with per-family kernels, DFlash2
  drafts and the full feature set (vision, ANE, sparse KV).
- **Tier B — generic**: standard dense and MoE transformers whose shapes fit
  the existing operators (Llama/Mistral/Qwen dense, DeepSeek-style MoE,
  Gemma). They load through a declarative family pack: model types, dimension
  predicates, tensor naming, chat format, tokenizer profile and kernel shape
  rows. They serve without a draft; a draft is a bonus.
- **Tier C — experimental**: architectures needing new operators (linear
  attention variants, MLA, exotic MoE routing). They are gated by an explicit
  `--experimental-family` and never selected by `auto`.

A family pack is data plus a layout type, not a new engine. The registry
answers three questions without compiling new C++: does this configuration
match (dimension predicates), which tokenizer/chat profile does it use, and
which kernels' shape rows does it add. This is how the range grows from five
to dozens without a fork per model.

Concretely, the first expansion candidates after the infrastructure lands:

1. dense Qwen2/Qwen3 and Llama 3.x GGUFs and MLX 4-bit (Tier B, no draft);
2. DeepSeek-V3/V4-style MoE with built-in MTP (Tier B, `mtp` builtin);
3. Gemma 3/4 and Mistral Small (Tier B, no draft; vision later);
4. DSpark bundles for DeepSeek-V4 (Tier C, experimental).

## Implementation phases

Each phase is independently shippable and leaves the current path working.

### P0 — plan and no-draft backbone (~1–2 weeks)

- Record schema version 2, `speculation` object, version-1 synthesis.
- `SpeculationPlan` in C++ (`ModelDescriptor.hpp`), plan-aware
  `inspectModelRoot`, `valid()` and `loadModel`.
- `NoneSpeculator`, optional draft state/memory, `speculation none` decode
  path and the selector's forced row.
- Installer: `--speculation`/`--no-draft`, `--require-draft`; no-draft
  assemblies; family registry may return `None`.
- Status fields and cache namespace.
- Exit: a supported GGUF and a supported MLX model serve with no draft at the
  same greedy output as `--verify-rows all` with a draft whose proposals are
  all rejected, and the memory audit passes.

### P1 — draft formats and sources (~1–2 weeks)

- `DraftSource` abstraction over safetensors and GGUF.
- GGUF DFlash2 plan check, tensor mapping, loading and split support.
- `z-lab` mirror candidates and `--draft-format`.
- Exit: a quantized GGUF target pairs with a GGUF DFlash2 draft; a
  safetensors target pairs with the same draft in either format.

### P2 — built-in and external MTP (~2–3 weeks)

- `MtpLayout`, tensor extraction from GGUF/MLX/Model Optimizer checkpoints.
- `MtpSpeculator`, its KV, commit and state layout.
- Alignment records and external-head installation.
- Exit: a DeepSeek-style target serves with its built-in MTP head and with an
  external head; a mismatched head falls back to `none` with a reason.

### P3 — DSpark (~3–4 weeks)

- Bundle ABI, plan check, parallel backbone and Markov head.
- Confidence scheduler wired to the verify budget.
- Exit: an experimental DSpark bundle serves a supported target, with
  acceptance at least that of DFlash2 on the release prompts.

### P4 — family packs (~ongoing)

- Declarative registry, Tier B generic layouts, tokenizer/chat profiles.
- First three candidate families.
- Exit: adding a Tier B family touches no engine source file.

## Testing

- Unit: resolution order, record round-trip, version-1 synthesis, fallback
  reasons, format auto-selection, alignment comparison.
- Engine: no-draft greedy decode equals the target-only oracle; a draft whose
  proposals are always rejected commits the same tokens; `verifyRows == 1`
  never reads draft buffers; zero draft state passes the memory audit; cache
  namespaces differ between `none` and `dflash2`.
- Installer: a no-draft `prepare` and `verify`; a GGUF draft's plan check and
  split selection; an incompatible draft falls back only when allowed.
- Integration: `dev/tests/engine` runs each provider's golden prompts;
  `make test-engine-metal` covers the no-draft path against the existing
  DFlash tests.
- Performance: report decode tok/s for `none` beside each draft provider, so
  the cost of speculation is visible rather than assumed.

## Risks and mitigations

- **Output drift.** A no-draft path must not change sampling, constraints or
  stop handling. Mitigation: it reuses the target verify graph with one
  committed row and is tested against the target-only oracle, including
  seeded and constrained requests.
- **Memory accounting holes.** Zero draft bytes must be a valid category, not
  a missing measurement. Mitigation: the audit keys on the plan, and the
  warmup report carries the plan's provider.
- **Cache collisions.** State layouts with and without draft rings are not
  interchangeable. Mitigation: the plan is in the persistent namespace and in
  the assembly identity.
- **Kernel coverage.** A GGUF draft's quantized projections may not have a
  draft-kernel instance. Mitigation: a supported-type list per phase, with a
  fallback reason naming the type; no host dequantization in production.
- **Silent quality loss.** A fallback must be visible. Mitigation: one warning
  and `fallback_reason` in status; `--require-draft` for benchmarks and CI.
- **Supply chain.** Draft repositories are data, never code. Mitigation: the
  existing rule stays: the installer reads config and tensors only, and
  `model-check` validates before download.

## Open questions

1. Should a draft's tokenizer digest be mandatory for external MTP and DSpark,
   or only for DFlash2?
2. Is `--verify-rows` meaningful for `none` / MTP, or should it become a
   provider option?
3. Should a no-draft installation still allocate the eight-row decode arena,
   or grow a one-row arena for `none` (memory vs. code paths)?
4. Which `z-lab` repositories are official mirrors, and should the family
   registry name them or leave them to explicit selection?
5. Does DSpark's Markov head warrant a new state kind in
   `CompositeStateLayout`, or can it reuse the draft context window?
