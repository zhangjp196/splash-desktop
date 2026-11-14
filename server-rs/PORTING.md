# Porting `server/` to Rust

The plan, the order, and the state. The Python server stays the oracle:
every module's port is wrong until the Python tests' behavior is
reproduced, and the whole server is not replaced until the port serves
what the Python one serves.

## The end-to-end path (the second session's milestone)

`src/serve/pipeline.rs` runs one Chat request from the body to the
assembled response: the messages normalized, the template rendered, the
prompt tokenized, the wire frame submitted to the engine client, the
output's tokens collected through the call's events, and the answer
assembled as OpenAI's dialect answers. `src/serve/stream.rs` runs one
request's stream: the role chunk at the start, the output's text as
content deltas through the streamer's incremental decode, the finish
chunk with the timings, the usage chunk when asked, and `[DONE]` — the
events written as OpenAI's SSE chunks as they arrive, the completion
callback writing the finish, as the Python's exchange collects.
`src/http/` carries the minimal HTTP/1.1 server the Python's stdlib
frontend mirrors, and `src/serve/server.rs` is the network entry: a
listener the caller binds, one thread per connection, the routes for
POST /v1/chat/completions (complete or streamed, as the body's `stream`
says) and GET /status. The tests run every path against a fake engine —
over socket pairs for the pipeline and the stream, and over real TCP
for the server: the complete response carries the engine's answer with
its usage and the reasoning kept apart from it, and the SSE stream
carries the role chunk, the reasoning and content deltas, the finish
chunk, the usage chunk and [DONE].

## Where this stands

The wire protocol is byte-exact against its golden file. The Rust listener
serves Chat Completions (JSON and SSE) and a `/status` document: the
engine's own status JSON — its memory, cache, request, fan, throttle and
speculation sections, the rates `metrics` reports — beside the server's
readiness, preparation slots and transport counters, which is what the
desktop's live board reads. Chat prompt preparation is limited by the
engine's reported concurrency. A request's `timeout` includes body upload,
queueing, and prompt preparation. The stream splits the output's reasoning
from its answer as the Python's exchange does, so a thinking model's
`</think>` never reaches a client as content, and its usage chunk carries
the request's metrics — the decode rate the chat page shows as tok/s —
beside the timings. POST `/v1/thermal` sets the running engine's prefill
duty in place, as the desktop's thermal pane asks. A prepared model is
resolved through the installer's selection link (`.selections/<digest of the
selection>`), so a directory with several prepared assemblies serves the one
the command's options name, and the streamer counts characters where the
Python does, so a multi-byte reply never splits inside a character.

The global `--request-timeout` and request-body `timeout` now share one
deadline from the start of HTTP parsing through prompt preparation, engine
submission and generation; an engine deadline failure returns a timeout
error in complete and streamed Chat responses. The listener bounds
connection workers and reserves a shared request-body byte budget.
Non-streaming Chat supports up to four independent choices and aggregates
their usage, while retaining the first choice's metrics and timings. It can
bind to a selected IP, requires an API key beyond loopback, and validates
configured hosts and browser origins. Absolute deadlines while a client
trickles request bytes are enforced by the global timeout; complete-response
disconnects now cancel all submitted choices, and SSE write failures cancel
their engine call. A per-request body timeout is applied after the body is
parsed. The engine client probes liveness while requests are pending and
fails the current generation if probes stop answering; the CLI now replaces
failed engine generations with bounded retries and backoff. Image-aware
compaction and PDF rendering run before acquiring the prompt preparation
slot.
The desktop DMG launches the Rust CLI directly and embeds the native engine.
The CLI loads existing model assemblies and binds to a selected IP with
Host and Origin validation and API-key checks for protected routes.
It resolves an explicit DFlash2 draft against the selected assembly's
recorded draft source, including version 1 assembly records.
The listener serves the bundled chat UI, Chat token counts, model listing,
health and readiness probes, tokenize/detokenize, apply-template, and memory
release. Anthropic Messages serves complete and live SSE responses, with
persistent thinking signatures on macOS. Text Completions supports complete
and live SSE responses; Responses supports complete and live SSE responses
with bounded in-memory storage. Judgments and Systemone use native score-only
inference. Assembly validation, full verification, atomic publication, and
selection-link publication are implemented; a bounded GGUF v3 reader maps
the four supported text architectures and the Qwen3VL merger vision config.
Rust also generates tokenizer assets for the two Python-supported GGUF BPE
profiles. `prepare-gguf` now prepares a local single-file or complete same-
directory split GGUF, optionally with one vision projector. It checks shard
names and numbering, split metadata, per-shard tensor layouts, duplicate
tensors and total tensor count, then runs the engine model check before it
publishes a verified assembly containing every shard. Draft conversion is
not supported by this path. `install-cached`
also installs single-file and complete split GGUF targets already present in
the local Hugging Face cache without network access, retains their snapshot
refs, and skips incomplete splits, draft, and MLX repositories. `prepare-mlx`
now prepares local MLX-affine directories with one `model.safetensors`, the
required tokenizer files, and optional validated vision weights/processor;
weights stay linked rather than copied. Sharded MLX indexes remain unsupported.
Hub downloads, full Hub-cache lifecycle, sharded MLX preparation, and DFlash
conversion are still pending. The
content-addressed `.generated` cache has local orphan collection that
preserves files referenced by published assemblies and keeps a one-day grace
period for in-progress preparation.
Responses custom and namespace tools normalize into native tool
schemas and preserve their public names and inputs in complete and streamed
output; Python's `background=true` request is rejected with its matching 400
error.

The current Chat path also prepares inline images for vision-capable
models and supports tool-call and JSON-schema constraints through token
masks. Inline PDF file parts now render in a supervised child process and
expand into text and page images before prompt normalization. Configure a
PDFium library with `--pdfium PATH` or `SPLASH_PDFIUM_PATH`; the macOS app also
discovers `runtime/engine/libpdfium.dylib`, which the DMG packager can include
when `SPLASH_PDFIUM_PATH` points to a compatible library. The
server never searches a Python environment or system library path. Linux
workers receive a 1 GiB address-space limit; macOS workers are monitored for
resident and physical footprint and are killed above 1 GiB. Unix workers also
have CPU limits, disabled core dumps, and a cleared environment. Workers are
terminated after the 30 second render deadline and their IPC response is
capped at 64 MiB. PDFium is optional and can be supplied to the packaging
script explicitly. Windows and other non-Linux/non-macOS targets lack an
enforced memory cap. Pillow BICUBIC resize output is byte-matched against its
oracle fixtures. PDF form initialization still differs from Python. macOS
`sandbox-exec` isolation is not yet ported; the managed environment refused
to apply a sandbox profile.
The remaining serving gaps include unported installer behavior. The
desktop DMG has an ad hoc signature and is not notarized; this Rust release
does not yet replace every feature of the Python server.

The Rust server suite now has a standalone `make check-rust-server` target
and its own CI job; `make test` runs it alongside the existing tests. The
Python compatibility suite stays enabled as the oracle and still covers the
installer and developer tools that have not been ported.

The Rust CLI now wires `--no-webui`, served-model aliases and announcement,
default reasoning effort, loop guard, thinking-end text, expert-memory growth,
and disabling the ANE mixer. It validates and normalizes memory/cache sizes,
context and request-size limits, idle durations, cache-eviction protection,
decode share and prefill controls, and persistent-cache dependencies before
starting the engine. Server-only choices stay in `Server`; only expert-memory
growth, ANE mixer disablement, loop guard, cache and thermal controls enter
the native command. CLI help now groups options by purpose, and path expansion
handles `~`, `~/`, and `~user/`. Bounded asynchronous
console diagnostics and atomic serve records cover startup, loading, ready
and failed phases. Chat complete and SSE requests log safe Started/Done
summaries, including cache, timing, speculation, and tool-signature metrics;
request text, tool argument values, and arbitrary error messages are omitted.
Anthropic Messages and Responses also reuse the same safe summaries for
complete and streamed requests.

## Why this order

Each module ports only against the modules below it, so the order is the
dependency order, and the riskiest core goes first: the wire protocol is
the byte-exact boundary with the engine, and its golden file pins both
sides.

| # | Module (`server/`) | Lines | Rust home | State |
|---|---|---|---|---|
| 1 | `protocol.py` | 1,496 | `src/protocol/` | **done** — golden bytes match, 4/4 tests |
| 2 | `json_codec.py`, `errors.py`, `lru.py` | ~700 | `src/support/` | **done** — 12 tests; one documented deviation: lone-surrogate JSON escapes are refused at parse (serde_json), the Python retains them; the dialect layer decides how such requests are answered |
| 3 | `runtime.py` (engine subprocess client) | 1,554 | `src/engine/` | **core done** — the call state machine and one-generation client (spawn, reader thread, dispatch, submit, cancel, status); pending requests use periodic liveness probes and the CLI recreates failed engine generations with bounded retries; bounded CPU mask workers remain |
| 4 | `fast_tokenizer.py`, `tokenization.py` | ~900 | `src/tokenization/` (tokenizers crate) | **encoding surface and prompt cache done** (5 tests, real Qwen tokenizer); the chat-template half of `fast_tokenizer.py` is module 5 |
| 5 | `chat_format.py`, `chat_templates.py`, `chat_messages.py` | ~2,300 | `src/chat_format.rs`, `src/templates/`, `src/chat_messages.rs` | **format model, rendering core, template loading and text message normalization done**; template probing/patching and multimodal normalization remain |
| 6 | `output.py`, `streamer.py`, `thinking.py` | ~1,100 | `src/output/`, `src/thinking.rs` | **stop search, reasoning splitter, tool-call projector, streamer and Fernet thinking signatures done**; the projector's schema-derived `ParameterTypes` are in module 11; regex evaluation uses a backtrack limit rather than Python's 50 ms deadline |
| 7 | `constraints.py` | ~600 | `src/constraints/` (llguidance crate) | **grammar tokenizer env, factory with cache, and token masks done** (4 tests, each against the Python's own output); the executor's parallel mask fill and the strict-tool schemas are tool_schema's (module 11) |
| 8 | `dialects/` (openai, anthropic, responses, scoring) | ~4,000 | `src/dialects/` | **Chat, Text Completions, Anthropic Messages and Responses serve complete and live SSE responses; Responses custom/namespace tools are normalized and restored; Judgments and Systemone use native scoring**; `background=true` is rejected as in Python; thinking signatures use macOS CommonCrypto |
| 9 | `http_server.py`, `http_handler.py`, `http_security.py`, `connections.py`, `origins.py`, `routing.py` | ~2,000 | `src/http/`, `src/serve/server.rs` | **bounded connection handlers and request-body byte reservations; configurable bind IP, host/origin policy, CORS preflight and response headers; Chat, Text Completions, Responses, Anthropic Messages, Judgments, Systemone, models, probes, prompt tools, memory and thermal controls routed; absolute global read deadline; generation disconnect cancellation**; runtime startup/shutdown details and a few response headers still differ |
| 10 | `frontend.py`, `backend.py`, `job.py`, `generation_options.py`, `request_fields.py` | ~2,500 | `src/serve/` | **Chat preparation slots and end-to-end deadlines, including the global default; temperature, top-p, top-k, min-p, penalties, stop sequences, random/explicit seed, priority, Python-compatible cache-salt hashing, logit bias, ignore-EOS, thinking budgets, output budgets, non-streaming logprobs, non-streaming Chat multiple choices, and image-aware compaction with generated/reused notes are handled** |
| 11 | `judgments.py`, `schema_validation.py`, `tool_schema.py`, `compaction.py` | ~1,500 | `src/serve/scoring.rs`, `src/tool_schema/`, `src/serve/compaction.rs` | **native score-only judgments and Systemone; text/image-aware compaction; generated notes and durable note reuse; strict schema-derived parameter types implemented**; remaining schema validation edge cases are pending |
| 12 | `images.py`, `documents.py`, `document_worker.py` | ~1,200 | `src/documents.rs`, `src/images/`, `src/gguf.rs` | **Rust image/document preparation; Pillow-compatible BICUBIC resize; bounded PDF IPC with AcroForm initialization and rendering; Linux and macOS worker memory caps; stable random template marker; GGUF v3 metadata/model-config parsing, tokenizer artifact generation, verified local and cached single/split GGUF assembly publication**; XFA/JavaScript, macOS sandbox profile and non-Linux/non-macOS memory cap remain |
| 13 | `metrics.py`, `latency.py`, `diagnostics.py`, `serve_record.py`, `serve_options.py`, `server.py` | ~1,500 | `src/main.rs`, `src/metrics.rs`, `src/diagnostics.rs`, `src/serve_record.rs` | **engine restart supervision, live `/status`, cached `/metrics`, upload, preparation queue/work, template, tokenization, image, grammar, native queue, HTTP time to first token and output interval latency, CLI option checks and grouped help, bounded asynchronous diagnostics, atomic serve records, and safe Started/Done summaries for Chat, Anthropic and Responses implemented** |
| — | `install/assembly.py` (startup reads it) | 504 | `src/assembly.rs`, `src/main.rs` | **existing assembly loading, full verification, atomic assembly publication, selection-link publication, and local single/split GGUF preparation** |
| — | installer (`install/` rest) | ~5,700 | `src/gguf.rs`, `src/cached_install.rs`, `src/mlx_local.rs` (partial) | Local single/split GGUF prepare; offline Hub-cache discovery/pinning for cached single/split GGUFs; and local single-file MLX-affine preparation are ported. Hub download, full cache lifecycle, sharded MLX preparation, DFlash conversion, and complete install orchestration remain Python |

## What the port keeps

- The engine subprocess and its wire protocol: the Rust server talks to
  the same `build/splash` binary with the same frames, so the engine is
  untouched.
- The tokenizers, llguidance and safetensors work: those libraries are
  Rust already; the Python was their FFI glue.
- The chat templates' semantics: transformers' rendering environment
  (trim_blocks, lstrip_blocks, the `{% generation %}` scope, `tojson`,
  `raise_exception`, `strftime_now`), which `server/fast_tokenizer.py`
  pins and the tests oracle against transformers.
- The tests: each ported module carries the Python tests' cases, and the
  golden files (`protocol_golden.txt`, the weight goldens, the status
  golden) stay the shared truth.

## What the port drops

- transformers, numpy, Pillow, pypdfium2 as Python packages: the Rust
  equivalents are the crates those packages wrapped or replaced.
- The `.venv`, the bytecode dance and the launcher's bootstrap: the
  release ships one binary beside the engine.
- huggingface_hub: the downloader stays Python (or moves to the desktop
  app) and hands the server a prepared assembly.

## Working conventions

- `cargo test` runs from `server-rs/`; the golden files are read from the
  repository root, so the crate tests the checkout it lives in.
- No dependencies in the protocol core: the wire code is std-only, so the
  byte-exactness has nothing to hide behind.
- A module's port is complete when its Python test file's cases pass in
  Rust; the Python tests stay green meanwhile (both servers run from the
  same tree until the switch).
