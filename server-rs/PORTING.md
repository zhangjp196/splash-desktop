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

The next work in the exchange is to carry the deadline through engine
submission and generation, add per-choice jobs and cancellation, and port
compaction. Preparation for token counts and documents still needs its
callers, including releasing the preparation slot while a PDF worker runs.
The desktop DMG launches the Rust CLI directly and embeds the native engine.
The CLI loads existing model assemblies and binds to IPv4 loopback with
Host and Origin validation and optional API-key checks for POST requests.
It resolves an explicit DFlash2 draft against the selected assembly's
recorded draft source, including version 1 assembly records.
The listener serves the bundled chat UI, Chat token counts and a model list.
Model downloading and assembly creation, full HTTP admission, LAN access
and the other dialects are pending.

The current Chat path also prepares inline images for vision-capable
models and supports tool-call and JSON-schema constraints through token
masks. PDF documents, generation deadlines and disconnect cancellation,
and the remaining API routes are still pending. The desktop DMG has an
ad hoc signature and is not notarized; this Rust release does not yet
replace every feature of the Python server.

## Why this order

Each module ports only against the modules below it, so the order is the
dependency order, and the riskiest core goes first: the wire protocol is
the byte-exact boundary with the engine, and its golden file pins both
sides.

| # | Module (`server/`) | Lines | Rust home | State |
|---|---|---|---|---|
| 1 | `protocol.py` | 1,496 | `src/protocol/` | **done** — golden bytes match, 4/4 tests |
| 2 | `json_codec.py`, `errors.py`, `lru.py` | ~700 | `src/support/` | **done** — 12 tests; one documented deviation: lone-surrogate JSON escapes are refused at parse (serde_json), the Python retains them; the dialect layer decides how such requests are answered |
| 3 | `runtime.py` (engine subprocess client) | 1,554 | `src/engine/` | **core done** — the call state machine and the one-generation client (spawn, reader thread, dispatch, submit, cancel, status) with 16 tests against a fake engine; restarts, liveness probes and the CPU mask workers remain |
| 4 | `fast_tokenizer.py`, `tokenization.py` | ~900 | `src/tokenization/` (tokenizers crate) | **encoding surface and prompt cache done** (5 tests, real Qwen tokenizer); the chat-template half of `fast_tokenizer.py` is module 5 |
| 5 | `chat_format.py`, `chat_templates.py`, `chat_messages.py` | ~2,300 | `src/chat_format.rs`, `src/templates/`, `src/chat_messages.rs` | **format model, rendering core, template loading and text message normalization done**; template probing/patching and multimodal normalization remain |
| 6 | `output.py`, `streamer.py`, `thinking.py` | ~1,100 | `src/output/` | **stop search, reasoning splitter, tool-call projector and streamer done** (20 tests, each against the Python's own output); the splitter rides the pipeline's complete answer and the SSE stream, so `reasoning_content` is reported rather than a raw `</think>`; `thinking.py` remains; the projector's full `ParameterTypes` (schema reading) is module 11 |
| 7 | `constraints.py` | ~600 | `src/constraints/` (llguidance crate) | **grammar tokenizer env, factory with cache, and token masks done** (4 tests, each against the Python's own output); the executor's parallel mask fill and the strict-tool schemas are tool_schema's (module 11) |
| 8 | `dialects/` (openai, anthropic, responses, scoring) | ~4,000 | `src/dialects/` | **chat_messages, the OpenAI and Anthropic assemblies, the Anthropic request conversion, and the Responses input and output done** (28 tests, each against the Python's own output); the exchange/job machinery and the remaining request conversion remain |
| 9 | `http_server.py`, `http_handler.py`, `http_security.py`, `connections.py`, `origins.py`, `routing.py` | ~2,000 | `src/http/`, `src/serve/server.rs` | **minimal listener and unit-tested security and admission primitives**, with Chat completions, token counts, models, `/status` and `/v1/thermal` routed; wiring the rest of the security primitives into the listener, request I/O limits, connection watching, and remaining routes are pending |
| 10 | `frontend.py`, `backend.py`, `job.py`, `generation_options.py`, `request_fields.py` | ~2,500 | `src/serve/` | **Chat preparation slots and preparation deadlines started**; generation deadlines, disconnect and choice-job handling, generation options, and request-field parity are pending |
| 11 | `judgments.py`, `schema_validation.py`, `tool_schema.py`, `compaction.py` | ~1,500 | `src/judgments/` | not started |
| 12 | `images.py`, `documents.py`, `document_worker.py` | ~1,200 | `src/documents/` (pdfium, image) | not started |
| 13 | `metrics.py`, `latency.py`, `diagnostics.py`, `serve_record.py`, `serve_options.py`, `server.py` | ~1,500 | `src/main.rs` | **standalone CLI starts and stops the native engine**; `/status` serves the engine's own document with the server's transport accounting, so the live board reads it; the status cache, full options, metrics and diagnostics remain |
| — | `install/assembly.py` (startup reads it) | 504 | `src/main.rs` | **reads existing model assemblies**; full verification and publication are pending |
| — | installer (`install/` rest) | ~5,700 | — | Python remains in the checkout; desktop bundle excludes it and uses prepared models |

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
