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
its usage, and the SSE stream carries the role chunk, the content
delta, the finish chunk, the usage chunk and [DONE].

## Where this stands (end of the first porting session)

5,702 lines of Rust (plus 2,152 of tests) port the first six modules'
cores: the wire protocol byte-exact against its golden file, the
support layer, the engine client's call state machine and subprocess
core, the tokenizer's encoding surface with the prompt cache, the chat
format's model with the templates' rendering and loading, and the
output's stop search, reasoning splitter and tool-call projector.
62 tests are green, every one against the Python implementation's own
output, and the build carries no warnings.

The next session picks up at the first `not started` row below. The
highest-value next slices, in order: the streamer's incremental decode
(`streamer.py`, small), the constraints (module 7, the llguidance crate
direct), then the dialects and the HTTP layer, which make an
end-to-end serving path. The projector's full `ParameterTypes` arrives
with tool_schema (module 11). Each module's port is complete when its
Python test file's cases pass in Rust.

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
| 5 | `chat_format.py`, `chat_templates.py`, `chat_messages.py` | ~2,300 | `src/chat_format.rs`, `src/templates/` (minijinja) | **format model, rendering core and template loading done** (12 tests); `chat_messages.py` (message normalization) and `chat_templates.py`'s probing/patching remain |
| 6 | `output.py`, `streamer.py`, `thinking.py` | ~1,100 | `src/output/` | **stop search, reasoning splitter, tool-call projector and streamer done** (20 tests, each against the Python's own output); `thinking.py` remains; the projector's full `ParameterTypes` (schema reading) is module 11 |
| 7 | `constraints.py` | ~600 | `src/constraints/` (llguidance crate) | **grammar tokenizer env, factory with cache, and token masks done** (4 tests, each against the Python's own output); the executor's parallel mask fill and the strict-tool schemas are tool_schema's (module 11) |
| 8 | `dialects/` (openai, anthropic, responses, scoring) | ~4,000 | `src/dialects/` | **chat_messages, the OpenAI and Anthropic assemblies, the Anthropic request conversion, and the Responses input normalization done** (25 tests, each against the Python's own output); the exchange/job machinery, the Responses output items and the remaining request conversion remain |
| 9 | `http_server.py`, `http_handler.py`, `http_security.py`, `connections.py`, `origins.py`, `routing.py` | ~2,000 | `src/http/` (axum or hyper) | not started |
| 10 | `frontend.py`, `backend.py`, `generation_options.py`, `request_fields.py` | ~2,500 | `src/serve/` | not started |
| 11 | `judgments.py`, `schema_validation.py`, `tool_schema.py`, `compaction.py` | ~1,500 | `src/judgments/` | not started |
| 12 | `images.py`, `documents.py`, `document_worker.py` | ~1,200 | `src/documents/` (pdfium, image) | not started |
| 13 | `metrics.py`, `latency.py`, `diagnostics.py`, `serve_record.py`, `serve_options.py`, `server.py` | ~1,500 | `src/main.rs` | not started |
| — | `install/assembly.py` (startup reads it) | 504 | `src/assembly/` | not started |
| — | installer (`install/` rest) | ~5,700 | **stays Python** | excluded by decision |

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
