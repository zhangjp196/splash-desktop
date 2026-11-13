# Porting `server/` to Rust

The plan, the order, and the state. The Python server stays the oracle:
every module's port is wrong until the Python tests' behavior is
reproduced, and the whole server is not replaced until the port serves
what the Python one serves.

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
| 6 | `output.py`, `streamer.py`, `thinking.py` | ~1,100 | `src/output/` | **stop search and reasoning splitter done** (8 tests, each against the Python's own output); the tool-call projector, the blocks and the streamer remain |
| 7 | `constraints.py` | ~600 | `src/constraints/` (llguidance crate) | not started |
| 8 | `dialects/` (openai, anthropic, responses, scoring) | ~4,000 | `src/dialects/` | not started |
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
