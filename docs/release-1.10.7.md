# Splash Desktop 1.10.7

Version 1.10.7 completes the desktop distribution's move from the Python
server to the Rust server. The macOS DMG bundles the Rust server and native
inference engine; running the app no longer requires a Python runtime. The
Python implementation remains in the repository as a compatibility oracle
for behavior that has not yet moved to Rust.

## What changed from the Python desktop build

- **Native Rust server.** The app launches the bundled Rust CLI directly.
  OpenAI Chat Completions, Text Completions, Anthropic Messages and Responses
  support complete and streamed responses. The server also serves the chat
  UI, model and health endpoints, tokenization, templates, status, and native
  scoring routes.
- **Local model preparation.** Rust validates and prepares local single-file
  or complete split GGUF targets, including optional vision projectors, and
  local MLX-affine targets. It can install complete GGUF targets already in
  the Hugging Face cache without network access. Prepared assemblies are
  verified and published atomically.
- **Vision and documents.** Inline images work with vision-capable models.
  Inline PDF pages are rendered in a supervised worker when a compatible
  PDFium library is supplied. Pillow-compatible image resizing matches the
  Python implementation's fixtures.
- **Server safety and diagnostics.** The Rust server enforces request
  deadlines, bounded connection and preparation capacity, body budgets,
  protected-route API keys, host and browser-origin validation, and cancels
  generation after stream disconnects. Safe startup and request summaries,
  readiness records and live engine metrics are available to the desktop.
- **Desktop behavior.** Chat streams reasoning separately from the answer,
  displays request throughput, and passes the selected thinking effort to the
  model. The Live and Thermal views read the Rust engine's status and control
  endpoints.
- **Packaging and checks.** The DMG contains no Python runtime. Rust server
  checks run through `make check-rust-server` and a dedicated CI job.

## Current limitations

- Hub downloads and the full Python installer lifecycle have not yet moved to
  Rust. Model preparation currently uses local files or complete GGUF targets
  already in the Hugging Face cache.
- Sharded MLX preparation, DFlash draft conversion, and PDF form
  initialization parity are not implemented. PDF support requires PDFium.
- macOS builds are ad hoc signed and are not notarized. The DMG requires
  macOS 26.4 or later.

The app icon was refreshed for this build. Verify the DMG with the adjacent
`.sha256` file before installation.
