//! Splash's serving process, in Rust.
//!
//! The port follows `server/`'s module order, one module at a time, each
//! against the Python tests it replaces (`PORTING.md` holds the plan). The
//! wire protocol is first: it is the byte-exact boundary with the engine,
//! and `dev/tests/engine/protocol_golden.txt` pins both sides' bytes.

pub mod chat_format;
pub mod assembly;
pub mod chat_messages;
pub mod cli_paths;
pub mod cached_install;
pub mod constraints;
pub mod documents;
pub mod diagnostics;
pub mod dialects;
pub mod http;
pub mod images;
pub mod metrics;
pub mod mlx_local;
pub mod local_draft;
pub mod output;
pub mod engine;
pub mod gguf;
pub mod protocol;
pub mod serve;
pub mod serve_record;
pub mod support;
pub mod templates;
pub mod tokenization;
pub mod tool_schema;
pub mod thinking;
