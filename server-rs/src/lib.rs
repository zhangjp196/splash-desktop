//! Splash's serving process, in Rust.
//!
//! The port follows `server/`'s module order, one module at a time, each
//! against the Python tests it replaces (`PORTING.md` holds the plan). The
//! wire protocol is first: it is the byte-exact boundary with the engine,
//! and `dev/tests/engine/protocol_golden.txt` pins both sides' bytes.

pub mod chat_format;
pub mod chat_messages;
pub mod constraints;
pub mod dialects;
pub mod output;
pub mod engine;
pub mod protocol;
pub mod support;
pub mod templates;
pub mod tokenization;
