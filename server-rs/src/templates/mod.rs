//! The chat templates: the rendering environment transformers' chat
//! templates assume, and the template files' loading and selection.
//! Ported from `server/fast_tokenizer.py`'s rendering half.

pub mod render;

pub use render::{ChatTemplateError, ChatTemplateRenderer};
