//! The tokenizer the server serves by, read straight from its files, and
//! the prompt tokenizer's bounded prefix reuse. Ported from
//! `server/fast_tokenizer.py` and `server/tokenization.py` on the
//! `tokenizers` crate, which is the same library the Python wrapped.
//!
//! The chat template's rendering is the next module; this one carries the
//! tokenizer's encoding surface and the prompt cache.

pub mod fast;
pub mod prompt;

pub use fast::FastTokenizer;
pub use prompt::PromptTokenizer;
