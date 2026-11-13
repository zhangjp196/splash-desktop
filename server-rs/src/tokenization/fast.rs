//! The tokenizer's encoding surface, ported from `server/fast_tokenizer.py`
//! without its chat-template half (the templates are the next module): a
//! `tokenizers::Tokenizer` loaded from a model's `tokenizer.json`, with the
//! calls the server makes — encode and decode, the vocabulary and
//! special-token lookups.

use std::path::Path;
use std::sync::Mutex;

use tokenizers::Tokenizer;

/// A server-side value could not be encoded or decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerError(pub String);

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for TokenizerError {}

use std::fmt;

pub struct FastTokenizer {
    tokenizer: Tokenizer,
    /// Encoding runs on one thread at a time: the tokenizers crate's
    /// encode takes `&self` but its caches are not synchronized, so the
    /// Python's single-threaded use is reproduced with a lock.
    encode_lock: Mutex<()>,
}

impl FastTokenizer {
    /// Load a model's `tokenizer` directory: `tokenizer.json` for the
    /// vocabulary and merges.
    pub fn from_file(directory: &Path) -> Result<FastTokenizer, TokenizerError> {
        let path = directory.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&path)
            .map_err(|error| TokenizerError(format!("{}: {error}", path.display())))?;
        Ok(FastTokenizer {
            tokenizer,
            encode_lock: Mutex::new(()),
        })
    }

    /// The token ids of `text`, with no special token added.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, TokenizerError> {
        let _guard = self.encode_lock.lock().expect("encode lock");
        let encoding = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|error| TokenizerError(format!("encode failed: {error}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// The token ids of `text`, with no special token added, and each
    /// token's span in it. The spans are byte offsets into `text`, as the
    /// tokenizers crate's `encode` reports them; the image pipeline ties
    /// each image to its placeholder by the span.
    pub fn encode_with_offsets(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<(Vec<u32>, Vec<(usize, usize)>), TokenizerError> {
        let _guard = self.encode_lock.lock().expect("encode lock");
        let encoding = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|error| TokenizerError(format!("encode failed: {error}")))?;
        Ok((encoding.get_ids().to_vec(), encoding.get_offsets().to_vec()))
    }

    /// The text `token_ids` decode to, with special tokens skipped as
    /// `skip_special_tokens` says.
    pub fn decode(
        &self,
        token_ids: &[u32],
        skip_special_tokens: bool,
    ) -> Result<String, TokenizerError> {
        self.tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(|error| TokenizerError(format!("decode failed: {error}")))
    }

    /// The vocabulary, added tokens included: token to id.
    pub fn get_vocab(&self) -> HashMap<String, u32> {
        self.tokenizer.get_vocab(true)
    }

    /// The id of one token, or `None` when the vocabulary has none.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    /// The tokenizer's EOS token id, read from `tokenizer_config.json`'s
    /// `eos_token` (a string, or an object with `content`).
    pub fn eos_token_id(&self, directory: &Path) -> Option<u32> {
        let config: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(directory.join("tokenizer_config.json")).ok()?)
                .ok()?;
        let eos = config.get("eos_token")?;
        let content = match eos {
            serde_json::Value::String(text) => Some(text.clone()),
            serde_json::Value::Object(map) => map.get("content").and_then(|c| c.as_str()).map(String::from),
            _ => None,
        }?;
        self.token_to_id(&content)
    }

    /// The ids of the tokenizer's added tokens.
    pub fn added_token_ids(&self) -> Vec<u32> {
        self.tokenizer.get_added_tokens_decoder().keys().copied().collect()
    }

    /// The tokenizer's added tokens, as the boundary check reads them.
    pub fn added_tokens(&self) -> Vec<(u32, tokenizers::AddedToken)> {
        self.tokenizer.get_added_tokens_decoder().into_iter().collect()
    }
}

use std::collections::HashMap;
