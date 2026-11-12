//! Bounded token reuse at literal chat-message boundaries, ported from
//! `server/tokenization.py`: a prompt's tokens up to its last `marker` are
//! cached, and a longer prompt reuses them and encodes only its tail.

use std::path::Path;
use std::sync::{Arc, Mutex};

use serde_json::Value;

use crate::support::lru::LruCache;
use crate::tokenization::fast::FastTokenizer;

/// How wide a prefix must be before its tokens are worth caching.
pub const MIN_PREFIX_CHARS: usize = 4096;
/// Wide enough that concurrent coding agents' conversations, each a few
/// hundred K tokens of ids, keep their prefixes cached beside each other:
/// an evicted prefix re-encodes whole, which a 150K-token conversation
/// pays on every admission until it is cached again.
pub const BUDGET_BYTES: usize = 32 * 1024 * 1024;
pub const CAPACITY: usize = 16;

/// The prompt tokenizer: encodes prompts reusing a cached prompt's tokens
/// up to its last `marker`, the chat template's special token at a message
/// boundary, where the tokenizer splits its input.
pub struct PromptTokenizer {
    tokenizer: Arc<FastTokenizer>,
    marker: String,
    pub enabled: bool,
    pub marker_id: Option<u32>,
    entries: Mutex<LruCache<String, Vec<u8>>>,
    hits: Mutex<usize>,
    reused_tokens: Mutex<usize>,
}

/// The boundary check for a synthetic pipeline document, as the tests use.
#[doc(hidden)]
pub fn supports_boundaries_for_test(config: &Value, marker: &str) -> bool {
    supports_boundaries(config, marker)
}

/// Whether the tokenizer's pipeline splits its input at literal added
/// tokens, read from `tokenizer.json`: a BPE model without dropout, a
/// pre-tokenizer of regex splits ending in byte-level BPE without a prefix
/// space, an NFC normalizer or none, a byte-level post-processor or none,
/// and exactly one added token that is the marker, literal and
/// non-stripping. Another pipeline may not split where the marker does.
fn supports_boundaries(config: &Value, _marker: &str) -> bool {
    let model = config.get("model");
    if model.and_then(|model| model.get("type")).and_then(|t| t.as_str()) != Some("BPE") {
        return false;
    }
    if model
        .and_then(|model| model.get("dropout"))
        .map(|dropout| !dropout.is_null())
        .unwrap_or(false)
    {
        return false;
    }
    let pre_tokenizer = config.get("pre_tokenizer");
    if pre_tokenizer.is_none() || pre_tokenizer == Some(&Value::Null) {
        return false;
    }
    let pre = pre_tokenizer.expect("checked above");
    let parts: Vec<&Value> = match pre.get("type").and_then(|t| t.as_str()) {
        Some("Sequence") => pre
            .get("pretokenizers")
            .and_then(|parts| parts.as_array())
            .map(|parts| parts.iter().collect())
            .unwrap_or_default(),
        _ => vec![pre],
    };
    let expected: Vec<&str> = std::iter::repeat("Split")
        .take(parts.len().saturating_sub(1))
        .chain(std::iter::once("ByteLevel"))
        .collect();
    let kinds: Vec<&str> = parts
        .iter()
        .map(|part| part.get("type").and_then(|t| t.as_str()).unwrap_or(""))
        .collect();
    if kinds != expected {
        return false;
    }
    if parts
        .last()
        .and_then(|part| part.get("add_prefix_space"))
        .and_then(|flag| flag.as_bool())
        .unwrap_or(false)
    {
        return false;
    }
    for (component, allowed) in [
        (config.get("normalizer"), ["NFC"]),
        (config.get("post_processor"), ["ByteLevel"]),
    ] {
        let kind = component
            .filter(|component| !component.is_null())
            .and_then(|component| component.get("type"))
            .and_then(|kind| kind.as_str());
        if let Some(kind) = kind {
            if !allowed.contains(&kind) {
                return false;
            }
        }
    }
    // The added tokens: exactly one that contains the marker, equal to it,
    // literal and non-stripping. The crate's own decoder is the truth the
    // encoder runs by, so the caller passes its tokens.
    true
}

/// The added-token rules of the boundary check, over the crate's tokens.
fn marker_is_literal(tokens: &[(u32, tokenizers::AddedToken)], marker: &str) -> bool {
    let mut markers = 0usize;
    for (_, token) in tokens {
        if !token.content.contains(marker) {
            continue;
        }
        if token.content != marker {
            return false;
        }
        if token.normalized || token.lstrip || token.rstrip || token.single_word {
            return false;
        }
        markers += 1;
    }
    markers == 1
}

impl PromptTokenizer {
    /// Load the tokenizer and its boundary support from a model's
    /// `tokenizer` directory.
    pub fn from_directory(
        tokenizer: Arc<FastTokenizer>,
        directory: &Path,
        marker: &str,
    ) -> PromptTokenizer {
        let config: Option<Value> = std::fs::read_to_string(directory.join("tokenizer.json"))
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok());
        let enabled = config
            .map(|config| supports_boundaries(&config, marker))
            .unwrap_or(false)
            && marker_is_literal(&tokenizer.added_tokens(), marker);
        let marker_id = if enabled {
            tokenizer.token_to_id(marker)
        } else {
            None
        };
        PromptTokenizer {
            tokenizer,
            marker: marker.to_string(),
            enabled,
            marker_id,
            entries: Mutex::new(LruCache::new(BUDGET_BYTES, Some(CAPACITY))),
            hits: Mutex::new(0),
            reused_tokens: Mutex::new(0),
        }
    }

    /// The token ids of `text`, with no special token added.
    pub fn encode_all_public(&self, text: &str) -> Vec<u32> {
        self.encode_all(text)
    }

    /// The token ids of `text`, with no special token added.
    fn encode_all(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text, false).unwrap_or_default()
    }

    /// The tokens of `text`, reusing a cached prefix's tokens up to its
    /// last marker.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let boundary = text.rfind(&self.marker);
        if !self.enabled || boundary.is_none() {
            return self.encode_all(text);
        }
        let boundary = boundary.expect("checked above") + self.marker.len();
        if boundary < MIN_PREFIX_CHARS {
            return self.encode_all(text);
        }
        let prefix = &text[..boundary];
        let (key, cached) = {
            let mut entries = self.entries.lock().expect("entries lock");
            let key = entries
                .keys()
                .filter(|key| prefix.starts_with(*key))
                .max_by_key(|key| key.len())
                .cloned()
                .unwrap_or_default();
            let cached = entries.get(&key).map(|bytes| {
                *self.hits.lock().expect("hits lock") += 1;
                let tokens = bytes
                    .chunks_exact(4)
                    .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("u32 chunk")))
                    .collect::<Vec<u32>>();
                *self.reused_tokens.lock().expect("reused lock") += tokens.len();
                tokens
            });
            (key, cached)
        };
        let mut tokens = cached.unwrap_or_default();
        if key != prefix {
            tokens.extend(self.encode_all(&prefix[key.len()..]));
            // A different added token can consume part of the marker. In
            // that case this literal occurrence is not a tokenizer
            // boundary.
            match self.marker_id {
                Some(marker_id) if tokens.last() == Some(&marker_id) => {}
                _ => return self.encode_all(text),
            }
            let token_bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
            let size = prefix.len() + token_bytes.len();
            if size <= BUDGET_BYTES {
                let mut entries = self.entries.lock().expect("entries lock");
                // An extension replaces its earlier prefix; unrelated
                // concurrent conversations retain their own LRU entries.
                entries.pop(&key);
                entries.put(prefix.to_string(), token_bytes, size);
            }
        }
        tokens.extend(self.encode_all(&text[boundary..]));
        tokens
    }

    /// `text` cut just past its last marker where the tokenizer splits it
    /// there: the tokens before the cut, and the text after it, which
    /// encodes on its own. No tokens and all of `text` where it does not.
    pub fn split(&self, text: &str) -> (Vec<u32>, String) {
        let boundary = text.rfind(&self.marker);
        if self.enabled {
            if let Some(boundary) = boundary {
                let boundary = boundary + self.marker.len();
                let head = self.encode_all(&text[..boundary]);
                // As in encode, another added token may consume part of the
                // marker, which is then no boundary.
                if let Some(marker_id) = self.marker_id {
                    if head.last() == Some(&marker_id) {
                        return (head, text[boundary..].to_string());
                    }
                }
            }
        }
        (Vec::new(), text.to_string())
    }

    /// The cache's counters, as the status document reports them.
    pub fn stats(&self) -> PromptCacheStats {
        let entries = self.entries.lock().expect("entries lock");
        PromptCacheStats {
            enabled: self.enabled,
            entries: entries.len(),
            bytes: entries.bytes,
            budget_bytes: entries.budget_bytes,
            capacity: entries.capacity,
            hits: *self.hits.lock().expect("hits lock"),
            reused_tokens: *self.reused_tokens.lock().expect("reused lock"),
        }
    }
}

/// The prompt cache's counters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptCacheStats {
    pub enabled: bool,
    pub entries: usize,
    pub bytes: usize,
    pub budget_bytes: usize,
    pub capacity: Option<usize>,
    pub hits: usize,
    pub reused_tokens: usize,
}
