//! The grammar tokenizer's environment: the token bytes llguidance reads,
//! built from a model's tokenizer.json the way the Python's
//! `llguidance.hf.from_tokenizer` builds them — the vocabulary's
//! byte-level strings mapped back to their bytes, the added tokens as
//! their literal text, and the vocabulary padded to the model's.

use std::path::Path;
use std::sync::Arc;

use toktrie::{TokEnv, TokRxInfo, TokTrie};

/// The GPT-2 byte-level mapping's inverse: each character of a vocabulary
/// token's string back to the byte it stands for. The forward map sends
/// each printable byte to itself and every other byte to the first
/// non-printable code point in order.
fn byte_level_inverse() -> Vec<Option<u8>> {
    let mut inverse = vec![None; 0x110000];
    let mut mapped = [false; 256];
    for byte in 0..=255u8 {
        let c = byte as char;
        if ('!'..='~').contains(&c) || ('\u{a1}'..='\u{ac}').contains(&c) || ('\u{ae}'..='\u{ff}').contains(&c) {
            mapped[byte as usize] = true;
            inverse[byte as usize] = Some(byte);
        }
    }
    // The forward map's code points run from 0x100 upward over the
    // non-printable bytes, in byte order.
    let mut next = 0x100u32;
    for byte in 0..=255u8 {
        if !mapped[byte as usize] {
            inverse[next as usize] = Some(byte);
            next += 1;
        }
    }
    inverse
}

/// The token bytes of a tokenizer.json's vocabulary: each token's
/// byte-level string mapped back to its bytes; the added tokens' bytes
/// are their literal text's.
fn token_bytes(config: &serde_json::Value) -> Vec<Vec<u8>> {
    let inverse = byte_level_inverse();
    let vocab_size = config
        .get("model")
        .and_then(|model| model.get("vocab_size"))
        .and_then(|size| size.as_u64())
        .unwrap_or(0) as usize;
    let mut tokens: Vec<Vec<u8>> = Vec::new();
    if let Some(vocab) = config
        .get("model")
        .and_then(|model| model.get("vocab"))
        .and_then(|vocab| vocab.as_object())
    {
        for (token, id) in vocab {
            let Some(id) = id.as_u64() else { continue };
            let id = id as usize;
            if tokens.len() <= id {
                tokens.resize(id + 1, Vec::new());
            }
            tokens[id] = token
                .chars()
                .filter_map(|c| inverse.get(c as usize).copied().flatten())
                .collect();
        }
    }
    if let Some(added) = config
        .get("added_tokens")
        .and_then(|added| added.as_array())
    {
        for token in added {
            let Some(id) = token.get("id").and_then(|id| id.as_u64()) else { continue };
            let Some(content) = token.get("content").and_then(|c| c.as_str()) else { continue };
            let id = id as usize;
            if tokens.len() <= id {
                tokens.resize(id + 1, Vec::new());
            }
            // Tool framing is addressed by name in the generated Lark
            // grammar even when the tokenizer marks it as non-special.
            // llguidance indexes named tokens under its special prefix.
            let named = token.get("special").and_then(|v| v.as_bool()).unwrap_or(false)
                || matches!(content, "<tool_call>" | "</tool_call>");
            let mut bytes = Vec::with_capacity(content.len() + usize::from(named));
            if named {
                bytes.push(TokTrie::SPECIAL_TOKEN_MARKER);
            }
            bytes.extend_from_slice(content.as_bytes());
            tokens[id] = bytes;
        }
    }
    if tokens.len() < vocab_size {
        tokens.resize(vocab_size, Vec::new());
    }
    tokens
}

/// The grammar tokenizer: the environment llguidance's matchers read, over
/// the model's padded vocabulary and the engine's stop tokens as extra
/// ends.
pub struct ConstraintTokenizer {
    pub env: TokEnv,
}

impl ConstraintTokenizer {
    /// Build the environment from a model's `tokenizer` directory and the
    /// chat format's vocabulary and stop ids.
    pub fn from_directory(
        directory: &Path,
        vocabulary: u32,
        stop_ids: &[u32],
    ) -> Result<ConstraintTokenizer, String> {
        let text = std::fs::read_to_string(directory.join("tokenizer.json"))
            .map_err(|error| format!("{}: {error}", directory.join("tokenizer.json").display()))?;
        let config: serde_json::Value =
            serde_json::from_str(&text).map_err(|error| format!("tokenizer.json: {error}"))?;
        let mut tokens = token_bytes(&config);
        if tokens.len() < vocabulary as usize {
            tokens.resize(vocabulary as usize, Vec::new());
        }
        let eos = stop_ids.first().copied().unwrap_or(0);
        let trie = TokTrie::from(&TokRxInfo::new(vocabulary, eos), &tokens);
        let trie = trie.with_eos_tokens(stop_ids);
        Ok(ConstraintTokenizer {
            env: Arc::new(toktrie::ApproximateTokEnv::new(trie)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_tool_tokens_are_indexed_even_when_not_marked_special() {
        let config = serde_json::json!({
            "model": {"vocab": {"word": 0}},
            "added_tokens": [
                {"id": 1, "content": "<tool_call>", "special": false},
                {"id": 2, "content": "</tool_call>", "special": false},
                {"id": 3, "content": "ordinary", "special": false}
            ]
        });
        let tokens = token_bytes(&config);
        let trie = TokTrie::from(&TokRxInfo::new(4, 0), &tokens);
        assert_eq!(trie.get_special_token("<tool_call>"), Some(1));
        assert_eq!(trie.get_special_token("</tool_call>"), Some(2));
        assert_eq!(tokens[3], b"ordinary");
    }
}
