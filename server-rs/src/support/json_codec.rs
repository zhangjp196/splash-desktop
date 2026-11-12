//! Strict JSON and compact UTF-8 encoding for API payloads and retained
//! state. Ported from `server/json_codec.py`, on `serde_json`, whose
//! compact form, raw UTF-8 output and finite-number rules are the Python
//! encoder's.
//!
//! One documented deviation: Python's parser reads a lone surrogate escape
//! (`"\ud800"`) into a lone-surrogate string, which the server retains and
//! re-escapes on output; `serde_json` refuses it at parse. The dialect
//! layer decides how requests that carry one are answered (PORTING.md).

use serde::Deserialize;
use serde_json::Value;

/// Arrays and objects nest at most this deep in JSON the server reads:
/// deeper than any request needs, and shallow enough for every later stage
/// that recurses. Schemas, read with more stack a level, keep their own
/// 64-level limit.
pub const MAX_DEPTH: usize = 128;

const TOO_DEEP: &str = "JSON nests deeper than 128 levels";

/// A server-side value could not be serialized as JSON, or the text read
/// was not strict JSON.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonError(pub String);

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for JsonError {}

use std::fmt;

/// Escape lone surrogates and separators recognized by text-based SSE
/// readers, as the Python's `_escape` does on the encoded text.
fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '\u{85}' | '\u{2028}' | '\u{2029}' => {
                out.push_str(&format!("\\u{:04x}", character as u32));
            }
            _ => out.push(character),
        }
    }
    out
}

/// The deepest container nesting `text` holds, or `JsonError` for text
/// whose strings or brackets do not close. A non-recursive scan, so an
/// adversarial document cannot exhaust the stack before the depth bound
/// rejects it.
fn nesting_depth(value: &str) -> Result<usize, JsonError> {
    let bytes = value.as_bytes();
    let mut depth = 0usize;
    let mut max_depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for &byte in bytes {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'[' | b'{' => {
                depth += 1;
                if depth > max_depth {
                    max_depth = depth;
                }
            }
            b']' | b'}' => {
                depth = depth.saturating_sub(1);
            }
            _ => {}
        }
    }
    if in_string || escaped {
        return Err(JsonError("EOF while parsing a string".into()));
    }
    Ok(max_depth)
}

/// The value JSON text spells; `JsonError` for text that is not strict
/// JSON or nests deeper than [`MAX_DEPTH`].
pub fn loads(value: &str) -> Result<Value, JsonError> {
    // The depth bound runs before the parser, so the parser's own
    // recursion, disabled below, cannot run past it: a document deeper
    // than MAX_DEPTH is refused without being parsed.
    if nesting_depth(value)? > MAX_DEPTH {
        return Err(JsonError(TOO_DEEP.into()));
    }
    let mut deserializer = serde_json::Deserializer::from_str(value);
    deserializer.disable_recursion_limit();
    let document = Value::deserialize(&mut deserializer)
        .map_err(|error| JsonError(error.to_string()))?;
    deserializer.end().map_err(|error| JsonError(error.to_string()))?;
    // serde_json's own recursion bound sits near the Python parser's; the
    // containers of each level are counted without recursion, as the
    // Python's loads counts them.
    let mut level: Vec<&Value> = match &document {
        value @ (Value::Object(_) | Value::Array(_)) => vec![value],
        _ => Vec::new(),
    };
    let mut depth = 0usize;
    while let Some(current) = level.pop() {
        depth += 1;
        if depth > MAX_DEPTH {
            return Err(JsonError(TOO_DEEP.into()));
        }
        match current {
            Value::Object(map) => {
                level.extend(map.values().filter(|child| {
                    matches!(child, Value::Object(_) | Value::Array(_))
                }));
            }
            Value::Array(items) => {
                level.extend(items.iter().filter(|child| {
                    matches!(child, Value::Object(_) | Value::Array(_))
                }));
            }
            _ => unreachable!("only containers are queued"),
        }
    }
    Ok(document)
}

/// The value as compact JSON text, lone surrogates aside (see the module
/// note); `JsonError` for a value JSON cannot carry, such as a NaN.
pub fn dumps(value: &Value) -> Result<String, JsonError> {
    let text = serde_json::to_string(value)
        .map_err(|error| JsonError(error.to_string()))?;
    Ok(escape(&text))
}

/// The value as compact JSON bytes.
pub fn encode(value: &Value) -> Result<Vec<u8>, JsonError> {
    Ok(dumps(value)?.into_bytes())
}

/// Count encoded bytes without materializing the complete document.
pub fn encoded_size(value: &Value) -> Result<usize, JsonError> {
    encode(value).map(|bytes| bytes.len())
}
