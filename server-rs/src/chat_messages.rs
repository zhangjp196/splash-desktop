//! Chat message normalization, which every API's request reaches.
//! Ported from `server/chat_messages.py`.
//!
//! One documented deviation: a PDF file part is refused until the
//! documents module ports (`server/documents.py` renders it); images are
//! kept in their canonical form for the chat template.

use serde_json::{json, Map, Value};

/// The image placeholder the chat template places where an image is.
pub const IMAGE_PAD_TOKEN: &str = "<|image_pad|>";

/// A refusal the client may read: its HTTP status, message and code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MessageError {
    pub status: u16,
    pub message: String,
    pub code: String,
}

impl MessageError {
    pub fn bad(message: impl Into<String>) -> MessageError {
        MessageError {
            status: 400,
            message: message.into(),
            code: "invalid_request_error".into(),
        }
    }
}

impl fmt::Display for MessageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.message, self.code)
    }
}

impl std::error::Error for MessageError {}

use std::fmt;

const TEXT_PARTS: [&str; 4] = ["text", "input_text", "output_text", "refusal"];

fn part_text(part: &Map<String, Value>) -> Result<String, MessageError> {
    let kind = part.get("type").and_then(|t| t.as_str()).unwrap_or("");
    let key = if kind == "refusal" { "refusal" } else { "text" };
    let text = part.get(key).and_then(|t| t.as_str());
    match text {
        Some(text) => Ok(text.to_string()),
        None => Err(MessageError::bad("text message content must be a string")),
    }
}

fn text_content(content: &Value) -> Result<String, MessageError> {
    match content {
        Value::Null => Ok(String::new()),
        Value::String(text) => Ok(text.clone()),
        Value::Array(parts) => {
            let mut out = String::new();
            for part in parts {
                let Some(map) = part.as_object() else {
                    return Err(MessageError::bad("only text message content is supported"));
                };
                let kind = map.get("type").and_then(|t| t.as_str()).unwrap_or("");
                if !TEXT_PARTS.contains(&kind) {
                    return Err(MessageError::bad("only text message content is supported"));
                }
                out.push_str(&part_text(map)?);
            }
            Ok(out)
        }
        _ => Err(MessageError::bad("message content must be text")),
    }
}

fn image_part(url: &Value) -> Result<Value, MessageError> {
    let Some(url) = url.as_str() else {
        return Err(MessageError::bad("invalid image content part"));
    };
    Ok(json!({"type": "image_url", "image_url": {"url": url}}))
}

/// Canonical parts, or their text when they carry no image.
fn media_content(parts: Vec<Value>) -> Value {
    if parts.iter().any(|part| {
        part.get("type").and_then(|t| t.as_str()) != Some("text")
    }) {
        Value::Array(parts)
    } else {
        let text: String = parts
            .iter()
            .filter_map(|part| part.get("text").and_then(|t| t.as_str()))
            .collect();
        Value::String(text)
    }
}

/// Text for plain user content; a canonical parts list when it carries
/// images, so the chat template places each image where the author put
/// it. The one place media is accepted or rejected: every API shape
/// converts its image and PDF parts to image_url and file parts that
/// reach this point before anything is decoded or rendered.
fn user_content(content: &Value, vision: bool) -> Result<Value, MessageError> {
    let Some(parts) = content.as_array() else {
        return text_content(content).map(Value::String);
    };
    let mut canonical = Vec::new();
    for part in parts {
        let Some(map) = part.as_object() else {
            return Err(MessageError::bad("message content parts must be objects"));
        };
        let kind = map.get("type").and_then(|t| t.as_str()).unwrap_or("");
        match kind {
            "image_url" | "input_image" => {
                if !vision {
                    return Err(MessageError::bad(
                        "image input is not supported: this model is serving without vision (started with --language-only)",
                    ));
                }
                let url = match map.get("image_url") {
                    Some(Value::Object(inner)) => inner.get("url").cloned().unwrap_or(Value::Null),
                    Some(other) => other.clone(),
                    None => Value::Null,
                };
                canonical.push(image_part(&url)?);
            }
            kind if TEXT_PARTS.contains(&kind) => {
                canonical.push(json!({"type": "text", "text": part_text(map)?}));
            }
            "file" => {
                // The documents module renders a PDF to its pages; until it
                // ports, a file part is refused with the vision note.
                return Err(MessageError::bad(
                    "PDF input is not supported: this model is serving without vision (started with --language-only)",
                ));
            }
            "video" | "video_url" | "input_audio" => {
                return Err(MessageError::bad(format!("{kind} content is not supported")));
            }
            _ => return Err(MessageError::bad("unsupported message content part")),
        }
    }
    Ok(media_content(canonical))
}

/// A history call's arguments: an object, as sent or as JSON text of one;
/// otherwise their text, the string as sent or another value's JSON,
/// which template_messages keeps as text, so that a conversation
/// continues past a call the output limit cut or the model wrote wrong.
fn call_arguments(arguments: &Value) -> Value {
    match arguments {
        Value::Null => json!({}),
        Value::String(text) => {
            match crate::support::json_codec::loads(text.trim().is_empty().then(|| "{}").unwrap_or(text.trim()))
            {
                Ok(parsed @ Value::Object(_)) => parsed,
                _ => Value::String(text.clone()),
            }
        }
        Value::Object(_) => arguments.clone(),
        other => {
            crate::support::json_codec::dumps(other).unwrap_or_else(|_| other.to_string())
                .into()
        }
    }
}

/// `messages` checked and in canonical form. Each PDF would render to its
/// pages here; until the documents module ports, a file part is refused.
pub fn normalize_messages(messages: &Value, vision: bool) -> Result<Vec<Value>, MessageError> {
    let Some(items) = messages.as_array() else {
        return Err(MessageError::bad("messages must be a non-empty array"));
    };
    if items.is_empty() {
        return Err(MessageError::bad("messages must be a non-empty array"));
    }
    let mut normalized: Vec<Value> = Vec::new();
    let mut has_user = false;
    for message in items {
        let Some(map) = message.as_object() else {
            return Err(MessageError::bad("each message must be an object"));
        };
        let role = map.get("role").and_then(|r| r.as_str()).unwrap_or("");
        if role == "system" || role == "developer" {
            let text = text_content(map.get("content").unwrap_or(&Value::Null))?;
            if normalized.len() == 1
                && normalized[0].get("role").and_then(|r| r.as_str()) == Some("system")
            {
                // Responses instructions and developer items, or an
                // Anthropic system and a system message, lead as one
                // system message.
                let previous = normalized[0].get("content").and_then(|c| c.as_str()).unwrap_or("");
                let joined = [previous, &text]
                    .iter()
                    .filter(|part| !part.is_empty())
                    .map(|part| part.to_string())
                    .collect::<Vec<_>>()
                    .join("\n\n");
                if let Some(entry) = normalized[0].as_object_mut() {
                    entry.insert("content".into(), Value::String(joined));
                }
            } else {
                normalized.push(json!({"role": "system", "content": text}));
            }
            continue;
        }
        if !matches!(role, "user" | "assistant" | "tool") {
            return Err(MessageError::bad(format!("unsupported message role: {role}")));
        }
        has_user |= role == "user";
        let content = map.get("content").unwrap_or(&Value::Null);
        let mut item = Map::new();
        item.insert("role".into(), Value::String(role.to_string()));
        item.insert(
            "content".into(),
            if role == "user" || role == "tool" {
                user_content(content, vision)?
            } else {
                text_content(content)?.into()
            },
        );
        if role == "assistant" {
            // Clients send an assistant's reasoning back as Splash's
            // reasoning_content or as reasoning text.
            let reasoning = match (map.get("reasoning_content"), map.get("reasoning")) {
                (Some(Value::Null), Some(Value::String(text))) => Some(Value::String(text.clone())),
                (Some(reasoning), _) if !reasoning.is_null() => Some(reasoning.clone()),
                _ => None,
            };
            if let Some(reasoning) = reasoning {
                item.insert(
                    "reasoning_content".into(),
                    text_content(&reasoning)?.into(),
                );
            }
        }
        if role == "assistant" && !map.get("tool_calls").map(|c| c.is_null()).unwrap_or(true) {
            let Some(calls) = map.get("tool_calls").and_then(|c| c.as_array()) else {
                return Err(MessageError::bad("assistant tool_calls must be an array"));
            };
            let mut normalized_calls = Vec::new();
            for call in calls {
                let function = call.get("function").and_then(|f| f.as_object());
                let name = function
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str());
                let Some(name) = name else {
                    return Err(MessageError::bad("invalid assistant tool call"));
                };
                let mut normalized_call = json!({
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": call_arguments(
                            function.and_then(|f| f.get("arguments")).unwrap_or(&Value::Null),
                        ),
                    }
                });
                if let Some(id) = call.get("id") {
                    if !id.is_string() {
                        return Err(MessageError::bad("tool call id must be a string"));
                    }
                    normalized_call["id"] = id.clone();
                }
                normalized_calls.push(normalized_call);
            }
            item.insert("tool_calls".into(), Value::Array(normalized_calls));
        }
        if role == "tool" {
            if let Some(tool_call_id) = map.get("tool_call_id") {
                if !tool_call_id.is_string() {
                    return Err(MessageError::bad("tool_call_id must be a string"));
                }
                item.insert("tool_call_id".into(), tool_call_id.clone());
            }
        }
        normalized.push(Value::Object(item));
    }
    if !has_user {
        return Err(MessageError::bad("messages must include a user message"));
    }
    Ok(normalized)
}

/// Keep a call whose arguments are text, not an object, as text, without
/// presenting it as a complete call.
pub fn template_messages(messages: &[Value]) -> Vec<Value> {
    let mut result = Vec::new();
    for message in messages {
        let calls = message
            .get("tool_calls")
            .and_then(|c| c.as_array())
            .cloned()
            .unwrap_or_default();
        if !calls
            .iter()
            .any(|call| call["function"]["arguments"].is_string())
        {
            result.push(message.clone());
            continue;
        }
        let mut current = message
            .as_object()
            .map(|map| {
                map.iter()
                    .filter(|(key, _)| key.as_str() != "tool_calls")
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect::<Map<String, Value>>()
            })
            .unwrap_or_default();
        for call in &calls {
            if !call["function"]["arguments"].is_string() {
                current
                    .entry("tool_calls".to_string())
                    .or_insert_with(|| Value::Array(Vec::new()))
                    .as_array_mut()
                    .expect("array")
                    .push(call.clone());
                continue;
            }
            if ["content", "reasoning_content", "tool_calls"]
                .iter()
                .any(|key| current.get(*key).map(|v| !v.is_null() && !v.as_str().map(|s| s.is_empty()).unwrap_or(true)).unwrap_or(false))
            {
                result.push(Value::Object(current.clone()));
            }
            result.push(json!({
                "role": "assistant",
                "content": format!(
                    "Incomplete tool call:\n{}",
                    serde_json::to_string(&call["function"]).unwrap_or_default()
                ),
            }));
            current = json!({"role": "assistant", "content": ""}).as_object().unwrap().clone();
        }
        if ["content", "reasoning_content", "tool_calls"]
            .iter()
            .any(|key| current.get(*key).map(|v| !v.is_null() && !v.as_str().map(|s| s.is_empty()).unwrap_or(true)).unwrap_or(false))
        {
            result.push(Value::Object(current));
        }
    }
    result
}
