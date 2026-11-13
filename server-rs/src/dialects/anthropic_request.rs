//! A Messages request as a Chat body, ported from
//! `server/dialects/anthropic.py`'s request conversion: the system text,
//! the messages' content blocks, the thinking's effort and display, and
//! the sampling fields.

use serde_json::{json, Map, Value};

use crate::chat_messages::MessageError;

fn bad(message: impl Into<String>) -> MessageError {
    MessageError::bad(message)
}

/// The thinking a request carries, as the response shows it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Thinking {
    /// The reasoning effort the request resolves to: "none" when disabled.
    pub effort: String,
    /// How the response shows the reasoning: summarized or omitted.
    pub display: String,
}

/// The reasoning effort and display of a Messages request's thinking,
/// which is off when omitted.
pub fn anthropic_thinking(thinking: &Value, effort: &str) -> Result<Thinking, MessageError> {
    if thinking.is_null() {
        return Ok(Thinking {
            effort: "none".into(),
            display: "summarized".into(),
        });
    }
    let Some(map) = thinking.as_object() else {
        return Err(bad("thinking.type must be enabled, disabled, or adaptive"));
    };
    let thinking_type = map.get("type").and_then(|t| t.as_str()).unwrap_or("");
    if !matches!(thinking_type, "enabled" | "disabled" | "adaptive") {
        return Err(bad("thinking.type must be enabled, disabled, or adaptive"));
    }
    let display = map.get("display");
    if let Some(display) = display {
        if thinking_type == "disabled" {
            return Err(bad("thinking.display requires enabled or adaptive thinking"));
        }
        let display = display.as_str().unwrap_or("");
        if !matches!(display, "summarized" | "omitted" | "updates") {
            return Err(bad(
                "thinking.display must be summarized, omitted, updates, or null",
            ));
        }
    }
    let display_value = display.and_then(|d| d.as_str()).unwrap_or("");
    Ok(Thinking {
        effort: if thinking_type == "disabled" {
            "none".into()
        } else {
            effort.to_string()
        },
        display: if display_value == "omitted" || display_value == "updates" {
            "omitted".into()
        } else {
            "summarized".into()
        },
    })
}

/// System text without billing-header blocks; `None` when nothing
/// remains.
fn anthropic_system_text(value: &Value, label: &str) -> Result<Option<String>, MessageError> {
    let blocks = match value {
        Value::String(text) => vec![json!({"type": "text", "text": text})],
        Value::Array(blocks) => blocks.clone(),
        _ => return Err(bad(format!("{label} must be text content"))),
    };
    let mut parts = Vec::new();
    for block in &blocks {
        let text = block
            .get("type")
            .and_then(|t| t.as_str())
            .filter(|kind| *kind == "text")
            .and_then(|_| block.get("text"))
            .and_then(|text| text.as_str());
        let Some(text) = text else {
            return Err(bad(format!("{label} contains unsupported content")));
        };
        // Claude Code prepends a billing-header text block; it is not user
        // text.
        if !text.starts_with("x-anthropic-billing-header") {
            parts.push(text.to_string());
        }
    }
    let joined = parts.concat();
    Ok((!joined.is_empty()).then_some(joined))
}

/// Text for plain Anthropic content; canonical parts in document order
/// when its blocks carry images. User messages and tool results share it,
/// so the template places each image where the author put it.
fn anthropic_content(value: &Value, label: &str) -> Result<Value, MessageError> {
    if let Some(text) = value.as_str() {
        return Ok(Value::String(text.to_string()));
    }
    let Some(blocks) = value.as_array() else {
        return Err(bad(format!("{label} must be text or content blocks")));
    };
    let mut parts = Vec::new();
    for block in blocks {
        let kind = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
        if kind == "text" && block.get("text").and_then(|t| t.as_str()).is_some() {
            parts.push(json!({"type": "text", "text": block["text"]}));
        } else if kind == "image" {
            let source = block.get("source");
            let valid = source
                .and_then(|s| s.as_object())
                .map(|s| {
                    s.get("type").and_then(|t| t.as_str()) == Some("base64")
                        && s.get("media_type").and_then(|m| m.as_str()).is_some()
                        && s.get("data").and_then(|d| d.as_str()).is_some()
                })
                .unwrap_or(false);
            if !valid {
                return Err(bad("Anthropic image blocks must be base64 sources"));
            }
            let url = format!(
                "data:{};base64,{}",
                source.unwrap()["media_type"].as_str().unwrap(),
                source.unwrap()["data"].as_str().unwrap()
            );
            parts.push(json!({"type": "image_url", "image_url": {"url": url}}));
        } else {
            return Err(bad(format!("{label} contains unsupported content")));
        }
    }
    // Canonical parts, or their text when they carry no image.
    if parts
        .iter()
        .any(|part| part.get("type").and_then(|t| t.as_str()) != Some("text"))
    {
        Ok(Value::Array(parts))
    } else {
        let text: String = parts
            .iter()
            .filter_map(|part| part.get("text").and_then(|t| t.as_str()))
            .collect();
        Ok(Value::String(text))
    }
}

/// A Messages request's stop_sequences, as Chat's stop.
fn stop_sequences(stops: &Value) -> Result<Option<Value>, MessageError> {
    if stops.is_null() {
        return Ok(None);
    }
    let Some(items) = stops.as_array() else {
        return Err(bad("stop_sequences must be an array of strings"));
    };
    if !items.iter().all(|stop| stop.is_string()) {
        return Err(bad("stop_sequences must be an array of strings"));
    }
    if items
        .iter()
        .any(|stop| stop.as_str().map(|s| s.trim().is_empty()).unwrap_or(true))
    {
        return Err(bad("each of stop_sequences must contain non-whitespace"));
    }
    if items.len() > 4 {
        return Err(bad("stop_sequences must be up to four strings"));
    }
    Ok(Some(stops.clone()))
}

/// The Chat messages one Messages message stands for: a system message's
/// text; an assistant's text, reasoning and calls as one message; a
/// user's tool results, then the rest of its content.
fn anthropic_message(
    message: &Value,
    thinking_resolver: &dyn Fn(&str) -> Result<String, MessageError>,
) -> Result<Vec<Value>, MessageError> {
    let Some(map) = message.as_object() else {
        return Err(bad("Anthropic messages must be objects"));
    };
    let role = map.get("role").and_then(|r| r.as_str()).unwrap_or("");
    let content = map.get("content").unwrap_or(&Value::Null);
    if !matches!(role, "user" | "assistant" | "system") {
        return Err(bad("Anthropic messages require user/assistant/system roles"));
    }
    if role == "system" {
        let text = anthropic_system_text(content, "system message")?;
        return Ok(match text {
            Some(text) => vec![json!({"role": "system", "content": text})],
            None => Vec::new(),
        });
    }
    if let Some(text) = content.as_str() {
        return Ok(vec![json!({"role": role, "content": text})]);
    }
    let Some(blocks) = content.as_array() else {
        return Err(bad("Anthropic message content must be text or blocks"));
    };
    let mut text_parts = Vec::new();
    let mut user_blocks = Vec::new();
    let mut reasoning_parts = Vec::new();
    let mut tool_calls = Vec::new();
    let mut tool_results = Vec::new();
    for block in blocks {
        let Some(_block_map) = block.as_object() else {
            return Err(bad("Anthropic content blocks must be objects"));
        };
        let kind = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
        if kind == "text" && block.get("text").and_then(|t| t.as_str()).is_some() {
            text_parts.push(block["text"].as_str().unwrap().to_string());
            user_blocks.push(block.clone());
        } else if role == "user" && (kind == "image" || kind == "document") {
            user_blocks.push(block.clone());
        } else if role == "assistant" && kind == "thinking" {
            let Some(thinking) = block.get("thinking").and_then(|t| t.as_str()) else {
                return Err(bad("unsupported Anthropic content block: thinking"));
            };
            let signature = block.get("signature").and_then(|s| s.as_str()).unwrap_or("");
            if signature.is_empty() {
                reasoning_parts.push(thinking.to_string());
            } else {
                match thinking_resolver(signature) {
                    Ok(reasoning) => {
                        if !reasoning.is_empty() {
                            reasoning_parts.push(reasoning);
                        }
                    }
                    Err(error) => {
                        // Other providers' signatures are opaque, as are
                        // ours under another key. Preserve their visible
                        // history; hidden content is dropped, as
                        // redacted_thinking is.
                        if error.code != "invalid_thinking_signature" {
                            return Err(error);
                        }
                        reasoning_parts.push(thinking.to_string());
                    }
                }
            }
        } else if role == "assistant" && kind == "redacted_thinking" {
            // Provider-encrypted reasoning cannot be rendered in this
            // model's prompt. Keep the surrounding visible history.
        } else if role == "assistant" && kind == "tool_use" {
            let id = block.get("id").and_then(|i| i.as_str());
            let name = block.get("name").and_then(|n| n.as_str());
            let input = block.get("input").and_then(|i| i.as_object());
            let (Some(id), Some(name), Some(input)) = (id, name, input) else {
                return Err(bad("invalid Anthropic tool_use block"));
            };
            tool_calls.push(json!({
                "id": id,
                "type": "function",
                "function": {"name": name, "arguments": input},
            }));
        } else if role == "user" && kind == "tool_result" {
            tool_results.push(anthropic_tool_result(block)?);
        } else {
            return Err(bad(format!("unsupported Anthropic content block: {kind}")));
        }
    }
    if role == "assistant" {
        let mut item = Map::new();
        item.insert("role".into(), json!(role));
        item.insert("content".into(), json!(text_parts.concat()));
        if !reasoning_parts.is_empty() {
            item.insert("reasoning_content".into(), json!(reasoning_parts.concat()));
        }
        if !tool_calls.is_empty() {
            item.insert("tool_calls".into(), json!(tool_calls));
        }
        return Ok(vec![Value::Object(item)]);
    }
    if !user_blocks.is_empty() || tool_results.is_empty() {
        let user = json!({
            "role": role,
            "content": anthropic_content(&Value::Array(user_blocks), "user message")?,
        });
        let mut out = tool_results;
        out.push(user);
        return Ok(out);
    }
    Ok(tool_results)
}

/// A tool result as a Chat tool message: its content as text, its call's
/// id.
fn anthropic_tool_result(block: &Value) -> Result<Value, MessageError> {
    let tool_call_id = block.get("tool_use_id").and_then(|id| id.as_str());
    let Some(tool_call_id) = tool_call_id else {
        return Err(bad("invalid Anthropic tool_result block"));
    };
    let content = block.get("content").unwrap_or(&Value::Null);
    let text = match content {
        Value::String(text) => text.clone(),
        Value::Array(parts) => {
            let mut out = String::new();
            for part in parts {
                let kind = part.get("type").and_then(|t| t.as_str()).unwrap_or("");
                if kind == "text" {
                    out.push_str(part.get("text").and_then(|t| t.as_str()).unwrap_or(""));
                } else {
                    return Err(bad("unsupported Anthropic tool_result content"));
                }
            }
            out
        }
        _ => String::new(),
    };
    Ok(json!({"role": "tool", "tool_call_id": tool_call_id, "content": text}))
}

/// A Messages generation request as a Chat body, and how its response
/// shows reasoning.
pub fn anthropic_to_chat_body(
    body: &Value,
    thinking_resolver: &dyn Fn(&str) -> Result<String, MessageError>,
) -> Result<(Value, String), MessageError> {
    let max_tokens = body.get("max_tokens").and_then(|t| t.as_u64());
    let Some(max_tokens) = max_tokens else {
        return Err(bad("max_tokens must be a positive integer"));
    };
    if max_tokens == 0 {
        return Err(bad("max_tokens must be a positive integer"));
    }
    if !body.get("stream").map(|s| s.is_boolean()).unwrap_or(true) {
        return Err(bad("stream must be a boolean"));
    }
    let (chat, thinking_display) = anthropic_chat(body, thinking_resolver)?;
    // Anthropic continues a final assistant message (a prefill); Splash
    // would close that turn and start another, so it refuses it.
    let last_role = chat["messages"]
        .as_array()
        .and_then(|messages| messages.last())
        .and_then(|last| last.get("role"))
        .and_then(|role| role.as_str());
    if last_role == Some("assistant") {
        return Err(bad(
            "a final assistant message (prefill) is not supported; end messages with a user turn",
        ));
    }
    let mut result = Map::new();
    result.insert("model".into(), chat["model"].clone());
    result.insert("messages".into(), chat["messages"].clone());
    if let Some(effort) = chat.get("reasoning_effort") {
        result.insert("reasoning_effort".into(), effort.clone());
    }
    result.insert("max_completion_tokens".into(), json!(max_tokens));
    if let Some(stop) = stop_sequences(&body.get("stop_sequences").cloned().unwrap_or(Value::Null))? {
        result.insert("stop".into(), stop);
    }
    for field in ["temperature", "top_p", "top_k", "seed"] {
        if let Some(value) = body.get(field) {
            if !value.is_null() {
                result.insert(field.into(), value.clone());
            }
        }
    }
    Ok((Value::Object(result), thinking_display))
}

/// A Messages request's prompt as a Chat body, and how its response shows
/// reasoning.
fn anthropic_chat(
    body: &Value,
    thinking_resolver: &dyn Fn(&str) -> Result<String, MessageError>,
) -> Result<(Value, String), MessageError> {
    let model = body.get("model").and_then(|m| m.as_str());
    let Some(model) = model.filter(|m| !m.is_empty()) else {
        return Err(bad("model must be a non-empty string"));
    };
    for field in ["grammar", "regex"] {
        if !body.get(field).map(|v| v.is_null()).unwrap_or(true) {
            return Err(bad(format!(
                "{field} is supported on Chat Completions and text completions alone"
            )));
        }
    }
    let empty = json!({});
    let output_config = body.get("output_config").unwrap_or(&empty);
    let Some(output_config) = output_config.as_object() else {
        return Err(bad("output_config must be an object"));
    };
    let effort = output_config
        .get("effort")
        .and_then(|e| e.as_str())
        .unwrap_or("high");
    if !matches!(effort, "low" | "medium" | "high" | "xhigh" | "max") {
        return Err(bad("output_config.effort is invalid"));
    }
    let messages = body.get("messages").and_then(|m| m.as_array());
    let Some(messages) = messages else {
        return Err(bad("messages must be a non-empty array"));
    };
    if messages.is_empty() {
        return Err(bad("messages must be a non-empty array"));
    }
    let thinking = anthropic_thinking(&body.get("thinking").cloned().unwrap_or(Value::Null), effort)?;
    let mut chat_messages = Vec::new();
    for message in messages {
        chat_messages.extend(anthropic_message(message, thinking_resolver)?);
    }
    let system_text = match body.get("system") {
        None | Some(Value::Null) => None,
        Some(system) => anthropic_system_text(system, "system")?,
    };
    let mut all = Vec::new();
    if let Some(system_text) = system_text {
        all.push(json!({"role": "system", "content": system_text}));
    }
    all.extend(chat_messages);
    let mut chat = Map::new();
    chat.insert("model".into(), json!(model));
    chat.insert("messages".into(), json!(all));
    if thinking.effort != "high" {
        chat.insert("reasoning_effort".into(), json!(thinking.effort));
    }
    Ok((Value::Object(chat), thinking.display))
}
