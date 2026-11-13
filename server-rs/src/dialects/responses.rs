//! The Responses API's input normalization, ported from
//! `server/dialects/responses.py`'s `normalize_responses_input`: the
//! instructions lead as the system message, and the input items become
//! the Chat messages — user and system messages as they are, an
//! assistant's reasoning and calls accumulating into one message, and a
//! call's output as a tool message.

use serde_json::{json, Value};

use crate::chat_messages::MessageError;

fn bad(message: impl Into<String>) -> MessageError {
    MessageError::bad(message)
}

/// The text of one content part's kinds.
fn part_text(part: &Value, kinds: &[&str]) -> Result<String, MessageError> {
    let kind = part.get("type").and_then(|t| t.as_str()).unwrap_or("");
    if !kinds.contains(&kind) {
        return Err(bad("only text message content is supported"));
    }
    part.get("text")
        .and_then(|t| t.as_str())
        .map(String::from)
        .ok_or_else(|| bad("text message content must be a string"))
}

/// The text of a Responses content: a string, or its parts' text.
fn responses_text(value: &Value, kinds: &[&str]) -> Result<String, MessageError> {
    match value {
        Value::Null => Ok(String::new()),
        Value::String(text) => Ok(text.clone()),
        Value::Array(parts) => {
            let mut out = String::new();
            for part in parts {
                out.push_str(&part_text(part, kinds)?);
            }
            Ok(out)
        }
        _ => Err(bad("message content must be text")),
    }
}

/// The content of a user message: its text, or its parts'.
fn responses_content(content: &Value) -> Result<Value, MessageError> {
    match content {
        Value::String(text) => Ok(Value::String(text.clone())),
        Value::Array(parts) => {
            let mut out = String::new();
            for part in parts {
                out.push_str(&part_text(part, &["input_text", "text"])?);
            }
            Ok(Value::String(out))
        }
        _ => Err(bad("message content must be text")),
    }
}

/// The instructions lead as the system message, and the input items
/// become the Chat messages.
pub fn normalize_responses_input(
    instructions: &Value,
    items: &Value,
) -> Result<Vec<Value>, MessageError> {
    let instructions = match instructions {
        Value::Null => String::new(),
        Value::String(text) => text.clone(),
        _ => return Err(bad("instructions must be a string")),
    };
    let Some(items) = items.as_array() else {
        return Err(bad("input must be a non-empty string or array"));
    };
    if items.is_empty() {
        return Err(bad("input must be a non-empty string or array"));
    }
    let mut messages: Vec<Value> = Vec::new();
    if !instructions.is_empty() {
        messages.push(json!({"role": "system", "content": instructions}));
    }
    let mut pending: Option<Value> = None;
    let flush = |messages: &mut Vec<Value>, pending: &mut Option<Value>| {
        if let Some(current) = pending.take() {
            messages.push(current);
        }
    };
    for item in items {
        let Some(map) = item.as_object() else {
            return Err(bad("each input item must be an object"));
        };
        let kind = map.get("type").and_then(|t| t.as_str()).unwrap_or("message");
        if kind == "message" {
            let role = map.get("role").and_then(|r| r.as_str()).unwrap_or("");
            if role == "user" {
                flush(&mut messages, &mut pending);
                let content = responses_content(map.get("content").unwrap_or(&Value::Null))?;
                messages.push(json!({"role": role, "content": content}));
                continue;
            }
            let text = responses_text(
                map.get("content").unwrap_or(&Value::Null),
                &["input_text", "output_text", "text"],
            )?;
            if role == "system" || role == "developer" {
                flush(&mut messages, &mut pending);
                messages.push(json!({"role": "system", "content": text}));
            } else if role == "assistant" {
                let current = pending.get_or_insert_with(|| json!({"role": "assistant", "content": ""}));
                let existing = current["content"].as_str().unwrap_or("").to_string();
                current["content"] = json!(format!("{existing}{text}"));
            } else {
                return Err(bad(format!("unsupported message role: {role}")));
            }
        } else if kind == "reasoning" {
            let content = map.get("content");
            let text = match content {
                Some(content) if !content.is_null() => {
                    responses_text(content, &["reasoning_text", "text"])?
                }
                _ => String::new(),
            };
            let text = if text.is_empty() {
                responses_text(map.get("summary").unwrap_or(&json!([])), &["summary_text"])?
            } else {
                text
            };
            if !text.is_empty() {
                let current = pending.get_or_insert_with(|| json!({"role": "assistant", "content": ""}));
                let existing = current["reasoning_content"].as_str().unwrap_or("").to_string();
                current["reasoning_content"] = json!(format!("{existing}{text}"));
            }
        } else if kind == "function_call" {
            let call_id = map.get("call_id").and_then(|c| c.as_str());
            let name = map.get("name").and_then(|n| n.as_str());
            let (Some(call_id), Some(name)) = (call_id, name) else {
                return Err(bad("function_call requires call_id and name"));
            };
            if call_id.is_empty() || name.is_empty() {
                return Err(bad("function_call requires call_id and name"));
            }
            let Some(arguments) = map.get("arguments").and_then(|a| a.as_str()) else {
                return Err(bad("function_call arguments must be JSON text"));
            };
            let current = pending.get_or_insert_with(|| json!({"role": "assistant", "content": ""}));
            current["tool_calls"]
                .as_array_mut()
                .map(|calls| {
                    calls.push(json!({
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }))
                })
                .unwrap_or_else(|| {
                    current["tool_calls"] = json!([{
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }]);
                });
        } else if kind == "custom_tool_call" {
            let call_id = map.get("call_id").and_then(|c| c.as_str());
            let name = map.get("name").and_then(|n| n.as_str());
            let (Some(call_id), Some(name)) = (call_id, name) else {
                return Err(bad("custom_tool_call requires call_id and name"));
            };
            let Some(text) = map.get("input").and_then(|t| t.as_str()) else {
                return Err(bad("custom_tool_call input must be a string"));
            };
            let current = pending.get_or_insert_with(|| json!({"role": "assistant", "content": ""}));
            // The call as the function the model made it.
            current["tool_calls"]
                .as_array_mut()
                .map(|calls| {
                    calls.push(json!({
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": {"__custom_input": text}},
                    }))
                })
                .unwrap_or_else(|| {
                    current["tool_calls"] = json!([{
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": {"__custom_input": text}},
                    }]);
                });
        } else if kind == "function_call_output" || kind == "custom_tool_call_output" {
            let call_id = map.get("call_id").and_then(|c| c.as_str());
            let Some(call_id) = call_id.filter(|id| !id.is_empty()) else {
                return Err(bad(format!("{kind} requires call_id")));
            };
            let output = map.get("output").unwrap_or(&Value::Null);
            let output = output.get("content").unwrap_or(output);
            flush(&mut messages, &mut pending);
            let content = responses_content(output)?;
            messages.push(json!({
                "role": "tool",
                "tool_call_id": call_id,
                "content": content,
            }));
        } else {
            return Err(bad(
                "only message, reasoning, function_call, function_call_output, \
                 custom_tool_call and custom_tool_call_output input items are \
                 supported",
            ));
        }
    }
    flush(&mut messages, &mut pending);
    Ok(messages)
}
