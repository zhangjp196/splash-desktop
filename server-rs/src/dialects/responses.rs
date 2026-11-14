//! The Responses API's input normalization, ported from
//! `server/dialects/responses.py`'s `normalize_responses_input`: the
//! instructions lead as the system message, and the input items become
//! the Chat messages — user and system messages as they are, an
//! assistant's reasoning and calls accumulating into one message, and a
//! call's output as a tool message.

use std::sync::{Arc, Mutex};
use std::collections::HashMap;

use serde_json::{json, Value};

use crate::chat_messages::MessageError;
use crate::support::lru::LruCache;

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
        Value::Null => Ok(Value::String(String::new())),
        Value::String(text) => Ok(Value::String(text.clone())),
        Value::Array(parts) => {
            let mut output = Vec::new();
            for part in parts {
                let kind = part.get("type").and_then(Value::as_str).unwrap_or("");
                match kind {
                    "input_image" => {
                        let image = part.get("image_url").cloned().unwrap_or(Value::Null);
                        let url = image.get("url").cloned().unwrap_or(image);
                        output.push(json!({"type":"image_url", "image_url":{"url":url}}));
                    }
                    "input_file" => {
                        let mut file = part.as_object().cloned().unwrap_or_default();
                        file.remove("type");
                        output.push(json!({"type":"file", "file":Value::Object(file)}));
                    }
                    "input_text" | "output_text" | "text" => {
                        output.push(json!({"type":"text", "text":part_text(part, &["input_text", "output_text", "text"])?}));
                    }
                    _ => return Err(bad("only text, image and inline PDF Responses content is supported")),
                }
            }
            if output.iter().any(|part| part.get("type").and_then(Value::as_str) != Some("text")) {
                Ok(Value::Array(output))
            } else {
                Ok(Value::String(output.iter().filter_map(|part| part.get("text").and_then(Value::as_str)).collect()))
            }
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

/// A Responses input string is shorthand for one user message.
pub fn canonical_responses_input(input: &Value) -> Result<Vec<Value>, MessageError> {
    match input {
        Value::String(text) => Ok(vec![json!({"type": "message", "role": "user", "content": text})]),
        Value::Array(items) if !items.is_empty() => Ok(items.clone()),
        Value::Array(_) => Err(bad("input must be a non-empty string or array")),
        _ => Err(bad("input must be a non-empty string or array")),
    }
}

/// Convert a Responses request and its continued history into the Chat
/// body accepted by the existing generation pipeline.
pub fn responses_to_chat_body(body: &Value, items: &[Value]) -> Result<Value, MessageError> {
    responses_to_chat_body_with_tools(body, items).map(|(chat, _)| chat)
}

/// Normalize a Responses request and retain the custom/namespace mapping
/// needed to turn generated Chat tool calls back into Responses items.
pub fn responses_to_chat_body_with_tools(body: &Value, items: &[Value]) -> Result<(Value, Value), MessageError> {
    let object = body.as_object().ok_or_else(|| bad("request body must be an object"))?;
    if object.get("conversation").is_some_and(|value| !value.is_null()) {
        return Err(bad("conversation is not supported"));
    }
    if object.get("background").is_some_and(|value| value != &Value::Bool(false) && !value.is_null()) {
        return Err(bad("background responses are not supported"));
    }
    if object.get("truncation").is_some_and(|value| value != &Value::String("disabled".into()) && !value.is_null()) {
        return Err(bad("only disabled truncation is supported"));
    }
    for field in ["grammar", "regex"] {
        if object.get(field).is_some_and(|value| !value.is_null()) {
            return Err(bad(format!("{field} is supported on Chat Completions and text completions alone")));
        }
    }
    if object.get("stream").is_some_and(|value| !value.is_boolean() && !value.is_null()) {
        return Err(bad("stream must be a boolean"));
    }

    let mut tool_metadata = json!({});
    let mut chat = json!({
        "messages": normalize_responses_input(object.get("instructions").unwrap_or(&Value::Null), &Value::Array(items.to_vec()))?,
        "parallel_tool_calls": object.get("parallel_tool_calls").cloned().unwrap_or(Value::Null),
    });
    if let Some(tools) = object.get("tools").filter(|value| !value.is_null()) {
        let Some(tools) = tools.as_array() else { return Err(bad("tools must be an array")) };
        let (converted, metadata) = normalize_responses_tools(tools)?;
        tool_metadata = metadata;
        chat["tools"] = Value::Array(converted);
    }
    if let Some(choice) = object.get("tool_choice").filter(|value| !value.is_null()) {
        if let Some(choice) = choice.as_str() {
            if !matches!(choice, "auto" | "none" | "required") {
                return Err(bad("invalid tool_choice"));
            }
            chat["tool_choice"] = json!(choice);
        } else if matches!(choice.get("type").and_then(Value::as_str), Some("function" | "custom")) {
            let original_name = choice.get("name").and_then(Value::as_str)
                .ok_or_else(|| bad("invalid named tool_choice"))?;
            let name = if choice.get("type").and_then(Value::as_str) == Some("custom") {
                if choice.get("namespace").is_some_and(|namespace| !namespace.is_null())
                    || tool_metadata.get(original_name).and_then(|metadata| metadata.get("kind")).and_then(Value::as_str) != Some("custom")
                { return Err(bad("invalid named tool_choice")); }
                original_name.to_owned()
            } else if let Some(namespace) = choice.get("namespace").filter(|value| !value.is_null()) {
                let namespace = namespace.as_str().ok_or_else(|| bad("invalid namespace tool name"))?;
                let alias = namespace_alias(namespace, original_name)?;
                let mapping = tool_metadata.get(&alias).ok_or_else(|| bad("invalid named tool_choice"))?;
                if mapping.get("kind").and_then(Value::as_str) != Some("namespace")
                    || mapping.get("name").and_then(Value::as_str) != Some(original_name)
                    || mapping.get("namespace").and_then(Value::as_str) != Some(namespace)
                { return Err(bad("invalid named tool_choice")); }
                alias
            } else {
                original_name.to_owned()
            };
            chat["tool_choice"] = json!({"type":"function", "function":{"name":name}});
        } else {
            return Err(bad("invalid named tool_choice"));
        }
    }
    if let Some(reasoning) = object.get("reasoning").filter(|value| !value.is_null()) {
        let Some(reasoning) = reasoning.as_object() else { return Err(bad("reasoning must be an object")) };
        if let Some(effort) = reasoning.get("effort").filter(|value| !value.is_null()) {
            chat["reasoning_effort"] = effort.clone();
        }
    }
    if let Some(text) = object.get("text").filter(|value| !value.is_null()) {
        let Some(text) = text.as_object() else { return Err(bad("text.format must be an object")) };
        if let Some(format) = text.get("format").filter(|value| !value.is_null()) {
            let Some(format) = format.as_object() else { return Err(bad("text.format must be an object")) };
            match format.get("type").and_then(Value::as_str).unwrap_or("text") {
                "text" => {}
                "json_object" => chat["response_format"] = json!({"type":"json_object"}),
                "json_schema" => {
                    let mut schema = json!({});
                    for field in ["name", "description", "schema", "strict"] {
                        if let Some(value) = format.get(field) { schema[field] = value.clone(); }
                    }
                    chat["response_format"] = json!({"type":"json_schema", "json_schema":schema});
                }
                _ => return Err(bad("only text and JSON text formats are supported")),
            }
        }
    }
    for (source, target) in [
        ("max_output_tokens", "max_completion_tokens"),
        ("model", "model"), ("temperature", "temperature"), ("top_p", "top_p"),
        ("top_k", "top_k"), ("min_p", "min_p"), ("frequency_penalty", "frequency_penalty"),
        ("presence_penalty", "presence_penalty"), ("logit_bias", "logit_bias"),
        ("seed", "seed"), ("timeout", "timeout"), ("priority", "priority"),
        ("cache_salt", "cache_salt"), ("stop", "stop"), ("ignore_eos", "ignore_eos"),
        ("chat_template_kwargs", "chat_template_kwargs"),
        ("preserve_thinking", "preserve_thinking"),
        ("thinking_token_budget", "thinking_token_budget"),
    ] {
        if let Some(value) = object.get(source).filter(|value| !value.is_null()) {
            chat[target] = value.clone();
        }
    }
    Ok((chat, tool_metadata))
}

fn normalize_responses_tools(tools: &[Value]) -> Result<(Vec<Value>, Value), MessageError> {
    let mut converted = Vec::new();
    let mut metadata = serde_json::Map::new();
    for tool in tools {
        let Some(tool) = tool.as_object() else { return Err(bad("only function and custom tools are supported")); };
        match tool.get("type").and_then(Value::as_str) {
            Some("function") => converted.push(response_function(tool, None)?),
            Some("namespace") => {
                let namespace = tool.get("name").and_then(Value::as_str).ok_or_else(|| bad("invalid namespace tool name"))?;
                let children = tool.get("tools").and_then(Value::as_array).ok_or_else(|| bad("namespace tools must be an array"))?;
                for child in children {
                    let child = child.as_object().filter(|child| child.get("type").and_then(Value::as_str) == Some("function"))
                        .ok_or_else(|| bad("only function namespace tools are supported"))?;
                    let name = child.get("name").and_then(Value::as_str).ok_or_else(|| bad("invalid namespace tool name"))?;
                    let alias = namespace_alias(namespace, name)?;
                    if metadata.contains_key(&alias) { return Err(bad(format!("duplicate tool name: {alias}"))); }
                    metadata.insert(alias.clone(), json!({"kind":"namespace", "namespace":namespace, "name":name}));
                    converted.push(response_function(child, Some(&alias))?);
                }
            }
            Some("custom") => {
                let name = tool.get("name").and_then(Value::as_str).unwrap_or("");
                if !crate::tool_schema::is_tag_name(name) {
                    return Err(bad(format!("custom tool name must be {}", crate::tool_schema::TAG_NAME_RULE)));
                }
                if metadata.contains_key(name) { return Err(bad(format!("duplicate tool name: {name}"))); }
                let description = tool.get("description").cloned().unwrap_or(json!(""));
                if !description.is_string() { return Err(bad(format!("custom tool {name} description must be a string"))); }
                let format = tool.get("format").cloned().unwrap_or(json!({"type":"text"}));
                let format = format.as_object().ok_or_else(|| bad(format!("custom tool {name} format must be text or a grammar")))?;
                match format.get("type").and_then(Value::as_str) {
                    Some("text") => {}
                    Some("grammar") if format.get("syntax").and_then(Value::as_str).is_some() && format.get("definition").and_then(Value::as_str).is_some() => {}
                    _ => return Err(bad(format!("custom tool {name} format must be text or a grammar"))),
                }
                let explained = if format.get("type").and_then(Value::as_str) == Some("grammar") {
                    format!("The tool's input, as raw text that this {} grammar accepts:\\n{}", format["syntax"].as_str().unwrap(), format["definition"].as_str().unwrap())
                } else { "The tool's input, as raw text.".to_owned() };
                converted.push(json!({"type":"function", "function":{
                    "name":name,
                    "description":description,
                    "parameters":{"type":"object","properties":{"input":{"type":"string","description":explained}},"required":["input"],"additionalProperties":false}
                }}));
                metadata.insert(name.to_owned(), json!({"kind":"custom", "name":name}));
            }
            _ => {
                let kind = tool.get("type").map(|value| format!("{value:?}")).unwrap_or_else(|| "None".into());
                return Err(bad(format!("only function and custom tools are supported, not {kind}")));
            }
        }
    }
    Ok((converted, Value::Object(metadata)))
}

fn response_function(tool: &serde_json::Map<String, Value>, name: Option<&str>) -> Result<Value, MessageError> {
    let source = tool.get("function").and_then(Value::as_object).unwrap_or(tool);
    let mut function = json!({});
    if let Some(tool_name) = name.or_else(|| source.get("name").and_then(Value::as_str)) {
        function["name"] = json!(tool_name);
    }
    for field in ["description", "parameters", "strict"] {
        if let Some(value) = source.get(field) { function[field] = value.clone(); }
    }
    Ok(json!({"type":"function", "function":function}))
}

fn namespace_alias(namespace: &str, name: &str) -> Result<String, MessageError> {
    if namespace.is_empty() || name.is_empty() { return Err(bad("invalid namespace tool name")); }
    Ok(format!("{namespace}__{name}"))
}

/// Bounded in-memory history for `store` and `previous_response_id`.
#[derive(Clone)]
pub struct StoredResponse {
    pub response: Value,
    pub history: Vec<Value>,
}

#[derive(Clone)]
pub struct ResponseStore {
    entries: Arc<Mutex<LruCache<String, StoredResponse>>>,
}

impl Default for ResponseStore {
    fn default() -> Self { Self::new() }
}

impl ResponseStore {
    pub fn new() -> Self {
        Self { entries: Arc::new(Mutex::new(LruCache::new(64 * 1024 * 1024, Some(128)))) }
    }

    pub fn get(&self, id: &str) -> Option<StoredResponse> {
        self.entries.lock().expect("response store lock").get(&id.to_owned()).cloned()
    }

    pub fn put(&self, id: String, response: Value, history: Vec<Value>) {
        let stored = StoredResponse { response, history };
        let size = serde_json::to_vec(&stored.response).map_or(0, |bytes| bytes.len())
            .saturating_add(serde_json::to_vec(&stored.history).map_or(0, |bytes| bytes.len()));
        self.entries.lock().expect("response store lock").put(id, stored, size);
    }

    pub fn delete(&self, id: &str) -> bool {
        self.entries.lock().expect("response store lock").pop(&id.to_owned()).is_some()
    }

}

/// Convert the generated Chat choice into ordered Responses output items.
pub fn output_from_chat(public_id: &str, chat_response: &Value) -> Result<Vec<Value>, MessageError> {
    output_from_chat_with_tools(public_id, chat_response, &Value::Null)
}

pub fn output_from_chat_with_tools(public_id: &str, chat_response: &Value, tool_metadata: &Value) -> Result<Vec<Value>, MessageError> {
    let message = chat_response.pointer("/choices/0/message")
        .ok_or_else(|| bad("chat generation returned no assistant message"))?;
    let mut blocks = Vec::new();
    if let Some(reasoning) = message.get("reasoning_content").and_then(Value::as_str).filter(|text| !text.is_empty()) {
        blocks.push(OutputBlock { kind: "reasoning".into(), text: reasoning.into(), call_id: None, name: None, status: "completed".into() });
    }
    if let Some(text) = message.get("content").and_then(Value::as_str).filter(|text| !text.is_empty()) {
        blocks.push(OutputBlock { kind: "text".into(), text: text.into(), call_id: None, name: None, status: "completed".into() });
    }
    if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
        for call in calls {
            let function = call.get("function").ok_or_else(|| bad("chat generation returned an invalid tool call"))?;
            blocks.push(OutputBlock {
                kind: "tool".into(),
                text: function.get("arguments").and_then(Value::as_str).unwrap_or("").into(),
                call_id: call.get("id").and_then(Value::as_str).map(str::to_owned),
                name: function.get("name").and_then(Value::as_str).map(str::to_owned),
                status: "completed".into(),
            });
        }
    }
    responses_output_with_tools(public_id, &blocks, tool_metadata)
}

/// Responses-shaped completion around a Chat completion result.
pub fn response_from_chat(
    model: &str,
    public_id: &str,
    created_at: i64,
    request: &Value,
    chat_response: &Value,
    store: bool,
    previous_response_id: Option<&str>,
) -> Result<Value, MessageError> {
    response_from_chat_with_tools(model, public_id, created_at, request, chat_response, store, previous_response_id, &Value::Null)
}

pub fn response_from_chat_with_tools(
    model: &str,
    public_id: &str,
    created_at: i64,
    request: &Value,
    chat_response: &Value,
    store: bool,
    previous_response_id: Option<&str>,
    tool_metadata: &Value,
) -> Result<Value, MessageError> {
    let output = output_from_chat_with_tools(public_id, chat_response, tool_metadata)?;
    response_from_output(model, public_id, created_at, request, chat_response, output, store, previous_response_id)
}

pub fn response_from_output(
    model: &str,
    public_id: &str,
    created_at: i64,
    request: &Value,
    chat_response: &Value,
    output: Vec<Value>,
    store: bool,
    previous_response_id: Option<&str>,
) -> Result<Value, MessageError> {
    let finish = chat_response.pointer("/choices/0/finish_reason").and_then(Value::as_str).unwrap_or("stop");
    let status = match finish {
        "length" => "incomplete",
        "cancelled" => "cancelled",
        _ => "completed",
    };
    let usage = chat_response.get("usage").map(|usage| {
        let prompt = usage.get("prompt_tokens").and_then(Value::as_u64).unwrap_or(0);
        let completion = usage.get("completion_tokens").and_then(Value::as_u64).unwrap_or(0);
        let cached = usage.pointer("/prompt_tokens_details/cached_tokens").and_then(Value::as_u64).unwrap_or(0);
        let reasoning = usage.pointer("/completion_tokens_details/reasoning_tokens").and_then(Value::as_u64).unwrap_or(0);
        json!({
            "input_tokens": prompt,
            "input_tokens_details": {"cached_tokens": cached, "cache_write_tokens": 0},
            "output_tokens": completion,
            "output_tokens_details": {"reasoning_tokens": reasoning},
            "total_tokens": prompt.saturating_add(completion),
        })
    });
    let format = request.pointer("/text/format").cloned().unwrap_or(json!({"type":"text"}));
    let end_turn = finish == "stop" && output.iter().all(|item| !matches!(item.get("type").and_then(Value::as_str), Some("function_call" | "custom_tool_call")));
    let mut response = json!({
        "id": format!("resp_{public_id}"),
        "object": "response",
        "created_at": created_at,
        "status": status,
        "incomplete_details": if status == "incomplete" { json!({"reason":"max_output_tokens"}) } else { Value::Null },
        "error": Value::Null,
        "model": model,
        "output": output,
        "parallel_tool_calls": request.get("parallel_tool_calls").and_then(Value::as_bool).unwrap_or(true),
        "store": store,
        "previous_response_id": previous_response_id,
        "text": {"format": format},
        "usage": usage,
        "end_turn": end_turn,
    });
    if let Some(timings) = chat_response.get("timings") { response["timings"] = timings.clone(); }
    if let Some(compaction) = chat_response.get("compaction") { response["compaction"] = compaction.clone(); }
    Ok(response)
}

pub fn in_progress_response(
    model: &str,
    public_id: &str,
    created_at: i64,
    request: &Value,
    store: bool,
    previous_response_id: Option<&str>,
) -> Value {
    json!({
        "id":format!("resp_{public_id}"), "object":"response", "created_at":created_at,
        "status":"in_progress", "incomplete_details":Value::Null, "error":Value::Null,
        "model":model, "output":[],
        "parallel_tool_calls":request.get("parallel_tool_calls").and_then(Value::as_bool).unwrap_or(true),
        "store":store, "previous_response_id":previous_response_id,
        "text":{"format":request.pointer("/text/format").cloned().unwrap_or(json!({"type":"text"}))},
        "usage":Value::Null,
    })
}

/// Translates the existing live Chat stream chunks into named Responses SSE
/// events. Each chunk is handled as it arrives; no completion is buffered for
/// later replay.
pub struct ResponsesStreamState {
    model: String,
    public_id: String,
    created_at: i64,
    request: Value,
    tool_metadata: Value,
    store_enabled: bool,
    previous_response_id: Option<String>,
    history: Vec<Value>,
    response_store: ResponseStore,
    sequence: u64,
    started: bool,
    output: Vec<OutputBlock>,
    tool_indexes: HashMap<usize, usize>,
    finish_reason: String,
    usage: Option<Value>,
    timings: Option<Value>,
    error: Option<Value>,
}

impl ResponsesStreamState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: String,
        public_id: String,
        created_at: i64,
        request: Value,
        store_enabled: bool,
        previous_response_id: Option<String>,
        history: Vec<Value>,
        response_store: ResponseStore,
    ) -> Self {
        Self::new_with_tools(model, public_id, created_at, request, store_enabled, previous_response_id, history, response_store, Value::Null)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_tools(
        model: String,
        public_id: String,
        created_at: i64,
        request: Value,
        store_enabled: bool,
        previous_response_id: Option<String>,
        history: Vec<Value>,
        response_store: ResponseStore,
        tool_metadata: Value,
    ) -> Self {
        Self {
            model, public_id, created_at, request, tool_metadata, store_enabled,
            previous_response_id, history, response_store, sequence: 0,
            started: false, output: Vec::new(), tool_indexes: HashMap::new(),
            finish_reason: "stop".into(), usage: None, timings: None, error: None,
        }
    }

    /// Parse one Chat SSE payload and return zero or more Responses events.
    /// The special `[DONE]` payload closes every item and emits the final
    /// response event.
    pub fn consume(&mut self, payload: &str) -> Result<Vec<(String, Value)>, String> {
        let mut events = Vec::new();
        if payload == "[DONE]" {
            self.begin(&mut events);
            self.finish(&mut events)?;
            return Ok(events);
        }
        let chunk: Value = serde_json::from_str(payload).map_err(|error| format!("invalid chat stream chunk: {error}"))?;
        if let Some(error) = chunk.get("error") {
            self.error = Some(error.clone());
            return Ok(events);
        }
        if let Some(timings) = chunk.get("timings") { self.timings = Some(timings.clone()); }
        if let Some(usage) = chunk.get("usage") { self.usage = Some(usage.clone()); }
        let Some(choice) = chunk.pointer("/choices/0") else { return Ok(events) };
        let delta = choice.get("delta").unwrap_or(&Value::Null);
        if delta.get("role").is_some() { self.begin(&mut events); }
        if let Some(text) = delta.get("reasoning_content").and_then(Value::as_str).filter(|text| !text.is_empty()) {
            self.begin(&mut events);
            let index = self.find_or_open("reasoning", None, None, None, &mut events)?;
            self.output[index].text.push_str(text);
            let item_id = responses_item_id(&self.public_id, "reasoning", index).map_err(|error| error.to_string())?;
            self.emit(&mut events, "response.reasoning_summary_text.delta", json!({
                "item_id": item_id, "output_index": index, "summary_index": 0, "delta": text,
            }));
        }
        if let Some(text) = delta.get("content").and_then(Value::as_str).filter(|text| !text.is_empty()) {
            self.begin(&mut events);
            let index = self.find_or_open("text", None, None, None, &mut events)?;
            self.output[index].text.push_str(text);
            let item_id = responses_item_id(&self.public_id, "text", index).map_err(|error| error.to_string())?;
            self.emit(&mut events, "response.output_text.delta", json!({
                "item_id": item_id, "output_index": index, "content_index": 0, "delta": text, "logprobs": [],
            }));
        }
        if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
            self.begin(&mut events);
            for call in calls {
                let tool_index = call.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                let call_id = call.get("id").and_then(Value::as_str);
                let name = call.pointer("/function/name").and_then(Value::as_str);
                let index = self.find_or_open("tool", Some(tool_index), call_id, name, &mut events)?;
                if let Some(id) = call_id { self.output[index].call_id = Some(id.into()); }
                if let Some(name) = name { self.output[index].name = Some(name.into()); }
                if let Some(arguments) = call.pointer("/function/arguments").and_then(Value::as_str).filter(|text| !text.is_empty()) {
                    self.output[index].text.push_str(arguments);
                    if !is_custom_tool(&self.tool_metadata, self.output[index].name.as_deref()) {
                        let item_id = responses_item_id(&self.public_id, "tool", index).map_err(|error| error.to_string())?;
                        self.emit(&mut events, "response.function_call_arguments.delta", json!({
                            "item_id": item_id, "output_index": index, "delta": arguments,
                        }));
                    }
                }
            }
        }
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            self.finish_reason = reason.to_owned();
        }
        Ok(events)
    }

    fn begin(&mut self, events: &mut Vec<(String, Value)>) {
        if self.started { return; }
        self.started = true;
        let response = self.snapshot("in_progress", json!([]), Value::Null, Value::Null);
        self.emit(events, "response.created", json!({"response": response}));
        self.emit(events, "response.in_progress", json!({"response": response}));
    }

    fn find_or_open(&mut self, kind: &str, tool_index: Option<usize>, call_id: Option<&str>, name: Option<&str>, events: &mut Vec<(String, Value)>) -> Result<usize, String> {
        if let Some(index) = tool_index.and_then(|index| self.tool_indexes.get(&index).copied()) { return Ok(index); }
        if tool_index.is_none() {
            if let Some(index) = self.output.iter().position(|block| block.kind == kind) { return Ok(index); }
        }
        let block = OutputBlock {
            kind: kind.into(), text: String::new(), call_id: call_id.map(str::to_owned),
            name: name.map(str::to_owned), status: "in_progress".into(),
        };
        let index = self.output.len();
        let item = responses_item_with_tools(&self.public_id, &block, index, &self.tool_metadata).map_err(|error| error.to_string())?;
        self.emit(events, "response.output_item.added", json!({"output_index": index, "item": item}));
        if kind == "reasoning" {
            self.emit(events, "response.reasoning_summary_part.added", json!({
                "item_id": item["id"], "output_index": index, "summary_index": 0,
                "part": {"type":"summary_text", "text":""},
            }));
        } else if kind == "text" {
            self.emit(events, "response.content_part.added", json!({
                "item_id": item["id"], "output_index": index, "content_index": 0,
                "part": {"type":"output_text", "text":"", "annotations":[]},
            }));
        }
        self.output.push(block);
        if let Some(tool_index) = tool_index { self.tool_indexes.insert(tool_index, index); }
        Ok(index)
    }

    fn finish(&mut self, events: &mut Vec<(String, Value)>) -> Result<(), String> {
        if let Some(error) = self.error.take() {
            let output = responses_output_with_tools(&self.public_id, &self.output, &self.tool_metadata).map_err(|error| error.to_string())?;
            let response = self.snapshot("failed", json!(output), error, self.usage.clone().unwrap_or(Value::Null));
            self.emit(events, "response.failed", json!({"response":response}));
            return Ok(());
        }
        let item_status = if self.finish_reason == "length" { "incomplete" } else { "completed" };
        let mut closing = Vec::<(&'static str, Value)>::new();
        for (index, block) in self.output.iter_mut().enumerate() {
            block.status = item_status.into();
            let item = responses_item_with_tools(&self.public_id, block, index, &self.tool_metadata).map_err(|error| error.to_string())?;
            let id = item["id"].clone();
            match block.kind.as_str() {
                "reasoning" => {
                    closing.push(("response.reasoning_summary_text.done", json!({
                        "item_id":id, "output_index":index, "summary_index":0, "text":block.text,
                    })));
                    let mut part_done = json!({
                        "item_id":id, "output_index":index, "summary_index":0,
                        "part":{"type":"summary_text", "text":block.text},
                    });
                    if item_status == "incomplete" { part_done["status"] = json!("incomplete"); }
                    closing.push(("response.reasoning_summary_part.done", part_done));
                }
                "text" => {
                    let part = item.pointer("/content/0").cloned().unwrap_or(json!({"type":"output_text","text":block.text,"annotations":[]}));
                    closing.push(("response.output_text.done", json!({
                        "item_id":id, "output_index":index, "content_index":0, "text":block.text, "logprobs":[],
                    })));
                    closing.push(("response.content_part.done", json!({
                        "item_id":id, "output_index":index, "content_index":0, "part":part,
                    })));
                }
                "tool" if is_custom_tool(&self.tool_metadata, block.name.as_deref()) => {
                    let input = custom_input(&block.text);
                    closing.push(("response.custom_tool_call_input.delta", json!({
                        "item_id":id, "output_index":index, "delta":input,
                    })));
                    closing.push(("response.custom_tool_call_input.done", json!({
                        "item_id":id, "output_index":index, "input":input,
                    })));
                }
                "tool" => closing.push(("response.function_call_arguments.done", json!({
                    "item_id":id, "output_index":index, "name":item["name"], "arguments":block.text,
                }))),
                _ => {}
            }
            closing.push(("response.output_item.done", json!({"output_index":index,"item":item})));
        }
        for (event, payload) in closing { self.emit(events, event, payload); }
        let output = responses_output_with_tools(&self.public_id, &self.output, &self.tool_metadata).map_err(|error| error.to_string())?;
        let calls: Vec<Value> = self.output.iter().filter(|block| block.kind == "tool").map(|block| json!({
            "id":block.call_id, "type":"function", "function":{"name":block.name, "arguments":block.text}
        })).collect();
        let reasoning = self.output.iter().filter(|block| block.kind == "reasoning").map(|block| block.text.as_str()).collect::<String>();
        let content = self.output.iter().filter(|block| block.kind == "text").map(|block| block.text.as_str()).collect::<String>();
        let finish_reason = self.finish_reason.clone();
        let chat_response = json!({
            "choices":[{"finish_reason":finish_reason, "message":{"reasoning_content":reasoning, "content":content, "tool_calls":calls}}],
            "usage":self.usage.clone().unwrap_or(Value::Null), "timings":self.timings.clone().unwrap_or(Value::Null),
        });
        let status = if self.finish_reason == "length" { "incomplete" } else { "completed" };
        let response = response_from_output(
            &self.model, &self.public_id, self.created_at, &self.request, &chat_response,
            output.clone(), self.store_enabled, self.previous_response_id.as_deref(),
        ).map_err(|error| error.to_string())?;
        if self.store_enabled {
            let id = response["id"].as_str().unwrap_or_default().to_owned();
            self.response_store.put(id, response.clone(), self.history.clone());
        }
        self.emit(events, &format!("response.{status}"), json!({"response":response}));
        Ok(())
    }

    fn snapshot(&self, status: &str, output: Value, error: Value, usage: Value) -> Value {
        json!({
            "id":format!("resp_{}",self.public_id), "object":"response", "created_at":self.created_at,
            "status":status, "incomplete_details":Value::Null, "error":error, "model":self.model,
            "output":output, "parallel_tool_calls":self.request.get("parallel_tool_calls").and_then(Value::as_bool).unwrap_or(true),
            "store":self.store_enabled, "previous_response_id":self.previous_response_id,
            "text":{"format":self.request.pointer("/text/format").cloned().unwrap_or(json!({"type":"text"}))},
            "usage":usage,
        })
    }

    fn emit(&mut self, events: &mut Vec<(String, Value)>, event: &str, mut payload: Value) {
        self.sequence += 1;
        payload["type"] = json!(event);
        payload["sequence_number"] = json!(self.sequence - 1);
        events.push((event.to_owned(), payload));
    }
}

/// The Responses output item of one output block, ported from
/// `server/dialects/responses.py`'s `responses_item`: the item id's
/// prefix by kind, the status, and the text or arguments an item in
/// progress does not carry yet.
#[derive(Debug, Clone, PartialEq)]
pub struct OutputBlock {
    pub kind: String,
    pub text: String,
    pub call_id: Option<String>,
    pub name: Option<String>,
    pub status: String,
}

/// The id prefix of each kind's item.
fn item_prefix(kind: &str) -> Option<&'static str> {
    match kind {
        "reasoning" => Some("rs"),
        "text" => Some("msg"),
        "tool" => Some("fc"),
        "custom" => Some("ctc"),
        _ => None,
    }
}

/// The Responses output item of `block` at `index`. An item in progress
/// carries no text or arguments yet.
pub fn responses_item(
    public_id: &str,
    block: &OutputBlock,
    index: usize,
) -> Result<Value, MessageError> {
    responses_item_with_tools(public_id, block, index, &Value::Null)
}

pub fn responses_item_with_tools(
    public_id: &str,
    block: &OutputBlock,
    index: usize,
    tool_metadata: &Value,
) -> Result<Value, MessageError> {
    let custom = block.kind == "tool" && is_custom_tool(tool_metadata, block.name.as_deref());
    let prefix = item_prefix(if custom { "custom" } else { &block.kind })
        .ok_or_else(|| bad(format!("unsupported output block: {}", block.kind)))?;
    let item_id = format!("{prefix}_{public_id}_{index}");
    let in_progress = block.status == "in_progress";
    let text = if in_progress { "" } else { &block.text };
    match block.kind.as_str() {
        "reasoning" => Ok(json!({
            "id": item_id,
            "type": "reasoning",
            "status": block.status,
            "summary": if text.is_empty() {
                json!([])
            } else {
                json!([{"type": "summary_text", "text": text}])
            },
            "content": if text.is_empty() {
                json!([])
            } else {
                json!([{"type": "reasoning_text", "text": text}])
            },
            "encrypted_content": null,
        })),
        "text" => Ok(json!({
            "id": item_id,
            "type": "message",
            "status": block.status,
            "role": "assistant",
            "content": if in_progress {
                json!([])
            } else {
                json!([{"type": "output_text", "text": text, "annotations": []}])
            },
        })),
        "tool" if custom => Ok(json!({
            "id": item_id,
            "type": "custom_tool_call",
            "status": block.status,
            "call_id": block.call_id,
            "name": block.name,
            "input": custom_input(text),
        })),
        "tool" => {
            let mapping = tool_metadata.get(block.name.as_deref().unwrap_or(""));
            let name = mapping.and_then(|mapping| mapping.get("name")).cloned().or_else(|| block.name.clone().map(Value::String)).unwrap_or(Value::Null);
            let mut item = json!({
                "id": item_id,
                "type": "function_call",
                "status": block.status,
                "call_id": block.call_id,
                "name": name,
                "arguments": text,
            });
            if mapping.and_then(|mapping| mapping.get("kind")).and_then(Value::as_str) == Some("namespace") {
                item["namespace"] = mapping.and_then(|mapping| mapping.get("namespace")).cloned().unwrap_or(Value::Null);
            }
            Ok(item)
        }
        "custom" => Ok(json!({
            "id": item_id,
            "type": "custom_tool_call",
            "status": block.status,
            "call_id": block.call_id,
            "name": block.name,
            "input": text,
        })),
        _ => unreachable!("prefix checked above"),
    }
}

/// The deterministic id of one output item.
pub fn responses_item_id(public_id: &str, kind: &str, index: usize) -> Result<String, MessageError> {
    let prefix = item_prefix(kind).ok_or_else(|| bad(format!("unsupported output block: {kind}")))?;
    Ok(format!("{prefix}_{public_id}_{index}"))
}

/// The Responses output items of the output's blocks, in order.
pub fn responses_output(
    public_id: &str,
    blocks: &[OutputBlock],
) -> Result<Vec<Value>, MessageError> {
    responses_output_with_tools(public_id, blocks, &Value::Null)
}

pub fn responses_output_with_tools(
    public_id: &str,
    blocks: &[OutputBlock],
    tool_metadata: &Value,
) -> Result<Vec<Value>, MessageError> {
    blocks
        .iter()
        .enumerate()
        .map(|(index, block)| responses_item_with_tools(public_id, block, index, tool_metadata))
        .collect()
}

fn is_custom_tool(metadata: &Value, name: Option<&str>) -> bool {
    name.and_then(|name| metadata.get(name))
        .and_then(|mapping| mapping.get("kind"))
        .and_then(Value::as_str) == Some("custom")
}

fn custom_input(arguments: &str) -> String {
    serde_json::from_str::<Value>(arguments).ok()
        .and_then(|value| value.get("input").and_then(Value::as_str).map(str::to_owned))
        .unwrap_or_else(|| arguments.to_owned())
}
