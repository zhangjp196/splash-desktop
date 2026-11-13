//! Anthropic's Messages: the response assembly. Ported from
//! `server/dialects/anthropic.py`'s stop reasons, usage, content blocks
//! and response envelope, over the collected output the frontend's
//! exchange gathers.

use serde_json::{json, Value};

use crate::dialects::openai::CollectedResult;

/// One content block of the output, as the sequencer reads it.
#[derive(Debug, Clone, PartialEq)]
pub struct OutputBlock {
    pub kind: String,
    pub text: String,
    /// A tool_use block's id and name.
    pub call_id: Option<String>,
    pub name: Option<String>,
    /// Whether a tool_use block is still in progress.
    pub in_progress: bool,
}

/// Anthropic's stop_reason: the output's whole tool calls count only when
/// the token limit did not end it.
pub fn anthropic_stop(
    result: &CollectedResult,
    called: bool,
    stop_sequence: Option<&str>,
    output_clamped_to_context: bool,
) -> &'static str {
    if called && result.reason != "length" {
        return "tool_use";
    }
    if result.reason == "length" {
        // Anthropic's reason when the context window, not max_tokens, ends
        // the response: max_tokens was lowered to what the context leaves,
        // or the engine ended it at what it holds of the context.
        if output_clamped_to_context || result.bounded {
            return "model_context_window_exceeded";
        }
        return "max_tokens";
    }
    if stop_sequence.is_some() {
        return "stop_sequence";
    }
    "end_turn"
}

/// Messages usage: the input past the cached prefix, the cached prefix,
/// and the output; with thinking tokens, also the output tokens the
/// response spent thinking, in output_tokens_details as Anthropic reports
/// them (Chat and Responses report them as reasoning_tokens).
pub fn anthropic_usage(
    prompt_tokens: u64,
    output_tokens: u64,
    cache_matched: u64,
    thinking_tokens: Option<u64>,
) -> Value {
    let mut usage = json!({
        "input_tokens": prompt_tokens - cache_matched,
        "cache_read_input_tokens": cache_matched,
        "output_tokens": output_tokens,
    });
    if let Some(thinking_tokens) = thinking_tokens {
        usage["output_tokens_details"] = json!({"thinking_tokens": thinking_tokens});
    }
    usage
}

/// The Messages content block of one output block, as a complete response
/// shows it. A call no longer in progress has complete arguments: the
/// projector writes each call it closes as JSON, and a complete response
/// leaves out a call the token limit or a stop sequence cut.
pub fn anthropic_block(block: &OutputBlock, signature: &str) -> Value {
    match block.kind.as_str() {
        "reasoning" => json!({
            "type": "thinking",
            "thinking": block.text,
            "signature": signature,
        }),
        "text" => json!({"type": "text", "text": block.text}),
        _ => {
            let input = if block.in_progress {
                json!({})
            } else {
                crate::support::json_codec::loads(&block.text).unwrap_or(json!({}))
            };
            json!({
                "type": "tool_use",
                "id": block.call_id,
                "name": block.name,
                "input": input,
            })
        }
    }
}

/// The Messages response envelope: the content blocks, the stop reason
/// and sequence, and the usage. An all-thinking content gains an empty
/// text block, as Anthropic's response carries.
pub fn anthropic_response(
    model: &str,
    public_id: &str,
    blocks: &[OutputBlock],
    result: &CollectedResult,
    called: bool,
    stop_sequence: Option<&str>,
    thinking_signature: &str,
    reasoning_tokens: u64,
    output_clamped_to_context: bool,
) -> Value {
    let mut content: Vec<Value> = blocks
        .iter()
        .map(|block| anthropic_block(block, thinking_signature))
        .collect();
    if !content.is_empty() && content.iter().all(|item| item["type"] == json!("thinking")) {
        content.push(json!({"type": "text", "text": ""}));
    }
    json!({
        "id": format!("msg_{public_id}"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": anthropic_stop(result, called, stop_sequence, output_clamped_to_context),
        "stop_sequence": stop_sequence.map(Value::from).unwrap_or(Value::Null),
        "usage": anthropic_usage(
            result.prompt_tokens,
            result.completion_tokens,
            result.cache_matched_tokens,
            Some(reasoning_tokens),
        ),
        "timings": crate::dialects::openai::timings_dict(result),
    })
}
