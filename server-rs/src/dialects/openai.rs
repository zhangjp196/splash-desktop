//! OpenAI's Chat and text completions: their answers. Ported from
//! `server/dialects/openai.py`'s response assembly, over the collected
//! result the frontend's exchange gathers.

use serde_json::{json, Map, Value};

/// The result one request's output carries, as the response assembly
/// reads it: the counts, the cache's matched prefix, the speculative
/// counts, and the latency intervals the timings read.
#[derive(Debug, Clone)]
pub struct CollectedResult {
    pub reason: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub prefill_tokens: u64,
    pub cache_matched_tokens: u64,
    pub drafted_tokens: u64,
    pub accepted_draft_tokens: u64,
    pub first_token_batch_tokens: u64,
    pub start_to_first_token_ms: f64,
    pub first_token_to_done_ms: f64,
    pub request_wall_ms: f64,
    pub stream_tokens_per_second: f64,
    /// A LENGTH end at a bound of the engine's own, which Anthropic's
    /// stop reason reads as the context window's.
    pub bounded: bool,
    /// The metrics object the response carries as it is.
    pub metrics: Value,
}

/// What one request's exchange collected: the output's text, its
/// reasoning, its whole tool calls, and the result.
#[derive(Debug, Clone, Default)]
pub struct Collected {
    pub content: String,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<Value>,
    pub result: Option<CollectedResult>,
    /// Whether the output called a tool.
    pub called: bool,
}

/// What one request carries past its output: the ids and the reasoning
/// token count the usage reads.
#[derive(Debug, Clone)]
pub struct JobInfo {
    pub public_id: String,
    pub created_at: i64,
    pub reasoning_tokens: u64,
}

/// OpenAI finish_reason: the output's whole tool calls count only when
/// the token limit did not end it.
pub fn finish_reason(result: &CollectedResult, called: bool) -> String {
    if called && result.reason != "length" {
        "tool_calls".to_string()
    } else {
        result.reason.clone()
    }
}

/// The usage object: the counts, the cache's matched prefix and the
/// reasoning tokens.
pub fn usage_dict(result: &CollectedResult, job: &JobInfo) -> Value {
    json!({
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.prompt_tokens + result.completion_tokens,
        "prompt_tokens_details": {"cached_tokens": result.cache_matched_tokens},
        "completion_tokens_details": {"reasoning_tokens": job.reasoning_tokens},
    })
}

/// The timings object: llama-server-compatible counts and request
/// lifecycle timings. Rates exclude cached prompt tokens and the first
/// emission respectively.
pub fn timings_dict(result: &CollectedResult) -> Value {
    let prompt_ms = result.start_to_first_token_ms;
    let prompt_rate = if prompt_ms != 0.0 {
        result.prefill_tokens as f64 * 1000.0 / prompt_ms
    } else {
        0.0
    };
    let mut timings = json!({
        "prompt_n": result.prompt_tokens,
        "prompt_ms": prompt_ms,
        "prompt_per_second": if prompt_rate.is_finite() { prompt_rate } else { 0.0 },
        "predicted_n": result.completion_tokens,
        "predicted_ms": result.first_token_to_done_ms,
        "predicted_per_second": result.stream_tokens_per_second,
        "cache_n": result.cache_matched_tokens,
    });
    if result.drafted_tokens != 0 {
        timings["draft_n"] = json!(result.drafted_tokens);
        timings["draft_n_accepted"] = json!(result.accepted_draft_tokens);
    }
    timings
}

/// One choice of a Chat completion: the assistant message the job
/// generated, its finish reason and, asked for, its logprobs.
pub fn chat_choice(collected: &Collected, _job: &JobInfo, index: usize) -> Value {
    let result = collected.result.as_ref().expect("a collected result");
    let mut message = Map::new();
    message.insert("role".into(), json!("assistant"));
    message.insert(
        "content".into(),
        if collected.content.is_empty() {
            Value::Null
        } else {
            json!(collected.content)
        },
    );
    if let Some(reasoning) = &collected.reasoning {
        message.insert("reasoning_content".into(), json!(reasoning));
    }
    if !collected.tool_calls.is_empty() {
        message.insert("tool_calls".into(), json!(collected.tool_calls));
    }
    let mut choice = Map::new();
    choice.insert("index".into(), json!(index));
    choice.insert("message".into(), Value::Object(message));
    choice.insert(
        "finish_reason".into(),
        json!(finish_reason(result, collected.called)),
    );
    Value::Object(choice)
}

/// The response envelope: the ids, the usage, the metrics and the
/// timings. The choices are the caller's, so an n>1 request builds it
/// once rather than once per choice.
pub fn completion_response(
    model: &str,
    job: &JobInfo,
    result: &CollectedResult,
    choices: Vec<Value>,
) -> Value {
    let mut response = json!({
        "id": format!("chatcmpl-{}", job.public_id),
        "object": "chat.completion",
        "created": job.created_at,
        "model": model,
        "choices": choices,
        "usage": usage_dict(result, job),
        "metrics": result.metrics,
        "timings": timings_dict(result),
    });
    response["usage"]["prompt_tokens"] = json!(result.prompt_tokens);
    response
}

/// One chunk of a Chat completion stream.
pub fn stream_chunk(
    model: &str,
    request_id: &str,
    created: i64,
    delta: Value,
    finish_reason: Option<String>,
    usage: Option<Value>,
    metrics: Option<Value>,
    timings: Option<Value>,
) -> Value {
    let mut chunk = json!({
        "id": format!("chatcmpl-{request_id}"),
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason.map(Value::String).unwrap_or(Value::Null),
        }],
    });
    if let Some(usage) = usage {
        chunk["choices"] = json!([]);
        chunk["usage"] = usage;
    }
    if let Some(metrics) = metrics {
        chunk["metrics"] = metrics;
    }
    if let Some(timings) = timings {
        chunk["timings"] = timings;
    }
    chunk
}
