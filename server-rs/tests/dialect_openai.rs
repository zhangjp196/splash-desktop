//! The OpenAI dialect's response assembly, each expectation checked
//! against the Python implementation's own output.

use serde_json::{json, Value};
use splash_server::dialects::openai::{
    chat_choice, completion_response, finish_reason, stream_chunk, timings_dict, usage_dict,
    Collected, CollectedResult, JobInfo,
};

fn result() -> CollectedResult {
    CollectedResult {
        reason: "stop".into(),
        prompt_tokens: 10,
        completion_tokens: 5,
        prefill_tokens: 10,
        cache_matched_tokens: 3,
        drafted_tokens: 0,
        accepted_draft_tokens: 0,
        first_token_batch_tokens: 1,
        start_to_first_token_ms: 12.5,
        first_token_to_done_ms: 30.0,
        request_wall_ms: 45.0,
        stream_tokens_per_second: 41.0,
        bounded: false,
        metrics: json!({
            "request_latency": {
                "start_to_first_token_ms": 12.5,
                "first_token_to_done_ms": 30.0,
                "stream_tokens_per_second": 41.0,
                "wall_ms": 45.0,
            }
        }),
    }
}

fn collected() -> Collected {
    Collected {
        content: "The answer.".into(),
        reasoning: None,
        tool_calls: vec![],
        result: Some(result()),
        called: false,
        logprobs: None,
    }
}

fn job() -> JobInfo {
    JobInfo {
        public_id: "abc123".into(),
        created_at: 1_700_000_000,
        reasoning_tokens: 2,
    }
}

#[test]
fn the_finish_reason_counts_whole_calls_but_a_length_end() {
    assert_eq!(finish_reason(&result(), false), "stop");
    let mut length = result();
    length.reason = "length".into();
    assert_eq!(finish_reason(&length, true), "length");
    let mut stop = result();
    stop.reason = "stop".into();
    assert_eq!(finish_reason(&stop, true), "tool_calls");
}

#[test]
fn a_chat_choice_carries_the_message_and_the_reason() {
    let choice = chat_choice(&collected(), &job(), 0);
    assert_eq!(
        choice,
        json!({
            "index": 0,
            "message": {"role": "assistant", "content": "The answer."},
            "finish_reason": "stop"
        })
    );
}

#[test]
fn the_response_carries_the_usage_the_metrics_and_the_timings() {
    let choice = chat_choice(&collected(), &job(), 0);
    let response = completion_response("Qwen", &job(), &result(), vec![choice]);
    assert_eq!(
        response,
        json!({
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "Qwen",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "The answer."},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "prompt_tokens_details": {"cached_tokens": 3},
                "completion_tokens_details": {"reasoning_tokens": 2}
            },
            "metrics": {
                "request_latency": {
                    "start_to_first_token_ms": 12.5,
                    "first_token_to_done_ms": 30.0,
                    "stream_tokens_per_second": 41.0,
                    "wall_ms": 45.0
                }
            },
            "timings": {
                "prompt_n": 10,
                "prompt_ms": 12.5,
                "prompt_per_second": 800.0,
                "predicted_n": 5,
                "predicted_ms": 30.0,
                "predicted_per_second": 41.0,
                "cache_n": 3
            }
        })
    );
}

#[test]
fn a_stream_chunk_carries_the_delta_and_the_usage_chunk_the_counts() {
    let chunk = stream_chunk(
        "Qwen",
        "req1",
        1_700_000_000,
        json!({"role": "assistant", "content": ""}),
        None,
        None,
        None,
        None,
    );
    assert_eq!(
        chunk,
        json!({
            "id": "chatcmpl-req1",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "Qwen",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]
        })
    );
    let usage_chunk = stream_chunk(
        "Qwen",
        "req1",
        1_700_000_000,
        json!({}),
        None,
        Some(json!({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})),
        None,
        None,
    );
    assert_eq!(
        usage_chunk,
        json!({
            "id": "chatcmpl-req1",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "Qwen",
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        })
    );
}

#[test]
fn the_usage_and_timings_read_the_counts() {
    let usage = usage_dict(&result(), &job());
    assert_eq!(usage["total_tokens"], json!(15));
    assert_eq!(usage["prompt_tokens_details"]["cached_tokens"], json!(3));
    let timings = timings_dict(&result());
    assert_eq!(timings["prompt_per_second"], json!(800.0));
    assert_eq!(timings["predicted_per_second"], json!(41.0));
    assert_eq!(timings["cache_n"], json!(3));
    // A speculative output carries its draft counts.
    let mut speculative = result();
    speculative.drafted_tokens = 350;
    speculative.accepted_draft_tokens = 280;
    let timings = timings_dict(&speculative);
    assert_eq!(timings["draft_n"], json!(350));
    assert_eq!(timings["draft_n_accepted"], json!(280));
    // A choice with reasoning and calls carries them.
    let mut with_calls = collected();
    with_calls.reasoning = Some("the reasoning".into());
    with_calls.tool_calls = vec![json!({"id": "call_1", "type": "function",
        "function": {"name": "run_tests", "arguments": "{}"}})];
    with_calls.called = true;
    let choice = chat_choice(&with_calls, &job(), 0);
    assert_eq!(choice["message"]["reasoning_content"], json!("the reasoning"));
    assert_eq!(choice["message"]["tool_calls"][0]["function"]["name"], json!("run_tests"));
    assert_eq!(choice["finish_reason"], json!("tool_calls"));
    // An empty content reads as null, as OpenAI's message carries.
    let mut empty = collected();
    empty.content = String::new();
    let choice = chat_choice(&empty, &job(), 0);
    assert_eq!(choice["message"]["content"], Value::Null);
}
