//! The Anthropic dialect's response assembly, each expectation checked
//! against the Python implementation's own output.

use serde_json::json;
use splash_server::dialects::anthropic::{
    anthropic_block, anthropic_response, anthropic_stop, anthropic_usage, OutputBlock,
};
use splash_server::dialects::openai::CollectedResult;

fn result() -> CollectedResult {
    CollectedResult {
        reason: "stop".into(),
        prompt_tokens: 10,
        completion_tokens: 5,
        prefill_tokens: 10,
        cache_matched_tokens: 3,
        drafted_tokens: 0,
        accepted_draft_tokens: 0,
        first_token_batch_tokens: 0,
        start_to_first_token_ms: 12.5,
        first_token_to_done_ms: 30.0,
        request_wall_ms: 45.0,
        stream_tokens_per_second: 41.0,
        bounded: false,
        metrics: json!({}),
    }
}

#[test]
fn the_stop_reasons_map_as_anthropic_names_them() {
    assert_eq!(anthropic_stop(&result(), false, None, false), "end_turn");
    assert_eq!(anthropic_stop(&result(), true, None, false), "tool_use");
    let mut length = result();
    length.reason = "length".into();
    assert_eq!(anthropic_stop(&length, false, None, false), "max_tokens");
    length.bounded = true;
    assert_eq!(
        anthropic_stop(&length, false, None, false),
        "model_context_window_exceeded"
    );
    assert_eq!(
        anthropic_stop(&result(), false, Some("STOP"), false),
        "stop_sequence"
    );
}

#[test]
fn the_usage_splits_the_cached_prefix_and_carries_the_thinking() {
    let usage = anthropic_usage(10, 5, 3, Some(2));
    assert_eq!(
        usage,
        json!({
            "input_tokens": 7,
            "cache_read_input_tokens": 3,
            "output_tokens": 5,
            "output_tokens_details": {"thinking_tokens": 2}
        })
    );
    let plain = anthropic_usage(10, 5, 3, None);
    assert!(plain.get("output_tokens_details").is_none());
}

#[test]
fn the_blocks_read_as_anthropic_carries_them() {
    let text = anthropic_block(
        &OutputBlock {
            kind: "text".into(),
            text: "hi".into(),
            call_id: None,
            name: None,
            in_progress: false,
        },
        "",
    );
    assert_eq!(text, json!({"type": "text", "text": "hi"}));
    let tool_use = anthropic_block(
        &OutputBlock {
            kind: "tool_use".into(),
            text: r#"{"a": 1}"#.into(),
            call_id: Some("c1".into()),
            name: Some("f".into()),
            in_progress: false,
        },
        "",
    );
    assert_eq!(
        tool_use,
        json!({"type": "tool_use", "id": "c1", "name": "f", "input": {"a": 1}})
    );
    // A call still in progress carries an empty input.
    let in_progress = anthropic_block(
        &OutputBlock {
            kind: "tool_use".into(),
            text: String::new(),
            call_id: Some("c1".into()),
            name: Some("f".into()),
            in_progress: true,
        },
        "",
    );
    assert_eq!(in_progress["input"], json!({}));
    let thinking = anthropic_block(
        &OutputBlock {
            kind: "reasoning".into(),
            text: "thought".into(),
            call_id: None,
            name: None,
            in_progress: false,
        },
        "sig",
    );
    assert_eq!(
        thinking,
        json!({"type": "thinking", "thinking": "thought", "signature": "sig"})
    );
}

#[test]
fn the_response_envelope_carries_the_message_shape() {
    let blocks = vec![
        OutputBlock {
            kind: "text".into(),
            text: "The answer.".into(),
            call_id: None,
            name: None,
            in_progress: false,
        },
    ];
    let response = anthropic_response(
        "Qwen",
        "abc123",
        &blocks,
        &result(),
        false,
        None,
        "",
        2,
        false,
    );
    assert_eq!(
        response,
        json!({
            "id": "msg_abc123",
            "type": "message",
            "role": "assistant",
            "model": "Qwen",
            "content": [{"type": "text", "text": "The answer."}],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 7,
                "cache_read_input_tokens": 3,
                "output_tokens": 5,
                "output_tokens_details": {"thinking_tokens": 2}
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
fn an_all_thinking_content_gains_an_empty_text_block() {
    let blocks = vec![OutputBlock {
        kind: "reasoning".into(),
        text: "thought".into(),
        call_id: None,
        name: None,
        in_progress: false,
    }];
    let response = anthropic_response(
        "Qwen",
        "abc123",
        &blocks,
        &result(),
        false,
        None,
        "sig",
        0,
        false,
    );
    let content = response["content"].as_array().expect("content");
    assert_eq!(content.len(), 2);
    assert_eq!(content[1]["type"], json!("text"));
    assert_eq!(content[1]["text"], json!(""));
}
