//! The Anthropic request conversion's tests, each expectation checked
//! against the Python implementation's own output.

use serde_json::{json, Value};
use splash_server::dialects::anthropic_request::{anthropic_thinking, anthropic_to_chat_body};
use splash_server::chat_messages::MessageError;

fn resolver(signature: &str) -> Result<String, MessageError> {
    Ok(format!("resolved: {signature}"))
}

#[test]
fn a_messages_request_converts_to_a_chat_body() {
    let body = json!({
        "model": "claude",
        "max_tokens": 100,
        "system": "You are helpful.",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "Run the tests."},
            ]},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "hidden thought", "signature": "sig1"},
                {"type": "text", "text": "Running."},
                {"type": "tool_use", "id": "c1", "name": "run_tests", "input": {"path": "src"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "c1", "content": "ok"},
                {"type": "text", "text": "Now memoize."},
            ]},
        ],
        "stop_sequences": ["STOP"],
        "temperature": 0.5,
    });
    let (chat, thinking_display) = anthropic_to_chat_body(&body, &resolver).expect("converts");
    assert_eq!(thinking_display, "summarized");
    assert_eq!(chat["model"], json!("claude"));
    assert_eq!(chat["max_completion_tokens"], json!(100));
    assert_eq!(chat["stop"], json!(["STOP"]));
    assert_eq!(chat["temperature"], json!(0.5));
    let messages = chat["messages"].as_array().expect("messages");
    assert_eq!(messages.len(), 5);
    assert_eq!(messages[0], json!({"role": "system", "content": "You are helpful."}));
    assert_eq!(messages[1], json!({"role": "user", "content": "Run the tests."}));
    // The assistant's thinking resolves through the resolver, its text and
    // call stay as one message.
    assert_eq!(messages[2]["content"], json!("Running."));
    assert_eq!(
        messages[2]["reasoning_content"],
        json!("resolved: sig1")
    );
    let calls = messages[2]["tool_calls"].as_array().expect("calls");
    assert_eq!(calls[0]["function"]["name"], json!("run_tests"));
    assert_eq!(calls[0]["function"]["arguments"]["path"], json!("src"));
    // The tool result leads the user turn.
    assert_eq!(messages[3]["role"], json!("tool"));
    assert_eq!(messages[3]["tool_call_id"], json!("c1"));
    assert_eq!(messages[3]["content"], json!("ok"));
    assert_eq!(messages[4]["content"], json!("Now memoize."));
}

#[test]
fn the_thinking_switches_the_effort_and_the_display() {
    // Omitted thinking is off, with a summarized display.
    let thinking = anthropic_thinking(&Value::Null, "high").expect("thinking");
    assert_eq!(thinking.effort, "none");
    assert_eq!(thinking.display, "summarized");
    // Disabled is off.
    let thinking = anthropic_thinking(&json!({"type": "disabled"}), "high").expect("thinking");
    assert_eq!(thinking.effort, "none");
    // Enabled keeps the effort.
    let thinking = anthropic_thinking(&json!({"type": "enabled"}), "high").expect("thinking");
    assert_eq!(thinking.effort, "high");
    // An omitted display shows nothing.
    let thinking = anthropic_thinking(
        &json!({"type": "enabled", "display": "omitted"}),
        "high",
    )
    .expect("thinking");
    assert_eq!(thinking.display, "omitted");
    // An invalid type is refused.
    assert!(anthropic_thinking(&json!({"type": "wat"}), "high").is_err());
}

#[test]
fn the_refusals_read_as_the_python_names_them() {
    // A prefill is refused.
    let body = json!({
        "model": "claude",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "partial"},
        ],
    });
    let error = anthropic_to_chat_body(&body, &resolver).expect_err("prefill");
    assert!(error.message.contains("prefill"), "{error}");
    // max_tokens must be positive.
    let body = json!({
        "model": "claude",
        "max_tokens": 0,
        "messages": [{"role": "user", "content": "hi"}],
    });
    assert!(anthropic_to_chat_body(&body, &resolver).is_err());
    // Grammar and regex are Chat's alone.
    let body = json!({
        "model": "claude",
        "max_tokens": 100,
        "grammar": "root",
        "messages": [{"role": "user", "content": "hi"}],
    });
    let error = anthropic_to_chat_body(&body, &resolver).expect_err("grammar");
    assert!(error.message.contains("grammar"), "{error}");
    // A billing-header system block is dropped.
    let body = json!({
        "model": "claude",
        "max_tokens": 100,
        "system": [
            {"type": "text", "text": "x-anthropic-billing-header: costs"},
            {"type": "text", "text": "You are helpful."},
        ],
        "messages": [{"role": "user", "content": "hi"}],
    });
    let (chat, _) = anthropic_to_chat_body(&body, &resolver).expect("converts");
    assert_eq!(chat["messages"][0]["content"], json!("You are helpful."));
}
