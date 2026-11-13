//! The message normalization's tests, ported from the Python tests that
//! drive `normalize_messages` and `template_messages`
//! (`dev/tests/server/test_chat_messages.py`'s cases).

use serde_json::{json, Value};
use splash_server::chat_messages::{normalize_messages, template_messages};

#[test]
fn plain_messages_normalize() {
    let messages = json!([
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Write a fib function."},
        {"role": "assistant", "content": "def fib(n): ..."},
        {"role": "user", "content": "Now memoize it."},
    ]);
    let normalized = normalize_messages(&messages, false).expect("normalizes");
    assert_eq!(normalized.len(), 4);
    assert_eq!(normalized[0]["role"], json!("system"));
    assert_eq!(normalized[2]["role"], json!("assistant"));
}

#[test]
fn system_and_developer_lead_as_one_system_message() {
    let messages = json!([
        {"role": "system", "content": "First."},
        {"role": "developer", "content": "Second."},
        {"role": "user", "content": "hi"},
    ]);
    let normalized = normalize_messages(&messages, false).expect("normalizes");
    assert_eq!(normalized.len(), 2);
    assert_eq!(normalized[0]["content"], json!("First.\n\nSecond."));
}

#[test]
fn content_parts_read_as_text() {
    let messages = json!([
        {"role": "user", "content": [
            {"type": "text", "text": "one "},
            {"type": "input_text", "text": "two"},
        ]},
    ]);
    let normalized = normalize_messages(&messages, false).expect("normalizes");
    assert_eq!(normalized[0]["content"], json!("one two"));
}

#[test]
fn an_image_part_needs_vision() {
    let messages = json!([
        {"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,xx"}},
        ]},
    ]);
    let error = normalize_messages(&messages, false).expect_err("refused without vision");
    assert!(error.message.contains("without vision"), "{error}");
    // With vision, the part is canonical and the text stays text.
    let normalized = normalize_messages(&messages, true).expect("normalizes");
    let content = &normalized[0]["content"];
    assert!(content.is_array());
    assert_eq!(content[1]["type"], json!("image_url"));
    assert_eq!(content[0]["type"], json!("text"));
}

#[test]
fn unsupported_parts_and_roles_are_refused() {
    for messages in [
        json!([{"role": "user", "content": [{"type": "video_url", "video_url": "x"}]}]),
        json!([{"role": "user", "content": [{"type": "wat", "text": "x"}]}]),
        json!([{"role": "wat", "content": "x"}]),
        json!([]),
        json!("not a list"),
    ] {
        assert!(
            normalize_messages(&messages, false).is_err(),
            "accepted {messages}"
        );
    }
}

#[test]
fn a_user_message_is_required() {
    let messages = json!([
        {"role": "system", "content": "You are helpful."},
        {"role": "assistant", "content": "I am."},
    ]);
    let error = normalize_messages(&messages, false).expect_err("no user");
    assert!(error.message.contains("user message"), "{error}");
}

#[test]
fn an_assistant_message_carries_reasoning_and_calls() {
    let arguments_text = "{\"path\": \"src\"}";
    let messages = json!([
        {"role": "assistant", "content": "thinking", "reasoning_content": "the reasoning",
         "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "run_tests", "arguments": arguments_text}},
         ]},
        {"role": "tool", "content": "ok", "tool_call_id": "call_1"},
        {"role": "user", "content": "thanks"},
    ]);
    let normalized = normalize_messages(&messages, false).expect("normalizes");
    assert_eq!(normalized[0]["reasoning_content"], json!("the reasoning"));
    let calls = &normalized[0]["tool_calls"];
    assert_eq!(calls[0]["function"]["name"], json!("run_tests"));
    // The arguments text reads as an object.
    assert_eq!(calls[0]["function"]["arguments"]["path"], json!("src"));
    assert_eq!(normalized[1]["tool_call_id"], json!("call_1"));
}

#[test]
fn a_history_call_with_text_arguments_stays_text() {
    let arguments_text = "not json";
    let messages = json!([
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "run_tests", "arguments": arguments_text}},
        ]},
        {"role": "user", "content": "again"},
    ]);
    let normalized = normalize_messages(&messages, false).expect("normalizes");
    // The arguments that spell no object stay the text as sent.
    assert_eq!(
        normalized[0]["tool_calls"][0]["function"]["arguments"],
        json!("not json")
    );
    // template_messages keeps such a call as text, without presenting it
    // as a complete call.
    let templated = template_messages(&normalized);
    assert_eq!(templated.len(), 2);
    assert_eq!(
        templated[0]["content"],
        Value::String(
            "Incomplete tool call:\n{\"name\":\"run_tests\",\"arguments\":\"not json\"}".to_string()
        )
    );
    assert!(templated[0].get("tool_calls").is_none());
}

#[test]
fn a_complete_call_stays_whole() {
    let messages = json!([
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "run_tests", "arguments": {"path": "src"}}},
        ]},
        {"role": "user", "content": "thanks"},
    ]);
    let normalized = normalize_messages(&messages, false).expect("normalizes");
    let templated = template_messages(&normalized);
    assert_eq!(templated.len(), 2);
    assert_eq!(templated[0], normalized[0]);
}
