//! The Responses input normalization's tests, each expectation checked
//! against the Python implementation's own output.

use serde_json::{json, Value};
use splash_server::dialects::responses::normalize_responses_input;

#[test]
fn the_input_items_become_chat_messages() {
    let items = json!([
        {"type": "message", "role": "user", "content": "Run the tests."},
        {"type": "reasoning", "summary": [
            {"type": "summary_text", "text": "thought"}
        ]},
        {"type": "function_call", "call_id": "c1", "name": "run_tests",
         "arguments": "{\"path\": \"src\"}"},
        {"type": "function_call_output", "call_id": "c1", "output": "ok"},
        {"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": "Running."}
        ]},
        {"type": "message", "role": "user", "content": "Now memoize."},
    ]);
    let messages = normalize_responses_input(&json!("You are helpful."), &items)
        .expect("normalizes");
    assert_eq!(
        messages,
        vec![
            json!({"role": "system", "content": "You are helpful."}),
            json!({"role": "user", "content": "Run the tests."}),
            json!({
                "role": "assistant",
                "content": "",
                "reasoning_content": "thought",
                "tool_calls": [{
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "run_tests", "arguments": "{\"path\": \"src\"}"}
                }]
            }),
            json!({"role": "tool", "tool_call_id": "c1", "content": "ok"}),
            json!({"role": "assistant", "content": "Running."}),
            json!({"role": "user", "content": "Now memoize."}),
        ]
    );
}

#[test]
fn consecutive_assistant_items_accumulate_into_one() {
    let items = json!([
        {"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": "one"}
        ]},
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thought"}]},
        {"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": "two"}
        ]},
        {"type": "message", "role": "user", "content": "go on"},
    ]);
    let messages = normalize_responses_input(&Value::Null, &items).expect("normalizes");
    assert_eq!(messages.len(), 2);
    assert_eq!(
        messages[0]["content"],
        json!("onetwo"),
        "the assistant's texts accumulate"
    );
    assert_eq!(messages[0]["reasoning_content"], json!("thought"));
    assert_eq!(messages[1]["role"], json!("user"));
}

#[test]
fn a_system_item_leads_as_its_own_message() {
    let items = json!([
        {"type": "message", "role": "system", "content": [
            {"type": "input_text", "text": "You are helpful."}
        ]},
        {"type": "message", "role": "user", "content": "hi"},
    ]);
    let messages = normalize_responses_input(&Value::Null, &items).expect("normalizes");
    assert_eq!(messages[0], json!({"role": "system", "content": "You are helpful."}));
}

#[test]
fn the_refusals_read_as_the_python_names_them() {
    // An empty input.
    assert!(normalize_responses_input(&Value::Null, &json!([])).is_err());
    // A non-string instructions.
    assert!(normalize_responses_input(&json!(5), &json!([{"role": "user", "content": "hi"}])).is_err());
    // An unsupported item kind.
    let items = json!([{"type": "wat"}]);
    let error = normalize_responses_input(&Value::Null, &items).expect_err("unsupported");
    assert!(error.message.contains("only message, reasoning"), "{error}");
    // A function_call without its ids.
    let items = json!([{"type": "function_call", "name": "f", "arguments": "{}"}]);
    assert!(normalize_responses_input(&Value::Null, &items).is_err());
    // An unsupported role.
    let items = json!([{"type": "message", "role": "wat", "content": "x"}]);
    let error = normalize_responses_input(&Value::Null, &items).expect_err("role");
    assert!(error.message.contains("unsupported message role"), "{error}");
}

/// The Responses output items' tests, each expectation checked against
/// the Python implementation's own output.
mod output_items {
    use serde_json::json;
    use splash_server::dialects::responses::{responses_item, responses_output, OutputBlock};

    #[test]
    fn the_blocks_read_as_responses_items() {
        let text = responses_item(
            "abc123",
            &OutputBlock {
                kind: "text".into(),
                text: "The answer.".into(),
                call_id: None,
                name: None,
                status: "done".into(),
            },
            0,
        )
        .expect("item");
        assert_eq!(
            text,
            json!({
                "id": "msg_abc123_0",
                "type": "message",
                "status": "done",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "The answer.", "annotations": []}]
            })
        );
        let tool = responses_item(
            "abc123",
            &OutputBlock {
                kind: "tool".into(),
                text: r#"{"a": 1}"#.into(),
                call_id: Some("c1".into()),
                name: Some("f".into()),
                status: "done".into(),
            },
            1,
        )
        .expect("item");
        assert_eq!(
            tool,
            json!({
                "id": "fc_abc123_1",
                "type": "function_call",
                "status": "done",
                "call_id": "c1",
                "name": "f",
                "arguments": "{\"a\": 1}"
            })
        );
        let reasoning = responses_item(
            "abc123",
            &OutputBlock {
                kind: "reasoning".into(),
                text: "thought".into(),
                call_id: None,
                name: None,
                status: "done".into(),
            },
            2,
        )
        .expect("item");
        assert_eq!(
            reasoning,
            json!({
                "id": "rs_abc123_2",
                "type": "reasoning",
                "status": "done",
                "summary": [{"type": "summary_text", "text": "thought"}],
                "content": [{"type": "reasoning_text", "text": "thought"}],
                "encrypted_content": null
            })
        );
    }

    #[test]
    fn an_item_in_progress_carries_no_text_yet() {
        let item = responses_item(
            "abc123",
            &OutputBlock {
                kind: "text".into(),
                text: "The answer.".into(),
                call_id: None,
                name: None,
                status: "in_progress".into(),
            },
            3,
        )
        .expect("item");
        assert_eq!(item["content"], json!([]));
        assert_eq!(item["status"], json!("in_progress"));
    }

    #[test]
    fn the_output_is_the_blocks_in_order() {
        let output = responses_output(
            "abc123",
            &[
                OutputBlock {
                    kind: "text".into(),
                    text: "The answer.".into(),
                    call_id: None,
                    name: None,
                    status: "done".into(),
                },
                OutputBlock {
                    kind: "tool".into(),
                    text: r#"{"a": 1}"#.into(),
                    call_id: Some("c1".into()),
                    name: Some("f".into()),
                    status: "done".into(),
                },
            ],
        )
        .expect("output");
        assert_eq!(output.len(), 2);
        assert_eq!(output[0]["id"], json!("msg_abc123_0"));
        assert_eq!(output[1]["id"], json!("fc_abc123_1"));
    }
}
