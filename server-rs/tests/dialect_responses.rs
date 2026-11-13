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
