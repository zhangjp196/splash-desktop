//! The Anthropic request conversion's tests, each expectation checked
//! against the Python implementation's own output.

use serde_json::{json, Value};
use splash_server::dialects::anthropic_request::{anthropic_thinking, anthropic_to_chat_body};
use splash_server::chat_messages::MessageError;

fn resolver(signature: &str) -> Result<String, MessageError> {
    Ok(format!("resolved: {signature}"))
}

#[test]
fn pdf_document_blocks_reach_the_shared_file_path() {
    let body = json!({"model": "claude", "max_tokens": 20, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Read this"},
        {"type": "document", "title": "Report", "source": {
            "type": "base64", "media_type": "application/pdf", "data": "JVBERg=="
        }}
    ]}]});
    let (chat, _) = anthropic_to_chat_body(&body, &resolver).unwrap();
    let content = &chat["messages"][0]["content"];
    assert_eq!(content[0]["text"], "Read this");
    assert_eq!(content[1]["text"], "Report\n");
    assert_eq!(content[2]["file"]["file_data"], "data:application/pdf;base64,JVBERg==");
}

#[test]
fn tool_results_preserve_media_references_and_failure() {
    let body = json!({"model": "claude", "max_tokens": 20, "messages": [
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call1", "is_error": true,
            "content": [{"type": "tool_reference", "tool_name": "lookup"},
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERg=="}}]}]}
    ]});
    let (chat, _) = anthropic_to_chat_body(&body, &resolver).unwrap();
    let content = &chat["messages"][0]["content"];
    assert_eq!(content[0]["text"], "Tool execution failed:\n");
    assert_eq!(content[1]["text"], "\nAvailable tool: lookup\n");
    assert_eq!(content[2]["file"]["file_data"], "data:application/pdf;base64,JVBERg==");
    let mut invalid = body;
    invalid["messages"][0]["content"][0]["is_error"] = json!("yes");
    assert!(anthropic_to_chat_body(&invalid, &resolver).is_err());
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

#[test]
fn omitted_thinking_roundtrips_through_a_fernet_signature() {
    let codec = splash_server::thinking::ThinkingCodec::new(
        b"MDEyMzQ1Njc4OWFiY2RlZmZlZGNiYTk4NzY1NDMyMTA=",
    )
    .expect("Fernet key");
    let hidden = "private omitted reasoning";
    let signature = codec.encode(hidden).expect("signs hidden reasoning");
    let body = json!({
        "model": "claude",
        "max_tokens": 100,
        "thinking": {"type": "enabled", "display": "omitted"},
        "messages": [
            {"role": "user", "content": "continue"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "", "signature": signature},
                {"type": "text", "text": "visible answer"}
            ]},
            {"role": "user", "content": "next"}
        ]
    });
    let resolver = |signature: &str| {
        codec.decode(signature).map_err(|error| MessageError {
            status: error.status,
            message: error.message,
            code: error.code,
        })
    };
    let (chat, display) = anthropic_to_chat_body(&body, &resolver).expect("valid signature");
    assert_eq!(display, "omitted");
    assert_eq!(chat["messages"][1]["reasoning_content"], json!(hidden));
}
