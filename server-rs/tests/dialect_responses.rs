//! The Responses input normalization's tests, each expectation checked
//! against the Python implementation's own output.

use serde_json::{json, Value};
use splash_server::dialects::responses::{normalize_responses_input, canonical_responses_input, responses_to_chat_body, ResponseStore};

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

#[test]
fn responses_request_maps_to_chat_and_accepts_string_input() {
    let items = canonical_responses_input(&json!("hi")).expect("canonical input");
    let chat = responses_to_chat_body(&json!({
        "instructions":"Be concise.",
        "max_output_tokens":48,
        "tools":[{"type":"function", "name":"lookup", "parameters":{"type":"object"}}],
        "text":{"format":{"type":"json_object"}}
    }), &items).expect("chat body");
    assert_eq!(chat["messages"], json!([
        {"role":"system", "content":"Be concise."},
        {"role":"user", "content":"hi"}
    ]));
    assert_eq!(chat["max_completion_tokens"], 48);
    assert_eq!(chat["tools"][0]["function"]["name"], "lookup");
    assert_eq!(chat["response_format"]["type"], "json_object");
}

#[test]
fn responses_namespace_and_custom_tools_match_python_normalization() {
    use splash_server::dialects::responses::responses_to_chat_body_with_tools;
    let items = canonical_responses_input(&json!("run it")).unwrap();
    let (chat, metadata) = responses_to_chat_body_with_tools(&json!({
        "tools":[
            {"type":"namespace","name":"store","tools":[
                {"type":"function","name":"lookup","description":"Find an item","parameters":{"type":"object"}}
            ]},
            {"type":"custom","name":"shell","description":"Run a command","format":{"type":"grammar","syntax":"GBNF","definition":"root ::= \\\"ok\\\""}}
        ],
        "tool_choice":{"type":"function","namespace":"store","name":"lookup"}
    }), &items).expect("normalizes tools");
    assert_eq!(chat["tools"][0]["function"]["name"], "store__lookup");
    assert_eq!(chat["tools"][1]["function"]["parameters"]["required"], json!(["input"]));
    assert_eq!(chat["tools"][1]["function"]["parameters"]["properties"]["input"]["type"], "string");
    assert_eq!(chat["tool_choice"]["function"]["name"], "store__lookup");
    assert_eq!(metadata["store__lookup"], json!({"kind":"namespace","namespace":"store","name":"lookup"}));
    assert_eq!(metadata["shell"]["kind"], "custom");
}

#[test]
fn responses_reject_unsupported_custom_and_namespace_shapes() {
    use splash_server::dialects::responses::responses_to_chat_body_with_tools;
    let items = canonical_responses_input(&json!("x")).unwrap();
    for tools in [
        json!([{"type":"custom","name":"bad<name"}]),
        json!([{"type":"custom","name":"ok","format":{"type":"grammar","syntax":"GBNF"}}]),
        json!([{"type":"namespace","name":"ns","tools":[{"type":"custom","name":"x"}]}]),
        json!([{"type":"computer_use"}]),
    ] {
        assert_eq!(responses_to_chat_body_with_tools(&json!({"tools":tools}), &items).unwrap_err().status, 400);
    }
}

#[test]
fn responses_output_restores_namespace_names_and_custom_input() {
    use splash_server::dialects::responses::{responses_output_with_tools, OutputBlock};
    let metadata = json!({
        "files__search":{"kind":"namespace","namespace":"files","name":"search"},
        "shell":{"kind":"custom","name":"shell"}
    });
    let output = responses_output_with_tools("req_tools", &[
        OutputBlock { kind:"tool".into(), text:r#"{"query":"notes"}"#.into(), call_id:Some("c1".into()), name:Some("files__search".into()), status:"completed".into() },
        OutputBlock { kind:"tool".into(), text:r#"{"input":"echo hi"}"#.into(), call_id:Some("c2".into()), name:Some("shell".into()), status:"completed".into() },
    ], &metadata).unwrap();
    assert_eq!(output[0], json!({"id":"fc_req_tools_0","type":"function_call","status":"completed","call_id":"c1","name":"search","arguments":"{\"query\":\"notes\"}","namespace":"files"}));
    assert_eq!(output[1], json!({"id":"ctc_req_tools_1","type":"custom_tool_call","status":"completed","call_id":"c2","name":"shell","input":"echo hi"}));
}

#[test]
fn responses_reject_background_execution_like_python() {
    let items = canonical_responses_input(&json!("hi")).expect("canonical input");
    let error = responses_to_chat_body(&json!({"background": true}), &items)
        .expect_err("background execution is unsupported");
    assert_eq!(error.status, 400);
    assert_eq!(error.message, "background responses are not supported");
}

#[test]
fn stored_responses_can_be_retrieved_and_deleted() {
    let store = ResponseStore::new();
    store.put("resp_abc".into(), json!({"id":"resp_abc"}), vec![json!({"role":"user","content":"hi"})]);
    let response = store.get("resp_abc").expect("stored response");
    assert_eq!(response.response["id"], "resp_abc");
    assert_eq!(response.history.len(), 1);
    assert!(store.delete("resp_abc"));
    assert!(store.get("resp_abc").is_none());
}

#[test]
fn image_and_file_inputs_keep_their_canonical_media_parts() {
    let items = json!([{
        "type":"message", "role":"user", "content":[
            {"type":"input_text", "text":"See these."},
            {"type":"input_image", "image_url":"data:image/png;base64,AA=="},
            {"type":"input_file", "file_data":"data:application/pdf;base64,JVBERi0="},
        ]
    }]);
    let messages = normalize_responses_input(&Value::Null, &items).expect("normalizes");
    assert_eq!(messages[0]["content"][0], json!({"type":"text", "text":"See these."}));
    assert_eq!(messages[0]["content"][1], json!({
        "type":"image_url", "image_url":{"url":"data:image/png;base64,AA=="}
    }));
    assert_eq!(messages[0]["content"][2], json!({
        "type":"file", "file":{"file_data":"data:application/pdf;base64,JVBERi0="}
    }));
}

#[test]
fn chat_completion_becomes_a_responses_completion_with_usage() {
    use splash_server::dialects::responses::response_from_chat;
    let request = json!({"text":{"format":{"type":"json_object"}}});
    let chat = json!({
        "choices":[{"finish_reason":"length", "message":{"content":"{}", "reasoning_content":"thinking"}}],
        "usage":{"prompt_tokens":10,"completion_tokens":4,"prompt_tokens_details":{"cached_tokens":2},"completion_tokens_details":{"reasoning_tokens":2}},
        "timings":{"prompt_ms":1.0}
    });
    let response = response_from_chat("model", "req1", 123, &request, &chat, true, None).expect("converts");
    assert_eq!(response["id"], "resp_req1");
    assert_eq!(response["status"], "incomplete");
    assert_eq!(response["incomplete_details"]["reason"], "max_output_tokens");
    assert_eq!(response["usage"]["input_tokens"], 10);
    assert_eq!(response["usage"]["input_tokens_details"]["cached_tokens"], 2);
    assert_eq!(response["output"][0]["type"], "reasoning");
    assert_eq!(response["output"][1]["type"], "message");
    assert_eq!(response["timings"]["prompt_ms"], 1.0);
}

#[test]
fn responses_stream_maps_live_chat_chunks_to_named_events_and_stores_final() {
    use splash_server::dialects::responses::{ResponseStore, ResponsesStreamState};
    let store = ResponseStore::new();
    let mut state = ResponsesStreamState::new(
        "model".into(), "req_stream".into(), 456,
        json!({"store":true, "input":"hi"}), true, None,
        vec![json!({"type":"message", "role":"user", "content":"hi"})],
        store.clone(),
    );
    let events = state.consume(r#"{"choices":[{"delta":{"role":"assistant"}}]}"#).unwrap();
    assert_eq!(events.iter().map(|(event, _)| event.as_str()).collect::<Vec<_>>(), [
        "response.created", "response.in_progress",
    ]);
    let events = state.consume(r#"{"choices":[{"delta":{"content":"Hello"}}]}"#).unwrap();
    assert_eq!(events.iter().map(|(event, _)| event.as_str()).collect::<Vec<_>>(), [
        "response.output_item.added", "response.content_part.added", "response.output_text.delta",
    ]);
    assert_eq!(events[2].1["delta"], "Hello");
    state.consume(r#"{"choices":[{"delta":{},"finish_reason":"stop"}],"timings":{"prompt_ms":2.0}}"#).unwrap();
    state.consume(r#"{"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}"#).unwrap();
    let events = state.consume("[DONE]").unwrap();
    assert_eq!(events.last().unwrap().0, "response.completed");
    let response = &events.last().unwrap().1["response"];
    assert_eq!(response["output"][0]["content"][0]["text"], "Hello");
    assert_eq!(response["usage"]["total_tokens"], 4);
    assert_eq!(store.get("resp_req_stream").unwrap().response, *response);
}

#[test]
fn responses_stream_emits_reasoning_summary_item_lifecycle() {
    use splash_server::dialects::responses::{ResponseStore, ResponsesStreamState};
    let mut state = ResponsesStreamState::new(
        "model".into(), "req_reason".into(), 456, json!({}), false, None, vec![], ResponseStore::new(),
    );
    state.consume(r#"{"choices":[{"delta":{"role":"assistant"}}]}"#).unwrap();
    let opened = state.consume(r#"{"choices":[{"delta":{"reasoning_content":"thinking"}}]}"#).unwrap();
    assert_eq!(opened.iter().map(|(event, _)| event.as_str()).collect::<Vec<_>>(), [
        "response.output_item.added",
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_text.delta",
    ]);
    assert_eq!(opened[2].1["delta"], "thinking");
    state.consume(r#"{"choices":[{"delta":{},"finish_reason":"length"}]}"#).unwrap();
    let closed = state.consume("[DONE]").unwrap();
    assert_eq!(closed.iter().map(|(event, _)| event.as_str()).collect::<Vec<_>>(), [
        "response.reasoning_summary_text.done",
        "response.reasoning_summary_part.done",
        "response.output_item.done",
        "response.incomplete",
    ]);
    assert_eq!(closed[2].1["item"]["status"], "incomplete");
    assert_eq!(closed[3].1["response"]["status"], "incomplete");
}

#[test]
fn responses_stream_emits_custom_tool_input_only_when_the_call_closes() {
    use splash_server::dialects::responses::{ResponseStore, ResponsesStreamState};
    let metadata = json!({"shell":{"kind":"custom","name":"shell"}});
    let mut state = ResponsesStreamState::new_with_tools(
        "model".into(), "req_custom".into(), 456, json!({}), false, None, vec![],
        ResponseStore::new(), metadata,
    );
    let opened = state.consume(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"shell","arguments":""}}]}}]}"#).unwrap();
    assert_eq!(opened[0].0, "response.created");
    assert_eq!(opened.iter().find(|(event, _)| event == "response.output_item.added").unwrap().1["item"]["type"], "custom_tool_call");
    let argument = state.consume(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"input\":\"echo hi\"}"}}]}}]}"#).unwrap();
    assert!(!argument.iter().any(|(event, _)| event.contains("arguments.delta")));
    let done = state.consume("[DONE]").unwrap();
    let custom_events: Vec<_> = done.iter().filter(|(event, _)| event.starts_with("response.custom_tool_call_input.")).collect();
    assert_eq!(custom_events.len(), 2);
    assert_eq!(custom_events[0].1["delta"], "echo hi");
    assert_eq!(custom_events[1].1["input"], "echo hi");
    assert_eq!(done.last().unwrap().1["response"]["output"][0]["id"], "ctc_req_custom_0");
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
