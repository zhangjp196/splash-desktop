//! The tool-call projector's tests, ported from the Python tests that
//! drive `StreamingToolCallProjector` (`dev/tests/tool_output.py`'s
//! `project` and `dev/tests/server/test_partial_tool_output.py`'s cases),
//! each expectation checked against the Python's own output.

use std::collections::{BTreeMap, HashMap, HashSet};

use splash_server::chat_format::{qwen, TOOL_CALL_OPEN};
use splash_server::output::{
    convert_value, ProjectorEvent, StaticTypes, StreamingToolCallProjector, ToolCall, Types,
};

fn types(names: &[&str]) -> Types {
    names.iter().map(|name| name.to_string()).collect()
}

/// The provider of a request that offers `run_tests` with a string path
/// and an integer count, as the Python test's policy reads.
fn run_tests_types() -> StaticTypes {
    let mut tools = BTreeMap::new();
    let mut parameters = HashMap::new();
    parameters.insert("path".to_string(), types(&["string"]));
    parameters.insert("count".to_string(), types(&["integer"]));
    tools.insert("run_tests".to_string(), (parameters, None));
    StaticTypes { tools }
}

fn projector() -> StreamingToolCallProjector {
    StreamingToolCallProjector::new(
        Box::new(run_tests_types()),
        7,
        &qwen(),
        false,
    )
}

/// The call the Python's projector produces for the whole output.
fn expected_call() -> ToolCall {
    ToolCall {
        id: "call_7_0".into(),
        name: "run_tests".into(),
        arguments: r#"{"path":"src/main.rs","count":3}"#.into(),
    }
}

#[test]
fn a_call_projects_into_argument_deltas() {
    let mut projector = projector();
    let output = format!(
        "{TOOL_CALL_OPEN}\n<function=run_tests>\n<parameter=path>\nsrc/main.rs\n</parameter>\n<parameter=count>\n3\n</parameter>\n</function>\n"
    );
    let mut events = projector.put(&output);
    let (content, calls, owed) = projector.finish(false);
    events.extend(owed);
    // The whitespace that only framed the call is dropped.
    assert_eq!(content, "");
    assert_eq!(calls, vec![expected_call()]);
    // The stream: the call's opening, then the arguments in deltas.
    assert_eq!(
        events[0],
        ProjectorEvent::ToolCall {
            index: 0,
            id: "call_7_0".into(),
            name: "run_tests".into(),
        }
    );
    let deltas: Vec<String> = events
        [1..]
        .iter()
        .filter_map(|event| match event {
            ProjectorEvent::ToolDelta { chunk, .. } => Some(chunk.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        deltas.concat(),
        r#"{"path":"src/main.rs","count":3}"#,
        "the deltas assemble the arguments"
    );
    // The string value streams as it is written; the integer converts at
    // its end.
    assert!(deltas.contains(&&"src/main.rs".to_string()));
    assert!(deltas.contains(&&r#","count":3"#.to_string()));
}

#[test]
fn a_call_cut_short_keeps_its_arguments() {
    let mut projector = projector();
    let output = format!(
        "{TOOL_CALL_OPEN}\n<function=run_tests>\n<parameter=path>\nsrc/ma"
    );
    projector.put(&output);
    let (content, calls, _owed) = projector.finish(true);
    // Cut output keeps the open call with the arguments it has, and no
    // call whose name it cut; the content is what the stream published.
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].arguments, r#"{"path":"src/ma"#);
    assert!(content.contains("src/ma") || content.is_empty());
}

#[test]
fn without_a_policy_a_call_opening_is_text() {
    let mut projector = StreamingToolCallProjector::without_policy(1, &qwen());
    let mut events = projector.put("hello ");
    events.extend(projector.put(&format!(
        "{TOOL_CALL_OPEN}\n<function=run_tests>\n"
    )));
    events.extend(projector.put(&format!("{} tail", "\u{003c}\u{002f}\u{0074}\u{0068}\u{0069}\u{006e}\u{006b}\u{003e}")));
    let (content, calls, _owed) = projector.finish(false);
    // Without a tool policy the output has no calls: a call's opening is
    // text, and only the reasoning's closes are dropped.
    assert!(calls.is_empty());
    assert_eq!(
        content,
        format!(
            "hello {TOOL_CALL_OPEN}\n<function=run_tests>\n tail"
        )
    );
    let texts: Vec<String> = events
        .iter()
        .filter_map(|event| match event {
            ProjectorEvent::Content(text) => Some(text.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(texts.concat(), content);
}

#[test]
fn a_repeated_parameter_keeps_its_first_value() {
    let mut projector = projector();
    let output = format!(
        "{TOOL_CALL_OPEN}\n<function=run_tests>\n<parameter=path>\nfirst\n</parameter>\n<parameter=path>\nsecond\n</parameter>\n</function>\n"
    );
    projector.put(&output);
    let (_content, calls, _owed) = projector.finish(false);
    assert_eq!(calls[0].arguments, r#"{"path":"first"}"#);
}

#[test]
fn values_convert_by_their_declared_types() {
    // A string value reads as the text; an integer as the JSON it spells.
    assert_eq!(
        convert_value("src/main.rs", &types(&["string"])),
        serde_json::json!("src/main.rs")
    );
    assert_eq!(convert_value("3", &types(&["integer"])), serde_json::json!(3));
    // A number declared reads an integer too.
    assert_eq!(convert_value("3", &types(&["number"])), serde_json::json!(3));
    // Empty types never reach convert_value in production (the value
    // streams); the function itself reads the JSON the text spells.
    assert_eq!(convert_value("3", &Types::new()), serde_json::json!(3));
    // A boolean declared without a string may be spelled as Python does.
    assert_eq!(
        convert_value("True", &types(&["boolean"])),
        serde_json::json!(true)
    );
    // Text that spells no JSON of a declared type is its text.
    assert_eq!(
        convert_value("not json", &types(&["integer"])),
        serde_json::json!("not json")
    );
}
