//! The chat templates' golden test: the Rust renderer must reproduce the
//! Python renderer's bytes for the real Qwen template, case by case, and
//! refuse what it refuses. The fixture is
//! `tests/fixtures/chat_template_golden.json`, which the Python
//! implementation wrote.

use std::collections::BTreeMap;
use std::path::PathBuf;

use splash_server::templates::{ChatTemplateError, ChatTemplateRenderer};

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/chat_template_golden.json")
}

#[derive(serde::Deserialize)]
struct GoldenCase {
    name: String,
    messages: serde_json::Value,
    options: BTreeMap<String, serde_json::Value>,
    rendered: Option<String>,
    error: Option<String>,
}

#[derive(serde::Deserialize)]
struct Golden {
    template_source: String,
    cases: Vec<GoldenCase>,
    rejection: Option<String>,
}

fn load_golden() -> Golden {
    let text = std::fs::read_to_string(fixture_path()).expect("golden fixture");
    serde_json::from_str(&text).expect("golden fixture parses")
}

/// The options the renderer takes as arguments, and the rest as kwargs.
fn split_options(options: &BTreeMap<String, serde_json::Value>) -> (Option<serde_json::Value>, bool, BTreeMap<String, serde_json::Value>) {
    let mut kwargs = options.clone();
    let tools = kwargs.remove("tools");
    let add_generation_prompt = kwargs
        .remove("add_generation_prompt")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    (tools, add_generation_prompt, kwargs)
}

#[test]
fn the_qwen_template_renders_as_the_python_does() {
    let golden = load_golden();
    let renderer = ChatTemplateRenderer::new(&golden.template_source)
        .expect("the template compiles");
    for case in &golden.cases {
        let (tools, add_generation_prompt, kwargs) = split_options(&case.options);
        let rendered = renderer.render(&case.messages, tools.as_ref(), None, add_generation_prompt, &kwargs);
        match (&case.rendered, &case.error) {
            (Some(expected), _) => {
                let actual = rendered.unwrap_or_else(|error| {
                    panic!("case {} failed to render: {error}", case.name)
                });
                assert_eq!(actual, *expected, "case {} rendered differently", case.name);
            }
            (None, Some(expected_error)) => {
                let error = rendered.expect_err(&case.name);
                assert!(
                    expected_error.contains(&error.0) || error.0.contains(expected_error),
                    "case {} refused differently:\n  python: {expected_error}\n  rust:   {error}",
                    case.name
                );
            }
            (None, None) => panic!("case {} has neither a render nor an error", case.name),
        }
    }
}

#[test]
fn the_rejection_case_refuses_with_the_templates_message() {
    let golden = load_golden();
    let renderer = ChatTemplateRenderer::new(&golden.template_source).expect("compiles");
    let messages = &golden.cases[0].messages;
    let mut kwargs = BTreeMap::new();
    kwargs.insert("reasoning_effort".to_string(), serde_json::json!("high"));
    let error = renderer
        .render(messages, None, None, true, &kwargs)
        .expect_err("the template refuses effort high");
    assert!(
        golden.rejection.as_ref().map(|expected| error.0.contains(expected)).unwrap_or(false),
        "the refusal differs:\n  python: {:?}\n  rust:   {error}",
        golden.rejection
    );
}

#[test]
fn the_generation_tags_are_stripped_and_the_body_stays() {
    // A template with a generation block renders its body where the block
    // was, as the string path does.
    let source = "{% generation %}ANSWER{% endgeneration %}";
    let renderer = ChatTemplateRenderer::new(source).expect("compiles");
    let rendered = renderer
        .render(&serde_json::json!([]), None, None, false, &BTreeMap::new())
        .expect("renders");
    assert_eq!(rendered, "ANSWER");
}

#[test]
fn tojson_keeps_the_keys_order_and_the_python_separators() {
    let renderer = ChatTemplateRenderer::new("{{ tools|tojson }}").expect("compiles");
    let tools = serde_json::json!([
        {"type": "function", "function": {"name": "run_tests", "parameters": {"type": "object"}}}
    ]);
    let rendered = renderer
        .render(&serde_json::json!([]), Some(&tools), None, false, &BTreeMap::new())
        .expect("renders");
    assert_eq!(
        rendered,
        r#"[{"type": "function", "function": {"name": "run_tests", "parameters": {"type": "object"}}}]"#
    );
}

#[test]
fn a_template_error_reads_as_a_template_error() {
    let renderer = ChatTemplateRenderer::new("{{ raise_exception('nope') }}").expect("compiles");
    let error = renderer
        .render(&serde_json::json!([]), None, None, false, &BTreeMap::new())
        .expect_err("raises");
    assert!(error.0.contains("nope"), "{error}");
    assert!(matches!(error, ChatTemplateError(_)));
}
