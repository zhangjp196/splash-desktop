//! The chat format's derived shapes and the template loader's selection,
//! ported from the Python tests that read them.

use splash_server::chat_format::{flint, qwen, GLM_ARGS, QWEN_XML, TOOL_CALL_OPEN};
use splash_server::templates::{load_chat_templates, Templates};

#[test]
fn the_qwen_format_derives_its_shapes() {
    let format = qwen();
    assert_eq!(format.vocabulary, 248320);
    assert_eq!(
        format.stop_ids(),
        vec![248044, 248046]
    );
    assert_eq!(format.think_end_id, 248069);
    assert_eq!(format.think_start_id, None);
    assert_eq!(format.call_open_id, 248058);
    // A generation prompt opens the think block, so the request switches
    // the reasoning, and a call's opening ends it.
    assert!(format.reasoning_switch());
    assert!(format.calls_end_reasoning());
    assert_eq!(format.message_boundary, "<|im_end|>");
}

#[test]
fn the_flint_format_opens_its_reasoning_itself() {
    let format = flint();
    assert_eq!(format.vocabulary, 150144);
    assert_eq!(format.stop_ids(), vec![150001, 150003, 150005, 150018]);
    assert_eq!(format.think_start_id, Some(150006));
    assert!(!format.reasoning_switch());
    assert!(!format.calls_end_reasoning());
}

#[test]
fn the_xml_syntax_lays_a_call_out_on_lines() {
    assert_eq!(QWEN_XML.call_open(), format!("{TOOL_CALL_OPEN}\n<function="));
    assert_eq!(
        QWEN_XML.function_opening("run_tests"),
        "\n<function=run_tests>\n"
    );
    // The name ends at the first character of name_close.
    assert_eq!(QWEN_XML.name_end(), '>');
    // The tags that may follow an argument: the next one's opening, the
    // arguments' close without its space, and the call's close.
    let tags = QWEN_XML.argument_tags();
    assert_eq!(tags.len(), 3);
    assert_eq!(tags[0], "<parameter=");
    assert_eq!(tags[1], "</function>");
    assert_eq!(tags[2], "\u{003c}\u{002f}\u{0074}\u{006f}\u{006f}\u{006c}\u{005f}\u{0063}\u{0061}\u{006c}\u{006c}\u{003e}");
    // A value ends at its close where a following tag follows it.
    let ends = QWEN_XML.value_ends();
    assert_eq!(ends.len(), 3);
    assert_eq!(ends[0], "\n</parameter>\n<parameter=");
    assert_eq!(ends[1], "\n</parameter>\n</function>");
}

#[test]
fn the_glm_syntax_writes_its_tags_as_tokens() {
    assert!(GLM_ARGS.tags_are_tokens);
    assert!(!QWEN_XML.tags_are_tokens);
    // The name ends at the first of the tags that follow it, where
    // name_close is empty.
    // The name ends at the first character of key_open, which is '<'.
    assert_eq!(GLM_ARGS.name_end(), '<');
    assert_eq!(GLM_ARGS.call_open(), TOOL_CALL_OPEN);
}

#[test]
fn the_loader_reads_the_jinja_file_and_the_named_templates() {
    let directory = std::env::temp_dir().join("splash-rs-loader-test");
    let _ = std::fs::remove_dir_all(&directory);
    let tokenizer_dir = directory.join("tokenizer");
    std::fs::create_dir_all(tokenizer_dir.join("additional_chat_templates")).expect("dirs");
    std::fs::write(
        tokenizer_dir.join("chat_template.jinja"),
        "DEFAULT TEMPLATE",
    )
    .expect("write default");
    std::fs::write(
        tokenizer_dir
            .join("additional_chat_templates")
            .join("tool_use.jinja"),
        "TOOL TEMPLATE",
    )
    .expect("write tool_use");

    let templates = load_chat_templates(&tokenizer_dir).expect("loads");
    match &templates {
        Templates::Named(named) => {
            assert_eq!(named.len(), 2);
            assert_eq!(named[0], (Some("default".into()), "DEFAULT TEMPLATE".to_string()));
            assert_eq!(named[1], (Some("tool_use".into()), "TOOL TEMPLATE".to_string()));
        }
        other => panic!("expected named templates, got {other:?}"),
    }
    // The default renders by default; the tools pick tool_use.
    assert_eq!(templates.select(None, None).unwrap(), "DEFAULT TEMPLATE");
    assert_eq!(
        templates
            .select(None, Some(&serde_json::json!([{"type": "function"}])))
            .unwrap(),
        "TOOL TEMPLATE"
    );
    // A request's own source passes through.
    assert_eq!(templates.select(Some("OWN"), None).unwrap(), "OWN");
    // A name picks its template.
    assert_eq!(templates.select(Some("tool_use"), None).unwrap(), "TOOL TEMPLATE");
    let _ = std::fs::remove_dir_all(&directory);
}

#[test]
fn the_legacy_config_templates_load() {
    let directory = std::env::temp_dir().join("splash-rs-loader-legacy");
    let _ = std::fs::remove_dir_all(&directory);
    let tokenizer_dir = directory.join("tokenizer");
    std::fs::create_dir_all(&tokenizer_dir).expect("dir");
    std::fs::write(
        tokenizer_dir.join("tokenizer_config.json"),
        r#"{"chat_template": [{"name": "default", "template": "LEGACY"}]}"#,
    )
    .expect("write config");
    let templates = load_chat_templates(&tokenizer_dir).expect("loads");
    assert_eq!(templates.select(None, None).unwrap(), "LEGACY");
    let _ = std::fs::remove_dir_all(&directory);
}

#[test]
fn no_template_is_an_error() {
    let directory = std::env::temp_dir().join("splash-rs-loader-empty");
    let _ = std::fs::remove_dir_all(&directory);
    let tokenizer_dir = directory.join("tokenizer");
    std::fs::create_dir_all(&tokenizer_dir).expect("dir");
    let templates = load_chat_templates(&tokenizer_dir).expect("loads");
    assert!(templates.is_empty());
    assert!(templates.select(None, None).is_err());
    let _ = std::fs::remove_dir_all(&directory);
}
