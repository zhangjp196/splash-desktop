//! The tool schema's tests, ported from the Python tests that drive
//! `server/tool_schema.py`, `server/schema_validation.py` and
//! `server/output.py`'s `validate_response_content`. The grammar strings the
//! fixtures hold are the Python implementation's own output for the same
//! inputs, so parity is pinned to the byte.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde_json::{json, Map, Value};
use serde_json::Value as Json;

use splash_server::chat_format::{flint, qwen};
use splash_server::constraints::Grammar;
use splash_server::output::projector::ParameterTypes;
use splash_server::support::errors::ApiError;
use splash_server::tool_schema::grammar::{argument_grammar, free_arguments, grammar_compatible_schema};
use splash_server::tool_schema::policy::{
    normalize_response_format, normalize_tools, check_tool_schema, ToolParameterTypes, ToolPolicy,
};
use splash_server::tool_schema::schema::{
    argument_shape, check_depth, is_tag_name, json_type_names, literal_texts, lookup_tool_reference,
    schema_types, strict_schema, tool_argument_schema,
    MAX_FRAMED_SCHEMA_BYTES,
};
use splash_server::tool_schema::validate::{
    build_validator, check_schema, validate_response_content, BuildError, ValidationFailure,
};
use splash_server::tool_schema::{json_grammar, tool_grammar};

fn fixture(name: &str) -> String {
    let path = format!(
        "{}/tests/fixtures/tool_grammar/{name}.txt",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("fixture {name}: {error}"))
}

/// Whether llguidance takes the grammar, as the Python's
/// `LLMatcher.validate_grammar` answers.
fn grammar_compiles(grammar: &str) -> bool {
    let Ok(top) = llguidance::api::TopLevelGrammar::from_lark_or_grammar_list(grammar) else {
        return false;
    };
    llguidance::api::GrammarInit::Serialized(top)
        .validate(None, llguidance::api::ParserLimits::default())
        .into_error()
        .is_none()
}

/// The policy of a request that offers each tool named in `schemas` with its
/// parameter schema, as `dev.tests.tool_output.tool_policy` builds one.
fn tool_policy(schemas: Value) -> ToolPolicy {
    let tools: Vec<Value> = schemas
        .as_object()
        .expect("schemas object")
        .iter()
        .map(|(name, schema)| {
            json!({"type": "function", "function": {"name": name, "parameters": schema}})
        })
        .collect();
    normalize_tools(Some(&Value::Array(tools)), Some(&json!("auto")), Some(&json!(true)), None)
        .expect("policy builds")
        .1
        .expect("policy present")
}

/// The argument grammar of a strict tool whose parameters are `schema`,
/// framed within a request's budget as ToolPolicy frames it.
fn argument_grammar_of(schema: Value) -> String {
    let mut schemas = Map::new();
    schemas.insert("tool".to_string(), schema);
    let policy = ToolPolicy::new(
        schemas,
        false,
        true,
        HashSet::from(["tool".to_string()]),
        true,
        json!({}),
    );
    let framed = policy.argument_schemas().expect("frames");
    argument_grammar(framed.get("tool").expect("framed"), &qwen().call_syntax).expect("grammar builds")
}

fn error_of(result: Result<(), ApiError>) -> (u16, String) {
    let error = result.expect_err("expected an error");
    (error.status, error.message)
}

fn types_of(types: Option<HashSet<String>>) -> Option<Vec<String>> {
    types.map(|types| {
        let mut names: Vec<String> = types.into_iter().collect();
        names.sort();
        names
    })
}

fn some_types(names: &[&str]) -> Option<Vec<String>> {
    Some(names.iter().map(|name| name.to_string()).collect())
}

fn some_texts(texts: &[&str]) -> Option<Vec<String>> {
    Some(texts.iter().map(|text| text.to_string()).collect())
}

// ---------------------------------------------------------------- names

#[test]
fn a_name_reads_back_as_the_grammar_writes_it() {
    for name in ["mcp.search", "web search", "天气", "n".repeat(256).as_str()] {
        assert!(is_tag_name(name), "{name}");
    }
    for name in ["", " text", "text ", "a<b", "a>b", "a\nb", "a\rb", "n".repeat(257).as_str()] {
        assert!(!is_tag_name(name), "{name}");
    }
}

// ------------------------------------------------------- normalize_tools

#[test]
fn a_grammar_applies_by_choice_and_strict_tools() {
    let lookup = json!({
        "type": "function", "function": {"name": "lookup", "parameters": {
            "type": "object", "properties": {"query": {"type": "string", "enum": ["alpha"]}},
            "required": ["query"], "additionalProperties": false}}});
    let finish = json!({"type": "function", "function": {"name": "finish", "parameters": {}}});
    let strict_finish = json!({
        "type": "function", "function": {"name": "finish", "parameters": {}, "strict": true}});
    let named = json!({"type": "function", "function": {"name": "finish"}});
    let cases: [(Value, Option<Value>, Option<Value>, Vec<&str>, bool); 5] = [
        (json!([lookup.clone(), finish.clone()]), None, Some(json!(true)), vec![], false),
        (json!([lookup.clone(), finish.clone()]), Some(json!("required")), Some(json!(true)), vec![], true),
        (json!([lookup.clone(), finish.clone()]), Some(named.clone()), Some(json!(true)), vec![], true),
        (json!([lookup.clone(), finish.clone()]), None, Some(json!(false)), vec![], true),
        (json!([lookup.clone(), strict_finish.clone()]), None, Some(json!(true)), vec!["finish"], true),
    ];
    for (tools, choice, parallel, strict, constrained) in cases {
        let policy = normalize_tools(Some(&tools), choice.as_ref(), parallel.as_ref(), None)
            .expect("policy builds")
            .1
            .expect("policy present");
        let mut strict_names: Vec<String> = policy.strict().iter().cloned().collect();
        strict_names.sort();
        assert_eq!(
            strict_names,
            strict.into_iter().map(str::to_string).collect::<Vec<_>>(),
            "strict tools of {tools}"
        );
        assert_eq!(policy.constrained, constrained, "constrained of {tools}");
    }
    let policy = normalize_tools(
        Some(&json!([lookup, finish])),
        Some(&named),
        Some(&json!(true)),
        None,
    )
    .expect("policy builds")
    .1
    .expect("policy present");
    let names: Vec<String> = policy.schemas().keys().cloned().collect();
    assert_eq!(names, vec!["finish"]);
    assert!(policy.required);
    assert!(!policy.parallel);
}

#[test]
fn none_keeps_the_tools_but_lets_no_call_start() {
    let tools = json!([
        {"type": "function", "function": {"name": "lookup"}},
        {"type": "function", "function": {"name": "finish"}},
    ]);
    let (offered, policy) =
        normalize_tools(Some(&tools), Some(&json!("none")), Some(&json!(true)), None).expect("builds");
    let names: Vec<String> = offered
        .expect("tools kept")
        .iter()
        .filter_map(|tool| tool.get("function").and_then(|f| f.get("name")).and_then(Json::as_str))
        .map(str::to_string)
        .collect();
    assert_eq!(names, vec!["lookup", "finish"]);
    let policy = policy.expect("policy present");
    assert!(policy.schemas().is_empty());
    assert!(!policy.required);
    assert!(policy.strict().is_empty());
    assert!(policy.constrained);
}

#[test]
fn normalize_tools_refuses_what_it_cannot_read() {
    let tools = json!([{"type": "function", "function": {"name": "f"}}]);
    for (tools, choice, parallel, expected) in [
        (None, Some(json!("required")), None, "tool_choice requires tools"),
        (None, Some(json!({"type": "function", "function": {"name": "f"}})), None, "tool_choice requires tools"),
        (Some(json!([])), Some(json!("required")), None, "tool_choice requires at least one tool"),
        (Some(json!([])), Some(json!({"type": "function"})), None, "tool_choice requires at least one tool"),
        (Some(json!(5)), None, None, "tools must be an array"),
        (Some(json!([{"type": "web_search"}])), None, None, "only function tools are supported"),
        (Some(json!([{"type": "function"}])), None, None, "a function tool needs a function object"),
        (Some(json!([{"type": "function", "function": {"name": "a<b"}}])), None, None, "tool name must be"),
        (Some(json!([{"type": "function", "function": {"name": "f"}}, {"type": "function", "function": {"name": "f"}}])), None, None, "duplicate tool name: f"),
        (Some(json!([{"type": "function", "function": {"name": "f", "parameters": 5}}])), None, None, "invalid tool schema for f"),
        (Some(json!([{"type": "function", "function": {"name": "f", "strict": "yes"}}])), None, None, "strict must be a boolean for tool f"),
        (Some(tools.clone()), None, Some(json!("yes")), "parallel_tool_calls must be a boolean"),
        (Some(tools.clone()), Some(json!("sometimes")), None, "invalid tool_choice"),
        (Some(tools.clone()), Some(json!({"type": "function", "function": {"name": "missing"}})), None, "invalid named tool_choice"),
        (Some(tools.clone()), Some(json!({"type": "custom", "function": {"name": "f"}})), None, "invalid named tool_choice"),
    ] {
        let (status, message) = error_of(
            normalize_tools(
                tools.as_ref().map(|tools| tools as &Json),
                choice.as_ref().map(|choice| choice as &Json),
                parallel.as_ref().map(|parallel| parallel as &Json),
                None,
            )
            .map(|_| ()),
        );
        assert_eq!(status, 400, "{message}");
        assert!(
            message.contains(expected),
            "expected {expected} in {message}"
        );
    }
}

#[test]
fn a_strict_tool_needs_a_schema_its_grammar_can_read() {
    fn deep_schema(depth: usize) -> Value {
        if depth == 0 {
            json!({"type": "string"})
        } else {
            json!({"type": "object", "properties": {"n": deep_schema(depth - 1)}})
        }
    }
    let deep = json!({"type": "object", "properties": {"a": deep_schema(70)}});
    // A strict tool's schema nested too deep is refused; a tool that is not
    // strict keeps the tools but declares no types.
    let (status, message) = error_of(check_tool_schema(
        "f",
        &json!({"type": "object", "properties": {"a": deep_schema(70)}}),
    ));
    assert_eq!(status, 400);
    assert!(message.contains("nested more than 64 levels deep"), "{message}");
    let tools = json!([
        {"type": "function", "function": {"name": "deep", "parameters": deep, "strict": true}},
    ]);
    let (status, message) = error_of(normalize_tools(Some(&tools), Some(&json!("auto")), Some(&json!(true)), None).map(|_| ()));
    assert_eq!(status, 400);
    assert!(message.contains("nested more than 64 levels deep"), "{message}");
    let tools = json!([
        {"type": "function", "function": {"name": "deep", "parameters": deep}},
    ]);
    let policy = normalize_tools(Some(&tools), Some(&json!("auto")), Some(&json!(true)), None)
        .expect("builds")
        .1
        .expect("policy present");
    assert!(policy.schemas().get("deep").expect("kept").is_null());
}

#[test]
fn remote_and_unresolved_references_are_request_errors() {
    let remote = json!({"type": "object", "properties": {"city": {"$ref": "https://example.com/city.json"}}});
    let (status, message) = error_of(check_tool_schema("w", &remote));
    assert_eq!(status, 400);
    assert_eq!(
        message,
        "remote tool schema reference is not allowed: https://example.com/city.json"
    );
    // A strict tool's references must resolve; a tool that is not strict
    // reads what it can.
    let unresolved = json!({"type": "object", "properties": {"a": {"$ref": "#/nope"}}});
    let tools = json!([
        {"type": "function", "function": {"name": "s", "parameters": unresolved, "strict": true}},
    ]);
    let (status, message) = error_of(normalize_tools(Some(&tools), Some(&json!("required")), Some(&json!(false)), None).map(|_| ()));
    assert_eq!(status, 400);
    assert_eq!(message, "unresolved tool parameter reference: #/nope");
    // A reference that is not a string is refused where it is read.
    let (status, message) = error_of(check_tool_schema(
        "t",
        &json!({"$schema": "http://json-schema.org/draft-04/schema#", "$ref": {"a": 1}}),
    ));
    assert_eq!((status, message.as_str()), (400, "$ref must be a string"));
    let (status, _) = error_of(check_tool_schema(
        "t",
        &json!({"$schema": "http://json-schema.org/draft-04/schema#", "properties": {"value": {"$ref": 5}}}),
    ));
    assert_eq!(status, 400);
}

// -------------------------------------------------------- strict_schema

#[test]
fn strict_closing_leaves_a_schema_that_others_extend() {
    let schema = json!({
        "type": "object",
        "properties": {
            "point": {"$ref": "#/$defs/Point"},
            "shape": {"anyOf": [{"type": "object", "properties": {"r": {}}}, {"type": "null"}]},
            "pair": {"type": "array", "items": [{"type": "integer"}]},
            "open": {"type": "object", "additionalProperties": true},
            "named": {"type": "object", "patternProperties": {"^x": {}}},
            "any": {},
        },
        "$defs": {"Point": {"type": "object", "properties": {"x": {}}}},
    });
    let strict = strict_schema(&schema);
    assert!(schema.get("additionalProperties").is_none());
    assert_eq!(strict.get("additionalProperties"), Some(&json!(false)));
    assert_eq!(
        strict
            .get("$defs")
            .and_then(|defs| defs.get("Point"))
            .and_then(|point| point.get("additionalProperties")),
        Some(&json!(false))
    );
    assert_eq!(
        strict.pointer("/properties/shape/anyOf/0/additionalProperties"),
        Some(&json!(false))
    );
    assert_eq!(
        strict.pointer("/properties/point"),
        Some(&json!({"$ref": "#/$defs/Point"}))
    );
    assert_eq!(
        strict.pointer("/properties/pair/additionalItems"),
        Some(&json!(false))
    );
    assert_eq!(
        strict.pointer("/properties/open/additionalProperties"),
        Some(&json!(true))
    );
    assert!(strict
        .pointer("/properties/named")
        .and_then(|named| named.as_object())
        .expect("named")
        .get("additionalProperties")
        .is_none());
    assert_eq!(strict.pointer("/properties/any"), Some(&json!({})));
    // An allOf extends its members: closing one would refuse the others'
    // properties, so the schema is left as declared.
    let extended = json!({
        "allOf": [{"type": "object", "properties": {"a": {}}}, {"properties": {"b": {}}}]
    });
    assert_eq!(strict_schema(&extended), extended);
}

// ---------------------------------------------------------- type reading

#[test]
fn schema_types_read_the_types_a_value_convert_by() {
    let read = |schema: Value| types_of(schema_types(&schema, &schema));
    assert_eq!(read(json!({"type": "integer"})), some_types(&["integer"]));
    assert_eq!(
        read(json!({"anyOf": [{"type": "string"}, {"type": "integer"}]})),
        some_types(&["integer", "string"])
    );
    // A number and an integer narrow to the integers they share.
    assert_eq!(
        read(json!({"allOf": [{"type": "number"}, {"type": "integer"}]})),
        some_types(&["integer"])
    );
    assert_eq!(
        read(json!({"enum": ["a", 1, null, true, 1.5, [1], {}]})),
        some_types(&[
            "array", "boolean", "integer", "null", "number", "object", "string",
        ])
    );
    // False admits no type; True constrains none.
    assert_eq!(read(json!(false)), Some(vec![]));
    assert_eq!(read(json!(true)), None);
    // A type name that stands for no JSON type constrains none, as an
    // unspelled alias does.
    assert_eq!(read(json!({"type": ["str", "int"]})), None);
    let spelled = json_type_names(&json!({"type": ["str", "int"]}));
    assert_eq!(types_of(schema_types(&spelled, &spelled)), some_types(&["integer", "string"]));
    // A reference narrows.
    let root = json!({
        "$defs": {"x": {"type": "number"}},
        "allOf": [{"$ref": "#/$defs/x"}],
        "type": "integer",
    });
    assert_eq!(types_of(schema_types(&root, &root)), some_types(&["integer"]));
}

#[test]
fn literal_texts_list_the_values_a_parameter_takes() {
    let read = |schema: Value| literal_texts(&schema, &schema);
    assert_eq!(
        read(json!({"enum": ["a", 2, null, [1], {"k": "v"}]})),
        some_texts(&["a", "2", "null", "[1]", "{\"k\":\"v\"}"])
    );
    assert_eq!(read(json!({"type": "boolean"})), some_texts(&["true", "false"]));
    assert_eq!(read(json!({"type": "null"})), some_texts(&["null"]));
    assert_eq!(
        read(json!({"anyOf": [{"enum": ["a"]}, {"enum": ["b", "a"]}]})),
        some_texts(&["a", "b"])
    );
    assert_eq!(
        read(json!({"enum": ["a", "b", "c"], "allOf": [{"enum": ["b", "c", "d"]}]})),
        some_texts(&["b", "c"])
    );
    let root = json!({"$defs": {"c": {"enum": ["x", "y"]}}, "$ref": "#/$defs/c"});
    assert_eq!(read(root), some_texts(&["x", "y"]));
    // A schema that does not list its values lists none.
    assert_eq!(read(json!({"type": "string"})), None);
}

// ------------------------------------------------------ argument framing

#[test]
fn argument_shape_projects_the_fields_a_tool_frames() {
    let mut budget = MAX_FRAMED_SCHEMA_BYTES;
    let shape = argument_shape(
        &json!({
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
            "required": ["a"],
        }),
        &mut budget,
    )
    .expect("projects");
    assert_eq!(
        shape,
        json!({
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
            "required": ["a"],
            "additionalProperties": true,
        })
    );
    // An if keeps every branch's fields, as assertions across fields go
    // unenforced.
    let mut budget = MAX_FRAMED_SCHEMA_BYTES;
    let shape = argument_shape(
        &json!({
            "type": "object",
            "properties": {"kind": {"type": "string"}},
            "if": {"properties": {"kind": {"const": "circle"}}},
            "then": {"properties": {"r": {"type": "number"}}, "required": ["r"]},
            "else": {"properties": {"side": {"type": "number"}}, "required": ["side"]},
        }),
        &mut budget,
    )
    .expect("projects");
    assert_eq!(
        shape.get("properties").and_then(|properties| properties.as_object()).map(|properties| {
            properties
                .iter()
                .map(|(name, value)| (name.clone(), value.is_boolean() || value == &json!({"type": "string"})))
                .collect::<Vec<_>>()
        }),
        Some(vec![
            ("kind".to_string(), true),
            ("r".to_string(), true),
            ("side".to_string(), true),
        ])
    );
    assert_eq!(shape.get("required"), Some(&json!([])));
    // An enum of objects frames each object's fields as constants.
    let mut budget = MAX_FRAMED_SCHEMA_BYTES;
    let shape = argument_shape(
        &json!({
            "type": "object",
            "properties": {"v": {"oneOf": [
                {"type": "object", "properties": {"t": {"const": "a"}, "x": {"type": "number"}},
                 "required": ["t", "x"], "additionalProperties": false},
                {"type": "object", "properties": {"t": {"const": "b"}},
                 "required": ["t"], "additionalProperties": false},
            ]}},
        }),
        &mut budget,
    )
    .expect("projects");
    assert_eq!(shape.pointer("/properties/v/oneOf/0/properties/t"), Some(&json!({"const": "a"})));
    assert_eq!(shape.pointer("/properties/v/oneOf/1/required"), Some(&json!(["t"])));
    // A schema that allows no object is refused.
    let mut budget = MAX_FRAMED_SCHEMA_BYTES;
    let (status, message) =
        error_of(argument_shape(&json!(false), &mut budget).map(|_| ()));
    assert_eq!((status, message.as_str()), (400, "tool parameters must allow a top-level JSON object"));
}

#[test]
fn framing_makes_each_field_a_schema_of_its_own() {
    let root = json!({
        "type": "object",
        "$defs": {"Name": {"type": "string", "enum": ["a"]}},
        "properties": {"n": {"$ref": "#/$defs/Name"}},
        "required": ["n"],
    });
    let mut budget = 10_000_000i64;
    let framed = tool_argument_schema(&root, &mut budget).expect("frames");
    assert_eq!(
        framed,
        json!({
            "properties": {
                "n": {
                    "$ref": "#/$defs/__splash_root/$defs/Name",
                    "$defs": {
                        "__splash_root": {
                            "type": "object",
                            "$defs": {"Name": {"type": "string", "enum": ["a"]}},
                            "properties": {"n": {"$ref": "#/$defs/__splash_root/$defs/Name"}},
                            "required": ["n"],
                        }
                    },
                }
            },
            "required": ["n"],
            "additionalProperties": true,
            "type": "object",
        })
    );
    assert_eq!(budget, 10_000_000 - 96);
    // A field that refers to nothing needs no copy of the root.
    let mut budget = 10_000_000i64;
    let framed = tool_argument_schema(
        &json!({"type": "object", "properties": {"n": {"type": "string"}}}),
        &mut budget,
    )
    .expect("frames");
    assert_eq!(
        framed.pointer("/properties/n"),
        Some(&json!({"type": "string"}))
    );
    assert!(budget > 10_000_000 - 20, "only the field was counted: {budget}");
}

#[test]
fn framing_refuses_what_the_budget_cannot_hold() {
    let root = json!({
        "type": "object",
        "properties": {"n": {"$ref": "#/$defs/Name"}},
        "$defs": {"Name": {"type": "string"}},
    });
    let mut budget = 10i64;
    let (status, message) = error_of(tool_argument_schema(&root, &mut budget).map(|_| ()));
    assert_eq!((status, message.as_str()), (400, "tool parameter schemas are too complex"));
}

// ------------------------------------------------------------- grammars

#[test]
fn the_grammars_are_the_python_implementations_own_output() {
    let syntax = qwen().call_syntax;
    assert_eq!(free_arguments(&syntax), fixture("free_qwen"));
    assert_eq!(free_arguments(&flint().call_syntax), fixture("free_glm"));

    let schema = json!({
        "type": "object",
        "properties": {"answer": {"const": 42}, "marker": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": false,
    });
    assert_eq!(json_grammar(&schema, false, &qwen()), fixture("json_think_false"));
    assert_eq!(json_grammar(&schema, true, &qwen()), fixture("json_think_true"));

    let mut schemas = Map::new();
    schemas.insert(
        "test".to_string(),
        json!({"type": "object", "properties": {"mode": {"enum": ["fast", 2, null]}}, "required": ["mode"]}),
    );
    let policy = ToolPolicy::new(
        schemas,
        true,
        false,
        HashSet::from(["test".to_string()]),
        true,
        json!({}),
    );
    assert_eq!(
        tool_grammar(&policy, false, None, &qwen()).expect("grammar builds"),
        fixture("tool_strict_enum_qwen")
    );

    let policy = tool_policy(json!({
        "lookup": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        "finish": {},
    }));
    assert_eq!(
        tool_grammar(&policy, false, Some(&schema), &qwen()).expect("grammar builds"),
        fixture("tool_auto_free_schema_qwen")
    );

    let mut schemas = Map::new();
    schemas.insert(
        "bash".to_string(),
        json!({"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}),
    );
    let policy = ToolPolicy::new(
        schemas,
        true,
        false,
        HashSet::from(["bash".to_string()]),
        true,
        json!({}),
    );
    assert_eq!(
        tool_grammar(&policy, true, None, &flint()).expect("grammar builds"),
        fixture("tool_strict_glm")
    );
}

#[test]
fn the_pinned_grammars_compile() {
    for name in [
        "free_qwen",
        "free_glm",
        "json_think_false",
        "json_think_true",
        "tool_strict_enum_qwen",
        "tool_auto_free_schema_qwen",
        "tool_strict_glm",
    ] {
        assert!(grammar_compiles(&fixture(name)), "{name} does not compile");
    }
}

#[test]
fn a_strict_tool_takes_only_the_arguments_it_declares() {
    let parameters = json!({
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "options": {"type": "object", "properties": {"depth": {"type": "integer"}}},
            "tags": {"type": "array"},
        },
        "required": ["path"],
    });
    let strict = argument_grammar_of(parameters.clone());
    // A strict tool's parameters come in schema order, each required one
    // present; its schema is closed, so no undeclared parameter is taken
    // and the grammar has no extra rule.
    assert!(strict.contains("parameter_0: \"<parameter=\" \"path\" \">\\n\" parameter_0_value"));
    assert!(strict.contains("parameter_1?"));
    assert!(strict.contains("parameter_2?"));
    assert!(!strict.contains("EXTRA_NAME"), "{strict}");
    // A tool that declares others takes them, their names but the declared
    // ones.
    let grammar = argument_grammar_of(json!({
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": {"type": "integer"},
    }));
    assert!(grammar.contains("EXTRA_NAME"), "{grammar}");
    assert!(grammar.contains("extra*"), "{grammar}");
    // A tool that is not strict takes any arguments.
    let mut schemas = Map::new();
    schemas.insert("search".to_string(), parameters);
    let policy = ToolPolicy::new(schemas, false, true, HashSet::new(), true, json!({}));
    let framed = policy.argument_schemas().expect("no strict tools to frame");
    assert!(framed.is_empty());
}

#[test]
fn strict_values_are_written_as_text_raw_or_json() {
    let grammar = argument_grammar_of(json!({
        "type": "object",
        "properties": {
            "mode": {"enum": ["fast", 2, null]},
            "time": {"type": ["string", "number"]},
            "note": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "count": {"type": "integer"},
        },
        "required": ["mode", "time", "note", "count"],
    }));
    for mode in ["\"fast\"", "\"2\"", "\"null\""] {
        assert!(grammar.contains(mode), "{mode} not in {grammar}");
    }
    assert_eq!(grammar.matches("/(?s:.*)/").count(), 2);
    assert!(grammar.contains("%json {\"type\":\"integer\""), "{grammar}");
}

#[test]
fn a_single_member_string_type_array_preserves_the_enum_grammar() {
    let scalar = json!({
        "type": "object",
        "properties": {"query": {"type": "string", "enum": ["alpha", "beta"]}},
        "required": ["query"],
    });
    let mut array = scalar.clone();
    array["properties"]["query"]["type"] = json!(["string"]);
    assert_eq!(argument_grammar_of(scalar), argument_grammar_of(array));
}

#[test]
fn the_grammar_leaves_out_what_it_cannot_compile() {
    let compatible = grammar_compatible_schema(&json!({
        "type": "object",
        "properties": {
            "p": {"type": "string", "pattern": "^ORD-[0-9]{4}$"},
            "q": {"type": "string", "pattern": "^(?!x)[a-z]+$"},
            "r": {"type": "object", "propertyNames": {"pattern": "^[a-z]+$"}},
        },
        "minItems": 1_000_000_000i64,
        "multipleOf": 1_000_000_000i64,
    }));
    assert_eq!(
        compatible,
        json!({
            "type": "object",
            "properties": {
                "p": {"type": "string", "pattern": "^ORD-[0-9]{4}$"},
                "q": {"type": "string"},
                "r": {"type": "object"},
            },
            "x-guidance": {
                "lenient": true,
                "whitespace_pattern": "[ \\t\\n\\r]{0,64}",
            },
        })
    );
    // Any value, with the whitespace between its tokens bounded; a false
    // schema stays false.
    assert_eq!(
        grammar_compatible_schema(&json!(true)),
        json!({"x-guidance": {"lenient": true, "whitespace_pattern": "[ \\t\\n\\r]{0,64}"}})
    );
    assert_eq!(grammar_compatible_schema(&json!(false)), json!(false));
}

#[test]
fn bounds_above_the_grammar_ceiling_are_left_out() {
    let limit = splash_server::tool_schema::MAX_GRAMMAR_BOUND;
    for keyword in ["minItems", "maxItems", "multipleOf"] {
        for bound in [limit, limit + 1] {
            let schema = json!({
                "type": "object",
                "properties": {"value": {"type": "array", keyword: bound}},
                "required": ["value"],
            });
            let grammar = argument_grammar_of(schema);
            assert_eq!(
                grammar.contains(&bound.to_string()),
                bound <= limit,
                "{keyword} {bound}: {grammar}"
            );
        }
    }
}

#[test]
fn an_enum_that_spells_the_framing_is_refused() {
    let mut schemas = Map::new();
    schemas.insert(
        "tool".to_string(),
        json!({"type": "object", "properties": {"v": {"enum": ["\n</parameter>\n"]}}, "required": ["v"]}),
    );
    let policy = ToolPolicy::new(
        schemas,
        true,
        false,
        HashSet::from(["tool".to_string()]),
        true,
        json!({}),
    );
    let framed = policy.argument_schemas().expect("frames");
    let (status, message) = error_of(
        argument_grammar(framed.get("tool").expect("framed"), &qwen().call_syntax).map(|_| ()),
    );
    assert_eq!(
        (status, message.as_str()),
        (400, "tool parameter enum contains XML framing")
    );
}

#[test]
fn a_parameter_that_cannot_take_a_value_is_refused_only_when_required() {
    let schema = json!({"type": "object", "properties": {"v": false}, "required": ["v"]});
    let mut schemas = Map::new();
    schemas.insert("tool".to_string(), schema.clone());
    let policy = ToolPolicy::new(
        schemas,
        true,
        false,
        HashSet::from(["tool".to_string()]),
        true,
        json!({}),
    );
    let framed = policy.argument_schemas().expect("frames");
    let (status, message) = error_of(
        argument_grammar(framed.get("tool").expect("framed"), &qwen().call_syntax).map(|_| ()),
    );
    assert_eq!(
        (status, message.as_str()),
        (400, "required tool parameter cannot have a value: v")
    );
    // Not required, the parameter is left out of the grammar.
    let mut schemas = Map::new();
    schemas.insert(
        "tool".to_string(),
        json!({"type": "object", "properties": {"v": false}}),
    );
    let policy = ToolPolicy::new(
        schemas,
        true,
        false,
        HashSet::from(["tool".to_string()]),
        true,
        json!({}),
    );
    let framed = policy.argument_schemas().expect("frames");
    let grammar = argument_grammar(framed.get("tool").expect("framed"), &qwen().call_syntax)
        .expect("grammar builds");
    assert!(grammar.contains("start:\n"), "{grammar}");
}

// ------------------------------------------------------- parameter types

#[test]
fn the_types_of_a_calls_values_convert_by_their_schema() {
    let policy = Arc::new(tool_policy(json!({
        "w": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        "n": {"type": "object", "properties": {"count": {"type": "integer"}}},
    })));
    let types = ToolParameterTypes::new(policy);
    let (parameters, other) = types.of("w");
    assert_eq!(
        parameters.get("city").map(|types| types_of(Some(types.clone()))),
        Some(Some(vec!["string".to_string()]))
    );
    assert_eq!(types_of(other), None);
    let (parameters, other) = types.of("n");
    assert_eq!(
        parameters.get("count").map(|types| types_of(Some(types.clone()))),
        Some(Some(vec!["integer".to_string()]))
    );
    assert_eq!(types_of(other), None);
    // A tool that was not offered declares no types.
    assert_eq!(types.of("nope"), (HashMap::new(), None));
}

#[test]
fn a_strict_tools_types_are_those_of_its_framed_arguments() {
    let tools = json!([
        {"type": "function", "function": {"name": "t", "parameters": {
            "type": "object",
            "properties": {"m": {"enum": ["fast", 2, null]}, "c": {"type": "integer"}},
            "required": ["m"],
        }, "strict": true}},
    ]);
    let policy = Arc::new(
        normalize_tools(Some(&tools), Some(&json!("required")), Some(&json!(false)), None)
            .expect("builds")
            .1
            .expect("policy present"),
    );
    let types = ToolParameterTypes::new(policy);
    let (parameters, other) = types.of("t");
    assert_eq!(
        parameters.get("m").map(|types| types_of(Some(types.clone()))),
        Some(Some(vec![
            "integer".to_string(),
            "null".to_string(),
            "string".to_string(),
        ]))
    );
    assert_eq!(
        parameters.get("c").map(|types| types_of(Some(types.clone()))),
        Some(Some(vec!["integer".to_string()]))
    );
    // A strict tool takes no other parameter: its schema is closed.
    assert_eq!(types_of(other), Some(vec![]));
}

#[test]
fn a_tool_whose_arguments_cannot_be_projected_declares_no_types() {
    let policy = Arc::new(tool_policy(json!({
        "c": {"type": "object", "properties": {"a": {"$ref": "#/properties/a"}}},
        "u": {"type": "object", "properties": {"a": {"$ref": "#/nope"}}},
    })));
    let types = ToolParameterTypes::new(policy);
    assert_eq!(types.of("c"), (HashMap::new(), None));
    assert_eq!(types.of("u"), (HashMap::new(), None));
}

// ------------------------------------------------------------- validation

#[test]
fn schemas_that_break_their_dialect_are_refused_and_others_are_not() {
    assert!(check_schema(&json!({"type": 7})).is_err());
    // An older declared dialect leaves later keywords unchecked.
    assert!(check_schema(&json!({
        "$schema": "http://json-schema.org/draft-03/schema#", "allOf": 5
    }))
    .is_ok());
    assert!(check_schema(&json!({
        "$schema": "http://json-schema.org/draft-07/schema#", "dependentSchemas": 5
    }))
    .is_ok());
    // Draft 3's required is a boolean of a property.
    assert!(check_schema(&json!({
        "$schema": "http://json-schema.org/draft-03/schema#", "required": true
    }))
    .is_ok());
    assert!(check_schema(&json!({
        "$schema": "http://json-schema.org/draft-03/schema#", "type": "x"
    }))
    .is_ok());
    assert!(check_schema(&json!({
        "$schema": "http://json-schema.org/draft-03/schema#", "divisibleBy": "x"
    }))
    .is_err());
}

#[test]
fn validators_refuse_what_they_cannot_evaluate() {
    let build = |schema: &Value| match build_validator(schema) {
        Ok(validator) => Ok(validator),
        Err(BuildError::Api(error)) => Err((error.status, error.message)),
        Err(BuildError::Schema(error)) => Err((400, error.0)),
    };
    // Draft 3 accepts any type name; the later drafts do not.
    assert!(build(&json!({
        "$schema": "http://json-schema.org/draft-03/schema#",
        "properties": {"value": {"type": ["any", {"type": "string"}]}}
    }))
    .is_ok());
    for (schema, expected) in [
        (
            json!({"$schema": "http://json-schema.org/draft-03/schema#", "type": "x"}),
            "unknown schema type: x",
        ),
        (
            json!({"$schema": "http://json-schema.org/draft-03/schema#", "disallow": "x"}),
            "unknown schema type: x",
        ),
        (
            json!({"type": "object", "patternProperties": {".*": {}}, "unevaluatedProperties": false}),
            "unevaluatedProperties with patternProperties is not supported",
        ),
        (
            json!({
                "$schema": "http://json-schema.org/draft-07/schema#",
                "properties": {"a": {"$schema": "https://json-schema.org/draft/2020-12/schema"}}
            }),
            "mixed schema dialects are not supported",
        ),
    ] {
        let (status, message) = build(&schema).expect_err("expected a refusal");
        assert_eq!(status, 400, "{message}");
        assert_eq!(message, expected);
    }
}

#[test]
fn a_draft3_schema_validates_as_its_keywords_read() {
    let validator = build_validator(&json!({
        "$schema": "http://json-schema.org/draft-03/schema#",
        "definitions": {"text": {"type": "string"}},
        "properties": {"value": {"extends": {"$ref": "#/definitions/text"}}},
    }))
    .expect("builds");
    assert!(validator.is_valid(&json!({"value": "x"})).expect("evaluates"));
    assert!(!validator.is_valid(&json!({"value": 1})).expect("evaluates"));
    // A property marked required is one the object must carry.
    let validator = build_validator(&json!({
        "$schema": "http://json-schema.org/draft-03/schema#",
        "properties": {"value": {"type": "string", "required": true}},
    }))
    .expect("builds");
    assert!(!validator.is_valid(&json!({})).expect("evaluates"));
    assert!(validator.is_valid(&json!({"value": "x"})).expect("evaluates"));
    // disallow refuses what any of its members would take.
    let validator = build_validator(&json!({
        "$schema": "http://json-schema.org/draft-03/schema#",
        "disallow": "string",
    }))
    .expect("builds");
    assert!(validator.is_valid(&json!(1)).expect("evaluates"));
    assert!(!validator.is_valid(&json!("s")).expect("evaluates"));
}

#[test]
fn an_answer_is_checked_against_its_schema() {
    let schema = json!({
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
        "additionalProperties": false,
    });
    let format = normalize_response_format(Some(&json!({
        "type": "json_schema", "json_schema": {"schema": schema}
    })))
    .expect("builds")
    .expect("format present");
    assert!(validate_response_content("{\"x\":1}", Some(&format.validator)).is_ok());
    let error = validate_response_content("{\"x\":\"s\"}", Some(&format.validator))
        .expect_err("invalid output");
    assert_eq!((error.status, error.code.as_str()), (500, "invalid_model_output"));
    assert!(error.message.starts_with("model returned invalid structured output"));
    let error = validate_response_content("not json", Some(&format.validator))
        .expect_err("not json");
    assert_eq!((error.status, error.code.as_str()), (500, "invalid_model_output"));
    // A request without a format checks nothing.
    assert!(validate_response_content("anything", None).is_ok());
}

#[test]
fn a_pattern_the_answer_breaks_is_the_models_error() {
    let schema = json!({"type": "string", "pattern": "^Paris$"});
    let format = normalize_response_format(Some(&json!({
        "type": "json_schema", "json_schema": {"schema": schema}
    })))
    .expect("builds")
    .expect("format present");
    assert!(validate_response_content("\"Paris\"", Some(&format.validator)).is_ok());
    let error = validate_response_content("\"London\"", Some(&format.validator))
        .expect_err("invalid output");
    assert_eq!((error.status, error.code.as_str()), (500, "invalid_model_output"));
}

#[test]
fn a_response_format_is_read_as_the_apis_declare_it() {
    assert!(normalize_response_format(None).expect("builds").is_none());
    assert!(normalize_response_format(Some(&json!(null))).expect("builds").is_none());
    assert!(normalize_response_format(Some(&json!({"type": "text"})))
        .expect("builds")
        .is_none());
    let format = normalize_response_format(Some(&json!({"type": "json_object"})))
        .expect("builds")
        .expect("format present");
    assert_eq!(format.schema, json!({"type": "object"}));
    let format = normalize_response_format(Some(&json!({
        "type": "json_schema", "json_schema": {"schema": true}
    })))
    .expect("builds")
    .expect("format present");
    assert_eq!(format.schema, json!(true));
    for (value, expected) in [
        (json!(5), "response_format must be an object"),
        (json!({"type": "nope"}), "unsupported response_format"),
        (
            json!({"type": "json_schema"}),
            "response_format.json_schema.schema is required",
        ),
        (
            json!({"type": "json_schema", "json_schema": {"schema": 5}}),
            "response_format.json_schema.schema is required",
        ),
    ] {
        let (status, message) = error_of(normalize_response_format(Some(&value)).map(|_| ()));
        assert_eq!((status, message.as_str()), (400, expected));
    }
    // A schema that nests too deep, or refers to another document, is
    // refused before anything reads it.
    let deep = json!({"type": "json_schema", "json_schema": {"schema": deep_schema(70)}});
    fn deep_schema(depth: usize) -> Value {
        if depth == 0 {
            json!({"type": "string"})
        } else {
            json!({"type": "object", "properties": {"n": deep_schema(depth - 1)}})
        }
    }
    let (status, message) = error_of(normalize_response_format(Some(&deep)).map(|_| ()));
    assert_eq!(status, 400);
    assert!(message.contains("invalid response schema"), "{message}");
    let remote = json!({
        "type": "json_schema",
        "json_schema": {"schema": {"type": "object", "properties": {"c": {"$ref": "https://example.com/city.json"}}}},
    });
    let (status, message) = error_of(normalize_response_format(Some(&remote)).map(|_| ()));
    assert_eq!(status, 400);
    assert_eq!(
        message,
        "remote schema reference is not allowed: https://example.com/city.json"
    );
}

#[test]
fn a_validator_the_crate_cannot_build_waits_for_the_output() {
    // A pattern the pattern engine refuses fails when the answer is
    // checked, as the Python's fails when it evaluates the pattern.
    let validator = build_validator(&json!({"type": "string", "pattern": "("}))
        .expect("the validator is built");
    let failure = validator
        .validate(&json!("x"))
        .expect_err("the pattern cannot be evaluated");
    assert!(
        matches!(failure, ValidationFailure::Evaluation(_)),
        "{failure:?}"
    );
    let error = validate_response_content("\"x\"", Some(&validator)).expect_err("fails");
    assert_eq!((error.status, error.code.as_str()), (500, "output_validation_failed"));
}

// ------------------------------------------------------------- references

#[test]
fn a_local_reference_names_its_target() {
    let root = json!({
        "type": "object",
        "$defs": {"Name": {"type": "string"}},
        "properties": {"n": {"$ref": "#/$defs/Name"}},
    });
    let target = lookup_tool_reference(&json!("#/$defs/Name"), &root).expect("resolves");
    assert_eq!(target, &json!({"type": "string"}));
    assert_eq!(
        lookup_tool_reference(&json!("#"), &root).expect("resolves"),
        &root
    );
    // An anchor names its node.
    let anchored = json!({"$anchor": "x", "type": "string"});
    let root = json!({"type": "object", "properties": {"a": anchored.clone()}});
    assert_eq!(
        lookup_tool_reference(&json!("#x"), &root).expect("resolves"),
        &anchored
    );
    for reference in ["https://example.com/x.json", "#/nope"] {
        let error = lookup_tool_reference(&json!(reference), &root)
            .expect_err("unresolved");
        assert_eq!(error.status, 400);
    }
    let error = lookup_tool_reference(&json!(5), &root).expect_err("not a string");
    assert_eq!((error.status, error.message.as_str()), (400, "unsupported tool parameter reference"));
}

#[test]
fn a_schema_nests_at_most_64_levels_deep() {
    fn deep(depth: usize) -> Value {
        if depth == 1 {
            json!({"type": "string"})
        } else {
            json!({"properties": deep(depth - 1)})
        }
    }
    assert!(check_depth(&deep(64), "schema").is_ok());
    let (status, message) = error_of(check_depth(&deep(65), "schema"));
    assert_eq!(status, 400);
    assert_eq!(message, "schema: nested more than 64 levels deep");
}

// --------------------------------------------------- the constraint factory

#[test]
fn a_grammar_list_compiles_as_one_grammar() {
    // The tool grammar is a grammar list; the factory compiles it whole,
    // side grammars and all, and starts at its first grammar.
    let grammar = Grammar::List(fixture("tool_strict_enum_qwen"));
    let text = match &grammar {
        Grammar::List(text) => text.clone(),
        _ => unreachable!("checked above"),
    };
    assert!(grammar_compiles(&text));
    // The list's size is its text's, as the other sources' is.
    assert_eq!(text.len(), fixture("tool_strict_enum_qwen").len());
}

// -------------------------------------------------------------- namespaces

#[test]
fn namespaces_are_kept_as_declared() {
    let tools = json!([{"type": "function", "function": {"name": "f"}}]);
    let policy = normalize_tools(
        Some(&tools),
        Some(&json!("auto")),
        Some(&json!(true)),
        Some(&json!({"mcp": {"server": "x"}})),
    )
    .expect("builds")
    .1
    .expect("policy present");
    assert_eq!(
        policy.namespaces,
        json!({"mcp": {"server": "x"}})
    );
    let policy = normalize_tools(Some(&tools), Some(&json!("auto")), Some(&json!(true)), None)
        .expect("builds")
        .1
        .expect("policy present");
    assert_eq!(policy.namespaces, json!({}));
}


#[test]
fn more_grammars_match_the_python_implementation() {
    // A reference with an array constraint, a compilable pattern, and a
    // bound too large to compile.
    let a = argument_grammar_of(json!({
        "type": "object",
        "$defs": {"Names": {"type": "array", "items": {"type": "string"}}},
        "properties": {
            "value": {"$ref": "#/$defs/Names", "maxItems": 1},
            "id": {"type": "string", "pattern": "^ORD-[0-9]{4}$"},
            "big": {"type": "array", "items": {"type": "string"}, "minItems": 1000000000i64},
        },
        "required": ["value", "id"],
    }));
    assert_eq!(a, fixture("parity_a"));
    assert!(grammar_compiles(&a), "{a}");
    // A keyword of a later draft, which Draft 3 leaves unchecked,
    // constrains nothing: the value is raw text.
    let tools = json!([
        {"type": "function", "function": {"name": "t", "parameters": {
            "$schema": "http://json-schema.org/draft-03/schema#", "type": "object",
            "properties": {"value": {"anyOf": 5}}}, "strict": true}},
    ]);
    let policy = normalize_tools(Some(&tools), Some(&json!("required")), Some(&json!(false)), None)
        .expect("builds")
        .1
        .expect("policy present");
    let b = tool_grammar(&policy, false, None, &qwen()).expect("grammar builds");
    assert_eq!(b, fixture("parity_b"));
    assert!(grammar_compiles(&b), "{b}");
    // A union whose string member lists its values takes those, the other
    // members' JSON and nothing else; a refinement of a type keeps it.
    let c = argument_grammar_of(json!({
        "type": "object", "$defs": {"color": {"enum": ["red", "green"]}},
        "properties": {
            "unit": {"anyOf": [{"type": "string", "enum": ["celsius", "kelvin"]}, {"type": "null"}]},
            "limit": {"anyOf": [{"enum": ["auto"]}, {"type": "integer", "minimum": 1}]},
            "count": {"allOf": [{"type": "integer"}, {"minimum": 0}]},
            "color": {"allOf": [{"$ref": "#/$defs/color"}]},
            "time": {"type": ["string", "number"]},
        }
    }));
    assert_eq!(c, fixture("parity_c"));
    assert!(grammar_compiles(&c), "{c}");
    // A look-around pattern is left out of the grammar; a compilable one
    // stays.
    let d = json_grammar(
        &json!({
            "type": "object",
            "properties": {"order": {"type": "string", "pattern": "^ORD-[0-9]{4}$"},
                           "code": {"type": "string", "pattern": "[A-Z]{3}"},
                           "path": {"type": "string", "pattern": "^(?!\\.\\.)[a-z/.]+$"}},
            "required": ["order", "code", "path"],
            "additionalProperties": false,
        }),
        false,
        &qwen(),
    );
    assert_eq!(d, fixture("parity_d"));
    assert!(grammar_compiles(&d), "{d}");
    assert!(!d.contains("?!"), "{d}");
}

#[test]
fn debug_named_choice_qwen_grammar() {
    let policy = tool_policy(json!({
        "get_weather": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }));
    let policy = splash_server::tool_schema::ToolPolicy::new(
        policy.schemas().clone(),
        true,
        false,
        std::collections::HashSet::new(),
        true,
        serde_json::json!({}),
    );
    let grammar = splash_server::tool_schema::tool_grammar(
        &policy, false, None, &splash_server::chat_format::qwen(),
    )
    .expect("grammar builds");
    std::fs::write("/tmp/named_choice_qwen.json", &grammar).expect("writes");
    let top = llguidance::api::TopLevelGrammar::from_lark_or_grammar_list(&grammar)
        .expect("parses");
    let error = llguidance::api::GrammarInit::Serialized(top)
        .validate(None, llguidance::api::ParserLimits::default())
        .into_error();
    println!("validate error: {error:?}");
}

#[test]
fn debug_named_choice_full_compile() {
    let policy = tool_policy(json!({
        "get_weather": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }));
    let policy = splash_server::tool_schema::ToolPolicy::new(
        policy.schemas().clone(),
        true,
        false,
        std::collections::HashSet::new(),
        true,
        serde_json::json!({}),
    );
    let grammar = splash_server::tool_schema::tool_grammar(
        &policy, false, None, &splash_server::chat_format::qwen(),
    )
    .expect("grammar builds");
    let base = std::env::var("HOME").expect("HOME");
    let snapshots = std::path::PathBuf::from(base)
        .join(".cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots");
    let Some(directory) = std::fs::read_dir(&snapshots)
        .ok()
        .and_then(|mut entries| entries.next())
        .and_then(|entry| entry.ok())
        .map(|entry| entry.path())
    else {
        eprintln!("no local Qwen tokenizer; skipped");
        return;
    };
    let format = splash_server::chat_format::qwen();
    let env = splash_server::constraints::ConstraintTokenizer::from_directory(
        &directory, format.vocabulary, &format.stop_ids(),
    )
    .expect("env builds");
    let factory = splash_server::constraints::ConstraintFactory::new(&env, &format)
        .expect("factory builds");
    match factory.create(&splash_server::constraints::Grammar::List(grammar)) {
        Ok(_constraint) => println!("compiled fine"),
        Err(error) => {
            for line in error.0.lines().take(6) {
                println!("ERR: {line}");
            }
        }
    }
}
