//! The output grammars, ported from `server/tool_schema.py`'s grammar
//! builders: what a constrained request's reasoning, tool calls and JSON
//! answer are generated as, in the chat template's layout, and the schema
//! constraints the grammar compiler takes.
//!
//! Grammar text is built to the byte the Python's builds it: JSON is
//! written as `json.dumps` writes it, with ASCII escapes, so a grammar the
//! Python server compiles and one this port compiles are the same string.

use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

use llguidance::api::{GrammarInit, ParserLimits, TopLevelGrammar};
use serde_json::{Map, Value};

use crate::chat_format::{ChatFormat, ToolCallSyntax, TOOL_CALL_CLOSE, TOOL_CALL_OPEN};
use crate::support::errors::ApiError;
use crate::support::lru::LruCache;

use super::policy::ToolPolicy;
use super::schema::{
    for_each_object_mut, for_each_subschema_mut, is_tag_name, literal_texts, schema_types,
    schema_with_root, SCHEMA_ANNOTATIONS, SCHEMA_IDENTIFIERS,
};

/// The grammar compiler expands these keywords into work proportional to their
/// values: a rule per required or optional array item, a state per divisor
/// residue. A tiny schema with a huge bound would exhaust memory, so larger
/// bounds are left out of the grammar; a response format checks them on the
/// complete output.
pub const GRAMMAR_BOUND_KEYWORDS: [&str; 3] = ["minItems", "maxItems", "multipleOf"];
/// The largest bound the grammar compiler takes.
pub const MAX_GRAMMAR_BOUND: i64 = 64;

/// Checking a pattern compiles a grammar, and a client sends the same tools and
/// output schema on every turn, so the answers for this many patterns are kept,
/// within this many bytes of them, as the server's other caches of request text
/// are.
pub const PATTERN_CHECK_CACHE_SIZE: usize = 1024;
pub const PATTERN_CHECK_CACHE_BYTES: usize = 8 * 1024 * 1024;

/// The whitespace a model chooses between the tokens of constrained output, in
/// JSON and around tool calls, is bounded: a model that prefers whitespace to
/// every token the grammar allows next would otherwise write it until
/// max_tokens, as Qwen models do, most of all under speculative decoding.
/// llama.cpp and Outlines bound it more tightly; 64 characters still take
/// pretty printing 15 levels deep at four spaces. Whitespace inside strings is
/// content and unbounded.
pub const MAX_WHITESPACE: i64 = 64;

/// The whitespace pattern the grammars bound the model's spacing with.
pub fn whitespace() -> String {
    format!("[ \\t\\n\\r]{{0,{MAX_WHITESPACE}}}")
}

/// The [`whitespace`] pattern as the grammars declare it.
pub fn whitespace_rule() -> String {
    format!("WS: /{}/", whitespace())
}

/// The text rule of the text outside calls: any text but a spelling of a
/// call's opening or the reasoning's close.
fn text_rule() -> String {
    format!(r"TEXT: /(?s:.*)/ & ~/(?s:.*)({TOOL_CALL_OPEN}|<\/think>)(?s:.*)/")
}

/// The text rule of the reasoning a generation prompt opened: any text but a
/// spelling of its close.
fn think_text_rule() -> String {
    r"TEXT: /(?s:.*)/ & ~/(?s:.*)<\/think>(?s:.*)/".to_string()
}

/// The value `value` spells as Python's `json.dumps` writes it, with the
/// separators `json.dumps` defaults to.
pub fn py_json_dumps(value: &Value) -> String {
    let mut out = String::new();
    write_py_json(&mut out, value, false);
    out
}

/// The value `value` spells as `json.dumps(value, separators=(",", ":"))`
/// writes it.
pub fn py_json_dumps_compact(value: &Value) -> String {
    let mut out = String::new();
    write_py_json(&mut out, value, true);
    out
}

fn write_py_json(out: &mut String, value: &Value, compact: bool) {
    let separator = if compact { "," } else { ", " };
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(number) => out.push_str(&number.to_string()),
        Value::String(text) => write_py_string(out, text),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(separator);
                }
                write_py_json(out, item, compact);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (index, (key, item)) in map.iter().enumerate() {
                if index > 0 {
                    out.push_str(separator);
                }
                write_py_string(out, key);
                out.push_str(if compact { ":" } else { ": " });
                write_py_json(out, item, compact);
            }
            out.push('}');
        }
    }
}

/// A string as `json.dumps` writes it with `ensure_ascii`: the controls and
/// the quotes escaped, and everything outside printable ASCII as `\uXXXX`,
/// a surrogate pair for a character beyond the basic plane.
fn write_py_string(out: &mut String, text: &str) {
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\u{0c}' => out.push_str("\\f"),
            '\r' => out.push_str("\\r"),
            _ if (character as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", character as u32));
            }
            _ if (character as u32) > 0x7e => {
                let code = character as u32;
                if code > 0xFFFF {
                    let high = 0xD800 + ((code - 0x10000) >> 10);
                    let low = 0xDC00 + ((code - 0x10000) & 0x3FF);
                    out.push_str(&format!("\\u{high:04x}\\u{low:04x}"));
                } else {
                    out.push_str(&format!("\\u{:04x}", code));
                }
            }
            character => out.push(character),
        }
    }
    out.push('"');
}

/// The pattern as a lark regex literal, its slashes escaped.
fn lark_regex(pattern: &str) -> String {
    format!("/{}/", pattern.replace('/', "\\/"))
}

/// The pattern as Python's `re.escape` writes it: the regex's special
/// characters backslashed, everything else as written.
fn re_escape(pattern: &str) -> String {
    let special = |character: char| {
        matches!(
            character,
            '(' | ')'
                | '['
                | ']'
                | '{'
                | '}'
                | '?'
                | '*'
                | '+'
                | '-'
                | '|'
                | '^'
                | '$'
                | '\\'
                | '.'
                | '&'
                | '~'
                | '#'
                | ' '
                | '\t'
                | '\n'
                | '\r'
                | '\u{0b}'
                | '\u{0c}'
        )
    };
    let mut out = String::with_capacity(pattern.len());
    for character in pattern.chars() {
        if special(character) {
            out.push('\\');
        }
        out.push(character);
    }
    out
}

/// A function's or parameter's name is at most this long, and reading strips
/// this space around it, so a name the grammars write neither starts nor ends
/// with it.
fn name_rule() -> String {
    lark_regex(&format!(
        r"[^<> \t\n\r]([^<>\n\r]{{0,{}}}[^<> \t\n\r])?",
        super::schema::MAX_NAME_LENGTH - 2
    ))
}

/// Any text, as a lark regex literal.
const RAW_VALUE: &str = "/(?s:.*)/";

/// Tags of `syntax` in a row, as a grammar writes them: each as its token
/// where the tags are tokens of the family's tokenizer (Flint's), which the
/// grammar compiler takes apart from text, so text cannot spell them; else
/// their text as one string.
fn tags(syntax: &ToolCallSyntax, tags: &[&str]) -> String {
    if syntax.tags_are_tokens {
        tags.iter()
            .filter(|tag| !tag.is_empty())
            .map(|tag| tag.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    } else {
        py_json_dumps(&Value::String(tags.concat()))
    }
}

/// What follows TOOL_CALL_OPEN in a grammar's call of tool `name`, up to
/// its arguments: the tags around the name, and the name as text.
fn function_opening(syntax: &ToolCallSyntax, name: &str) -> String {
    if syntax.tags_are_tokens {
        [
            syntax.name_open,
            py_json_dumps(&Value::String(name.to_string())).as_str(),
            syntax.name_close,
        ]
        .into_iter()
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
    } else {
        py_json_dumps(&Value::String(syntax.function_opening(name)))
    }
}

/// The tags around a parameter's name in a grammar, up to its value. The
/// name is a lexeme of its own: one spanning "<parameter=url>" would win
/// over another name that starts like it, and lexing cannot back off.
fn key_tags(syntax: &ToolCallSyntax) -> (String, String) {
    (
        tags(syntax, &[syntax.key_open]),
        tags(syntax, &[syntax.key_close, syntax.value_open]),
    )
}

/// The rules of a raw value, rule `rule`, with its close: where the tags
/// are text, any text up to the close's first spelling, its suffix; where
/// they are tokens, any text but a spelling of the close, then its token
/// ([`value_terminals`]).
fn raw_value(syntax: &ToolCallSyntax, rule: &str) -> Vec<String> {
    if syntax.tags_are_tokens {
        vec![format!("{rule}: RAW_VALUE {}", syntax.value_close)]
    } else {
        vec![format!(
            "{rule}[suffix={}]: {RAW_VALUE}",
            py_json_dumps(&Value::String(syntax.value_close.to_string()))
        )]
    }
}

/// The terminal the raw values of `syntax` read where its tags are tokens.
fn value_terminals(syntax: &ToolCallSyntax) -> Vec<String> {
    if !syntax.tags_are_tokens {
        return Vec::new();
    }
    let spelled = lark_regex(&format!("(?s:.*){}(?s:.*)", re_escape(syntax.value_close)));
    vec![format!("RAW_VALUE: {RAW_VALUE} & ~{spelled}")]
}

/// The arguments of a tool that is not strict: parameters in the chat
/// template's tags, each with any name and raw text.
pub fn free_arguments(syntax: &ToolCallSyntax) -> String {
    let (key_open, key_close) = key_tags(syntax);
    let mut rules = vec![
        "%llguidance {}".to_string(),
        "start: parameter*".to_string(),
        format!("parameter: {key_open} NAME {key_close} value"),
    ];
    rules.extend(raw_value(syntax, "value"));
    rules.push(format!("NAME: {}", name_rule()));
    rules.extend(value_terminals(syntax));
    rules.join("\n") + "\n"
}

/// The members of a schema that is only a union of them; otherwise the
/// schema itself.
fn union_members(schema: &Value) -> Vec<Value> {
    if let Some(map) = schema.as_object() {
        let members = map.get("anyOf").or_else(|| map.get("oneOf"));
        if let Some(Value::Array(items)) = members {
            if !items.is_empty() {
                let others: HashSet<&str> = map
                    .keys()
                    .map(String::as_str)
                    .filter(|key| {
                        !SCHEMA_ANNOTATIONS.contains(key) && !SCHEMA_IDENTIFIERS.contains(key)
                    })
                    .collect();
                if others.iter().all(|key| *key == "anyOf" || *key == "oneOf") {
                    return items.clone();
                }
            }
        }
    }
    vec![schema.clone()]
}

/// The rules of a parameter of a strict tool, its value written as the
/// template writes values, a string as its text and any other value as
/// JSON, then the syntax's value close: raw text when it may be any string,
/// else one of the values each member of its union lists, or JSON of a
/// member that lists none. `None` when no value is possible. Framing makes
/// each parameter schema self-contained, so it is its own reference
/// root.
fn parameter_rules(
    rule: &str,
    prefix: &str,
    value_schema: &Value,
    syntax: &ToolCallSyntax,
) -> Result<Option<Vec<String>>, ApiError> {
    let closing = tags(syntax, &[syntax.value_close]);
    if value_schema == &Value::Bool(false) {
        return Ok(None);
    }
    let mut rules: Vec<String> = Vec::new();
    let mut choices: Vec<String> = Vec::new();
    for member in union_members(value_schema) {
        let texts = literal_texts(&member, value_schema);
        if texts.is_none() {
            let types = schema_types(&member, value_schema);
            if types.is_none() || types.expect("checked above").contains("string") {
                // Any text may be the value; the listed ones are among it.
                let mut taken = vec![format!("{rule}: {prefix} {rule}_value")];
                taken.extend(raw_value(syntax, &format!("{rule}_value")));
                return Ok(Some(taken));
            }
            let mut member = member;
            if member != *value_schema {
                // A member refers to the parameter schema's definitions.
                member = schema_with_root(&member, value_schema).unwrap_or(member);
            }
            let member = grammar_compatible_schema(&member);
            choices.push(format!("%json {}", py_json_dumps_compact(&member)));
            continue;
        }
        for text in texts.expect("checked above") {
            if text.contains(syntax.value_close) {
                return Err(ApiError::new(
                    400,
                    "tool parameter enum contains XML framing",
                    "invalid_request_error",
                ));
            }
            if !text.is_empty() {
                choices.push(py_json_dumps(&Value::String(text)));
            } else {
                rules.push(format!("{rule}_empty_{}:", rules.len()));
                choices.push(format!("{rule}_empty_{}", rules.len() - 1));
            }
        }
    }
    if choices.is_empty() {
        return Ok(None);
    }
    let mut unique: Vec<String> = Vec::new();
    for choice in choices {
        if !unique.contains(&choice) {
            unique.push(choice);
        }
    }
    rules.push(format!("{rule}: {prefix} ({}) {closing}", unique.join(" | ")));
    Ok(Some(rules))
}

/// The arguments of a strict tool: the declared parameters in schema
/// order, each required one present, then any others the schema allows.
pub fn argument_grammar(schema: &Value, syntax: &ToolCallSyntax) -> Result<String, ApiError> {
    let (key_open, key_close) = key_tags(syntax);
    let properties = schema
        .get("properties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let required: HashSet<String> = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let mut rules: Vec<String> = Vec::new();
    let mut sequence: Vec<String> = Vec::new();
    for (index, (name, value_schema)) in properties.iter().enumerate() {
        let rule = format!("parameter_{index}");
        let prefix = format!(
            "{key_open} {} {key_close}",
            py_json_dumps(&Value::String(name.clone()))
        );
        if value_schema != &Value::Bool(false) && !is_tag_name(name) {
            return Err(ApiError::new(
                400,
                "invalid tool parameter name",
                "invalid_request_error",
            ));
        }
        let parameter_rules = parameter_rules(&rule, &prefix, value_schema, syntax)?;
        let Some(parameter_rules) = parameter_rules else {
            if required.contains(name) {
                return Err(ApiError::new(
                    400,
                    format!("required tool parameter cannot have a value: {name}"),
                    "invalid_request_error",
                ));
            }
            continue;
        };
        rules.extend(parameter_rules);
        sequence.push(format!(
            "{rule}{}",
            if required.contains(name) { "" } else { "?" }
        ));
    }
    let extra_rules = parameter_rules(
        "extra",
        &format!("{key_open} EXTRA_NAME {key_close}"),
        &schema
            .get("additionalProperties")
            .cloned()
            .unwrap_or(Value::Bool(true)),
        syntax,
    )?;
    if let Some(extra_rules) = extra_rules {
        // Other names are those the tool does not declare.
        let declared = properties
            .keys()
            .map(|name| py_json_dumps(&Value::String(name.clone())))
            .collect::<Vec<_>>()
            .join(" | ");
        rules.push(format!(
            "EXTRA_NAME: {}{}",
            name_rule(),
            if declared.is_empty() {
                String::new()
            } else {
                format!(" & ~({declared})")
            }
        ));
        rules.extend(extra_rules);
        sequence.push("extra*".to_string());
    }
    rules.extend(value_terminals(syntax));
    Ok(format!(
        "%llguidance {{}}\nstart:{}\n{}\n",
        sequence.iter().map(|item| format!(" {item}")).collect::<String>(),
        rules.join("\n")
    ))
}

/// The reasoning a grammar lets come before its answer, as the first item
/// of its start and the rule that item names: where the generation prompt
/// opened a think block (`thinking`), its text up to the block's close; where
/// the model opens one itself, an optional block from its opening to its
/// close; otherwise none. Each tag is its token, as the template writes it,
/// and the text between is TEXT, which cannot spell the close.
fn reasoning(thinking: bool, chat_format: &ChatFormat) -> (String, Option<String>) {
    let close = format!("<[{}]>", chat_format.think_end_id);
    if thinking {
        return ("think ".to_string(), Some(format!("think: TEXT {close}")));
    }
    if let Some(start) = chat_format.think_start_id {
        return (
            "think? ".to_string(),
            Some(format!("think: <[{start}]> TEXT {close}")),
        );
    }
    (String::new(), None)
}

/// The output grammar of a request with a response schema and no tools:
/// the reasoning chat_format lets come first ([`reasoning`]), then the JSON.
pub fn json_grammar(schema: &Value, thinking: bool, chat_format: &ChatFormat) -> String {
    let (think, think_rule) = reasoning(thinking, chat_format);
    let mut grammar = vec![
        "%llguidance {}".to_string(),
        format!(
            "start: {think}WS %json {}",
            py_json_dumps_compact(&grammar_compatible_schema(schema))
        ),
    ];
    if let Some(think_rule) = think_rule {
        grammar.push(think_rule);
        grammar.push(think_text_rule());
    }
    grammar.push(whitespace_rule());
    grammar.join("\n") + "\n"
}

/// Whether the grammar compiler takes a JSON Schema ``pattern``. It keeps
/// the pattern's search semantics but rejects look-around, word boundaries
/// and backreferences, which a response format checks on the complete
/// output.
fn grammar_takes_pattern(pattern: &str) -> bool {
    fn pattern_checks() -> &'static Mutex<LruCache<String, bool>> {
        static CACHE: OnceLock<Mutex<LruCache<String, bool>>> = OnceLock::new();
        CACHE.get_or_init(|| {
            Mutex::new(LruCache::new(
                PATTERN_CHECK_CACHE_BYTES,
                Some(PATTERN_CHECK_CACHE_SIZE),
            ))
        })
    }
    {
        let mut cache = pattern_checks().lock().expect("pattern check lock");
        if let Some(takes) = cache.get(&pattern.to_string()) {
            return *takes;
        }
    }
    let string = py_json_dumps(&Value::Object(Map::from_iter([
        ("type".to_string(), Value::String("string".into())),
        ("pattern".to_string(), Value::String(pattern.to_string())),
    ])));
    let grammar = format!("%llguidance {{}}\nstart: %json {string}\n");
    // json.dumps uses ASCII escapes, so its length bounds the pattern's
    // bytes.
    let size = string.len();
    let takes = (|| {
        let Ok(top) = TopLevelGrammar::from_lark_or_grammar_list(&grammar) else {
            return false;
        };
        GrammarInit::Serialized(top)
            .validate(None, ParserLimits::default())
            .into_error()
            .is_none()
    })();
    let mut cache = pattern_checks().lock().expect("pattern check lock");
    if let Some(kept) = cache.get(&pattern.to_string()) {
        return *kept;
    }
    cache.put(pattern.to_string(), takes, size);
    takes
}

/// The constraints of `schema` the grammar compiler takes.
pub fn grammar_compatible_schema(schema: &Value) -> Value {
    let mut output = schema.clone();
    for_each_subschema_mut(&mut output, &mut |node| {
        let Some(map) = node.as_object_mut() else {
            return;
        };
        map.remove("propertyNames");
        let pattern = map.remove("pattern");
        if let Some(Value::String(pattern)) = &pattern {
            if grammar_takes_pattern(pattern) {
                map.insert("pattern".into(), Value::String(pattern.clone()));
            }
        }
    });
    // A local reference can point anywhere in the document, so any object may
    // be compiled as a schema.
    for_each_object_mut(&mut output, &mut |node| {
        let Some(map) = node.as_object_mut() else {
            return;
        };
        for key in GRAMMAR_BOUND_KEYWORDS {
            let exceeds = map
                .get(key)
                .and_then(Value::as_number)
                .and_then(|bound| bound.as_f64())
                .map(|bound| bound > MAX_GRAMMAR_BOUND as f64)
                .unwrap_or(false);
            if exceeds {
                map.remove(key);
            }
        }
    });
    if output == Value::Bool(true) {
        // Any value, with the whitespace between its tokens bounded.
        output = Value::Object(Map::new());
    }
    if let Some(map) = output.as_object_mut() {
        map.insert(
            "x-guidance".into(),
            Value::Object(Map::from_iter([
                ("lenient".to_string(), Value::Bool(true)),
                ("whitespace_pattern".to_string(), Value::String(whitespace())),
            ])),
        );
    }
    output
}

/// The output grammar of a request whose tool calls are constrained, or
/// whose tools stand beside a response schema: after the reasoning
/// chat_format lets come first ([`reasoning`]), each call names an offered tool
/// in the chat template's layout (chat_format.call_syntax), with a strict
/// tool's arguments generated to its schema and any other's free. A
/// required choice starts with a call and a named one is that call; under
/// an auto choice text may come first. Parallel calls may be followed by
/// text and further calls, and a single call by whitespace only. Beside an
/// auto choice, a response schema admits its JSON answer instead of the
/// calls.
pub fn tool_grammar(
    policy: &ToolPolicy,
    thinking: bool,
    response_schema: Option<&Value>,
    chat_format: &ChatFormat,
) -> Result<String, ApiError> {
    let syntax = &chat_format.call_syntax;
    let argument_schemas = policy.argument_schemas()?;
    let mut strict_arguments: HashMap<String, String> = HashMap::new();
    for name in policy.schemas().keys() {
        if !policy.is_strict(name) {
            continue;
        }
        let schema = argument_schemas.get(name).expect("strict tools are framed");
        strict_arguments.insert(name.clone(), argument_grammar(schema, syntax)?);
    }
    let mut side_grammars: Vec<Value> = Vec::new();
    if strict_arguments.len() < policy.schemas().len() {
        side_grammars.push(Value::Object(Map::from_iter([
            ("name".to_string(), Value::String("free_arguments".into())),
            ("lark_grammar".to_string(), Value::String(free_arguments(syntax))),
        ])));
    }
    let closing = if syntax.arguments_close.is_empty() {
        String::new()
    } else {
        format!(" {}", tags(syntax, &[syntax.arguments_close]))
    };
    let mut calls: Vec<String> = Vec::new();
    for (index, name) in policy.schemas().keys().enumerate() {
        let arguments = match strict_arguments.get(name) {
            Some(grammar) => {
                let arguments = format!("arguments_{index}");
                side_grammars.push(Value::Object(Map::from_iter([
                    ("name".to_string(), Value::String(arguments.clone())),
                    ("lark_grammar".to_string(), Value::String(grammar.clone())),
                ])));
                arguments
            }
            None => "free_arguments".to_string(),
        };
        calls.push(format!(
            "call_{index}: {TOOL_CALL_OPEN} {} @{arguments}{closing} {TOOL_CALL_CLOSE}",
            function_opening(syntax, name),
        ));
    }
    let call = format!(
        "({})",
        (0..calls.len())
            .map(|index| format!("call_{index}"))
            .collect::<Vec<_>>()
            .join(" | ")
    );
    let body = if calls.is_empty() {
        // tool_choice "none": neither the text nor a JSON answer starts a call.
        if response_schema.is_none() {
            "TEXT".to_string()
        } else {
            "answer".to_string()
        }
    } else if policy.required || response_schema.is_some() {
        // A call comes first, set apart from the reasoning by whitespace.
        let mut body = format!(
            "WS {call} {}",
            if policy.parallel {
                format!("(TEXT {call})* TEXT")
            } else {
                "WS".to_string()
            }
        );
        if !policy.required {
            body = format!("(({body}) | answer)");
        }
        body
    } else if policy.parallel {
        format!("(TEXT {call})* TEXT")
    } else {
        format!("TEXT ({call} WS)?")
    };
    let (think, think_rule) = reasoning(thinking, chat_format);
    let mut main = vec!["%llguidance {}".to_string(), format!("start: {think}{body}")];
    if let Some(response_schema) = response_schema.filter(|_| !policy.required) {
        main.push(format!(
            "answer: WS %json {} WS",
            py_json_dumps_compact(&grammar_compatible_schema(response_schema))
        ));
    }
    if let Some(think_rule) = think_rule {
        main.push(think_rule);
    }
    main.extend(calls);
    main.push(whitespace_rule());
    main.push(text_rule());
    side_grammars.insert(
        0,
        Value::Object(Map::from_iter([
            ("name".to_string(), Value::String("tool_output".into())),
            ("lark_grammar".to_string(), Value::String(main.join("\n") + "\n")),
        ])),
    );
    Ok(py_json_dumps_compact(&Value::Object(Map::from_iter([(
        "grammars".to_string(),
        Value::Array(side_grammars),
    )]))))
}
