//! How a request's tool calls are generated and read, ported from
//! `server/tool_schema.py`'s `ToolPolicy`, `ParameterTypes` and the
//! normalizers: the tools a request offers, the choice among them, and the
//! types their arguments' values convert by.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use serde_json::{Map, Value};

use crate::output::projector::{ParameterTypes, Types};
use crate::support::errors::ApiError;

use super::schema::{
    argument_shape, check_depth, for_each_subschema, is_tag_name, json_type_names,
    lookup_tool_reference, remote_ref, strict_schema, tool_argument_schema, types_reader,
    TAG_NAME_RULE,
};
use super::validate::{build_validator, check_schema, BuildError, ResponseValidator};

/// How a request's tool calls are generated and read.
///
/// `schemas` holds the declared parameter schema of each tool a call may
/// name; a null schema for a tool whose schema cannot be read, which only
/// describes it to the model. `strict` holds the strict tools, whose
/// arguments are generated to their schemas. `constrained` says whether a
/// grammar constrains the output: under a required or named choice, with
/// parallel_tool_calls false, beside a strict tool, and under tool_choice
/// none, whose grammar lets no call start.
pub struct ToolPolicy {
    schemas: Map<String, Value>,
    pub required: bool,
    pub parallel: bool,
    strict: HashSet<String>,
    pub constrained: bool,
    pub namespaces: Value,
    /// The strict tools' framed arguments, computed once, under one shared
    /// budget.
    framed: Mutex<Option<Result<Map<String, Value>, ApiError>>>,
}

impl ToolPolicy {
    /// The policy of one request's tools.
    pub fn new(
        schemas: Map<String, Value>,
        required: bool,
        parallel: bool,
        strict: HashSet<String>,
        constrained: bool,
        namespaces: Value,
    ) -> ToolPolicy {
        ToolPolicy {
            schemas,
            required,
            parallel,
            strict,
            constrained,
            namespaces,
            framed: Mutex::new(None),
        }
    }

    /// The declared parameter schema of each tool a call may name, in the
    /// order the request offered them.
    pub fn schemas(&self) -> &Map<String, Value> {
        &self.schemas
    }

    /// The strict tools, whose arguments are generated to their schemas.
    pub fn strict(&self) -> &HashSet<String> {
        &self.strict
    }

    /// Whether tool `name` is strict.
    pub fn is_strict(&self, name: &str) -> bool {
        self.strict.contains(name)
    }

    /// The arguments of each strict tool, framed as its grammar generates
    /// them. The grammars are built together, so one budget bounds them.
    pub fn argument_schemas(&self) -> Result<Map<String, Value>, ApiError> {
        let mut framed = self.framed.lock().expect("argument schemas lock");
        if let Some(framed) = framed.as_ref() {
            return framed.clone();
        }
        let mut budget = super::schema::MAX_FRAMED_SCHEMA_BYTES;
        let mut framed_schemas = Map::new();
        let result = (|| {
            for (name, schema) in &self.schemas {
                if !self.strict.contains(name) {
                    continue;
                }
                let framed = tool_argument_schema(
                    &strict_schema(&json_type_names(schema)),
                    &mut budget,
                )?;
                framed_schemas.insert(name.clone(), framed);
            }
            Ok(framed_schemas)
        })();
        *framed = Some(result.clone());
        result
    }
}

/// The JSON types by which the values of a request's tool calls convert,
/// which the projector of its output reads as a call first names a tool of
/// its [`ToolPolicy`] and keeps for its later calls: those of each parameter
/// the tool declares, and those of any other; `None` where its schema
/// constrains no type, as for a tool that was not offered. A strict tool's
/// are those of the arguments its grammar generates. Any other tool's
/// arguments are projected from its schema, within what the budget of the
/// request's tools that are not strict has left, and their types read
/// against that schema; one whose schema cannot be read or whose arguments
/// cannot be projected declares no types.
pub struct ToolParameterTypes {
    policy: Arc<ToolPolicy>,
    /// The types of each tool a call has named, and what the budget of the
    /// tools that are not strict has left as calls name them.
    state: Mutex<TypesState>,
}

struct TypesState {
    types: HashMap<String, (HashMap<String, Types>, Option<Types>)>,
    budget: i64,
}

impl ToolParameterTypes {
    /// The types provider of one request's tool policy.
    pub fn new(policy: Arc<ToolPolicy>) -> ToolParameterTypes {
        ToolParameterTypes {
            policy,
            state: Mutex::new(TypesState {
                types: HashMap::new(),
                budget: super::schema::MAX_FRAMED_SCHEMA_BYTES,
            }),
        }
    }
}

impl ParameterTypes for ToolParameterTypes {
    fn of(&self, name: &str) -> (HashMap<String, Types>, Option<Types>) {
        let mut state = self.state.lock().expect("parameter types lock");
        if !state.types.contains_key(name) {
            let types = read_types(&self.policy, &mut state.budget, name);
            state.types.insert(name.to_string(), types);
        }
        state.types.get(name).cloned().expect("just inserted")
    }
}

/// The types of tool `name`'s parameters, and those of any other.
fn read_types(
    policy: &ToolPolicy,
    budget: &mut i64,
    name: &str,
) -> (HashMap<String, Types>, Option<Types>) {
    if policy.is_strict(name) {
        let Ok(framed) = policy.argument_schemas() else {
            // Framing failed; the grammar build refused the request before
            // any output could be projected.
            return (HashMap::new(), None);
        };
        let Some(shape) = framed.get(name) else {
            return (HashMap::new(), None);
        };
        // A framed value is a schema of its own.
        let read = |value: &Value| super::schema::schema_types(value, value);
        let properties = shape
            .get("properties")
            .and_then(Value::as_object)
            .map(|properties| {
                properties
                    .iter()
                    .filter_map(|(parameter, value)| {
                        read(value).map(|types| (parameter.clone(), types))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let other = shape.get("additionalProperties").and_then(|value| read(value));
        return (properties, other);
    }
    let Some(schema) = policy.schemas().get(name) else {
        return (HashMap::new(), None);
    };
    if schema.is_null() {
        // A schema nested too deep to read only describes the tool.
        return (HashMap::new(), None);
    }
    let schema = json_type_names(schema);
    let Ok(shape) = argument_shape(&schema, budget) else {
        return (HashMap::new(), None);
    };
    let mut reader = types_reader(&schema);
    let properties = shape
        .get("properties")
        .and_then(Value::as_object)
        .map(|properties| {
            properties
                .iter()
                .filter_map(|(parameter, value)| {
                    reader.read(value).map(|types| (parameter.clone(), types))
                })
                .collect()
        })
        .unwrap_or_default();
    let other = shape
        .get("additionalProperties")
        .and_then(|value| reader.read(value));
    (properties, other)
}

/// Refuse strict tool `name`'s schema, from which its grammar is built, if
/// it nests too deep, refers to another document or breaks its dialect's
/// meta-schema.
pub fn check_tool_schema(name: &str, schema: &Value) -> Result<(), ApiError> {
    check_depth(schema, &format!("invalid tool schema for {name}"))?;
    if let Some(reference) = remote_ref(schema)? {
        return Err(ApiError::new(
            400,
            format!("remote tool schema reference is not allowed: {reference}"),
            "invalid_request_error",
        ));
    }
    check_schema(schema).map_err(|error| {
        ApiError::new(
            400,
            format!("invalid tool schema for {name}: {error}"),
            "invalid_request_error",
        )
    })
}

/// A response format's schema and the validator its answers are checked
/// against.
pub struct ResponseFormat {
    pub schema: Value,
    pub validator: Arc<ResponseValidator>,
}

/// The response format of a request, or `None` for text or no format.
pub fn normalize_response_format(
    value: Option<&Value>,
) -> Result<Option<ResponseFormat>, ApiError> {
    let text = Value::Object(Map::from_iter([(
        "type".to_string(),
        Value::String("text".into()),
    )]));
    if matches!(value, None | Some(Value::Null)) || value == Some(&text) {
        return Ok(None);
    }
    let Some(map) = value.and_then(Value::as_object) else {
        return Err(ApiError::new(
            400,
            "response_format must be an object",
            "invalid_request_error",
        ));
    };
    let schema = match map.get("type").and_then(Value::as_str) {
        Some("json_object") => Value::Object(Map::from_iter([(
            "type".to_string(),
            Value::String("object".into()),
        )])),
        Some("json_schema") => {
            let schema = map
                .get("json_schema")
                .and_then(Value::as_object)
                .and_then(|wrapper| wrapper.get("schema"));
            match schema {
                Some(schema) if schema.is_object() || schema.is_boolean() => schema.clone(),
                _ => {
                    return Err(ApiError::new(
                        400,
                        "response_format.json_schema.schema is required",
                        "invalid_request_error",
                    ))
                }
            }
        }
        _ => {
            return Err(ApiError::new(
                400,
                "unsupported response_format",
                "invalid_request_error",
            ))
        }
    };
    check_depth(&schema, "invalid response schema")?;
    if let Some(reference) = remote_ref(&schema)? {
        return Err(ApiError::new(
            400,
            format!("remote schema reference is not allowed: {reference}"),
            "invalid_request_error",
        ));
    }
    let validator = build_validator(&schema).map_err(|error| match error {
        BuildError::Schema(error) => ApiError::new(
            400,
            format!("invalid response schema: {error}"),
            "invalid_request_error",
        ),
        BuildError::Api(error) => error,
    })?;
    Ok(Some(ResponseFormat { schema, validator }))
}

/// Whether a tool's or a tool choice's `type` names a function, the only
/// kind there is, or names none, which reads as one.
fn names_function(kind: Option<&Value>) -> bool {
    match kind {
        None | Some(Value::Null) => true,
        Some(Value::String(kind)) => kind == "function",
        _ => false,
    }
}

/// The tools of a request and its [`ToolPolicy`].
///
/// `tools` is the request's tools, `None` where it offered none;
/// `tool_choice` the choice among them, `None` where the request said
/// nothing; `parallel` the parallel_tool_calls flag, `None` where unset;
/// `namespaces` the tool namespaces the request declared, `None` where it
/// said nothing.
pub fn normalize_tools(
    tools: Option<&Value>,
    tool_choice: Option<&Value>,
    parallel: Option<&Value>,
    namespaces: Option<&Value>,
) -> Result<(Option<Vec<Value>>, Option<ToolPolicy>), ApiError> {
    let parallel = match parallel {
        None | Some(Value::Null) => true,
        Some(Value::Bool(parallel)) => *parallel,
        Some(_) => {
            return Err(ApiError::new(
                400,
                "parallel_tool_calls must be a boolean",
                "invalid_request_error",
            ))
        }
    };
    let namespaces = match namespaces {
        None | Some(Value::Null) => Value::Object(Map::new()),
        Some(Value::Object(namespaces)) if namespaces.is_empty() => {
            Value::Object(Map::new())
        }
        Some(namespaces) => namespaces.clone(),
    };
    let Some(tools) = tools else {
        match tool_choice {
            None | Some(Value::Null) => return Ok((None, None)),
            Some(Value::String(choice)) if choice == "none" || choice == "auto" => {
                return Ok((None, None))
            }
            _ => {
                return Err(ApiError::new(
                    400,
                    "tool_choice requires tools",
                    "invalid_request_error",
                ))
            }
        }
    };
    let Some(offered_tools) = tools.as_array() else {
        return Err(ApiError::new(
            400,
            "tools must be an array",
            "invalid_request_error",
        ));
    };
    let mut schemas: Map<String, Value> = Map::new();
    let mut marked: HashSet<String> = HashSet::new();
    let mut offered: Vec<Value> = Vec::new();
    for tool in offered_tools {
        let Some(fields) = tool.as_object() else {
            return Err(ApiError::new(
                400,
                "only function tools are supported",
                "invalid_request_error",
            ));
        };
        // A tool that names no type is a function, the only kind there is,
        // and the template declares it as one.
        if !names_function(fields.get("type")) {
            return Err(ApiError::new(
                400,
                "only function tools are supported",
                "invalid_request_error",
            ));
        }
        let mut tool = tool.clone();
        if matches!(fields.get("type"), None | Some(Value::Null)) {
            let mut declared = Map::new();
            declared.insert("type".to_string(), Value::String("function".into()));
            for (key, value) in fields {
                if key != "type" {
                    declared.insert(key.clone(), value.clone());
                }
            }
            tool = Value::Object(declared);
        }
        offered.push(tool.clone());
        let Some(function) = tool.get("function").and_then(Value::as_object) else {
            return Err(ApiError::new(
                400,
                "a function tool needs a function object",
                "invalid_request_error",
            ));
        };
        let Some(name) = function.get("name").and_then(Value::as_str) else {
            return Err(ApiError::new(
                400,
                format!("tool name must be {TAG_NAME_RULE}"),
                "invalid_request_error",
            ));
        };
        if !is_tag_name(name) {
            return Err(ApiError::new(
                400,
                format!("tool name must be {TAG_NAME_RULE}"),
                "invalid_request_error",
            ));
        }
        if schemas.contains_key(name) {
            return Err(ApiError::new(
                400,
                format!("duplicate tool name: {name}"),
                "invalid_request_error",
            ));
        }
        let mut schema = function
            .get("parameters")
            .filter(|schema| !schema.is_null())
            .cloned()
            .unwrap_or_else(|| Value::Object(Map::new()));
        if !schema.is_object() && !schema.is_boolean() {
            return Err(ApiError::new(
                400,
                format!("invalid tool schema for {name}"),
                "invalid_request_error",
            ));
        }
        let strict = match function.get("strict") {
            None | Some(Value::Null) => false,
            Some(Value::Bool(strict)) => *strict,
            Some(_) => {
                return Err(ApiError::new(
                    400,
                    format!("strict must be a boolean for tool {name}"),
                    "invalid_request_error",
                ))
            }
        };
        if strict {
            check_tool_schema(name, &schema)?;
        } else {
            // Only a strict tool's grammar is built from its schema. Any
            // other's gives the types its values convert by wherever it can
            // be read, valid or not; one nested too deep to read only
            // describes the tool, and its values convert as those of a
            // parameter it does not declare.
            if check_depth(&schema, &format!("invalid tool schema for {name}")).is_err() {
                schema = Value::Null;
            }
        }
        schemas.insert(name.to_string(), schema);
        if strict {
            marked.insert(name.to_string());
        }
    }
    let choice = match tool_choice {
        None | Some(Value::Null) => Value::String("auto".into()),
        Some(choice) => choice.clone(),
    };
    if offered_tools.is_empty() {
        let allowed = matches!(&choice, Value::String(choice) if choice == "auto" || choice == "none");
        if !allowed {
            return Err(ApiError::new(
                400,
                "tool_choice requires at least one tool",
                "invalid_request_error",
            ));
        }
        return Ok((None, None));
    }
    if choice.as_str() == Some("none") {
        // The prompt keeps every tool; the grammar lets no call start.
        return Ok((
            Some(offered),
            Some(ToolPolicy::new(
                Map::new(),
                false,
                parallel,
                HashSet::new(),
                true,
                namespaces,
            )),
        ));
    }
    let mut schemas = schemas;
    let mut parallel = parallel;
    if let Some(choice) = choice.as_object() {
        let function = choice.get("function").and_then(Value::as_object);
        let name = function
            .and_then(|function| function.get("name"))
            .and_then(Value::as_str);
        let named_function = names_function(choice.get("type"));
        let offered_name = name
            .filter(|name| schemas.contains_key(*name))
            .map(str::to_string);
        if !named_function || offered_name.is_none() {
            return Err(ApiError::new(
                400,
                "invalid named tool_choice",
                "invalid_request_error",
            ));
        }
        // The prompt keeps every tool; the grammar forces the call, exactly
        // one, as a forced function is defined.
        let name = offered_name.expect("checked above");
        let mut named = Map::new();
        named.insert(name.clone(), schemas.get(&name).expect("checked above").clone());
        schemas = named;
        parallel = false;
    } else if choice.as_str() != Some("auto") && choice.as_str() != Some("required") {
        return Err(ApiError::new(
            400,
            "invalid tool_choice",
            "invalid_request_error",
        ));
    }
    let strict: HashSet<String> = marked
        .intersection(&schemas.keys().cloned().collect::<HashSet<_>>())
        .cloned()
        .collect();
    // A strict tool's schema compiles into its grammar, so its references
    // must resolve.
    let mut failure: Option<ApiError> = None;
    for name in &strict {
        let schema = schemas.get(name).expect("strict tools are in schemas");
        for_each_subschema(schema, &mut |node| {
            if failure.is_some() {
                return;
            }
            if let Some(reference) = node.get("$ref") {
                if let Err(error) = lookup_tool_reference(reference, schema) {
                    failure = Some(error);
                }
            }
        });
    }
    if let Some(error) = failure {
        return Err(error);
    }
    let required = choice.as_str() != Some("auto");
    // A grammar holds the calls to the template's tags when the choice
    // requires one, when it may make at most one, and beside a strict tool,
    // whose arguments it generates; otherwise the model writes them freely.
    let constrained = required || !parallel || !strict.is_empty();
    let policy = ToolPolicy::new(schemas, required, parallel, strict, constrained, namespaces);
    Ok((Some(offered), Some(policy)))
}
