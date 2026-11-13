//! JSON Schema validation with bounded regular-expression evaluation,
//! ported from `server/schema_validation.py` and `server/output.py`'s
//! `validate_response_content`: a schema is checked against its dialect's
//! meta-schema, and an answer against its schema, on the `jsonschema`
//! crate, whose pattern engine (fancy-regex) bounds a pattern's evaluation
//! by its backtracking steps where the Python's bounds it by time.
//!
//! Draft 3, which the crate does not carry, is read by this module: its
//! schema keywords are checked against the draft's own rules, and its
//! schemas are translated to the keywords of the later drafts the crate
//! reads — `extends` as allOf, `disallow` as not, a type list's schema
//! members as anyOf, a property's `required` as its parent's, `divisibleBy`
//! as multipleOf, the exclusive bounds as numbers, tuple `items` as
//! prefixItems and `additionalItems` as items.

use std::collections::HashSet;
use std::sync::{Arc, Mutex, OnceLock};

use jsonschema::error::ValidationErrorKind;
use jsonschema::{Draft, ValidationError};
use serde_json::{Map, Value};

use crate::support::errors::ApiError;
use crate::support::json_codec;
use crate::support::lru::LruCache;

use super::grammar::py_json_dumps;
use super::schema::{for_each_object, for_each_subschema, for_each_subschema_mut};

/// A validator could not evaluate an already accepted schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluationError(pub String);

/// A schema that breaks its dialect's meta-schema: the request's error, as
/// `jsonschema`'s `SchemaError` is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaError(pub String);

impl std::fmt::Display for SchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// Why a validator could not be built: a schema that breaks its dialect, or
/// one the reading refuses.
#[derive(Debug, Clone, PartialEq)]
pub enum BuildError {
    Schema(SchemaError),
    Api(ApiError),
}

/// Why an answer is not a valid structured output: the schema could not be
/// evaluated, which is the server's failure, or the answer does not hold to
/// it, which is the model's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationFailure {
    Evaluation(String),
    Invalid(String),
}

/// The dialects a schema may declare, by the URI its `$schema` names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Dialect {
    Draft3,
    Draft4,
    Draft6,
    Draft7,
    Draft201909,
    Draft202012,
}

/// The dialect `node`'s `$schema` names; the latest draft where it names
/// nothing, as the Python's `validator_for` falls back to.
pub(crate) fn dialect_of_node(node: &Value) -> Dialect {
    match node.get("$schema").and_then(Value::as_str) {
        Some(uri) => dialect_of_uri(uri),
        None => Dialect::Draft202012,
    }
}

fn dialect_of_uri(uri: &str) -> Dialect {
    match uri.strip_suffix('#').unwrap_or(uri) {
        uri if known_dialect_uri(uri) => match uri {
            "http://json-schema.org/draft-03/schema" => Dialect::Draft3,
            "http://json-schema.org/draft-04/schema" => Dialect::Draft4,
            "http://json-schema.org/draft-06/schema" => Dialect::Draft6,
            "http://json-schema.org/draft-07/schema" => Dialect::Draft7,
            "https://json-schema.org/draft/2019-09/schema" => Dialect::Draft201909,
            _ => Dialect::Draft202012,
        },
        // The Python's validator_for falls back to the latest draft, with a
        // deprecation warning, where $schema names nothing it knows.
        _ => Dialect::Draft202012,
    }
}

/// Whether the URI is one of the dialects the meta-schemas are known for.
fn known_dialect_uri(uri: &str) -> bool {
    matches!(
        uri,
        "http://json-schema.org/draft-03/schema"
            | "http://json-schema.org/draft-04/schema"
            | "http://json-schema.org/draft-06/schema"
            | "http://json-schema.org/draft-07/schema"
            | "https://json-schema.org/draft/2019-09/schema"
            | "https://json-schema.org/draft/2020-12/schema"
    )
}

fn draft_of(dialect: Dialect) -> Draft {
    match dialect {
        Dialect::Draft4 => Draft::Draft4,
        Dialect::Draft6 => Draft::Draft6,
        Dialect::Draft7 => Draft::Draft7,
        Dialect::Draft201909 => Draft::Draft201909,
        Dialect::Draft202012 | Dialect::Draft3 => Draft::Draft202012,
    }
}

/// The type names each dialect's validator knows; Draft 3 accepts `any`.
fn known_type(dialect: Dialect, name: &str) -> bool {
    match dialect {
        Dialect::Draft3 => matches!(
            name,
            "any" | "array" | "boolean" | "integer" | "null" | "number" | "object" | "string"
        ),
        _ => matches!(
            name,
            "array" | "boolean" | "integer" | "null" | "number" | "object" | "string"
        ),
    }
}

/// The schemas `check_schema` has checked, by their canonical form, apart
/// from the validators response formats keep: `Ok`, or why the schema is
/// invalid.
fn checked_schemas() -> &'static Mutex<LruCache<String, Result<(), String>>> {
    static CACHE: OnceLock<Mutex<LruCache<String, Result<(), String>>>> = OnceLock::new();
    CACHE.get_or_init(|| {
        Mutex::new(LruCache::new(8 * 1024 * 1024, Some(1024)))
    })
}

/// The validators `build_validator` has built, by their schemas' canonical
/// form.
fn validators() -> &'static Mutex<LruCache<String, Arc<ResponseValidator>>> {
    static CACHE: OnceLock<Mutex<LruCache<String, Arc<ResponseValidator>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(LruCache::new(8 * 1024 * 1024, Some(256))))
}

/// The schema as a canonical string, its object keys sorted, as the caches
/// key them.
fn canonical_json(value: &Value) -> String {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut out = String::from("{");
            for (index, key) in keys.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                out.push_str(&py_json_dumps(&Value::String((*key).clone())));
                out.push(':');
                out.push_str(&canonical_json(&map[*key]));
            }
            out.push('}');
            out
        }
        Value::Array(items) => {
            let mut out = String::from("[");
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                out.push_str(&canonical_json(item));
            }
            out.push(']');
            out
        }
        other => py_json_dumps(other),
    }
}

/// Raise `SchemaError` unless `schema` is valid in the dialect it declares.
///
/// A strict tool's schema needs no other check, as its calls are not
/// validated, and a client sends its tools on every turn, so each outcome is
/// kept.
pub fn check_schema(schema: &Value) -> Result<(), SchemaError> {
    let key = canonical_json(schema);
    {
        let mut cache = checked_schemas().lock().expect("checked schemas lock");
        if let Some(outcome) = cache.get(&key) {
            return outcome.clone().map_err(SchemaError);
        }
    }
    let outcome = check_schema_uncached(schema).map_err(|error| error.0);
    let size = key.len();
    let mut cache = checked_schemas().lock().expect("checked schemas lock");
    if let Some(kept) = cache.get(&key) {
        return kept.clone().map_err(SchemaError);
    }
    cache.put(key, outcome.clone(), size);
    outcome.map_err(SchemaError)
}

fn check_schema_uncached(schema: &Value) -> Result<(), SchemaError> {
    match dialect_of_node(schema) {
        Dialect::Draft3 => check_schema_draft3(schema),
        _ => {
            // The crate detects the dialect from `$schema`; where it names
            // nothing the crate knows, a copy without it is checked against
            // the latest draft's meta-schema, as the Python's fallback does.
            let known = schema
                .get("$schema")
                .and_then(Value::as_str)
                .map(|uri| known_dialect_uri(uri.strip_suffix('#').unwrap_or(uri)))
                .unwrap_or(true);
            let document = if known {
                schema.clone()
            } else {
                let mut document = schema.clone();
                if let Some(map) = document.as_object_mut() {
                    map.remove("$schema");
                }
                document
            };
            jsonschema::meta::validate(&document)
                .map_err(|error| SchemaError(error.to_string()))
        }
    }
}

/// Draft 3's meta-schema, as far as its keywords' kinds go: a keyword of
/// another kind than it takes is refused, as the meta-schema refuses it, and
/// a keyword of a later draft, which Draft 3 leaves unchecked, is not.
fn check_schema_draft3(schema: &Value) -> Result<(), SchemaError> {
    let mut error: Option<String> = None;
    for_each_subschema(schema, &mut |node| {
        if error.is_some() {
            return;
        }
        let Some(map) = node.as_object() else {
            return;
        };
        for (key, value) in map {
            let kind = |expected: &str| {
                format!("{value} is not of type '{expected}'")
            };
            let mut fail = |message: String| {
                if error.is_none() {
                    error = Some(message);
                }
            };
            match key.as_str() {
                "type" | "disallow" => {
                    let ok = value.is_string()
                        || value.as_array().map(|items| {
                            items.iter().all(|item| item.is_string() || item.is_object())
                        }).unwrap_or(false);
                    if !ok {
                        fail(kind("string, array"));
                    }
                }
                "properties" | "patternProperties" => {
                    if !value.is_object() {
                        fail(kind("object"));
                    }
                }
                "items" => {
                    if !value.is_object() && !value.is_array() {
                        fail(kind("object, array"));
                    }
                }
                "additionalItems" | "additionalProperties" => {
                    if !value.is_object() && !value.is_boolean() {
                        fail(kind("object, boolean"));
                    }
                }
                "dependencies" => {
                    let ok = value.as_object().map(|dependencies| {
                        dependencies.values().all(|dependency| {
                            dependency.is_object()
                                || dependency.as_array().map(|items| {
                                    items.iter().all(Value::is_string)
                                }).unwrap_or(false)
                        })
                    }).unwrap_or(false);
                    if !ok {
                        fail(kind("object"));
                    }
                }
                "extends" => {
                    if !value.is_object() && !value.is_array() {
                        fail(kind("object, array"));
                    }
                }
                "enum" => {
                    if !value.is_array() {
                        fail(kind("array"));
                    }
                }
                "divisibleBy" | "minimum" | "maximum" => {
                    if !value.is_number() {
                        fail(kind("number"));
                    }
                }
                "exclusiveMinimum" | "exclusiveMaximum" => {
                    if !value.is_boolean() {
                        fail(kind("boolean"));
                    }
                }
                "minLength" | "maxLength" | "minItems" | "maxItems" => {
                    let ok = value
                        .as_number()
                        .and_then(|number| number.as_i64())
                        .map(|bound| bound >= 0)
                        .unwrap_or(false);
                    if !ok {
                        fail(kind("integer"));
                    }
                }
                "pattern" | "format" | "$ref" | "$schema" | "id" => {
                    if !value.is_string() {
                        fail(kind("string"));
                    }
                }
                "uniqueItems" | "required" => {
                    if !value.is_boolean() {
                        fail(kind("boolean"));
                    }
                }
                _ => {}
            }
        }
    });
    match error {
        Some(message) => Err(SchemaError(message)),
        None => Ok(()),
    }
}

/// A validator of `schema` whose regular expressions are bounded, refusing
/// a schema that breaks its dialect (`BuildError::Schema`) or that it could
/// not evaluate (`BuildError::Api`).
pub fn build_validator(schema: &Value) -> Result<Arc<ResponseValidator>, BuildError> {
    let key = canonical_json(schema);
    {
        let mut cache = validators().lock().expect("validator cache lock");
        if let Some(validator) = cache.get(&key) {
            return Ok(validator.clone());
        }
    }
    let built = build_validator_uncached(schema);
    let size = key.len();
    let mut cache = validators().lock().expect("validator cache lock");
    if let Some(kept) = cache.get(&key) {
        return Ok(kept.clone());
    }
    if let Ok(validator) = &built {
        cache.put(key, validator.clone(), size);
    }
    built
}

fn build_validator_uncached(schema: &Value) -> Result<Arc<ResponseValidator>, BuildError> {
    check_schema(schema).map_err(BuildError::Schema)?;
    let dialect = dialect_of_node(schema);
    // Draft 3 accepts any type name, and validation fails on one the dialect
    // does not define with an error that is not a validation error.
    let mut unknown: Option<ApiError> = None;
    for_each_subschema(schema, &mut |node| {
        if unknown.is_some() || !node.is_object() {
            return;
        }
        for keyword in ["type", "disallow"] {
            let Some(names) = node.get(keyword) else {
                continue;
            };
            let items: Vec<&Value> = match names {
                Value::Array(items) => items.iter().collect(),
                other => vec![other],
            };
            for name in items {
                if let Some(text) = name.as_str() {
                    if !known_type(dialect, text) {
                        unknown = Some(ApiError::new(
                            400,
                            format!("unknown schema type: {text}"),
                            "invalid_request_error",
                        ));
                        return;
                    }
                }
            }
        }
    });
    if let Some(error) = unknown {
        return Err(BuildError::Api(error));
    }
    // The validator matches patternProperties with its own engine to find
    // the properties unevaluatedProperties applies to. A reference can
    // reach any object, so one document cannot use both.
    let mut keywords: HashSet<String> = HashSet::new();
    for_each_object(schema, &mut |node| {
        if let Some(map) = node.as_object() {
            keywords.extend(map.keys().cloned());
        }
    });
    if keywords.contains("patternProperties") && keywords.contains("unevaluatedProperties") {
        return Err(BuildError::Api(ApiError::new(
            400,
            "unevaluatedProperties with patternProperties is not supported",
            "invalid_request_error",
        )));
    }
    let mut validated = schema.clone();
    // A document uses one dialect. Removing identical declarations keeps
    // the crate reading the dialect the root declared.
    let mut mixed: Option<ApiError> = None;
    for_each_subschema_mut(&mut validated, &mut |node| {
        if mixed.is_some() {
            return;
        }
        if node.get("$schema").is_some() {
            if dialect_of_node(node) != dialect {
                mixed = Some(ApiError::new(
                    400,
                    "mixed schema dialects are not supported",
                    "invalid_request_error",
                ));
                return;
            }
            if let Some(map) = node.as_object_mut() {
                map.remove("$schema");
            }
        }
    });
    if let Some(error) = mixed {
        return Err(BuildError::Api(error));
    }
    let (document, draft) = match dialect {
        Dialect::Draft3 => (translate_draft3(&validated), Draft::Draft202012),
        other => (validated, draft_of(other)),
    };
    match jsonschema::options().with_draft(draft).build(&document) {
        Ok(inner) => Ok(Arc::new(ResponseValidator {
            inner: Some(inner),
            evaluation_failure: None,
        })),
        Err(error) => {
            if is_regex_error(&error) {
                // A pattern the pattern engine refuses: the Python builds
                // its validator and fails when it evaluates the pattern, so
                // the refusal waits for the output too.
                Ok(Arc::new(ResponseValidator {
                    inner: None,
                    evaluation_failure: Some("schema pattern could not be evaluated".to_string()),
                }))
            } else {
                Err(BuildError::Schema(SchemaError(error.to_string())))
            }
        }
    }
}

/// Whether the validator refused a schema over one of its patterns.
fn is_regex_error(error: &ValidationError<'static>) -> bool {
    matches!(
        error.kind,
        ValidationErrorKind::Format { ref format } if format == "regex"
    )
}

/// A validator of one request's structured answers.
#[derive(Debug)]
pub struct ResponseValidator {
    inner: Option<jsonschema::Validator>,
    evaluation_failure: Option<String>,
}

impl ResponseValidator {
    /// The answer `value` holds to the schema, or why it does not.
    pub fn validate(&self, value: &Value) -> Result<(), ValidationFailure> {
        if let Some(message) = &self.evaluation_failure {
            return Err(ValidationFailure::Evaluation(message.clone()));
        }
        let Some(inner) = &self.inner else {
            return Err(ValidationFailure::Evaluation(
                "schema pattern could not be evaluated".to_string(),
            ));
        };
        match inner.validate(value) {
            Ok(()) => Ok(()),
            Err(error) => {
                if matches!(
                    error.kind,
                    ValidationErrorKind::BacktrackLimitExceeded { .. }
                ) {
                    Err(ValidationFailure::Evaluation(
                        "schema pattern exceeded its evaluation time limit".to_string(),
                    ))
                } else {
                    Err(ValidationFailure::Invalid(error.to_string()))
                }
            }
        }
    }

    /// Whether the answer `value` holds to the schema, or the evaluation
    /// failure that stopped the check.
    pub fn is_valid(&self, value: &Value) -> Result<bool, String> {
        match self.validate(value) {
            Ok(()) => Ok(true),
            Err(ValidationFailure::Invalid(_)) => Ok(false),
            Err(ValidationFailure::Evaluation(message)) => Err(message),
        }
    }
}

/// The structured answer `content` spells, checked against `validator`:
/// `Ok` where it holds, or the refusal the answer earns — the server's,
/// where the schema could not be evaluated, and the model's, where the
/// answer does not hold or is not JSON.
pub fn validate_response_content(
    content: &str,
    validator: Option<&ResponseValidator>,
) -> Result<(), ApiError> {
    let Some(validator) = validator else {
        return Ok(());
    };
    let value = json_codec::loads(content).map_err(|error| {
        ApiError::new(
            500,
            format!("model returned invalid structured output: {error}"),
            "invalid_model_output",
        )
    })?;
    match validator.validate(&value) {
        Ok(()) => Ok(()),
        Err(ValidationFailure::Evaluation(message)) => {
            Err(ApiError::new(500, message, "output_validation_failed"))
        }
        Err(ValidationFailure::Invalid(message)) => Err(ApiError::new(
            500,
            format!("model returned invalid structured output: {message}"),
            "invalid_model_output",
        )),
    }
}

/// Draft 3's schema keywords, which the validator reads as their later
/// equivalents (see the module note). Keywords of later drafts, which an
/// older dialect leaves unchecked, are left out.
fn translate_draft3(schema: &Value) -> Value {
    translate_draft3_node(schema)
}

fn translate_draft3_node(node: &Value) -> Value {
    let Some(map) = node.as_object() else {
        return node.clone();
    };
    // $ref replaces its siblings.
    if let Some(reference) = map.get("$ref") {
        let mut only = Map::new();
        only.insert("$ref".to_string(), reference.clone());
        return Value::Object(only);
    }
    let mut out = Map::new();
    let mut required: Vec<String> = Vec::new();
    for (key, value) in map {
        match key.as_str() {
            // Keywords of later drafts, which Draft 3 leaves unchecked.
            "allOf" | "anyOf" | "oneOf" | "not" | "if" | "then" | "else" | "const"
            | "contains" | "propertyNames" | "prefixItems" | "unevaluatedItems"
            | "unevaluatedProperties" | "dependentSchemas" | "dependentRequired"
            | "multipleOf" | "$recursiveRef" | "$recursiveAnchor" | "$dynamicRef"
            | "$dynamicAnchor" | "$anchor" | "$id" | "$schema" | "$comment" | "required" => {}
            "extends" => {
                let members: Vec<&Value> = match value {
                    Value::Array(items) => items.iter().collect(),
                    other => vec![other],
                };
                out.insert(
                    "allOf".to_string(),
                    Value::Array(members.into_iter().map(translate_draft3_node).collect()),
                );
            }
            "disallow" => {
                let members: Vec<&Value> = match value {
                    Value::Array(items) => items.iter().collect(),
                    other => vec![other],
                };
                let alternatives = members
                    .into_iter()
                    .map(draft3_type_member)
                    .collect::<Vec<_>>();
                let mut not = Map::new();
                not.insert("anyOf".to_string(), Value::Array(alternatives));
                out.insert("not".to_string(), Value::Object(not));
            }
            "type" => match value {
                Value::String(name) if name == "any" => {}
                Value::String(name) => {
                    out.insert("type".to_string(), Value::String(name.clone()));
                }
                Value::Array(items) => {
                    let alternatives = items.iter().map(draft3_type_member).collect::<Vec<_>>();
                    out.insert("anyOf".to_string(), Value::Array(alternatives));
                }
                _ => {}
            },
            "divisibleBy" => {
                out.insert("multipleOf".to_string(), value.clone());
            }
            "exclusiveMinimum" => {
                if value == &Value::Bool(true) {
                    if let Some(minimum) = map.get("minimum") {
                        out.insert("exclusiveMinimum".to_string(), minimum.clone());
                    }
                }
            }
            "exclusiveMaximum" => {
                if value == &Value::Bool(true) {
                    if let Some(maximum) = map.get("maximum") {
                        out.insert("exclusiveMaximum".to_string(), maximum.clone());
                    }
                }
            }
            "items" => match value {
                Value::Array(items) => {
                    let translated = items.iter().map(translate_draft3_node).collect::<Vec<_>>();
                    out.insert("prefixItems".to_string(), Value::Array(translated));
                }
                other => {
                    out.insert("items".to_string(), translate_draft3_node(other));
                }
            },
            "additionalItems" => {
                // Draft 3 checks additionalItems only beside tuple items.
                if map.get("items").map(Value::is_array).unwrap_or(false) {
                    match value {
                        Value::Bool(false) => {
                            out.insert("items".to_string(), Value::Bool(false));
                        }
                        Value::Bool(true) => {}
                        other => {
                            out.insert("items".to_string(), translate_draft3_node(other));
                        }
                    }
                }
            }
            "dependencies" => {
                if let Some(dependencies) = value.as_object() {
                    let mut dependent_required = Map::new();
                    let mut dependent_schemas = Map::new();
                    for (name, dependency) in dependencies {
                        match dependency {
                            Value::Array(_) => {
                                dependent_required.insert(name.clone(), dependency.clone());
                            }
                            other => {
                                dependent_schemas
                                    .insert(name.clone(), translate_draft3_node(other));
                            }
                        }
                    }
                    if !dependent_required.is_empty() {
                        out.insert(
                            "dependentRequired".to_string(),
                            Value::Object(dependent_required),
                        );
                    }
                    if !dependent_schemas.is_empty() {
                        out.insert(
                            "dependentSchemas".to_string(),
                            Value::Object(dependent_schemas),
                        );
                    }
                }
            }
            "properties" => {
                if let Some(properties) = value.as_object() {
                    let mut translated = Map::new();
                    for (name, sub) in properties {
                        if sub.get("required") == Some(&Value::Bool(true)) {
                            required.push(name.clone());
                        }
                        translated.insert(name.clone(), translate_draft3_node(sub));
                    }
                    out.insert("properties".to_string(), Value::Object(translated));
                }
            }
            "patternProperties" => {
                if let Some(properties) = value.as_object() {
                    let translated = properties
                        .iter()
                        .map(|(name, sub)| (name.clone(), translate_draft3_node(sub)))
                        .collect();
                    out.insert("patternProperties".to_string(), Value::Object(translated));
                }
            }
            "$defs" | "definitions" => {
                if let Some(definitions) = value.as_object() {
                    let translated = definitions
                        .iter()
                        .map(|(name, sub)| (name.clone(), translate_draft3_node(sub)))
                        .collect();
                    out.insert(key.clone(), Value::Object(translated));
                }
            }
            _ => {
                out.insert(key.clone(), value.clone());
            }
        }
    }
    if !required.is_empty() {
        out.insert(
            "required".to_string(),
            Value::Array(required.into_iter().map(Value::String).collect()),
        );
    }
    Value::Object(out)
}

/// A member of a Draft 3 `type` or `disallow` list: a type name as the
/// schema of that type, `any` as one that constrains nothing, a schema as
/// itself.
fn draft3_type_member(member: &Value) -> Value {
    match member {
        Value::String(name) if name == "any" => Value::Object(Map::new()),
        Value::String(name) => Value::Object(Map::from_iter([(
            "type".to_string(),
            Value::String(name.clone()),
        )])),
        other => translate_draft3_node(other),
    }
}
