//! Schema reading, ported from `server/tool_schema.py`'s readers and the
//! subschema walk of `server/schema_validation.py`: the checks a schema must
//! pass before anything reads it, the JSON types and literal texts a
//! parameter's value takes, and the projection a tool's fields frame
//! through.
//!
//! The readers memoize each node once per reading, as the Python's do by
//! object identity. `serde_json` values have no stable identity, so a node
//! is keyed by its canonical JSON text: two nodes that serialize alike read
//! once, which changes nothing a reading returns, and a schema that nests a
//! structural copy of itself stops at the copy where the Python's would
//! recurse until the interpreter's stack gave out.

use std::collections::{HashMap, HashSet};

use serde_json::{Map, Value};
use unicode_normalization::UnicodeNormalization;

use crate::support::errors::ApiError;

use super::grammar::py_json_dumps_compact;

/// A function's or parameter's name is at most this long, and reading strips
/// this space around it, so a name the grammars write neither starts nor ends
/// with it.
pub const MAX_NAME_LENGTH: usize = 256;
/// The space a name loses around it.
pub const NAME_SPACE: &str = " \t\n\r";
/// How [`is_tag_name`] describes itself to a client.
pub const TAG_NAME_RULE: &str = "1 to 256 characters in NFC, without <, >, line breaks, lone surrogates or surrounding whitespace";

/// Whether `name` reads back as written from a call's tags, where a
/// function's name and a strict tool's parameter's are written. A tokenizer
/// may normalize text to NFC, as Qwen's does, and none spells a lone
/// surrogate; a Rust string holds no lone surrogate to check for.
pub fn is_tag_name(name: &str) -> bool {
    let length = name.chars().count();
    0 < length
        && length <= MAX_NAME_LENGTH
        && name == name.trim_matches(|c| NAME_SPACE.contains(c))
        && !name.chars().any(|c| "<>\n\r".contains(c))
        && name.chars().nfc().collect::<String>() == name
}

/// Schemas are read recursively, by the validator and by this module, several
/// stack frames a level, and arrays nested twice this deep would exhaust the
/// stack: a schema nests objects and arrays at most this deep.
pub const MAX_SCHEMA_DEPTH: usize = 64;

/// Framing projects each tool's fields through schema composition, and for a
/// strict tool's grammar copies the root schema into every field that refers
/// to it. Pathological schemas make that quadratic or exponential in their
/// size, so framing may produce about this many bytes of schemas: for a
/// request's strict tools, whose grammars are built together, and as much for
/// its other tools, which are framed for their parameters' types alone.
pub const MAX_FRAMED_SCHEMA_BYTES: i64 = 16 * 1024 * 1024;

/// Refuse `schema`, as `invalid` names it, if it nests deeper than
/// [`MAX_SCHEMA_DEPTH`]. Read without recursion, before anything recurses.
pub fn check_depth(schema: &Value, invalid: &str) -> Result<(), ApiError> {
    let too_deep = || {
        ApiError::new(
            400,
            format!("{invalid}: nested more than {MAX_SCHEMA_DEPTH} levels deep"),
            "invalid_request_error",
        )
    };
    let mut pending = vec![(schema, 1usize)];
    while let Some((value, depth)) = pending.pop() {
        match value {
            Value::Object(map) => {
                if depth > MAX_SCHEMA_DEPTH {
                    return Err(too_deep());
                }
                pending.extend(map.values().map(|child| (child, depth + 1)));
            }
            Value::Array(items) => {
                if depth > MAX_SCHEMA_DEPTH {
                    return Err(too_deep());
                }
                pending.extend(items.iter().map(|child| (child, depth + 1)));
            }
            _ => {}
        }
    }
    Ok(())
}

/// Schema keywords whose values are annotations, which hold no assertion.
pub const SCHEMA_ANNOTATIONS: [&str; 7] = [
    "$comment",
    "title",
    "description",
    "default",
    "examples",
    "deprecated",
    "readOnly",
];

/// Keywords that apply further schemas to the instance a schema describes.
pub const IN_PLACE_APPLICATORS: [&str; 10] = [
    "allOf",
    "not",
    "if",
    "then",
    "else",
    "dependentSchemas",
    "dependencies",
    "extends",
    "$dynamicRef",
    "$recursiveRef",
];
/// Keywords that name the schema or its parts, which hold no assertion.
pub const SCHEMA_IDENTIFIERS: [&str; 6] =
    ["$schema", "$id", "$anchor", "$dynamicAnchor", "$defs", "definitions"];
/// Keywords by which an object says which properties it takes beyond those it
/// declares.
pub const MORE_PROPERTIES: [&str; 3] =
    ["additionalProperties", "unevaluatedProperties", "patternProperties"];

/// Whether other schemas apply to the instance `node` describes, so that
/// `node` may declare only some of its properties.
pub fn extended(node: &Value) -> bool {
    let Some(map) = node.as_object() else {
        return false;
    };
    let keys: HashSet<&str> = map
        .keys()
        .map(String::as_str)
        .filter(|key| !SCHEMA_ANNOTATIONS.contains(key) && !SCHEMA_IDENTIFIERS.contains(key))
        .collect();
    if keys.iter().any(|key| IN_PLACE_APPLICATORS.contains(key)) {
        return true;
    }
    if keys.contains("$ref") && keys.len() != 1 {
        return true;
    }
    let unions = keys.iter().filter(|key| **key == "anyOf" || **key == "oneOf").count();
    unions > 0
        && keys
            .iter()
            .any(|key| *key != "anyOf" && *key != "oneOf" && *key != "type")
}

/// The schema a strict tool's arguments are generated to, as vLLM and
/// SGLang generate them with XGrammar's strict mode: an object that does not
/// say which properties it takes beyond those it declares takes none, and
/// an array that does not say which items it takes beyond its leading ones
/// takes none.
///
/// A schema that other schemas of the same instance extend, as an allOf
/// does, may declare only some of its properties, and closing it would
/// refuse the others; a schema composed that way is left as declared.
pub fn strict_schema(schema: &Value) -> Value {
    let mut closes = false;
    for_each_subschema(schema, &mut |node| {
        if !closes && node.is_object() && extended(node) {
            closes = true;
        }
    });
    if closes {
        return schema.clone();
    }
    let mut output = schema.clone();
    for_each_subschema_mut(&mut output, &mut |node| {
        let Some(map) = node.as_object_mut() else {
            return;
        };
        // A union or a reference describes its instance by its alternatives
        // or its target, which are closed in their own places.
        if map.contains_key("anyOf") || map.contains_key("oneOf") || map.contains_key("$ref") {
            return;
        }
        let kind = map.get("type").filter(|kind| !kind.is_null()).cloned();
        let kinds: Vec<Value> = match &kind {
            Some(Value::Array(items)) => items.clone(),
            other => vec![other.clone().unwrap_or(Value::Null)],
        };
        let kind_is_object = kinds.iter().any(|kind| kind == "object");
        let kind_is_array = kinds.iter().any(|kind| kind == "array");
        let kind_is_none = kind.is_none();
        if kind_is_object || (kind_is_none && map.contains_key("properties")) {
            if !MORE_PROPERTIES.iter().any(|key| map.contains_key(*key)) {
                map.insert("additionalProperties".into(), Value::Bool(false));
            }
        }
        if kind_is_array || (kind_is_none && map.contains_key("prefixItems")) {
            if map.get("items").map(Value::is_array).unwrap_or(false) {
                if !map.contains_key("additionalItems") && !map.contains_key("unevaluatedItems") {
                    map.insert("additionalItems".into(), Value::Bool(false));
                }
            } else if !map.contains_key("items") && !map.contains_key("unevaluatedItems") {
                map.insert("items".into(), Value::Bool(false));
            }
        }
    });
    output
}

/// The `$defs` of `schema`; none where it holds no object there.
fn definitions(schema: &Value) -> Map<String, Value> {
    schema.get("$defs").and_then(Value::as_object).cloned().unwrap_or_default()
}

/// The reference `schema`'s first local reference spells, or `None` where it
/// holds none.
fn first_local_ref(schema: &Value) -> Option<String> {
    let mut found = None;
    for_each_subschema(schema, &mut |node| {
        if found.is_none() {
            if let Some(reference) = node.get("$ref").and_then(Value::as_str) {
                if reference.starts_with('#') {
                    found = Some(reference.to_string());
                }
            }
        }
    });
    found
}

/// `schema` with its local references rebased to a copy of `root` embedded
/// under `__splash_root`, so the framed value is a schema of its own; `None`
/// where `schema` holds no local reference and needs no rebasing.
pub fn schema_with_root(schema: &Value, root: &Value) -> Option<Value> {
    first_local_ref(schema)?;
    let root_defs = definitions(root);
    let schema_defs = definitions(schema);
    let mut name = "__splash_root".to_string();
    while root_defs.contains_key(&name) || schema_defs.contains_key(&name) {
        name.push('_');
    }
    let prefix = format!("#/$defs/{name}");

    let rebase = |value: &Value| {
        let mut output = value.clone();
        for_each_subschema_mut(&mut output, &mut |node| {
            let reference = match node.get("$ref").and_then(Value::as_str) {
                Some(reference) if reference == "#" || reference.starts_with("#/") => {
                    reference.to_string()
                }
                _ => return,
            };
            if let Some(map) = node.as_object_mut() {
                map.insert(
                    "$ref".into(),
                    Value::String(format!("{prefix}{}", &reference[1..])),
                );
            }
        });
        output
    };

    let mut output = rebase(schema);
    let mut defs = definitions(&output);
    defs.insert(name, rebase(root));
    output["$defs"] = Value::Object(defs);
    Some(output)
}

/// The values a schema node lists, as an enum or a const; `None` if it
/// lists none, as an enum that is not a list does.
pub fn enumeration(node: &Value) -> Option<Vec<Value>> {
    if let Some(constant) = node.get("const") {
        return Some(vec![constant.clone()]);
    }
    node.get("enum").and_then(Value::as_array).cloned()
}

/// The JSON type of each kind of value the codec reads, by its name.
pub const JSON_TYPES: [&str; 7] = ["null", "boolean", "integer", "number", "string", "array", "object"];

/// The JSON type a schema's type name stands for, whatever its case and the
/// space around it: a JSON type by its own name or by a common alias. Any
/// other name stands for no JSON type.
fn spelled_type(kind: &str) -> Option<&'static str> {
    match kind.trim().to_ascii_lowercase().as_str() {
        "null" => Some("null"),
        "boolean" | "bool" => Some("boolean"),
        "integer" | "int" => Some("integer"),
        "number" | "float" => Some("number"),
        "string" | "str" | "text" => Some("string"),
        "array" | "list" => Some("array"),
        "object" | "dict" => Some("object"),
        _ => None,
    }
}

/// A schema's type with each name that stands for a JSON type spelled as
/// that type's name, and anything else as written.
fn spelled_types(kind: &Value) -> Value {
    match kind {
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|item| match item {
                    Value::String(text) => spelled_type(text)
                        .map(|name| Value::String(name.into()))
                        .unwrap_or_else(|| item.clone()),
                    other => other.clone(),
                })
                .collect(),
        ),
        Value::String(text) => spelled_type(text)
            .map(|name| Value::String(name.into()))
            .unwrap_or_else(|| kind.clone()),
        other => other.clone(),
    }
}

/// `schema` with its types spelled as the JSON types they stand for
/// ([`spelled_types`]), as framing, the types it gives a tool's values and a
/// strict tool's grammar read it: a copy where any is spelled otherwise,
/// else `schema` itself. Data a reference reaches, which is not a schema,
/// keeps its type names as written.
pub fn json_type_names(schema: &Value) -> Value {
    let mut spelled = true;
    for_each_subschema(schema, &mut |node| {
        if spelled && node.is_object() {
            if let Some(kind) = node.get("type") {
                if spelled_types(kind) != *kind {
                    spelled = false;
                }
            }
        }
    });
    if spelled {
        return schema.clone();
    }
    let mut output = schema.clone();
    for_each_subschema_mut(&mut output, &mut |node| {
        if let Some(map) = node.as_object_mut() {
            if let Some(kind) = map.get_mut("type") {
                *kind = spelled_types(kind);
            }
        }
    });
    output
}

/// The JSON type a type name names in a schema whose types are spelled
/// as JSON types ([`json_type_names`]); `None` where it names none, as
/// anything but a name does.
fn json_type(name: &Value) -> Option<&'static str> {
    let text = name.as_str()?;
    JSON_TYPES.iter().copied().find(|kind| *kind == text)
}

/// The JSON type of a value the codec reads, as [`JSON_TYPES`] names it.
pub fn json_type_of_value(value: &Value) -> Option<&'static str> {
    match value {
        Value::Null => Some("null"),
        Value::Bool(_) => Some("boolean"),
        Value::Number(number) => {
            if number.is_i64() || number.is_u64() {
                Some("integer")
            } else {
                Some("number")
            }
        }
        Value::String(_) => Some("string"),
        Value::Array(_) => Some("array"),
        Value::Object(_) => Some("object"),
    }
}

/// The types of a value both `types` and `other` admit; `None` admits any.
fn narrowed(
    types: Option<HashSet<String>>,
    other: Option<HashSet<String>>,
) -> Option<HashSet<String>> {
    if types.is_none() || other.is_none() {
        return if types.is_none() { other } else { types };
    }
    let types = types.expect("checked above");
    let other = other.expect("checked above");
    let mut both: HashSet<String> = types.intersection(&other).cloned().collect();
    if (types.contains("number") && other.contains("integer"))
        || (types.contains("integer") && other.contains("number"))
    {
        both.insert("integer".into());
    }
    Some(both)
}

/// The JSON types a value of `schema` may take, by which a parameter's
/// text converts; `None` where it constrains no type. Those its type names
/// name ([`json_type`]), unless one of them names none: such a type
/// constrains none, as framing reads it. Else those of its enum or const
/// values, else those its anyOf or oneOf members take together, narrowed by
/// its allOf members and the target of its local reference. A member that
/// constrains no type, such as one met again inside its own reading,
/// neither widens nor narrows the others'.
pub fn schema_types(schema: &Value, root: &Value) -> Option<HashSet<String>> {
    types_reader(root).read(schema)
}

/// What reads [`schema_types`] of schemas within `root`, each node once
/// whatever schema it is read for: the parameters of a tool read the
/// definitions they share once.
pub(crate) fn types_reader(root: &Value) -> TypeReader<'_> {
    TypeReader { root, memo: HashMap::new() }
}

/// The reader [`types_reader`] returns.
pub(crate) struct TypeReader<'a> {
    root: &'a Value,
    memo: HashMap<String, Option<HashSet<String>>>,
}

impl TypeReader<'_> {
    pub(crate) fn read(&mut self, node: &Value) -> Option<HashSet<String>> {
        let key = py_json_dumps_compact(node);
        if let Some(types) = self.memo.get(&key) {
            return types.clone();
        }
        // Each node is read once; a node met again inside its own reading
        // constrains nothing yet.
        self.memo.insert(key.clone(), None);
        let types = self.own(node);
        self.memo.insert(key, types.clone());
        types
    }

    fn own(&mut self, node: &Value) -> Option<HashSet<String>> {
        let Some(map) = node.as_object() else {
            // False admits no type; any other non-schema constrains none.
            return if node == &Value::Bool(false) {
                Some(HashSet::new())
            } else {
                None
            };
        };
        let kind = map.get("type");
        let values = enumeration(node);
        let members = map.get("anyOf").or_else(|| map.get("oneOf"));
        let mut result = if matches!(kind, Some(Value::String(_)) | Some(Value::Array(_))) {
            let kinds: Vec<Value> = match kind {
                Some(Value::Array(items)) => items.clone(),
                Some(other) => vec![other.clone()],
                None => unreachable!("checked above"),
            };
            let mut named = HashSet::new();
            let mut unnamed = false;
            for item in &kinds {
                match json_type(item) {
                    Some(name) => {
                        named.insert(name.to_string());
                    }
                    None => unnamed = true,
                }
            }
            if unnamed {
                None
            } else {
                Some(named)
            }
        } else if let Some(values) = &values {
            let mut named = HashSet::new();
            for value in values {
                if let Some(name) = json_type_of_value(value) {
                    named.insert(name.to_string());
                }
            }
            Some(named)
        } else if let Some(Value::Array(members)) = members {
            if members.is_empty() {
                None
            } else {
                let mut union: HashSet<String> = HashSet::new();
                let mut any = false;
                for member in members {
                    if let Some(types) = self.read(member) {
                        any = true;
                        union.extend(types);
                    }
                }
                any.then_some(union)
            }
        } else {
            None
        };
        let mut narrowing: Vec<Value> = map
            .get("allOf")
            .and_then(Value::as_array)
            .map(|items| items.to_vec())
            .unwrap_or_default();
        if map.contains_key("$ref") {
            if let Some(target) = referenced(node, self.root) {
                narrowing.push(target.clone());
            }
        }
        for member in &narrowing {
            result = narrowed(result, self.read(member));
        }
        result
    }
}

/// The values a parameter of `schema` takes when the schema lists them
/// all, as the grammar writes each in a call's value tags: a string as its
/// text, any other value as JSON. An enum or const lists its values, a
/// schema whose only type is null or boolean lists that type's, a union
/// lists its members' and a local reference its target's, narrowed by allOf
/// members that list theirs. `None` when the schema does not list them.
pub fn literal_texts(schema: &Value, root: &Value) -> Option<Vec<String>> {
    let mut reader = TextReader { root, memo: HashMap::new() };
    reader.read(schema)
}

struct TextReader<'a> {
    root: &'a Value,
    memo: HashMap<String, Option<Vec<String>>>,
}

impl TextReader<'_> {
    fn read(&mut self, node: &Value) -> Option<Vec<String>> {
        let key = py_json_dumps_compact(node);
        if let Some(texts) = self.memo.get(&key) {
            return texts.clone();
        }
        // Each node is read once, and a node met again inside its own
        // reading lists nothing.
        self.memo.insert(key.clone(), None);
        let texts = self.own(node);
        self.memo.insert(key, texts.clone());
        texts
    }

    fn own(&mut self, node: &Value) -> Option<Vec<String>> {
        let map = node.as_object()?;
        let values = enumeration(node);
        let kind = match map.get("type") {
            Some(Value::Array(items)) if items.len() == 1 => Some(&items[0]),
            other => other,
        };
        let members = map.get("anyOf").or_else(|| map.get("oneOf"));
        let mut result: Option<Vec<String>> = if let Some(values) = &values {
            let mut texts: Vec<String> = Vec::new();
            for value in values {
                let text = match value {
                    Value::String(text) => text.clone(),
                    other => py_json_dumps_compact(other),
                };
                if !texts.contains(&text) {
                    texts.push(text);
                }
            }
            Some(texts)
        } else if kind == Some(&Value::String("null".into())) {
            Some(vec!["null".into()])
        } else if kind == Some(&Value::String("boolean".into())) {
            Some(vec!["true".into(), "false".into()])
        } else if let Some(Value::Array(members)) = members {
            if members.is_empty() {
                None
            } else {
                let mut texts: Option<Vec<String>> = Some(Vec::new());
                for member in members {
                    match self.read(member) {
                        None => {
                            texts = None;
                            break;
                        }
                        Some(member_texts) => {
                            if let Some(kept) = &mut texts {
                                for text in member_texts {
                                    if !kept.contains(&text) {
                                        kept.push(text);
                                    }
                                }
                            }
                        }
                    }
                }
                texts
            }
        } else if map.contains_key("$ref") {
            match referenced(node, self.root) {
                Some(target) => self.read(target),
                None => None,
            }
        } else {
            None
        };
        if let Some(narrowing) = map.get("allOf").and_then(Value::as_array) {
            for member in narrowing {
                if let Some(member_texts) = self.read(member) {
                    result = match result {
                        None => Some(member_texts),
                        Some(kept) => Some(
                            kept.into_iter()
                                .filter(|text| member_texts.contains(text))
                                .collect(),
                        ),
                    };
                }
            }
        }
        result
    }
}

/// The schema a local reference of `node` names, or `None`.
pub fn referenced<'a>(node: &Value, root: &'a Value) -> Option<&'a Value> {
    let reference = node.get("$ref")?;
    lookup_tool_reference(reference, root).ok()
}

/// The schema the local reference `reference` names within `root`, or the
/// request's error: a reference that leaves the document, or one that names
/// nothing in it, cannot be read.
pub fn lookup_tool_reference<'a>(
    reference: &Value,
    root: &'a Value,
) -> Result<&'a Value, ApiError> {
    let Some(text) = reference.as_str() else {
        return Err(ApiError::new(
            400,
            "unsupported tool parameter reference",
            "invalid_request_error",
        ));
    };
    lookup_reference_text(text, root)
}

fn lookup_reference_text<'a>(reference: &str, root: &'a Value) -> Result<&'a Value, ApiError> {
    if !reference.starts_with('#') {
        return Err(ApiError::new(
            400,
            "unsupported tool parameter reference",
            "invalid_request_error",
        ));
    }
    let fragment = unquote(&reference[1..]);
    if fragment.is_empty() {
        return Ok(root);
    }
    if let Some(pointer) = fragment.strip_prefix('/') {
        let mut current = root;
        for part in pointer.split('/') {
            let part = part.replace("~1", "/").replace("~0", "~");
            if let Value::Array(items) = current {
                let indexed = part == "0"
                    || (part.starts_with(|c: char| c.is_ascii_digit() && c != '0')
                        && part.chars().all(|c| c.is_ascii_digit()));
                if indexed {
                    let index: usize = part.parse().unwrap_or(usize::MAX);
                    if index < items.len() {
                        current = &items[index];
                        continue;
                    }
                }
            } else if let Value::Object(map) = current {
                if let Some(child) = map.get(&part) {
                    current = child;
                    continue;
                }
            }
            return Err(ApiError::new(
                400,
                format!("unresolved tool parameter reference: {reference}"),
                "invalid_request_error",
            ));
        }
        return Ok(current);
    }
    find_anchor(root, &fragment).ok_or_else(|| {
        ApiError::new(
            400,
            format!("unresolved tool parameter reference: {reference}"),
            "invalid_request_error",
        )
    })
}

/// The schema `root`'s schema positions anchor as `fragment`, depth first,
/// or `None` where none does.
fn find_anchor<'a>(root: &'a Value, fragment: &str) -> Option<&'a Value> {
    if root.is_object()
        && (root.get("$anchor").and_then(Value::as_str) == Some(fragment)
            || root.get("$dynamicAnchor").and_then(Value::as_str) == Some(fragment))
    {
        return Some(root);
    }
    for child in schema_children(root) {
        if let Some(found) = find_anchor(child, fragment) {
            return Some(found);
        }
    }
    None
}

/// The text a URI fragment's percent-escapes spell, as `urllib.parse.unquote`
/// reads it: each escape run decoded as UTF-8, invalid bytes replaced.
fn unquote(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%'
            && index + 2 < bytes.len()
            && bytes[index + 1].is_ascii_hexdigit()
            && bytes[index + 2].is_ascii_hexdigit()
        {
            let mut run = Vec::new();
            while index + 2 < bytes.len()
                && bytes[index] == b'%'
                && bytes[index + 1].is_ascii_hexdigit()
                && bytes[index + 2].is_ascii_hexdigit()
            {
                let hex = std::str::from_utf8(&bytes[index + 1..index + 3]).unwrap_or("00");
                run.push(u8::from_str_radix(hex, 16).unwrap_or(0));
                index += 3;
            }
            out.extend(String::from_utf8_lossy(&run).bytes());
        } else {
            out.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The reference a schema leaves the document by, if any: the first `$ref`,
/// `$dynamicRef` or `$recursiveRef` that does not name this document, walked
/// over the schema positions of `root`. A `$schema` that is not a string,
/// or one of the reference keywords that is not, is refused.
pub fn remote_ref(schema: &Value) -> Result<Option<String>, ApiError> {
    let mut remote: Option<String> = None;
    let mut error: Option<ApiError> = None;
    for_each_subschema(schema, &mut |node| {
        if remote.is_some() || error.is_some() || !node.is_object() {
            return;
        }
        if let Some(dialect) = node.get("$schema") {
            if !dialect.is_string() {
                error = Some(ApiError::new(
                    400,
                    "$schema must be a string",
                    "invalid_request_error",
                ));
                return;
            }
        }
        for key in ["$ref", "$dynamicRef", "$recursiveRef"] {
            let Some(reference) = node.get(key) else {
                continue;
            };
            let Some(text) = reference.as_str() else {
                error = Some(ApiError::new(
                    400,
                    format!("{key} must be a string"),
                    "invalid_request_error",
                ));
                return;
            };
            if !text.starts_with('#') {
                remote = Some(text.to_string());
                return;
            }
        }
    });
    match error {
        Some(error) => Err(error),
        None => Ok(remote),
    }
}

/// Keywords whose values are maps from names to schemas.
pub const SCHEMA_MAP_KEYWORDS: [&str; 5] =
    ["properties", "patternProperties", "$defs", "definitions", "dependentSchemas"];
/// Keywords whose values are single schemas or lists of schemas.
/// ``dependencies`` holds a schema or a list of property names per entry and
/// is told apart by shape. Draft 3's ``type`` and ``disallow`` lists may hold
/// schemas beside type names.
pub const SUBSCHEMA_KEYWORDS: [&str; 17] = [
    "items",
    "prefixItems",
    "additionalItems",
    "contains",
    "additionalProperties",
    "unevaluatedItems",
    "unevaluatedProperties",
    "propertyNames",
    "allOf",
    "anyOf",
    "oneOf",
    "not",
    "if",
    "then",
    "else",
    "contentSchema",
    "extends",
];

/// `schema` and, depth first, every schema nested under it.
///
/// Only schema positions are visited, so property names and literal const,
/// enum, default and examples data are never mistaken for schemas.
pub fn for_each_subschema(schema: &Value, visit: &mut dyn FnMut(&Value)) {
    visit(schema);
    walk_subschemas(schema, visit);
}

fn walk_subschemas(schema: &Value, visit: &mut dyn FnMut(&Value)) {
    for child in schema_children(schema) {
        visit(child);
        walk_subschemas(child, visit);
    }
}

/// The schemas nested directly under `schema`, at its schema positions, in
/// keyword order.
fn schema_children(schema: &Value) -> Vec<&Value> {
    let Value::Object(map) = schema else {
        return Vec::new();
    };
    let mut children: Vec<&Value> = Vec::new();
    for (key, item) in map {
        let key = key.as_str();
        if SCHEMA_MAP_KEYWORDS.contains(&key) && item.is_object() {
            if let Value::Object(named) = item {
                children.extend(named.values());
            }
        } else if key == "dependencies" && item.is_object() {
            if let Value::Object(named) = item {
                children.extend(named.values().filter(|child| !child.is_array()));
            }
        } else if SUBSCHEMA_KEYWORDS.contains(&key) {
            match item {
                Value::Array(items) => children.extend(items.iter()),
                other => children.push(other),
            }
        } else if (key == "type" || key == "disallow") && item.is_array() {
            if let Value::Array(items) = item {
                children.extend(items.iter().filter(|child| child.is_object()));
            }
        }
    }
    children
}

/// The mutable counterpart of [`for_each_subschema`], for the readers that
/// rewrite schemas in place.
pub fn for_each_subschema_mut(schema: &mut Value, visit: &mut dyn FnMut(&mut Value)) {
    visit(schema);
    walk_subschemas_mut(schema, visit);
}

fn walk_subschemas_mut(schema: &mut Value, visit: &mut dyn FnMut(&mut Value)) {
    let Value::Object(map) = schema else {
        return;
    };
    for (key, item) in map.iter_mut() {
        let key = key.as_str().to_string();
        let key = key.as_str();
        if SCHEMA_MAP_KEYWORDS.contains(&key) && item.is_object() {
            if let Value::Object(children) = item {
                for (_, child) in children.iter_mut() {
                    visit(child);
                    walk_subschemas_mut(child, visit);
                }
            }
        } else if key == "dependencies" && item.is_object() {
            if let Value::Object(children) = item {
                for (_, child) in children.iter_mut() {
                    if !child.is_array() {
                        visit(child);
                        walk_subschemas_mut(child, visit);
                    }
                }
            }
        } else if SUBSCHEMA_KEYWORDS.contains(&key) {
            if let Value::Array(children) = item {
                for child in children.iter_mut() {
                    visit(child);
                    walk_subschemas_mut(child, visit);
                }
            } else {
                visit(item);
                walk_subschemas_mut(item, visit);
            }
        } else if (key == "type" || key == "disallow") && item.is_array() {
            if let Value::Array(children) = item {
                for child in children.iter_mut() {
                    if child.is_object() {
                        visit(child);
                        walk_subschemas_mut(child, visit);
                    }
                }
            }
        }
    }
}

/// Every object in a JSON document, walked without recursion.
pub fn for_each_object(value: &Value, visit: &mut dyn FnMut(&Value)) {
    let mut pending = vec![value];
    while let Some(value) = pending.pop() {
        match value {
            Value::Object(map) => {
                visit(value);
                pending.extend(map.values());
            }
            Value::Array(items) => pending.extend(items.iter()),
            _ => {}
        }
    }
}

/// Every object in a JSON document, mutable, walked depth first.
pub fn for_each_object_mut(value: &mut Value, visit: &mut dyn FnMut(&mut Value)) {
    if value.is_object() {
        visit(value);
        if let Value::Object(map) = value {
            for (_, child) in map.iter_mut() {
                for_each_object_mut(child, visit);
            }
        }
    } else if let Value::Array(items) = value {
        for child in items.iter_mut() {
            for_each_object_mut(child, visit);
        }
    }
}

/// The combination keyword's identity: the value a combination of none
/// returns, `true` for allOf and `false` for anyOf.
fn combination_identity(keyword: &str) -> bool {
    keyword == "allOf"
}

/// The schema the values `values` combine to under `keyword`: the identity
/// where none combine, `false` where one refuses, the one value where a
/// single one is left, else a schema of them. A definition reached through
/// several references combines once.
fn schema_combination(keyword: &str, values: Vec<Value>) -> Value {
    let identity = combination_identity(keyword);
    let mut unique: Vec<Value> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for value in values {
        if value == Value::Bool(identity) {
            continue;
        }
        let key = py_json_dumps_compact(&value);
        if seen.insert(key) {
            unique.push(value);
        }
    }
    if unique.is_empty() {
        return Value::Bool(identity);
    }
    if unique.iter().any(Value::is_boolean) {
        return Value::Bool(!identity);
    }
    if unique.len() == 1 {
        return unique.into_iter().next().expect("checked above");
    }
    let items = unique.into_iter().collect();
    let mut map = Map::new();
    map.insert(keyword.to_string(), Value::Array(items));
    Value::Object(map)
}

/// Approximate serialized size of `value`, counted no further than `limit`.
pub fn json_size(value: &Value, limit: i64) -> usize {
    enum Pending<'a> {
        Node(&'a Value),
        Key(&'a str),
    }
    let mut size = 0usize;
    let mut pending = vec![Pending::Node(value)];
    while !pending.is_empty() && (size as i64) <= limit {
        match pending.pop().expect("checked above") {
            Pending::Node(Value::Object(map)) => {
                size += 1;
                pending.extend(map.keys().map(|key| Pending::Key(key.as_str())));
                pending.extend(map.values().map(Pending::Node));
            }
            Pending::Node(Value::Array(items)) => {
                size += 1;
                pending.extend(items.iter().map(Pending::Node));
            }
            Pending::Node(Value::String(text)) => size += text.len(),
            Pending::Node(_) => size += 1,
            Pending::Key(key) => size += key.len(),
        }
    }
    size
}

/// Take `size` from what a framing budget has left, refusing what it
/// cannot hold.
pub fn charge(budget: &mut i64, size: usize) -> Result<(), ApiError> {
    *budget -= size as i64;
    if *budget < 0 {
        return Err(ApiError::new(
            400,
            "tool parameter schemas are too complex",
            "invalid_request_error",
        ));
    }
    Ok(())
}

/// The shape a tool's arguments frame to: its properties, the required
/// names and the additionalProperties, each value a node of `root` or a
/// combination of its nodes.
///
/// The projection keeps every value a field may take rather than choosing a
/// branch before the model has written the discriminator or dependent
/// properties, so assertions across fields go unenforced.
///
/// A keyword can hold anything: an older declared dialect leaves a newer one
/// unchecked, a reference can reach data that is not a schema, and the
/// schema of a tool that is not strict is not checked. A keyword that holds
/// another kind of value than it takes constrains nothing, as one its
/// dialect does not define, and so do a value that is not a schema where
/// one belongs, a type name that stands for no JSON type, such as Draft 3's
/// "any", and a reference to nothing in the document.
pub fn argument_shape(root: &Value, budget: &mut i64) -> Result<Value, ApiError> {
    let mut references: HashMap<String, Option<Value>> = HashMap::new();
    let shape = project(
        Some(root),
        &mut HashSet::new(),
        budget,
        &mut references,
        root,
    )?;
    shape.ok_or_else(|| {
        ApiError::new(
            400,
            "tool parameters must allow a top-level JSON object",
            "invalid_request_error",
        )
    })
}

/// The shape the fields of `shapes` frame to together: their properties
/// combined, their required names united or held in common, and their
/// additionalProperties combined. `union` holds what any one of them allows;
/// otherwise a shape that allows nothing refuses them all.
fn combine(
    shapes: Vec<Option<Value>>,
    union: bool,
    budget: &mut i64,
) -> Result<Option<Value>, ApiError> {
    let keyword = if union { "anyOf" } else { "allOf" };
    let shapes: Vec<Value> = if union {
        let kept: Vec<Value> = shapes.into_iter().flatten().collect();
        if kept.is_empty() {
            return Ok(None);
        }
        kept
    } else {
        if shapes.iter().any(Option::is_none) {
            return Ok(None);
        }
        shapes.into_iter().flatten().collect()
    };
    if shapes.is_empty() {
        return Ok(Some(shape_of(Map::new(), Vec::new(), Value::Bool(true))));
    }
    let mut names: Vec<String> = Vec::new();
    for shape in &shapes {
        for name in shape
            .get("properties")
            .and_then(Value::as_object)
            .into_iter()
            .flat_map(|properties| properties.keys())
        {
            if !names.contains(name) {
                names.push(name.clone());
            }
        }
    }
    charge(budget, names.len() * shapes.len())?;
    let shape_required = |shape: &Value| {
        shape
            .get("required")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    };
    let mut required = shape_required(shapes.first().expect("checked above"));
    for shape in &shapes[1..] {
        let other: HashSet<String> = shape_required(shape).into_iter().collect();
        if union {
            required.retain(|name| other.contains(name));
        } else {
            for name in other {
                if !required.contains(&name) {
                    required.push(name);
                }
            }
        }
    }
    let properties: Map<String, Value> = names
        .into_iter()
        .map(|name| {
            let values = shapes
                .iter()
                .map(|shape| {
                    shape
                        .get("properties")
                        .and_then(Value::as_object)
                        .and_then(|properties| properties.get(&name))
                        .cloned()
                        .unwrap_or_else(|| {
                            shape
                                .get("additionalProperties")
                                .cloned()
                                .unwrap_or(Value::Bool(true))
                        })
                })
                .collect();
            (name, schema_combination(keyword, values))
        })
        .collect();
    let additional = schema_combination(
        keyword,
        shapes
            .iter()
            .map(|shape| {
                shape
                    .get("additionalProperties")
                    .cloned()
                    .unwrap_or(Value::Bool(true))
            })
            .collect(),
    );
    required.sort();
    Ok(Some(shape_of(properties, required, additional)))
}

fn shape_of(properties: Map<String, Value>, required: Vec<String>, additional: Value) -> Value {
    let mut map = Map::new();
    map.insert("properties".into(), Value::Object(properties));
    map.insert(
        "required".into(),
        Value::Array(required.into_iter().map(Value::String).collect()),
    );
    map.insert("additionalProperties".into(), additional);
    Value::Object(map)
}

fn project(
    node: Option<&Value>,
    visiting: &mut HashSet<String>,
    budget: &mut i64,
    references: &mut HashMap<String, Option<Value>>,
    root: &Value,
) -> Result<Option<Value>, ApiError> {
    // False allows no object; None, as any value that is not a schema,
    // constrains nothing.
    let empty_map = Map::new();
    let empty_value = Value::Object(Map::new());
    let node = match node {
        None => &empty_value,
        Some(node) if node == &Value::Bool(false) => return Ok(None),
        Some(node) => node,
    };
    let map: &Map<String, Value> = node.as_object().unwrap_or(&empty_map);
    let kind = map
        .get("type")
        .filter(|kind| kind.is_string() || kind.is_array())
        .cloned()
        .unwrap_or_else(|| Value::String("object".into()));
    let names: Vec<Value> = match &kind {
        Value::Array(items) => items.clone(),
        other => vec![other.clone()],
    };
    // A type allows no object only where it names JSON types alone, none
    // of them object.
    if names.iter().all(|name| {
        matches!(
            json_type(name),
            Some(spelled) if spelled != "object"
        )
    }) {
        return Ok(None);
    }
    let mut properties: Map<String, Value> = keyword_object(node, "properties").cloned().unwrap_or_default();
    let mut additional = map
        .get("additionalProperties")
        .cloned()
        .unwrap_or(Value::Bool(true));
    let patterns: Vec<Value> = keyword_object(node, "patternProperties")
        .map(|patterns| patterns.values().cloned().collect())
        .unwrap_or_default();
    if !patterns.is_empty() {
        let mut values = vec![additional];
        values.extend(patterns);
        additional = schema_combination("anyOf", values);
    }
    let required: Vec<String> = keyword_array(node, "required")
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    for name in &required {
        properties.entry(name.clone()).or_insert_with(|| additional.clone());
    }
    let mut shapes = vec![Some(shape_of(properties, required, additional))];
    if let Some(reference) = keyword_string(node, "$ref") {
        if visiting.contains(reference) {
            return Err(ApiError::new(
                400,
                "cyclic direct tool argument reference",
                "invalid_request_error",
            ));
        }
        if !references.contains_key(reference) {
            let mut inner = visiting.clone();
            inner.insert(reference.to_string());
            let target = referenced(node, root);
            let projected = project(target, &mut inner, budget, references, root)?;
            references.insert(reference.to_string(), projected);
        }
        shapes.push(references.get(reference).cloned().flatten());
    }
    if let Some(children) = keyword_array(node, "allOf") {
        for child in children {
            shapes.push(project(
                Some(child),
                &mut visiting.clone(),
                budget,
                references,
                root,
            )?);
        }
    }
    for keyword in ["anyOf", "oneOf"] {
        if let Some(children) = keyword_array(node, keyword) {
            let mut members = Vec::new();
            for child in children {
                members.push(project(
                    Some(child),
                    &mut visiting.clone(),
                    budget,
                    references,
                    root,
                )?);
            }
            shapes.push(combine(members, true, budget)?);
        }
    }
    if map.contains_key("if") {
        let then = project(
            map.get("then").or(Some(&empty_value)),
            &mut visiting.clone(),
            budget,
            references,
            root,
        )?;
        let otherwise = project(
            map.get("else").or(Some(&empty_value)),
            &mut visiting.clone(),
            budget,
            references,
            root,
        )?;
        shapes.push(combine(vec![then, otherwise], true, budget)?);
    }
    if let Some(children) = keyword_object(node, "dependentSchemas") {
        for child in children.values() {
            let projected = project(
                Some(child),
                &mut visiting.clone(),
                budget,
                references,
                root,
            )?;
            let unconstrained = project(
                Some(&empty_value),
                &mut visiting.clone(),
                budget,
                references,
                root,
            )?;
            shapes.push(combine(vec![projected, unconstrained], true, budget)?);
        }
    }
    if let Some(choices) = enumeration(node) {
        let objects: Vec<&Value> = choices.iter().filter(|value| value.is_object()).collect();
        if objects.is_empty() {
            return Ok(None);
        }
        let members = objects
            .into_iter()
            .map(|object| {
                let mut properties = Map::new();
                let mut required = Vec::new();
                if let Value::Object(fields) = object {
                    for (name, value) in fields {
                        let mut constant = Map::new();
                        constant.insert("const".into(), value.clone());
                        properties.insert(name.clone(), Value::Object(constant));
                        required.push(name.clone());
                    }
                }
                shape_of(properties, required, Value::Bool(false))
            })
            .map(Some)
            .collect();
        shapes.push(combine(members, true, budget)?);
    }
    combine(shapes, false, budget)
}

/// The value of keyword `key` of `node`, which framing reads as an object;
/// `None`, as if it were absent, where it holds another kind.
fn keyword_object<'a>(node: &'a Value, key: &str) -> Option<&'a Map<String, Value>> {
    node.get(key).and_then(Value::as_object)
}

/// The value of keyword `key` of `node`, which framing reads as a list;
/// `None`, as if it were absent, where it holds another kind.
fn keyword_array<'a>(node: &'a Value, key: &str) -> Option<&'a Vec<Value>> {
    node.get(key).and_then(Value::as_array)
}

/// The value of keyword `key` of `node`, which framing reads as a string;
/// `None`, as if it were absent, where it holds another kind.
fn keyword_string<'a>(node: &'a Value, key: &str) -> Option<&'a str> {
    node.get(key).and_then(Value::as_str)
}

/// A strict tool's arguments as its grammar generates them: their
/// projection ([`argument_shape`]), each value made a schema of its own,
/// with a copy of the root schema where it refers to the root.
pub fn tool_argument_schema(root: &Value, budget: &mut i64) -> Result<Value, ApiError> {
    let framed = |value: &Value, budget: &mut i64| -> Result<Value, ApiError> {
        // Shared definitions repeat in the serialized grammar; count them all.
        charge(budget, json_size(value, *budget))?;
        match schema_with_root(value, root) {
            None => Ok(value.clone()),
            Some(framed_value) => {
                charge(budget, json_size(root, *budget))?;
                Ok(framed_value)
            }
        }
    };
    let mut shape = argument_shape(root, budget)?;
    if let Some(map) = shape.as_object_mut() {
        map.insert("type".into(), Value::String("object".into()));
        if let Some(Value::Object(properties)) = map.get_mut("properties") {
            for (_, value) in properties.iter_mut() {
                *value = framed(value, budget)?;
            }
        }
        let additional = map
            .get("additionalProperties")
            .cloned()
            .unwrap_or(Value::Bool(true));
        map.insert("additionalProperties".into(), framed(&additional, budget)?);
    }
    Ok(shape)
}
