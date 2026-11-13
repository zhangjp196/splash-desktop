//! The rendering environment, ported from `server/fast_tokenizer.py`'s
//! `apply_chat_template`: the same whitespace behavior, the same
//! `tojson` filter, the same globals, and the `{% generation %}` block
//! rendered as a plain scope.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use minijinja::value::{Enumerator, Object, ObjectRepr, Value, ValueKind};
use minijinja::{Environment, Error as MiniError, ErrorKind};

/// A template that cannot render what the request asked for, as Jinja's
/// TemplateError carries: the caller retries with its own fallbacks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTemplateError(pub String);

impl fmt::Display for ChatTemplateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for ChatTemplateError {}

impl From<MiniError> for ChatTemplateError {
    fn from(error: MiniError) -> ChatTemplateError {
        ChatTemplateError(error.to_string())
    }
}

/// A JSON object that keeps its keys in the order it was built with, as
/// Python's dicts do and minijinja's own maps do not: a chat template
/// renders a tool's schema as JSON, and the model reads the key order it
/// was trained on.
#[derive(Debug)]
struct OrderedObject {
    pairs: Vec<(String, Value)>,
}

impl Object for OrderedObject {
    fn repr(self: &Arc<Self>) -> ObjectRepr {
        ObjectRepr::Map
    }

    fn get_value(self: &Arc<Self>, key: &Value) -> Option<Value> {
        let key = key.as_str()?;
        self.pairs
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value.clone())
    }

    fn enumerate(self: &Arc<Self>) -> Enumerator {
        Enumerator::Iter(Box::new(
            self.pairs
                .iter()
                .map(|(name, _)| Value::from(name.clone()))
                .collect::<Vec<_>>()
                .into_iter(),
        ))
    }
}

/// The value a JSON document becomes, keeping objects' key order.
pub fn to_template_value(value: &serde_json::Value) -> Value {
    match value {
        serde_json::Value::Null => Value::from(()),
        serde_json::Value::Bool(flag) => Value::from(*flag),
        serde_json::Value::Number(number) => {
            if let Some(int) = number.as_i64() {
                Value::from(int)
            } else if let Some(uint) = number.as_u64() {
                Value::from(uint)
            } else {
                Value::from(number.as_f64().unwrap_or(0.0))
            }
        }
        serde_json::Value::String(text) => Value::from(text.clone()),
        serde_json::Value::Array(items) => {
            Value::from(items.iter().map(to_template_value).collect::<Vec<_>>())
        }
        serde_json::Value::Object(map) => Value::from_object(OrderedObject {
            pairs: map
                .iter()
                .map(|(name, value)| (name.clone(), to_template_value(value)))
                .collect(),
        }),
    }
}

/// Write a JSON document as Python's `json.dumps` writes it: raw UTF-8,
/// the control characters escaped, maps in their own order, `, ` and `: `
/// between the parts.
fn write_json(value: &serde_json::Value, out: &mut String) {
    match value {
        serde_json::Value::Null => out.push_str("null"),
        serde_json::Value::Bool(flag) => {
            out.push_str(if *flag { "true" } else { "false" });
        }
        serde_json::Value::Number(number) => {
            if let Some(int) = number.as_i64() {
                out.push_str(&int.to_string());
            } else if let Some(uint) = number.as_u64() {
                out.push_str(&uint.to_string());
            } else {
                let float = number.as_f64().unwrap_or(0.0);
                // Python's repr: the shortest round-tripping form. Rust's
                // Display is the same shortest form; the exponent's shape
                // differs for very large and very small magnitudes, which
                // a tool schema's numbers do not reach.
                out.push_str(&format!("{float}"));
            }
        }
        serde_json::Value::String(text) => {
            // serde_json's string escaping is the JSON one: the quote, the
            // backslash and the control characters escaped, everything
            // else raw UTF-8.
            let escaped = serde_json::to_string(text).unwrap_or_default();
            out.push_str(&escaped);
        }
        serde_json::Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_json(item, out);
            }
            out.push(']');
        }
        serde_json::Value::Object(map) => {
            out.push('{');
            for (index, (name, item)) in map.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                let escaped = serde_json::to_string(name).unwrap_or_default();
                out.push_str(&escaped);
                out.push_str(": ");
                write_json(item, out);
            }
            out.push('}');
        }
    }
}

/// The `tojson` filter, as transformers' templates read it: Python's
/// `json.dumps` with `ensure_ascii=False` and no indent. The value
/// serializes through serde, which keeps the ordered objects' order.
fn tojson(value: Value) -> Result<String, MiniError> {
    let json = serde_json::to_value(&value)
        .map_err(|error| MiniError::new(ErrorKind::InvalidOperation, error.to_string()))?;
    let mut out = String::new();
    write_json(&json, &mut out);
    Ok(out)
}

/// The `{% generation %}` and `{% endgeneration %}` tags, which bound the
/// tokens the model itself generates. The string path renders the block's
/// body as a plain scope, so the tags come out and the body stays.
fn strip_generation_tags(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let bytes = source.as_bytes();
    let mut position = 0usize;
    while position < bytes.len() {
        match source[position..].find("{%") {
            Some(offset) if position + offset + 2 <= bytes.len() => {
                let tag_start = position + offset;
                let tag_end = match source[tag_start..].find("%}") {
                    Some(end) => tag_start + end + 2,
                    None => {
                        out.push_str(&source[position..]);
                        return out;
                    }
                };
                let tag = source[tag_start..tag_end].trim();
                let inner = tag
                    .trim_start_matches("{%")
                    .trim_end_matches("%}")
                    .trim()
                    .trim_start_matches('-')
                    .trim_end_matches('-')
                    .trim();
                if inner == "generation" || inner == "endgeneration" {
                    out.push_str(&source[position..tag_start]);
                } else {
                    out.push_str(&source[position..tag_end]);
                }
                position = tag_end;
            }
            _ => {
                out.push_str(&source[position..]);
                break;
            }
        }
    }
    out
}

/// One template's renderer: the environment built once, rendered per
/// request.
pub struct ChatTemplateRenderer {
    environment: Environment<'static>,
}

impl ChatTemplateRenderer {
    /// Build the environment for one template source: transformers'
    /// whitespace behavior, the `tojson` filter, the `raise_exception` and
    /// `strftime_now` globals.
    pub fn new(source: &str) -> Result<ChatTemplateRenderer, ChatTemplateError> {
        let mut environment = Environment::new();
        environment.set_syntax(
            minijinja::syntax::SyntaxConfig::builder()
                .trim_blocks(true)
                .lstrip_blocks(true)
                .build()
                .map_err(|error| ChatTemplateError(error.to_string()))?,
        );
        environment.add_filter("tojson", tojson);
        environment.set_unknown_method_callback(python_string_method);
        environment.add_global("raise_exception", minijinja::Value::from_function(raise_exception));
        environment.add_global("strftime_now", minijinja::Value::from_function(strftime_now));
        let stripped: &'static str = strip_generation_tags(source).leak();
        environment
            .add_template("chat", stripped)
            .map_err(|error| ChatTemplateError(error.to_string()))?;
        Ok(ChatTemplateRenderer { environment })
    }

    /// Render the template for one request: the messages, the tool
    /// schemas, the documents, the generation prompt's switch, and the
    /// request's own template variables.
    pub fn render(
        &self,
        messages: &serde_json::Value,
        tools: Option<&serde_json::Value>,
        documents: Option<&serde_json::Value>,
        add_generation_prompt: bool,
        kwargs: &BTreeMap<String, serde_json::Value>,
    ) -> Result<String, ChatTemplateError> {
        let template = self
            .environment
            .get_template("chat")
            .expect("the template was added");
        let mut context = BTreeMap::new();
        context.insert("messages".to_string(), to_template_value(messages));
        context.insert(
            "tools".to_string(),
            match tools {
                Some(tools) => to_template_value(tools),
                None => Value::UNDEFINED,
            },
        );
        context.insert(
            "documents".to_string(),
            match documents {
                Some(documents) => to_template_value(documents),
                None => Value::UNDEFINED,
            },
        );
        context.insert(
            "add_generation_prompt".to_string(),
            Value::from(add_generation_prompt),
        );
        for (name, value) in kwargs {
            context.insert(name.clone(), to_template_value(value));
        }
        template
            .render(Value::from_object(ContextObject {
                entries: context,
            }))
            .map_err(|error| ChatTemplateError(error.to_string()))
    }
}

/// The render context as an object, so its keys read like Jinja's globals.
#[derive(Debug)]
struct ContextObject {
    entries: BTreeMap<String, Value>,
}

impl Object for ContextObject {
    fn repr(self: &Arc<Self>) -> ObjectRepr {
        ObjectRepr::Map
    }

    fn get_value(self: &Arc<Self>, key: &Value) -> Option<Value> {
        let key = key.as_str()?;
        self.entries.get(key).cloned()
    }

    fn enumerate(self: &Arc<Self>) -> Enumerator {
        Enumerator::Str(
            self.entries
                .keys()
                .map(|key| Box::leak(key.clone().into_boxed_str()) as &'static str)
                .collect::<Vec<&'static str>>()
                .leak(),
        )
    }
}

/// The Python string methods templates call, which Jinja exposes on
/// Python's `str` and minijinja's strings do not carry. The ones the
/// served families' templates use, plus the common neighbors; anything
/// else stays an unknown method.
fn python_string_method(
    _state: &mut minijinja::State,
    value: &Value,
    method: &str,
    args: &[Value],
) -> Result<Value, MiniError> {
    if value.kind() != ValueKind::String {
        return Err(MiniError::from(ErrorKind::UnknownMethod));
    }
    let text = value.as_str().unwrap_or_default();
    let arg_str = |index: usize| -> Option<String> {
        args.get(index).and_then(|arg| arg.as_str()).map(String::from)
    };
    let result = match method {
        "startswith" => {
            let prefix = arg_str(0).ok_or_else(|| {
                MiniError::new(ErrorKind::MissingArgument, "startswith takes a prefix")
            })?;
            Value::from(text.starts_with(&prefix))
        }
        "endswith" => {
            let suffix = arg_str(0).ok_or_else(|| {
                MiniError::new(ErrorKind::MissingArgument, "endswith takes a suffix")
            })?;
            Value::from(text.ends_with(&suffix))
        }
        "upper" => Value::from(text.to_uppercase()),
        "lower" => Value::from(text.to_lowercase()),
        "strip" => Value::from(text.trim()),
        "lstrip" => Value::from(text.trim_start()),
        "rstrip" => Value::from(text.trim_end()),
        "replace" => {
            let old = arg_str(0).ok_or_else(|| {
                MiniError::new(ErrorKind::MissingArgument, "replace takes old and new")
            })?;
            let new = arg_str(1).unwrap_or_default();
            Value::from(text.replace(&old, &new))
        }
        "find" => {
            let needle = arg_str(0).unwrap_or_default();
            Value::from(text.find(&needle).map(|index| index as i64).unwrap_or(-1i64))
        }
        "count" => {
            let needle = arg_str(0).unwrap_or_default();
            Value::from(text.matches(&needle).count() as i64)
        }
        "join" => {
            let parts = args
                .first()
                .and_then(|value| value.try_iter().ok())
                .map(|iter| {
                    iter.filter_map(|part| part.as_str().map(String::from))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            Value::from(parts.join(&text))
        }
        "split" => {
            let needle = arg_str(0);
            let parts: Vec<String> = match needle {
                Some(needle) if !needle.is_empty() => {
                    text.split(&needle).map(String::from).collect()
                }
                _ => text.split_whitespace().map(String::from).collect(),
            };
            Value::from(parts)
        }
        "title" => Value::from(
            text.split_whitespace()
                .map(|word| {
                    let mut characters = word.chars();
                    match characters.next() {
                        Some(first) => first.to_uppercase().collect::<String>()
                            + &characters.as_str().to_lowercase(),
                        None => String::new(),
                    }
                })
                .collect::<Vec<_>>()
                .join(" "),
        ),
        "capitalize" => {
            let mut characters = text.chars();
            match characters.next() {
                Some(first) => Value::from(
                    first.to_uppercase().collect::<String>()
                        + &characters.as_str().to_lowercase(),
                ),
                None => Value::from(""),
            }
        }
        "isdigit" => Value::from(!text.is_empty() && text.chars().all(|c| c.is_ascii_digit())),
        "isalpha" => Value::from(!text.is_empty() && text.chars().all(|c| c.is_alphabetic())),
        "isupper" => {
            Value::from(text.chars().any(|c| c.is_uppercase()) && !text.chars().any(|c| c.is_lowercase()))
        }
        "islower" => {
            Value::from(text.chars().any(|c| c.is_lowercase()) && !text.chars().any(|c| c.is_uppercase()))
        }
        _ => return Err(MiniError::from(ErrorKind::UnknownMethod)),
    };
    Ok(result)
}

/// The templates' `raise_exception`, which fails the render with the
/// message the template names.
fn raise_exception(message: String) -> Result<Value, MiniError> {
    Err(MiniError::new(ErrorKind::InvalidOperation, message))
}

/// The templates' `strftime_now`, the local time as `format` spells it.
fn strftime_now(format: String) -> String {
    chrono::Local::now().format(&format).to_string()
}
