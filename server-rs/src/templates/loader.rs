//! The chat template files' loading and selection, ported from
//! `server/fast_tokenizer.py`: `chat_template.jinja` as the default,
//! `additional_chat_templates/*.jinja` as named templates beside it, else
//! the legacy `chat_template` entry of `tokenizer_config.json`; a request
//! renders by its own source, else the named template its name or the
//! tools pick, else the default.

use std::path::Path;

use serde_json::Value;

use super::render::ChatTemplateError;

/// The tokenizer's chat templates: `chat_template.jinja` as the default,
/// `additional_chat_templates/*.jinja` as named templates beside it, else
/// the legacy `chat_template` entry of `tokenizer_config.json`. Files take
/// priority over the config, as transformers reads them.
pub fn load_chat_templates(directory: &Path) -> Result<Templates, ChatTemplateError> {
    let mut templates: Vec<(Option<String>, String)> = Vec::new();
    let default = directory.join("chat_template.jinja");
    if default.is_file() {
        templates.push((Some("default".to_string()), std::fs::read_to_string(&default).map_err(|error| {
            ChatTemplateError(format!("{}: {error}", default.display()))
        })?));
    }
    let extra = directory.join("additional_chat_templates");
    if extra.is_dir() {
        let mut names: Vec<_> = std::fs::read_dir(&extra)
            .map_err(|error| ChatTemplateError(format!("{}: {error}", extra.display())))?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.extension().map(|ext| ext == "jinja").unwrap_or(false))
            .collect();
        names.sort();
        for path in names {
            let name = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .expect("jinja stem")
                .to_string();
            templates.push((
                Some(name),
                std::fs::read_to_string(&path).map_err(|error| {
                    ChatTemplateError(format!("{}: {error}", path.display()))
                })?,
            ));
        }
    }
    if templates.len() == 1 && templates[0].0.is_none() {
        return Ok(Templates::Single(templates.remove(0).1));
    }
    if !templates.is_empty() {
        return Ok(Templates::Named(templates));
    }
    let config: Value = serde_json::from_str(
        &std::fs::read_to_string(directory.join("tokenizer_config.json")).unwrap_or_default(),
    )
    .unwrap_or(Value::Null);
    match config.get("chat_template") {
        Some(Value::String(text)) => Ok(Templates::Single(text.clone())),
        Some(Value::Array(entries)) => {
            let named = entries
                .iter()
                .filter_map(|entry| {
                    let name = entry.get("name")?.as_str()?.to_string();
                    let template = entry.get("template")?.as_str()?.to_string();
                    Some((Some(name), template))
                })
                .collect();
            Ok(Templates::Named(named))
        }
        _ => Ok(Templates::None),
    }
}

/// The tokenizer's templates: one, or named ones with a default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Templates {
    None,
    Single(String),
    Named(Vec<(Option<String>, String)>),
}

impl Templates {
    /// Whether any template is defined.
    pub fn is_empty(&self) -> bool {
        matches!(self, Templates::None)
    }

    /// The template a request renders by: its own source, else the named
    /// template its name or the tools pick, else the default — as
    /// transformers' get_chat_template chooses.
    pub fn select(
        &self,
        requested: Option<&str>,
        tools: Option<&serde_json::Value>,
    ) -> Result<String, ChatTemplateError> {
        if let Some(requested) = requested {
            if let Templates::Named(named) = self {
                if let Some((_, source)) = named.iter().find(|(name, _)| name.as_deref() == Some(requested)) {
                    return Ok(source.clone());
                }
            }
            return Ok(requested.to_string());
        }
        match self {
            Templates::Single(source) => Ok(source.clone()),
            Templates::Named(named) => {
                if tools.is_some() {
                    if let Some((_, source)) = named.iter().find(|(name, _)| name.as_deref() == Some("tool_use")) {
                        return Ok(source.clone());
                    }
                }
                if let Some((_, source)) = named.iter().find(|(name, _)| name.as_deref() == Some("default")) {
                    return Ok(source.clone());
                }
                Err(ChatTemplateError(
                    "the model has multiple chat templates with no default".into(),
                ))
            }
            Templates::None => Err(ChatTemplateError(
                "the model's tokenizer defines no chat template".into(),
            )),
        }
    }
}
