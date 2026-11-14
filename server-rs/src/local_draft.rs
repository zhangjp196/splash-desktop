//! File selection for a local DFlash2 checkpoint.

use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

pub fn files(root: &Path) -> Result<Vec<(String, PathBuf)>, String> {
    let config = root.join("config.json");
    if !config.is_file() {
        return Err(format!("DFlash2 draft has no config.json: {}", config.display()));
    }
    let mut names = BTreeSet::new();
    let index = root.join("model.safetensors.index.json");
    if index.is_file() {
        let value: Value = serde_json::from_slice(
            &fs::read(&index).map_err(|error| format!("cannot read {}: {error}", index.display()))?,
        )
        .map_err(|error| format!("invalid DFlash2 shard index: {error}"))?;
        let weight_map = value
            .get("weight_map")
            .and_then(Value::as_object)
            .filter(|map| !map.is_empty())
            .ok_or("invalid DFlash2 shard index: empty weight_map")?;
        for filename in weight_map.values() {
            let filename = filename
                .as_str()
                .ok_or("invalid DFlash2 shard index: non-string filename")?;
            if filename.contains('/') || !filename.ends_with(".safetensors") {
                return Err(format!("unsupported DFlash2 shard filename: {filename}"));
            }
            names.insert(filename.to_owned());
        }
        names.insert("model.safetensors.index.json".to_owned());
    } else if root.join("model.safetensors").is_file() {
        names.insert("model.safetensors".to_owned());
    } else {
        return Err(format!("DFlash2 draft has no safetensors weights: {}", root.display()));
    }
    if root.join("mask_embedding.safetensors").is_file() {
        names.insert("mask_embedding.safetensors".to_owned());
    }
    let mut files = vec![("draft/config.json".to_owned(), config)];
    for name in names {
        let path = root.join(&name);
        if !path.is_file() {
            return Err(format!("DFlash2 shard is missing: {}", path.display()));
        }
        files.push((format!("draft/{name}"), path));
    }
    Ok(files)
}
