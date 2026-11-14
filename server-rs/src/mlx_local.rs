//! Offline preparation of a local MLX-affine checkpoint.
//!
//! The initial native slice deliberately supports the common single-file
//! `model.safetensors` layout. Large weights are linked into the assembly;
//! they are never copied.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

const MAX_HEADER_BYTES: u64 = 100_000_000;
const TOKENIZER_FILES: &[&str] = &[
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
];
const VISION_PREFIXES: &[&str] = &["vision_tower.", "model.visual."];

pub fn prepare_local(
    target_dir: &Path,
    models_root: &Path,
    model_id: &str,
    engine: &Path,
    language_only: bool,
) -> Result<PathBuf, String> {
    prepare_local_with_draft(target_dir, None, models_root, model_id, engine, language_only)
}

pub fn prepare_local_with_draft(
    target_dir: &Path,
    draft_dir: Option<&Path>,
    models_root: &Path,
    model_id: &str,
    engine: &Path,
    language_only: bool,
) -> Result<PathBuf, String> {
    if model_id.trim().is_empty() {
        return Err("model ID must not be empty".into());
    }
    let target_dir = fs::canonicalize(target_dir)
        .map_err(|error| format!("cannot resolve local MLX directory {}: {error}", target_dir.display()))?;
    if !target_dir.is_dir() {
        return Err("local MLX target must be a directory".into());
    }

    let config_path = target_dir.join("config.json");
    let tokenizer_path = target_dir.join("tokenizer.json");
    let tokenizer_config_path = target_dir.join("tokenizer_config.json");
    for path in [&config_path, &tokenizer_path, &tokenizer_config_path] {
        if !path.is_file() {
            return Err(format!("local MLX model is missing required file: {}", path.file_name().unwrap().to_string_lossy()));
        }
    }
    let config: Value = read_json(&config_path, "model config")?;
    let has_vision_config = config.get("vision_config").is_some_and(Value::is_object);
    let weights_path = target_dir.join("model.safetensors");
    if !weights_path.is_file() {
        if target_dir.join("model.safetensors.index.json").is_file() {
            return Err("sharded MLX safetensors indexes are not supported by native local preparation yet".into());
        }
        return Err("local MLX model is missing model.safetensors".into());
    }
    let tensor_names = safetensors_tensor_names(&weights_path)?;

    let vision_format = if has_vision_config && !language_only {
        let processor = image_processor(&target_dir)?;
        validate_processor(&processor)?;
        if !tensor_names.iter().any(|name| VISION_PREFIXES.iter().any(|prefix| name.starts_with(prefix))) {
            return Err("local MLX config describes vision, but model.safetensors has no vision tower tensors".into());
        }
        "safetensors"
    } else {
        "none"
    };
    let draft_root = draft_dir.map(fs::canonicalize).transpose()
        .map_err(|error| format!("cannot resolve DFlash2 draft: {error}"))?;
    let draft_config = draft_root.as_ref().map(|root| root.join("config.json"));
    if let Some(config) = &draft_config {
        if !config.is_file() {
            return Err(format!("DFlash2 draft has no config.json: {}", config.display()));
        }
    }
    let family = model_check(engine, vision_format, &config_path, draft_config.as_deref())?;

    let mut files = BTreeMap::new();
    files.insert("config.json".to_owned(), config_path.clone());
    files.insert("target/config.json".to_owned(), config_path.clone());
    files.insert("tokenizer/config.json".to_owned(), config_path.clone());
    files.insert("target/model.safetensors".to_owned(), weights_path.clone());
    for name in TOKENIZER_FILES {
        let path = target_dir.join(name);
        if path.is_file() {
            files.insert(format!("tokenizer/{name}"), path);
        }
    }
    if vision_format == "safetensors" {
        files.insert("vision/config.json".to_owned(), config_path.clone());
        // Python's assembler links mixed text+vision shards in both locations.
        // A single checkpoint file may contain both groups.
        files.insert("vision/model.safetensors".to_owned(), weights_path);
    }
    let record = if let Some(draft_root) = draft_root {
        for (name, path) in crate::local_draft::files(&draft_root)? {
            files.insert(name, path);
        }
        serde_json::json!({
            "version": 1,
            "model": model_id,
            "family": family,
            "target_format": "mlx-affine",
            "vision_format": vision_format,
            "sources": {
                "target": {"repo": target_dir.to_string_lossy(), "revision": null},
                "draft": {"repo": draft_root.to_string_lossy(), "revision": null}
            }
        })
    } else {
        serde_json::json!({
            "version": 2,
            "model": model_id,
            "family": family,
            "target_format": "mlx-affine",
            "vision_format": vision_format,
            "sources": {"target": {"repo": target_dir.to_string_lossy(), "revision": null}},
            "speculation": {
                "version": 1,
                "provider": "none",
                "format": "none",
                "source": {"kind": "none"},
                "proposal_rows": 1,
                "state": "none",
                "fallback": "none"
            }
        })
    };
    crate::assembly::publish(models_root, record, &files)
}

fn read_json(path: &Path, description: &str) -> Result<Value, String> {
    let bytes = fs::read(path).map_err(|error| format!("cannot read {description}: {error}"))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("invalid {description}: {error}"))
}

fn safetensors_tensor_names(path: &Path) -> Result<Vec<String>, String> {
    let mut file = File::open(path).map_err(|error| format!("cannot read safetensors checkpoint: {error}"))?;
    let file_bytes = file.metadata().map_err(|error| format!("cannot inspect safetensors checkpoint: {error}"))?.len();
    let mut length = [0u8; 8];
    file.read_exact(&mut length).map_err(|error| format!("invalid safetensors header: {error}"))?;
    let header_bytes = u64::from_le_bytes(length);
    if !(2..=MAX_HEADER_BYTES).contains(&header_bytes) || header_bytes + 8 > file_bytes {
        return Err("invalid safetensors header length".into());
    }
    let mut header = vec![0u8; header_bytes as usize];
    file.read_exact(&mut header).map_err(|error| format!("truncated safetensors header: {error}"))?;
    let header: Value = serde_json::from_slice(&header).map_err(|error| format!("invalid safetensors header JSON: {error}"))?;
    let object = header.as_object().ok_or("invalid safetensors header: expected an object")?;
    let mut names = Vec::new();
    let data_bytes = file_bytes - 8 - header_bytes;
    let mut ranges = Vec::new();
    for (name, entry) in object {
        if name == "__metadata__" { continue; }
        if name.is_empty() || name.chars().any(char::is_control) || !entry.is_object() {
            return Err("invalid safetensors tensor entry".into());
        }
        let dtype = entry.get("dtype").and_then(Value::as_str)
            .ok_or_else(|| format!("invalid safetensors tensor dtype: {name}"))?;
        let width = safetensors_dtype_width(dtype)
            .ok_or_else(|| format!("unsupported safetensors tensor dtype {dtype}: {name}"))?;
        let shape = entry.get("shape").and_then(Value::as_array)
            .ok_or_else(|| format!("invalid safetensors tensor shape: {name}"))?;
        let elements = shape.iter().try_fold(1u64, |size, dimension| {
            dimension.as_u64().and_then(|dimension| size.checked_mul(dimension))
        }).ok_or_else(|| format!("invalid or oversized safetensors tensor shape: {name}"))?;
        let expected_bytes = elements.checked_mul(width)
            .ok_or_else(|| format!("oversized safetensors tensor: {name}"))?;
        let offsets = entry.get("data_offsets").and_then(Value::as_array)
            .filter(|offsets| offsets.len() == 2)
            .ok_or_else(|| format!("invalid safetensors tensor offsets: {name}"))?;
        let start = offsets[0].as_u64().ok_or_else(|| format!("invalid safetensors tensor offsets: {name}"))?;
        let end = offsets[1].as_u64().ok_or_else(|| format!("invalid safetensors tensor offsets: {name}"))?;
        if end < start || end > data_bytes || end - start != expected_bytes {
            return Err(format!("safetensors tensor payload is invalid: {name}"));
        }
        ranges.push((start, end, name.as_str()));
        names.push(name.clone());
    }
    if names.is_empty() {
        return Err("safetensors checkpoint contains no tensors".into());
    }
    ranges.sort_by_key(|(start, end, _)| (*start, *end));
    let mut last_end = 0;
    for (start, end, name) in ranges {
        if start < last_end {
            return Err(format!("overlapping safetensors tensor payload: {name}"));
        }
        last_end = end;
    }
    Ok(names)
}

fn safetensors_dtype_width(dtype: &str) -> Option<u64> {
    Some(match dtype {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E4M3FN" | "F8_E5M2" => 1,
        "I16" | "U16" | "F16" | "BF16" => 2,
        "I32" | "U32" | "F32" => 4,
        "I64" | "U64" | "F64" => 8,
        _ => return None,
    })
}

fn image_processor(target_dir: &Path) -> Result<Value, String> {
    let preprocessor = target_dir.join("preprocessor_config.json");
    if preprocessor.is_file() {
        return read_json(&preprocessor, "preprocessor config");
    }
    let processor = target_dir.join("processor_config.json");
    if processor.is_file() {
        let config = read_json(&processor, "processor config")?;
        return config.get("image_processor").cloned()
            .ok_or_else(|| "processor_config.json has no image_processor".into());
    }
    Err("vision MLX model is missing preprocessor_config.json or processor_config.json".into())
}

fn validate_processor(config: &Value) -> Result<(), String> {
    let expected = [
        ("patch_size", Value::from(16)),
        ("temporal_patch_size", Value::from(2)),
        ("merge_size", Value::from(2)),
        ("image_mean", serde_json::json!([0.5, 0.5, 0.5])),
        ("image_std", serde_json::json!([0.5, 0.5, 0.5])),
    ];
    if !config.is_object() || expected.iter().any(|(key, value)| config.get(*key) != Some(value)) {
        return Err("unsupported vision preprocessing configuration".into());
    }
    Ok(())
}

fn model_check(engine: &Path, vision_format: &str, config: &Path, draft_config: Option<&Path>) -> Result<String, String> {
    let output = Command::new(engine)
        .args(["model-check", "mlx-affine", vision_format])
        .arg(config)
        .args(draft_config.into_iter())
        .output()
        .map_err(|error| format!("cannot run native model-check using {}: {error}", engine.display()))?;
    if !output.status.success() {
        let reason = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        return Err(if reason.is_empty() { format!("native model-check failed with {}", output.status) }
            else { format!("native model-check refused the MLX model: {reason}") });
    }
    let result: Value = serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("native model-check returned invalid JSON: {error}"))?;
    result["family"].as_str().map(str::to_owned)
        .ok_or_else(|| "native model-check returned no family".into())
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use serde_json::Map;
    use std::os::unix::fs::PermissionsExt;

    static NEXT_SCRATCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    struct Scratch(PathBuf);
    impl Scratch {
        fn new() -> Self {
            let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
            let sequence = NEXT_SCRATCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!("splash-mlx-local-{}-{nanos}-{sequence}", std::process::id()));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Scratch { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }

    fn write_safetensors(path: &Path, names: &[&str]) {
        let header = names.iter().enumerate().map(|(index, name)| (name.to_string(), serde_json::json!({
            "dtype": "F32", "shape": [1], "data_offsets": [index * 4, index * 4 + 4]
        }))).collect::<Map<_, _>>();
        let bytes = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut encoded = (bytes.len() as u64).to_le_bytes().to_vec();
        encoded.extend(bytes);
        encoded.extend(vec![0; names.len() * 4]);
        fs::write(path, encoded).unwrap();
    }

    fn fixture_dir(root: &Path, vision: bool) {
        fs::create_dir_all(root).unwrap();
        fs::write(root.join("config.json"), serde_json::to_vec(&serde_json::json!({
            "model_type": "qwen3_5", "vision_config": if vision { serde_json::json!({"hidden_size": 32}) } else { Value::Null }
        })).unwrap()).unwrap();
        fs::write(root.join("tokenizer.json"), b"{}").unwrap();
        fs::write(root.join("tokenizer_config.json"), b"{}").unwrap();
        if vision {
            fs::write(root.join("preprocessor_config.json"), serde_json::to_vec(&serde_json::json!({
                "patch_size":16,"temporal_patch_size":2,"merge_size":2,
                "image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5]
            })).unwrap()).unwrap();
            write_safetensors(&root.join("model.safetensors"), &["language_model.embed_tokens.weight", "vision_tower.embeddings.weight"]);
        } else {
            write_safetensors(&root.join("model.safetensors"), &["language_model.embed_tokens.weight"]);
        }
    }

    #[test]
    fn prepares_text_only_mlx_with_links_and_engine_family() {
        let scratch = Scratch::new();
        let target = scratch.0.join("model");
        fixture_dir(&target, false);
        let engine = scratch.0.join("engine");
        fs::write(&engine, "#!/bin/sh\n[ \"$1 $2 $3\" = 'model-check mlx-affine none' ] || exit 2\nprintf '{\"family\":\"Qwen3.5-Test\"}'\n").unwrap();
        let mut permissions = fs::metadata(&engine).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&engine, permissions).unwrap();
        let published = prepare_local(&target, &scratch.0.join("models"), "local/model", &engine, false).unwrap();
        let record: Value = serde_json::from_slice(&fs::read(published.join("model.json")).unwrap()).unwrap();
        assert_eq!(record["target_format"], "mlx-affine");
        assert_eq!(record["vision_format"], "none");
        assert_eq!(record["family"], "Qwen3.5-Test");
        assert!(published.join("target/model.safetensors").is_symlink());
        assert!(published.join("tokenizer/tokenizer.json").is_symlink());
    }

    #[test]
    fn prepares_mlx_with_local_dflash2_draft() {
        let scratch = Scratch::new();
        let target = scratch.0.join("target");
        fixture_dir(&target, false);
        let draft = scratch.0.join("draft");
        fs::create_dir_all(&draft).unwrap();
        fs::write(draft.join("config.json"), br#"{"model_type":"dflash2"}"#).unwrap();
        fs::write(draft.join("model.safetensors"), b"draft weights").unwrap();
        let engine = scratch.0.join("engine");
        fs::write(&engine, "#!/bin/sh\n[ \"$1 $2 $3\" = 'model-check mlx-affine none' ] || exit 2\ncase \"$5\" in */draft/config.json) ;; *) exit 3 ;; esac\nprintf '{\"family\":\"Qwen3.5-Test\"}'\n").unwrap();
        let mut permissions = fs::metadata(&engine).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&engine, permissions).unwrap();

        let published = prepare_local_with_draft(
            &target,
            Some(&draft),
            &scratch.0.join("models"),
            "local/target",
            &engine,
            false,
        ).unwrap();
        let record: Value = serde_json::from_slice(&fs::read(published.join("model.json")).unwrap()).unwrap();
        assert_eq!(record["version"], 1);
        assert_eq!(record["sources"]["draft"]["repo"], draft.canonicalize().unwrap().to_string_lossy().as_ref());
        assert!(published.join("draft/config.json").is_symlink());
        assert!(published.join("draft/model.safetensors").is_symlink());
    }

    #[test]
    fn rejects_unsupported_sharded_checkpoints_before_publication() {
        let scratch = Scratch::new();
        let target = scratch.0.join("model");
        fixture_dir(&target, false);
        fs::remove_file(target.join("model.safetensors")).unwrap();
        fs::write(target.join("model.safetensors.index.json"), b"{}").unwrap();
        let result = prepare_local(&target, &scratch.0.join("models"), "local/model", Path::new("unused"), false);
        assert!(result.unwrap_err().contains("sharded MLX safetensors indexes"));
        assert!(!scratch.0.join("models/.resolved").exists());
    }

    #[test]
    fn safetensors_header_rejects_out_of_bounds_and_overlapping_payloads() {
        let scratch = Scratch::new();
        let path = scratch.0.join("model.safetensors");
        let bad = serde_json::json!({
            "a": {"dtype":"F32", "shape":[1], "data_offsets":[0,8]}
        });
        let bytes = serde_json::to_vec(&bad).unwrap();
        let mut encoded = (bytes.len() as u64).to_le_bytes().to_vec();
        encoded.extend(bytes);
        encoded.extend_from_slice(&[0; 8]);
        fs::write(&path, encoded).unwrap();
        assert!(safetensors_tensor_names(&path).unwrap_err().contains("payload is invalid"));

        let overlap = serde_json::json!({
            "a": {"dtype":"F32", "shape":[1], "data_offsets":[0,4]},
            "b": {"dtype":"F32", "shape":[1], "data_offsets":[2,6]}
        });
        let bytes = serde_json::to_vec(&overlap).unwrap();
        let mut encoded = (bytes.len() as u64).to_le_bytes().to_vec();
        encoded.extend(bytes);
        encoded.extend_from_slice(&[0; 6]);
        fs::write(&path, encoded).unwrap();
        assert!(safetensors_tensor_names(&path).unwrap_err().contains("overlapping"));
    }

    #[test]
    fn validates_vision_processor_and_adds_vision_links() {
        let scratch = Scratch::new();
        let target = scratch.0.join("model");
        fixture_dir(&target, true);
        let engine = scratch.0.join("engine");
        fs::write(&engine, "#!/bin/sh\n[ \"$1 $2 $3\" = 'model-check mlx-affine safetensors' ] || exit 2\nprintf '{\"family\":\"Qwen3.5-Vision\"}'\n").unwrap();
        let mut permissions = fs::metadata(&engine).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&engine, permissions).unwrap();
        let published = prepare_local(&target, &scratch.0.join("models"), "local/model", &engine, false).unwrap();
        assert!(published.join("vision/model.safetensors").is_symlink());
        assert_eq!(fs::read_to_string(published.join("model.json")).unwrap().contains("safetensors"), true);
    }
}
