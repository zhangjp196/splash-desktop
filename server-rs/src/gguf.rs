//! Bounded GGUF v3 header reader. It reads metadata and tensor descriptors,
//! never tensor payloads, and refuses unreasonable counts before allocating.

use serde_json::{Number, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

pub const VERSION: u32 = 3;
pub const MAX_FIELDS: u64 = 16_384;
pub const MAX_BYTES: usize = 128 * 1024 * 1024;
pub const MAX_ITEMS: u64 = 1_000_000;
pub const MAX_DIMENSIONS: u32 = 4;

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    /// GGUF lists tensor extents innermost first.
    pub extents: Vec<u64>,
    pub kind: u32,
    pub offset: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Metadata {
    pub values: BTreeMap<String, Value>,
    pub tensors: BTreeMap<String, Tensor>,
    /// Aligned absolute position of the tensor payload, when descriptors were read.
    pub data_offset: Option<u64>,
}

impl Metadata {
    pub fn read(path: impl AsRef<std::path::Path>, include_tensors: bool) -> Result<Self, String> {
        let file = std::fs::File::open(path.as_ref())
            .map_err(|error| format!("cannot read GGUF {}: {error}", path.as_ref().display()))?;
        Self::from_reader(file, include_tensors)
    }

    pub fn from_reader(input: impl Read, include_tensors: bool) -> Result<Self, String> {
        let mut reader = BoundedReader { input, consumed: 0 };
        if reader.read(4)? != b"GGUF" || reader.u32()? != VERSION {
            return Err("unsupported GGUF header (expected little-endian v3)".into());
        }
        let tensor_count = reader.u64()?;
        let field_count = reader.u64()?;
        if tensor_count > MAX_FIELDS || field_count > MAX_FIELDS {
            return Err("GGUF metadata has too many fields".into());
        }
        let mut values = BTreeMap::new();
        for _ in 0..field_count {
            let key = reader.string()?;
            if values.contains_key(&key) {
                return Err(format!("duplicate GGUF metadata key: {key}"));
            }
            let kind = reader.u32()?;
            let value = reader.value(kind)?;
            values.insert(key, value);
        }
        let mut tensors = BTreeMap::new();
        if include_tensors {
            for _ in 0..tensor_count {
                let name = reader.string()?;
                let dimensions = reader.u32()?;
                if dimensions > MAX_DIMENSIONS {
                    return Err(format!("invalid GGUF tensor rank: {name}"));
                }
                let mut extents = Vec::with_capacity(dimensions as usize);
                for _ in 0..dimensions { extents.push(reader.u64()?); }
                let tensor = Tensor { extents, kind: reader.u32()?, offset: reader.u64()? };
                if tensors.insert(name.clone(), tensor).is_some() {
                    return Err(format!("duplicate GGUF tensor: {name}"));
                }
            }
        }
        let data_offset = if include_tensors {
            let alignment = values.get("general.alignment").and_then(Value::as_u64).unwrap_or(32);
            if alignment == 0 || alignment > 65_536 || !alignment.is_power_of_two() {
                return Err("invalid GGUF alignment".into());
            }
            let consumed = u64::try_from(reader.consumed).map_err(|_| "GGUF header is too large")?;
            Some(consumed.checked_add((alignment - consumed % alignment) % alignment)
                .ok_or("GGUF header is too large")?)
        } else { None };
        Ok(Self { values, tensors, data_offset })
    }

    /// The typed scalar envelope consumed by the native engine's
    /// `model-check gguf` command. Arrays and non-finite floats are omitted,
    /// matching the Python installer adapter.
    pub fn scalar_metadata(&self) -> Value {
        let (mut unsigned, mut float, mut string) = (serde_json::Map::new(), serde_json::Map::new(), serde_json::Map::new());
        for (key, value) in &self.values {
            match value {
                Value::Number(number) if number.is_u64() => { unsigned.insert(key.clone(), value.clone()); }
                Value::Number(number) if number.is_i64() && number.as_i64().unwrap_or(0) < 0 => {
                    let wrapped = number.as_i64().unwrap() as u64;
                    unsigned.insert(key.clone(), Value::Number(Number::from(wrapped)));
                }
                Value::Number(number) if number.is_f64() => { float.insert(key.clone(), value.clone()); }
                Value::String(_) => { string.insert(key.clone(), value.clone()); }
                _ => {}
            }
        }
        serde_json::json!({"unsigned": unsigned, "float": float, "string": string})
    }

    fn string_value(&self, key: &str) -> Result<&str, String> {
        self.values.get(key).and_then(Value::as_str)
            .ok_or_else(|| format!("missing or invalid GGUF metadata: {key}"))
    }

    fn positive(&self, key: &str) -> Result<u64, String> {
        let value = self.values.get(key).and_then(Value::as_u64)
            .ok_or_else(|| format!("missing or invalid GGUF metadata: {key}"))?;
        if value == 0 { return Err(format!("GGUF metadata must be positive: {key}")); }
        Ok(value)
    }

    fn integer(&self, key: &str) -> Result<i64, String> {
        self.values.get(key).and_then(Value::as_i64)
            .ok_or_else(|| format!("missing or invalid GGUF metadata: {key}"))
    }

    fn boolean(&self, key: &str) -> Result<bool, String> {
        self.values.get(key).and_then(Value::as_bool)
            .ok_or_else(|| format!("missing or invalid GGUF metadata: {key}"))
    }

    fn array(&self, key: &str) -> Result<&Vec<Value>, String> {
        self.values.get(key).and_then(Value::as_array)
            .ok_or_else(|| format!("missing or invalid GGUF metadata: {key}"))
    }
}

/// Validate that every descriptor has a supported ggml type and a complete,
/// aligned, non-overlapping payload range inside this GGUF shard.
pub fn validate_file_layout(path: &Path, metadata: &Metadata) -> Result<(), String> {
    require_single_file(metadata)?;
    validate_shard_layout(path, metadata)
}

fn validate_shard_layout(path: &Path, metadata: &Metadata) -> Result<(), String> {
    let data_offset = metadata.data_offset.ok_or("GGUF tensor descriptors were not loaded")?;
    let file_len = fs::metadata(path)
        .map_err(|error| format!("cannot inspect GGUF {}: {error}", path.display()))?.len();
    if data_offset > file_len { return Err("GGUF data section is truncated".into()); }
    let data_bytes = file_len - data_offset;
    let alignment = metadata.values.get("general.alignment").and_then(Value::as_u64).unwrap_or(32);
    let mut ranges = Vec::with_capacity(metadata.tensors.len());
    for (name, tensor) in &metadata.tensors {
        let (block_elements, block_bytes) = ggml_type_layout(tensor.kind)
            .ok_or_else(|| format!("unknown ggml type {} for {name}", tensor.kind))?;
        if tensor.extents.is_empty() || tensor.extents.len() > MAX_DIMENSIONS as usize || tensor.extents.iter().any(|&d| d == 0) {
            return Err(format!("invalid tensor dimensions for {name}"));
        }
        let columns = tensor.extents[0];
        if columns % block_elements != 0 { return Err(format!("tensor row is not block aligned: {name}")); }
        let rows = tensor.extents.iter().skip(1).try_fold(1u64, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| format!("GGUF tensor dimensions overflow: {name}"))?;
        let bytes = rows.checked_mul(columns / block_elements).and_then(|v| v.checked_mul(block_bytes))
            .ok_or_else(|| format!("GGUF tensor size overflow: {name}"))?;
        if tensor.offset % alignment != 0 { return Err(format!("tensor data is misaligned: {name}")); }
        if tensor.offset > data_bytes || bytes > data_bytes - tensor.offset {
            return Err(format!("tensor data runs past the end of the file: {name}"));
        }
        ranges.push((tensor.offset, tensor.offset + bytes, name));
    }
    ranges.sort_by_key(|range| (range.0, range.1));
    for pair in ranges.windows(2) {
        if pair[0].1 > pair[1].0 {
            return Err(format!("overlapping GGUF tensor data: {} and {}", pair[0].2, pair[1].2));
        }
    }
    Ok(())
}

/// Validate and publish a local single-file GGUF target as an immutable
/// assembly. Split targets use `prepare_local_parts` after shard discovery.
pub fn prepare_local(
    target_path: &Path,
    vision_path: Option<&Path>,
    models_root: &Path,
    model_id: &str,
    engine: &Path,
) -> Result<PathBuf, String> {
    prepare_local_with_draft(target_path, vision_path, None, models_root, model_id, engine)
}

/// Prepare a local GGUF target together with an optional local DFlash2
/// checkpoint. The draft is linked into the same immutable assembly so serve
/// does not need a prior Python installer cache.
pub fn prepare_local_with_draft(
    target_path: &Path,
    vision_path: Option<&Path>,
    draft_path: Option<&Path>,
    models_root: &Path,
    model_id: &str,
    engine: &Path,
) -> Result<PathBuf, String> {
    prepare_local_discovered_with_draft(
        target_path, vision_path, draft_path, models_root, model_id, engine,
    )
}

/// Discover all shards beside a selected local GGUF and publish them in
/// numeric order. The argument may name any one shard; a complete exact shard
/// set must be present in the same directory.
pub fn prepare_local_discovered(
    target_path: &Path,
    vision_path: Option<&Path>,
    models_root: &Path,
    model_id: &str,
    engine: &Path,
) -> Result<PathBuf, String> {
    prepare_local_discovered_with_draft(target_path, vision_path, None, models_root, model_id, engine)
}

pub fn prepare_local_discovered_with_draft(
    target_path: &Path,
    vision_path: Option<&Path>,
    draft_path: Option<&Path>,
    models_root: &Path,
    model_id: &str,
    engine: &Path,
) -> Result<PathBuf, String> {
    let parts = discover_split_parts(target_path)?;
    prepare_local_parts(&parts, vision_path, draft_path, models_root, model_id, engine)
}

fn prepare_local_parts(
    target_paths: &[PathBuf],
    vision_path: Option<&Path>,
    draft_path: Option<&Path>,
    models_root: &Path,
    model_id: &str,
    engine: &Path,
) -> Result<PathBuf, String> {
    if model_id.trim().is_empty() { return Err("model ID must not be empty".into()); }
    if target_paths.is_empty() { return Err("GGUF target has no files".into()); }
    let mut target_names = BTreeSet::new();
    let mut canonical_paths = Vec::with_capacity(target_paths.len());
    let mut headers = Vec::with_capacity(target_paths.len());
    for path in target_paths {
        // Keep the caller's `.gguf` basename, because Hub snapshot links often
        // point at extensionless blobs; resolve the linked source separately.
        let name = path.file_name().and_then(|name| name.to_str())
            .ok_or("target GGUF has no valid filename")?.to_owned();
        if !name.ends_with(".gguf") || name.chars().any(char::is_control) || !target_names.insert(name.clone()) {
            return Err(format!("invalid or duplicate GGUF shard filename: {name}"));
        }
        canonical_paths.push(fs::canonicalize(path)
            .map_err(|error| format!("cannot resolve target GGUF {}: {error}", path.display()))?);
        headers.push(Metadata::read(canonical_paths.last().unwrap(), true)?);
    }
    let target = merge_split_metadata(&headers)?;
    require_loadable(&target)?;
    for (path, header) in canonical_paths.iter().zip(&headers) {
        validate_shard_layout(path, header)?;
    }
    let vision_path = vision_path.map(fs::canonicalize).transpose()
        .map_err(|error| format!("cannot resolve vision GGUF: {error}"))?;
    let vision = vision_path.as_deref().map(|path| Metadata::read(path, true)).transpose()?;
    if let (Some(path), Some(metadata)) = (vision_path.as_deref(), vision.as_ref()) {
        require_single_file(metadata)?;
        require_vision_loadable(metadata)?;
        validate_file_layout(path, metadata)?;
    }
    let config = model_config(&target, vision.as_ref())?;
    let mut generated = tokenizer_files(&target)?;
    let config_bytes = pretty_json(&config)?;
    generated.insert("config.json".into(), config_bytes.clone());
    generated.insert("tokenizer/config.json".into(), config_bytes);
    let scalar_metadata = target.scalar_metadata();
    let scalar_bytes = pretty_json(&scalar_metadata)?;
    generated.insert("gguf-metadata.json".into(), scalar_bytes.clone());

    let draft_root = draft_path.map(fs::canonicalize).transpose()
        .map_err(|error| format!("cannot resolve DFlash2 draft: {error}"))?;
    let draft_config = draft_root.as_ref().map(|root| root.join("config.json"));
    if let Some(config) = &draft_config {
        if !config.is_file() { return Err(format!("DFlash2 draft has no config.json: {}", config.display())); }
    }
    let family = validate_with_engine(engine, &target, vision.as_ref(), draft_config.as_deref())?;
    let sources = crate::assembly::persist_generated(models_root, &generated)?;
    let metadata_key = crate::assembly::gguf_metadata_key_many(&canonical_paths, vision_path.as_deref())?;
    let mut files = BTreeMap::new();
    for name in generated.keys() {
        files.insert(name.clone(), sources[name].clone());
    }
    for (path, original) in canonical_paths.iter().zip(target_paths) {
        let name = original.file_name().and_then(|name| name.to_str()).ok_or("target GGUF has no valid filename")?;
        files.insert(format!("target/{name}"), path.clone());
    }
    if let Some(path) = vision_path {
        files.insert("vision/mmproj.gguf".into(), path);
    }
    let record = if let Some(draft_root) = draft_root {
        for (name, path) in crate::local_draft::files(&draft_root)? {
            files.insert(name, path);
        }
        serde_json::json!({
            "version": 1,
            "model": model_id,
            "family": family,
            "target_format": "gguf",
            "vision_format": if vision.is_some() { "gguf" } else { "none" },
            "sources": {
                "target": {"repo": canonical_paths[0].to_string_lossy(), "revision": null},
                "draft": {"repo": draft_root.to_string_lossy(), "revision": null}
            },
            "metadata": metadata_key
        })
    } else {
        serde_json::json!({
            "version": 2,
            "model": model_id,
            "family": family,
            "target_format": "gguf",
            "vision_format": if vision.is_some() { "gguf" } else { "none" },
            "sources": {"target": {"repo": canonical_paths[0].to_string_lossy(), "revision": null}},
            "metadata": metadata_key,
            "speculation": {
                "version": 1, "provider": "none", "format": "none",
                "source": {"kind": "none"}, "proposal_rows": 1,
                "state": "none", "fallback": "none"
            }
        })
    };
    crate::assembly::publish(models_root, record, &files)
}

fn discover_split_parts(selected: &Path) -> Result<Vec<PathBuf>, String> {
    let name = selected.file_name().and_then(|name| name.to_str())
        .ok_or("target GGUF has no valid filename")?;
    let Some((model, selected_no, expected_count)) = split_filename(name) else {
        return Ok(vec![selected.to_path_buf()]);
    };
    let directory = selected.parent().unwrap_or_else(|| Path::new("."));
    let mut found = BTreeMap::<u32, (u32, PathBuf)>::new();
    for entry in fs::read_dir(directory).map_err(|error| format!("cannot scan split GGUF directory {}: {error}", directory.display()))? {
        let entry = entry.map_err(|error| format!("cannot scan split GGUF directory: {error}"))?;
        let kind = entry.file_type().map_err(|error| format!("cannot inspect split GGUF part: {error}"))?;
        let path = entry.path();
        if !(kind.is_file() || (kind.is_symlink() && fs::metadata(&path).is_ok_and(|m| m.is_file()))) { continue; }
        let Some(candidate) = entry.file_name().to_str().map(str::to_owned) else { continue; };
        let Some((candidate_model, number, count)) = split_filename(&candidate) else { continue; };
        if candidate_model != model { continue; }
        if count != expected_count {
            return Err(format!("split GGUF shard count disagrees: {candidate}"));
        }
        if found.insert(number, (count, path)).is_some() {
            return Err(format!("duplicate split GGUF part number: {number}"));
        }
    }
    if selected_no == 0 || selected_no > expected_count {
        return Err(format!("invalid split GGUF filename: {name}"));
    }
    let numbers = found.keys().copied().collect::<Vec<_>>();
    let expected = (1..=expected_count).collect::<Vec<_>>();
    if numbers != expected {
        return Err(format!("split GGUF is incomplete or has extra parts (expected 1..={expected_count}, found {numbers:?})"));
    }
    Ok(found.into_values().map(|(_, path)| path).collect())
}

/// Resolve a selected GGUF file or a directory containing exactly one logical
/// GGUF target. A directory with several independent targets is ambiguous and
/// asks the caller to select a specific GGUF; split shards count as one.
pub fn select_local_target(path: &Path) -> Result<PathBuf, String> {
    if path.is_file() { return Ok(path.to_path_buf()); }
    if !path.is_dir() { return Err(format!("GGUF target does not exist: {}", path.display())); }
    let mut targets = Vec::new();
    for entry in fs::read_dir(path).map_err(|error| format!("cannot scan GGUF directory {}: {error}", path.display()))? {
        let entry = entry.map_err(|error| format!("cannot scan GGUF directory: {error}"))?;
        let file_type = entry.file_type().map_err(|error| format!("cannot inspect GGUF target: {error}"))?;
        if !(file_type.is_file() || (file_type.is_symlink() && fs::metadata(entry.path()).is_ok_and(|metadata| metadata.is_file()))) { continue; }
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else { continue; };
        if !name.to_ascii_lowercase().ends_with(".gguf") || name.to_ascii_lowercase().contains("mmproj") { continue; }
        match split_filename(&name) {
            Some((_, 1, _)) => targets.push(entry.path()),
            Some(_) => {},
            None => targets.push(entry.path()),
        }
    }
    targets.sort();
    targets.dedup();
    match targets.as_slice() {
        [target] => Ok(target.clone()),
        [] => Err(format!("no GGUF target found in {}", path.display())),
        _ => Err(format!("directory contains multiple GGUF targets; select one .gguf file: {}",
            targets.iter().map(|target| target.file_name().unwrap_or_default().to_string_lossy()).collect::<Vec<_>>().join(", "))),
    }
}

fn split_filename(name: &str) -> Option<(&str, u32, u32)> {
    let stem = name.strip_suffix(".gguf")?;
    let (prefix, count) = stem.rsplit_once("-of-")?;
    let (model, number) = prefix.rsplit_once('-')?;
    if number.len() != 5 || count.len() != 5
        || !number.bytes().all(|b| b.is_ascii_digit())
        || !count.bytes().all(|b| b.is_ascii_digit())
    { return None; }
    Some((model, number.parse().ok()?, count.parse().ok()?))
}

fn merge_split_metadata(headers: &[Metadata]) -> Result<Metadata, String> {
    let Some(first) = headers.first() else { return Err("GGUF target has no files".into()); };
    let count = match first.values.get("split.count") {
        None => 1,
        Some(value) => value.as_u64().ok_or("invalid GGUF metadata: split.count")?,
    };
    if usize::try_from(count).ok() != Some(headers.len()) {
        return Err("the GGUF's split metadata does not match its split.count".into());
    }
    if headers.len() == 1 {
        require_single_file(first)?;
        return Ok(first.clone());
    }
    let mut merged = first.clone();
    let mut tensors = BTreeMap::new();
    for (index, header) in headers.iter().enumerate() {
        if header.values.get("split.count").and_then(Value::as_u64) != Some(count)
            || header.values.get("split.no").and_then(Value::as_u64) != Some(index as u64)
        {
            return Err(format!("GGUF split file {} is out of order or has inconsistent metadata", index + 1));
        }
        for (name, tensor) in &header.tensors {
            if tensors.insert(name.clone(), tensor.clone()).is_some() {
                return Err(format!("duplicate GGUF tensor across split files: {name}"));
            }
        }
    }
    let total = match first.values.get("split.tensors.count") {
        Some(value) => value.as_u64().ok_or("invalid GGUF metadata: split.tensors.count")?,
        None => tensors.len() as u64,
    };
    if total != tensors.len() as u64 {
        return Err("the GGUF's split files hold another number of tensors".into());
    }
    merged.tensors = tensors;
    Ok(merged)
}

fn ggml_type_layout(kind: u32) -> Option<(u64, u64)> {
    Some(match kind {
        0 => (1,4), 1 => (1,2), 2 => (32,18), 3 => (32,20), 6 => (32,22), 7 => (32,24),
        8 => (32,34), 9 => (32,36), 10 => (256,84), 11 => (256,110), 12 => (256,144),
        13 => (256,176), 14 => (256,210), 15 => (256,292), 16 => (256,66), 17 => (256,74),
        18 => (256,98), 19 => (256,50), 20 => (32,18), 21 => (256,110), 22 => (256,82),
        23 => (256,136), 24 => (1,1), 25 => (1,2), 26 => (1,4), 27 => (1,8), 28 => (1,8),
        29 => (256,56), 30 => (1,2), 39 => (32,17), 40 => (64,36), 41 => (128,18),
        42 => (64,18), 142 => (128,34), 144 => (128,34), 145 => (128,26), 146 => (128,30),
        _ => return None,
    })
}

/// Convert one of the Python installer's explicitly supported GGUF text
/// architectures into the engine's model-check config shape.
pub fn model_config(metadata: &Metadata, vision: Option<&Metadata>) -> Result<Value, String> {
    let arch = metadata.string_value("general.architecture")?;
    let (model_type, moe, flat) = match arch {
        "qwen35" => ("qwen3_5_text", false, false),
        "qwen35moe" => ("qwen3_5_moe_text", true, false),
        "qwen4exp" => ("qwen4_exp_text", true, false),
        "flint" => ("flint", true, true),
        _ => return Err(format!("unsupported GGUF model architecture: {arch}")),
    };

    let mut text = serde_json::Map::new();
    for (field, suffix) in [
        ("hidden_size", "embedding_length"),
        ("max_position_embeddings", "context_length"),
        ("num_attention_heads", "attention.head_count"),
        ("num_key_value_heads", "attention.head_count_kv"),
        ("head_dim", "attention.key_length"),
    ] {
        text.insert(field.into(), Value::from(metadata.positive(&format!("{arch}.{suffix}"))?));
    }
    if moe {
        text.insert("num_experts".into(), Value::from(metadata.positive(&format!("{arch}.expert_count"))?));
    }
    text.insert("num_hidden_layers".into(), Value::from(loaded_layers(metadata, arch)?));
    text.insert("model_type".into(), Value::String(model_type.into()));
    text.insert("vocab_size".into(), Value::from(metadata.array("tokenizer.ggml.tokens")?.len() as u64));

    if arch == "flint" { add_flint_config(metadata, arch, &mut text)?; }
    let mut config = if flat {
        Value::Object(text)
    } else {
        let mut nested = serde_json::Map::new();
        nested.insert("model_type".into(), Value::String(model_type.strip_suffix("_text").unwrap_or(model_type).into()));
        nested.insert("text_config".into(), Value::Object(text));
        Value::Object(nested)
    };
    if let Some(vision) = vision {
        config.as_object_mut().unwrap().insert("vision_config".into(), vision_config(vision)?);
    }
    Ok(config)
}

/// Refuse split GGUF inputs at the config-preparation boundary until their
/// per-part metadata and tensor tables can be merged and validated together.
pub fn require_single_file(metadata: &Metadata) -> Result<(), String> {
    match metadata.values.get("split.count") {
        None => Ok(()),
        Some(value) if value.as_u64() == Some(1) => Ok(()),
        Some(value) if value.as_u64().is_some_and(|count| count > 1) => {
            Err("gguf-config refuses split GGUF targets until split metadata is merged".into())
        }
        Some(_) => Err("invalid GGUF metadata: split.count".into()),
    }
}

/// Refuse a GGUF header that the native loader cannot use. This mirrors the
/// installer's name/type screening without reading tensor payloads; the engine
/// still validates tensor dimensions and payload ranges when it loads them.
pub fn require_loadable(metadata: &Metadata) -> Result<(), String> {
    require_rotation(metadata)?;
    require_arrays(metadata)?;
    let loaded = expected_tensors(metadata)?;
    let format_types = format_types();
    let unquantized = ["BF16", "F16", "F32"].into_iter().collect::<BTreeSet<_>>();
    let mut projection_types = BTreeSet::<String>::new();
    let mut unsupported = BTreeMap::<(String, String), usize>::new();

    for (name, accepted) in &loaded {
        let kind = metadata.tensors.get(name).map(|tensor| tensor.kind);
        let found = match kind {
            None => "missing".to_owned(),
            Some(kind) => tensor_type_name(kind).map(str::to_owned).unwrap_or_else(|| format!("type {kind}")),
        };
        if accepted == &format_types && name != "token_embd.weight" {
            projection_types.insert(found.clone());
        }
        if !accepted.contains(found.as_str()) {
            let short = name.strip_prefix("blk.").and_then(|rest| rest.split_once('.').map(|(_, tail)| tail)).unwrap_or(name);
            *unsupported.entry((short.to_owned(), found.to_owned())).or_default() += 1;
        }
        if name.ends_with(".weight") {
            let scale = format!("{}.scale", name.trim_end_matches(".weight"));
            if metadata.tensors.contains_key(&scale) {
                let short = name.strip_prefix("blk.").and_then(|rest| rest.split_once('.').map(|(_, tail)| tail)).unwrap_or(name);
                if found != "NVFP4" {
                    *unsupported.entry((format!("{}.scale", short.trim_end_matches(".weight")), format!("beside {found}"))).or_default() += 1;
                }
            }
        }
    }

    if projection_types.iter().all(|kind| unquantized.contains(kind.as_str())) {
        return Err("this GGUF stores every projection and expert unquantized (BF16, F16 or F32): Splash loads quantized GGUFs; choose a quantized variant".into());
    }
    if !unsupported.is_empty() {
        let listed = unsupported.into_iter().map(|((name, found), count)| {
            format!("{name} {found} ({count} {})", if count == 1 { "tensor" } else { "tensors" })
        }).collect::<Vec<_>>().join(", ");
        return Err(format!("this GGUF stores tensors Splash cannot load: {listed}; choose another variant"));
    }
    Ok(())
}

fn tensor_type_name(kind: u32) -> Option<&'static str> {
    Some(match kind {
        0 => "F32", 1 => "F16", 2 => "Q4_0", 3 => "Q4_1", 6 => "Q5_0", 7 => "Q5_1", 8 => "Q8_0",
        10 => "Q2_K", 11 => "Q3_K", 12 => "Q4_K", 13 => "Q5_K", 14 => "Q6_K", 16 => "IQ2_XXS",
        17 => "IQ2_XS", 18 => "IQ3_XXS", 19 => "IQ1_S", 20 => "IQ4_NL", 21 => "IQ3_S", 22 => "IQ2_S",
        23 => "IQ4_XS", 29 => "IQ1_M", 30 => "BF16", 39 => "MXFP4", 40 => "NVFP4", 41 => "Q1_0",
        42 => "Q2_0", 142 => "PQ2_0", 144 => "TQ2_T", 145 => "TQK6", 146 => "TQK7", _ => return None,
    })
}

fn format_types() -> BTreeSet<&'static str> {
    ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1",
     "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_NL", "IQ4_XS",
     "MXFP4", "NVFP4", "Q1_0", "Q2_0", "PQ2_0", "TQ2_T", "TQK6", "TQK7", "BF16", "F16"]
        .into_iter().collect()
}

fn type_set(names: &[&'static str]) -> BTreeSet<&'static str> { names.iter().copied().collect() }

fn expected_tensors(metadata: &Metadata) -> Result<BTreeMap<String, BTreeSet<&'static str>>, String> {
    let arch = metadata.string_value("general.architecture")?;
    let is_moe = matches!(arch, "qwen35moe" | "qwen4exp" | "flint");
    if !matches!(arch, "qwen35" | "qwen35moe" | "qwen4exp" | "flint") {
        return Err(format!("unsupported GGUF model architecture: {arch}"));
    }
    let format = format_types();
    let norm = type_set(&["F32", "BF16", "F16"]);
    let float = type_set(&["F32"]);
    let float_projection = type_set(&["F32", "BF16"]);
    let float_rows = format.union(&float).copied().collect::<BTreeSet<_>>();
    let router = format.union(&float_projection).copied().collect::<BTreeSet<_>>();
    let mut tensors = BTreeMap::new();
    let mut add = |name: String, types: &BTreeSet<&'static str>| { tensors.insert(name, types.clone()); };
    add("token_embd.weight".into(), &format);
    if arch == "qwen4exp" { add("per_layer_token_embd.weight".into(), &format); }
    else { add("output_norm.weight".into(), &norm); }
    add("output.weight".into(), &format);
    if arch == "qwen4exp" {
        add("output_hc_norm.weight".into(), &norm);
        add("output_hc_down.weight".into(), &format);
        add("output_hc_up.weight".into(), &format);
    }

    let layers = loaded_layers(metadata, arch)?;
    let hybrid = arch != "flint";
    let period = if hybrid { metadata.positive(&format!("{arch}.full_attention_interval"))? } else { 1 };
    let dense = if arch == "flint" {
        let count = metadata.integer(&format!("{arch}.leading_dense_block_count"))?;
        if count < 0 || count as u64 >= layers { return Err(format!("invalid GGUF metadata: {arch}.leading_dense_block_count")); }
        count as u64
    } else { 0 };
    for layer in 0..layers {
        let prefix = format!("blk.{layer}.");
        let qwen4 = arch == "qwen4exp";
        let flint = arch == "flint";
        for (name, types) in if qwen4 { vec![("hc_attn_norm.weight", &norm), ("hc_attn_down.weight", &format), ("hc_attn_up.weight", &format), ("hc_attn_inject.weight", &float_rows), ("hc_ffn_norm.weight", &norm), ("hc_ffn_down.weight", &format), ("hc_ffn_up.weight", &format), ("hc_ffn_inject.weight", &float_rows)] }
            else if flint { vec![("attn_norm.weight", &norm), ("ffn_norm.weight", &norm)] }
            else { vec![("attn_norm.weight", &norm), ("post_attention_norm.weight", &norm)] } { add(format!("{prefix}{name}"), types); }

        let full_attention = !hybrid || (layer + 1) % period == 0;
        if full_attention {
            for (name, types) in [("attn_q.weight", &format), ("attn_k.weight", &format), ("attn_v.weight", &format), ("attn_q_norm.weight", &norm), ("attn_k_norm.weight", &norm), ("attn_output.weight", &format)] { add(format!("{prefix}{name}"), types); }
            if qwen4 {
                for (name, types) in [("indexer.q_proj.weight", &float_projection), ("indexer.k_proj.weight", &float_projection), ("indexer.q_norm.weight", &norm), ("indexer.k_norm.weight", &norm)] { add(format!("{prefix}{name}"), types); }
            }
            if flint { add(format!("{prefix}attn_sinks.weight"), &float); }
        } else {
            for (name, types) in [("attn_qkv.weight", &format), ("attn_gate.weight", &format), ("ssm_alpha.weight", &format), ("ssm_beta.weight", &format), ("ssm_conv1d.weight", &float), ("ssm_a", &float), ("ssm_dt.bias", &float), ("ssm_norm.weight", &norm), ("ssm_out.weight", &format)] { add(format!("{prefix}{name}"), types); }
        }

        let dense_ffn = !is_moe || layer < dense;
        if dense_ffn {
            for name in ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"] { add(format!("{prefix}{name}"), &format); }
        } else if metadata.tensors.contains_key(&format!("{prefix}ffn_gate_up_exps.weight")) {
            add(format!("{prefix}ffn_gate_inp.weight"), &router);
            add(format!("{prefix}ffn_gate_up_exps.weight"), &format);
            if !flint { add(format!("{prefix}ffn_gate_inp_shexp.weight"), &float_rows); }
            for name in ["ffn_down_exps.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"] { add(format!("{prefix}{name}"), &format); }
        } else {
            add(format!("{prefix}ffn_gate_inp.weight"), &router);
            for name in ["ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"] { add(format!("{prefix}{name}"), &format); }
            if !flint { add(format!("{prefix}ffn_gate_inp_shexp.weight"), &float_rows); }
            for name in ["ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"] { add(format!("{prefix}{name}"), &format); }
        }
        if qwen4 && layer == 1 {
            for (name, types) in [("ple_key.weight", &format), ("ple_value.weight", &format), ("ple_norm_key.weight", &norm), ("ple_norm_query.weight", &norm), ("ple_norm_conv.weight", &norm), ("ple_conv1d.weight", &norm)] { add(format!("{prefix}{name}"), types); }
        }
    }
    Ok(tensors)
}

fn require_arrays(metadata: &Metadata) -> Result<(), String> {
    let arch = metadata.string_value("general.architecture")?;
    if arch == "qwen4exp" {
        let layers = loaded_layers(metadata, arch)?;
        let period = metadata.positive(&format!("{arch}.full_attention_interval"))?;
        let expected = (0..layers).map(|layer| Value::from(if (layer + 1) % period == 0 { 4 } else { 0 })).collect::<Vec<_>>();
        let ratios = metadata.values.get(&format!("{arch}.attention.compress_ratios")).and_then(Value::as_array);
        if ratios.is_none_or(|ratios| ratios.len() < expected.len() || ratios[..expected.len()] != expected) {
            return Err("GGUF metadata does not match the target: attention.compress_ratios".into());
        }
        let ple = metadata.values.get(&format!("{arch}.ple.layers"));
        if ple != Some(&Value::Array(vec![Value::from(1)])) {
            return Err("GGUF metadata does not match the target: ple.layers".into());
        }
    }
    Ok(())
}

fn require_rotation(metadata: &Metadata) -> Result<(), String> {
    const PREFIX: &str = "prism.hadamard.";
    let keys = metadata.values.keys().filter_map(|key| key.strip_prefix(PREFIX)).collect::<BTreeSet<_>>();
    if keys.is_empty() { return Ok(()); }
    let arch = metadata.string_value("general.architecture")?;
    let allowed = ["version", "transform", "axis", "sign_mode", "weight_names", "inverse_weight_names", "sign_widths", "sign_values", "block_size", "gdn_v_grouped"].into_iter().collect::<BTreeSet<_>>();
    let common = [("version", Value::from(1)), ("transform", Value::from("normalized-sylvester-walsh-hadamard")), ("axis", Value::from("input-last-dimension")), ("sign_mode", Value::from("explicit"))];
    let expected = common.iter().all(|(key,value)| metadata.values.get(&format!("{PREFIX}{key}")) == Some(value));
    let block = metadata.values.get(&format!("{PREFIX}block_size")).and_then(Value::as_u64);
    let grouped = metadata.values.get(&format!("{PREFIX}gdn_v_grouped")).and_then(Value::as_bool);
    if !matches!(arch, "qwen35" | "qwen4exp") || keys.iter().any(|key| !allowed.contains(key)) || !expected
        || !matches!(block, Some(128 | 1024)) || (arch == "qwen35" && grouped != Some(true)) || (arch == "qwen4exp" && grouped.is_some()) {
        return Err("this GGUF's input rotation is not one Splash runs; choose another variant".into());
    }
    if arch == "qwen4exp" {
        let names = metadata.values.get(&format!("{PREFIX}weight_names")).and_then(Value::as_array);
        if metadata.values.contains_key(&format!("{PREFIX}inverse_weight_names")) || names.is_none() {
            return Err("this GGUF's input rotation does not name its routed experts alone; choose another variant".into());
        }
        let experts = metadata.tensors.keys().filter(|name| name.ends_with("_exps.weight")).map(String::as_str).collect::<BTreeSet<_>>();
        let named = names.unwrap().iter().filter_map(Value::as_str).collect::<BTreeSet<_>>();
        if named != experts { return Err("this GGUF's input rotation does not name its routed experts alone; choose another variant".into()); }
    }
    Ok(())
}

/// Derive tokenizer.json, tokenizer_config.json and chat_template.jinja from
/// supported GGUF GPT-2 BPE metadata. The tokenizer graph is built by the same
/// Rust `tokenizers` library used to serve the resulting JSON.
pub fn tokenizer_files(metadata: &Metadata) -> Result<BTreeMap<String, Vec<u8>>, String> {
    use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
    use tokenizers::models::bpe::{BPE, Vocab};
    use tokenizers::normalizers::{NFC, NormalizerWrapper};
    use tokenizers::pre_tokenizers::{
        byte_level::ByteLevel as ByteLevelPreTokenizer,
        sequence::Sequence as PreTokenizerSequence,
        split::{Split, SplitPattern},
        PreTokenizerWrapper,
    };
    use tokenizers::processors::{
        byte_level::ByteLevel as ByteLevelProcessor,
        PostProcessorWrapper,
    };
    use tokenizers::tokenizer::SplitDelimiterBehavior;
    use tokenizers::{AddedToken, Tokenizer};

    if metadata.string_value("tokenizer.ggml.model")? != "gpt2" {
        return Err("unsupported GGUF tokenizer profile; expected gpt2/qwen35 or gpt2/flint".into());
    }
    let profile = match metadata.string_value("tokenizer.ggml.pre")? {
        "qwen35" => TokenizerProfile::Qwen35,
        "flint" => TokenizerProfile::Flint,
        _ => return Err("unsupported GGUF tokenizer profile; expected gpt2/qwen35 or gpt2/flint".into()),
    };
    let tokens = metadata.array("tokenizer.ggml.tokens")?;
    let token_types = metadata.array("tokenizer.ggml.token_type")?;
    let merges = metadata.array("tokenizer.ggml.merges")?;
    if tokens.is_empty() || token_types.len() != tokens.len() {
        return Err("invalid or unsupported GGUF tokenizer vocabulary".into());
    }
    let mut token_strings = Vec::with_capacity(tokens.len());
    let mut vocab = Vocab::default();
    let mut added = Vec::new();
    for (index, (token, kind)) in tokens.iter().zip(token_types).enumerate() {
        let Some(token) = token.as_str().filter(|token| !token.is_empty()) else {
            return Err("invalid or unsupported GGUF tokenizer vocabulary".into());
        };
        let Some(kind) = kind.as_u64() else {
            return Err("invalid or unsupported GGUF tokenizer vocabulary".into());
        };
        if !matches!(kind, 1 | 3 | 4 | 5) || index > u32::MAX as usize || vocab.insert(token.to_owned(), index as u32).is_some() {
            return Err("invalid or unsupported GGUF tokenizer vocabulary".into());
        }
        if kind == 3 || kind == 4 {
            added.push((index, kind == 3, token.to_owned()));
        }
        token_strings.push(token.to_owned());
    }
    let byte_alphabet = ByteLevelPreTokenizer::alphabet();
    if byte_alphabet.iter().any(|symbol| !vocab.contains_key(&symbol.to_string())) {
        return Err("GGUF byte-level tokenizer is missing byte tokens".into());
    }

    let pairs = merges.iter().map(|merge| {
        let Some(merge) = merge.as_str() else { return Err("invalid GGUF BPE merge".to_owned()); };
        let parts: Vec<_> = merge.split(' ').collect();
        if parts.len() != 2 { return Err("invalid GGUF BPE merge".to_owned()); }
        Ok((parts[0].to_owned(), parts[1].to_owned()))
    }).collect::<Result<Vec<_>, _>>()?;
    for key in ["tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_eos_token"] {
        if metadata.values.get(key).is_some_and(|value| value != &Value::Bool(false)) {
            return Err(format!("GGUF automatic BOS/EOS insertion is unsupported: {key}"));
        }
    }
    let model = BPE::builder().vocab_and_merges(vocab, pairs).build()
        .map_err(|_| "invalid GGUF BPE vocabulary or merges".to_owned())?;
    let mut tokenizer = Tokenizer::new(model);
    if profile.nfc() {
        tokenizer.with_normalizer(Some(NormalizerWrapper::from(NFC)))
            .map_err(|_| "invalid GGUF BPE vocabulary or merges".to_owned())?;
    }
    let mut pretokenizers = profile.patterns().iter().map(|pattern| {
        Split::new(SplitPattern::Regex((*pattern).to_owned()), SplitDelimiterBehavior::Isolated, false)
            .map(PreTokenizerWrapper::from)
            .map_err(|_| "invalid or unsupported GGUF tokenizer regex".to_owned())
    }).collect::<Result<Vec<_>, _>>()?;
    pretokenizers.push(ByteLevelPreTokenizer::new(false, false, false).into());
    tokenizer.with_pre_tokenizer(Some(PreTokenizerSequence::new(pretokenizers)));
    tokenizer.with_post_processor(Some(PostProcessorWrapper::from(ByteLevelProcessor::new(false, false, false))));
    tokenizer.with_decoder(Some(tokenizers::decoders::DecoderWrapper::from(ByteLevelDecoder::new(false, false, false))));
    tokenizer.add_tokens(added.iter().map(|(_, special, token)| {
        AddedToken::from(token.clone(), *special).normalized(false)
    })).map_err(|_| "invalid or unsupported GGUF tokenizer vocabulary".to_owned())?;

    let bos = special_token(metadata, &token_strings, token_types, "bos", true)?;
    let eos = special_token(metadata, &token_strings, token_types, "eos", false)?;
    let pad = special_token(metadata, &token_strings, token_types, "padding", true)?;
    let template = metadata.string_value("tokenizer.chat_template")?;
    if template.trim().is_empty() { return Err("GGUF chat template is empty".into()); }
    let architecture = metadata.string_value("general.architecture")?;
    if !matches!(architecture, "qwen35" | "qwen35moe" | "qwen4exp" | "flint") {
        return Err(format!("unsupported GGUF model architecture: {architecture}"));
    }
    let context = metadata.positive(&format!("{architecture}.context_length"))?;
    let added_decoder = added.into_iter().map(|(index, special, token)| (
        index.to_string(),
        serde_json::json!({
            "content": token,
            "special": special,
            "normalized": false,
            "single_word": false,
            "lstrip": false,
            "rstrip": false,
        }),
    )).collect::<serde_json::Map<_, _>>();
    let tokenizer_config = serde_json::json!({
        "tokenizer_class": "TokenizersBackend",
        "model_max_length": context,
        "clean_up_tokenization_spaces": false,
        "bos_token": bos,
        "eos_token": eos,
        "pad_token": pad,
        "added_tokens_decoder": added_decoder,
    });
    let tokenizer_json = tokenizer.to_string(false)
        .map_err(|error| format!("cannot serialize GGUF tokenizer: {error}"))?;
    let tokenizer_config = pretty_json(&tokenizer_config)?;
    let mut files = BTreeMap::new();
    files.insert("tokenizer/tokenizer.json".into(), tokenizer_json.into_bytes());
    files.insert("tokenizer/tokenizer_config.json".into(), tokenizer_config);
    files.insert("tokenizer/chat_template.jinja".into(), template.as_bytes().to_vec());
    Ok(files)
}

fn special_token(
    metadata: &Metadata,
    tokens: &[String],
    types: &[Value],
    name: &str,
    optional: bool,
) -> Result<Option<String>, String> {
    let key = format!("tokenizer.ggml.{name}_token_id");
    let Some(value) = metadata.values.get(&key) else {
        if optional { return Ok(None); }
        return Err(format!("missing or invalid GGUF metadata: {key}"));
    };
    let Some(index) = value.as_u64().and_then(|index| usize::try_from(index).ok()) else {
        return Err(format!("missing or invalid GGUF metadata: {key}"));
    };
    if index >= tokens.len() || types[index].as_u64() != Some(3) {
        return Err(format!("invalid GGUF special token: {key}"));
    }
    Ok(Some(tokens[index].clone()))
}

#[derive(Clone, Copy)]
enum TokenizerProfile { Qwen35, Flint }
impl TokenizerProfile {
    fn nfc(self) -> bool { matches!(self, Self::Qwen35) }
    fn patterns(self) -> &'static [&'static str] {
        match self {
            Self::Qwen35 => &[r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"],
            Self::Flint => &[
                r"\p{N}{1,3}",
                "[一-龥぀-ゟ゠-ヿ]+",
                r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            ],
        }
    }
}

fn pretty_json(value: &Value) -> Result<Vec<u8>, String> {
    fn sorted(value: &Value) -> Value {
        match value {
            Value::Object(object) => {
                let mut keys = object.keys().collect::<Vec<_>>();
                keys.sort();
                let mut result = serde_json::Map::new();
                for key in keys { result.insert(key.clone(), sorted(&object[key])); }
                Value::Object(result)
            }
            Value::Array(values) => Value::Array(values.iter().map(sorted).collect()),
            _ => value.clone(),
        }
    }
    let mut bytes = serde_json::to_vec_pretty(&sorted(value)).map_err(|error| error.to_string())?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn loaded_layers(metadata: &Metadata, arch: &str) -> Result<u64, String> {
    let layers = metadata.positive(&format!("{arch}.block_count"))?;
    let nextn = match metadata.values.get(&format!("{arch}.nextn_predict_layers")) {
        None => 0,
        Some(value) => value.as_u64().ok_or("invalid GGUF NextN layer count")?,
    };
    if nextn >= layers { return Err("invalid GGUF NextN layer count".into()); }
    Ok(layers - nextn)
}

fn add_flint_config(metadata: &Metadata, arch: &str, text: &mut serde_json::Map<String, Value>) -> Result<(), String> {
    let gating = metadata.integer(&format!("{arch}.expert_gating_func"))?;
    let gating = match gating {
        1 => "softmax",
        2 => "sigmoid",
        other => return Err(format!("unsupported GGUF expert gating function: {other}")),
    };
    for (field, suffix) in [
        ("intermediate_size", "feed_forward_length"),
        ("moe_intermediate_size", "expert_feed_forward_length"),
        ("shared_expert_intermediate_size", "expert_shared_feed_forward_length"),
        ("num_experts_per_tok", "expert_used_count"),
        ("sliding_window", "attention.sliding_window"),
    ] {
        text.insert(field.into(), Value::from(metadata.positive(&format!("{arch}.{suffix}"))?));
    }
    text.insert("norm_topk_prob".into(), Value::Bool(metadata.boolean(&format!("{arch}.expert_weights_norm"))?));
    text.insert("router_score_function".into(), Value::String(gating.into()));
    let layers = loaded_layers(metadata, arch)?;
    let dense = metadata.integer(&format!("{arch}.leading_dense_block_count"))?;
    if dense < 0 || dense as u64 >= layers { return Err(format!("invalid GGUF metadata: {arch}.leading_dense_block_count")); }
    text.insert("mlp_only_layers".into(), Value::Array((0..dense).map(Value::from).collect()));
    let sliding = layer_pattern(metadata, arch, "attention.sliding_window_pattern", layers)?;
    text.insert("layer_types".into(), Value::Array(sliding.into_iter().map(|value| Value::String(if value { "sliding_attention" } else { "full_attention" }.into())).collect()));
    let rotated = layer_pattern(metadata, arch, "attention.rope_pattern", layers)?;
    text.insert("no_rope_layer_indices".into(), Value::Array(rotated.into_iter().enumerate().filter(|(_, rotates)| !*rotates).map(|(index, _)| Value::from(index as u64)).collect()));
    text.insert("eos_token_id".into(), Value::Array(flint_stop_tokens(metadata)?));
    Ok(())
}

fn layer_pattern(metadata: &Metadata, arch: &str, name: &str, layers: u64) -> Result<Vec<bool>, String> {
    let key = format!("{arch}.{name}");
    let values = metadata.array(&key)?;
    if values.len() as u64 != layers { return Err(format!("missing or invalid GGUF metadata: {key}")); }
    values.iter().map(|value| value.as_bool().ok_or_else(|| format!("missing or invalid GGUF metadata: {key}"))).collect()
}

fn flint_stop_tokens(metadata: &Metadata) -> Result<Vec<Value>, String> {
    if let Some(stops) = metadata.values.get("tokenizer.ggml.eog_token_ids") {
        let stops = stops.as_array().ok_or("invalid GGUF metadata: tokenizer.ggml.eog_token_ids")?;
        if stops.is_empty() || stops.iter().any(|stop| stop.as_i64().is_none()) { return Err("invalid GGUF metadata: tokenizer.ggml.eog_token_ids".into()); }
        return Ok(stops.clone());
    }
    let tokens = metadata.array("tokenizer.ggml.tokens")?;
    let mut stops = Vec::new();
    for stop in ["<|end_of_text|>", "<|user|>", "<|observation|>", "<|developer|>"] {
        let index = tokens.iter().position(|token| token.as_str() == Some(stop))
            .ok_or("missing GGUF metadata: tokenizer.ggml.eog_token_ids")?;
        stops.push(Value::from(index as u64));
    }
    Ok(stops)
}

pub fn vision_config(vision: &Metadata) -> Result<Value, String> {
    if vision.string_value("general.architecture")? != "clip"
        || vision.string_value("clip.projector_type")? != "qwen3vl_merger"
        || !vision.boolean("clip.use_gelu")? {
        return Err("unsupported GGUF vision architecture".into());
    }
    let fields = [
        ("depth", "block_count"),
        ("hidden_size", "embedding_length"),
        ("num_heads", "attention.head_count"),
        ("intermediate_size", "feed_forward_length"),
        ("out_hidden_size", "projection_dim"),
        ("patch_size", "patch_size"),
        ("spatial_merge_size", "spatial_merge_size"),
    ];
    let mut result = serde_json::Map::new();
    for (field, suffix) in fields {
        result.insert(field.into(), Value::from(vision.positive(&format!("clip.vision.{suffix}"))?));
    }
    let image_size = vision.positive("clip.vision.image_size")?;
    let patch_size = result["patch_size"].as_u64().unwrap();
    if image_size % patch_size != 0 { return Err("invalid GGUF vision position grid".into()); }
    if vision.array("clip.vision.is_deepstack_layers")?.iter().any(|layer| layer.as_bool() == Some(true)) {
        return Err("GGUF vision deepstack layers are unsupported".into());
    }
    result.insert("temporal_patch_size".into(), Value::from(2));
    result.insert("in_channels".into(), Value::from(3));
    let grid = image_size / patch_size;
    let positions = grid.checked_mul(grid).ok_or("invalid GGUF vision position grid")?;
    result.insert("num_position_embeddings".into(), Value::from(positions));
    result.insert("hidden_act".into(), Value::String("gelu_pytorch_tanh".into()));
    result.insert("deepstack_visual_indexes".into(), Value::Array(Vec::new()));
    Ok(Value::Object(result))
}

/// Validate a local Qwen3VL GGUF projector using the installer's accepted
/// precision and processor rules before exposing it to the native loader.
pub fn require_vision_loadable(vision: &Metadata) -> Result<(), String> {
    if vision.string_value("general.architecture")? != "clip" {
        return Err("unsupported GGUF vision projector architecture".into());
    }
    let types = vision.tensors.values().map(|tensor| {
        tensor_type_name(tensor.kind).unwrap_or("unknown")
    }).collect::<BTreeSet<_>>();
    let accepted = ["BF16", "F32", "F16", "Q8_0"].into_iter().collect::<BTreeSet<_>>();
    if types.is_empty() || !types.is_subset(&accepted) {
        let found = if types.is_empty() { "no tensors".to_owned() } else { types.iter().copied().collect::<Vec<_>>().join(", ") };
        return Err(format!("vision projector has unsupported tensor types: {found}"));
    }
    let config = vision_config(vision)?;
    let processor = [
        ("patch_size", Value::from(16)),
        ("temporal_patch_size", Value::from(2)),
        ("spatial_merge_size", Value::from(2)),
    ];
    for (key, expected) in processor {
        if config.get(key) != Some(&expected) {
            return Err("unsupported vision preprocessing configuration".into());
        }
    }
    let mean = vision.values.get("clip.vision.image_mean");
    let std = vision.values.get("clip.vision.image_std");
    let expected = Value::Array(vec![Value::from(0.5), Value::from(0.5), Value::from(0.5)]);
    if mean != Some(&expected) || std != Some(&expected) {
        return Err("unsupported vision preprocessing configuration".into());
    }
    Ok(())
}

/// Validate generated config against model-check, then check expected tensor
/// names/types and vision processor metadata. Payload ranges and generated
/// assembly sources remain separate checks before publication.
pub fn validate_with_engine(
    engine: &Path,
    target: &Metadata,
    vision: Option<&Metadata>,
    draft_config: Option<&Path>,
) -> Result<String, String> {
    require_loadable(target)?;
    if let Some(vision) = vision { require_vision_loadable(vision)?; }
    let directory = TempDirectory::new()?;
    let config_path = directory.0.join("config.json");
    let metadata_path = directory.0.join("gguf-metadata.json");
    write_json(&config_path, &model_config(target, None)?)?;
    write_json(&metadata_path, &target.scalar_metadata())?;
    let family = model_check(engine, "none", &config_path, &metadata_path, draft_config)?;
    if let Some(vision) = vision {
        write_json(&config_path, &model_config(target, Some(vision))?)?;
        let vision_family = model_check(engine, "gguf", &config_path, &metadata_path, draft_config)?;
        if family != vision_family {
            return Err("native model-check changed the GGUF family when adding vision".into());
        }
    }
    Ok(family)
}

fn model_check(
    engine: &Path,
    vision_format: &str,
    config: &Path,
    metadata: &Path,
    draft_config: Option<&Path>,
) -> Result<String, String> {
    let output = Command::new(engine)
        .args(["model-check", "gguf", vision_format])
        .arg(config)
        .arg(metadata)
        .args(draft_config.into_iter())
        .output()
        .map_err(|error| format!("cannot run native model-check using {}: {error}", engine.display()))?;
    if !output.status.success() {
        let reason = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        return Err(if reason.is_empty() { format!("native model-check failed with {}", output.status) }
            else { format!("native model-check refused the GGUF: {reason}") });
    }
    let result: Value = serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("native model-check returned invalid JSON: {error}"))?;
    result["family"].as_str().map(str::to_owned)
        .ok_or_else(|| "native model-check returned no family".into())
}

fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    let mut bytes = serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?;
    bytes.push(b'\n');
    fs::write(path, bytes).map_err(|error| format!("cannot write model-check input: {error}"))
}

struct TempDirectory(PathBuf);
impl TempDirectory {
    fn new() -> Result<Self, String> {
        for _ in 0..10 {
            let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
            let path = std::env::temp_dir().join(format!("splash-gguf-check-{}-{nonce}", std::process::id()));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self(path)),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(format!("cannot create model-check scratch directory: {error}")),
            }
        }
        Err("cannot allocate model-check scratch directory".into())
    }
}
impl Drop for TempDirectory {
    fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); }
}

struct BoundedReader<R> { input: R, consumed: usize }
impl<R: Read> BoundedReader<R> {
    fn read(&mut self, size: usize) -> Result<Vec<u8>, String> {
        if size > MAX_BYTES.saturating_sub(self.consumed) { return Err("GGUF metadata exceeds the size limit".into()); }
        let mut bytes = vec![0; size];
        self.input.read_exact(&mut bytes).map_err(|_| "truncated GGUF metadata".to_owned())?;
        self.consumed += size;
        Ok(bytes)
    }
    fn u8(&mut self) -> Result<u8, String> { Ok(self.read(1)?[0]) }
    fn u16(&mut self) -> Result<u16, String> { Ok(u16::from_le_bytes(self.read(2)?.try_into().unwrap())) }
    fn u32(&mut self) -> Result<u32, String> { Ok(u32::from_le_bytes(self.read(4)?.try_into().unwrap())) }
    fn u64(&mut self) -> Result<u64, String> { Ok(u64::from_le_bytes(self.read(8)?.try_into().unwrap())) }
    fn string(&mut self) -> Result<String, String> {
        let length = usize::try_from(self.u64()?).map_err(|_| "GGUF string is too large".to_owned())?;
        String::from_utf8(self.read(length)?).map_err(|_| "invalid UTF-8 in GGUF metadata".to_owned())
    }
    fn value(&mut self, kind: u32) -> Result<Value, String> {
        match kind {
            0 => Ok(Value::from(self.u8()?)),
            1 => Ok(Value::from(self.u8()? as i8)),
            2 => Ok(Value::from(self.u16()?)),
            3 => Ok(Value::from(self.u16()? as i16)),
            4 => Ok(Value::from(self.u32()?)),
            5 => Ok(Value::from(self.u32()? as i32)),
            6 => self.f32_value(),
            7 => Ok(Value::Bool(self.u8()? != 0)),
            8 => Ok(Value::String(self.string()?)),
            9 => {
                let element = self.u32()?;
                let count = self.u64()?;
                if !(matches!(element, 0..=8 | 10..=12)) || count > MAX_ITEMS {
                    return Err("unsupported or oversized GGUF metadata array".into());
                }
                let mut values = Vec::with_capacity(count as usize);
                for _ in 0..count { values.push(self.value(element)?); }
                Ok(Value::Array(values))
            }
            10 => Ok(Value::from(self.u64()?)),
            11 => Ok(Value::from(self.u64()? as i64)),
            12 => self.f64_value(),
            _ => Err("unknown GGUF metadata type".into()),
        }
    }
    fn f32_value(&mut self) -> Result<Value, String> {
        Ok(Number::from_f64(f32::from_le_bytes(self.read(4)?.try_into().unwrap()) as f64).map(Value::Number).unwrap_or(Value::Null))
    }
    fn f64_value(&mut self) -> Result<Value, String> {
        Ok(Number::from_f64(f64::from_le_bytes(self.read(8)?.try_into().unwrap())).map(Value::Number).unwrap_or(Value::Null))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn fixture() -> Vec<u8> {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes()); // tensor count
        bytes.extend_from_slice(&3u64.to_le_bytes()); // field count
        push_string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&8u32.to_le_bytes());
        push_string(&mut bytes, "qwen35");
        push_string(&mut bytes, "model.layers");
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&32u32.to_le_bytes());
        push_string(&mut bytes, "tokenizer.tokens");
        bytes.extend_from_slice(&9u32.to_le_bytes()); // array
        bytes.extend_from_slice(&8u32.to_le_bytes()); // strings
        bytes.extend_from_slice(&2u64.to_le_bytes());
        push_string(&mut bytes, "one");
        push_string(&mut bytes, "two");
        push_string(&mut bytes, "layer.0.weight");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&4u64.to_le_bytes());
        bytes.extend_from_slice(&8u64.to_le_bytes());
        bytes.extend_from_slice(&14u32.to_le_bytes());
        bytes.extend_from_slice(&16u64.to_le_bytes());
        bytes
    }

    fn encoded_gguf(mut metadata: Metadata) -> Vec<u8> {
        fn raw_string(bytes: &mut Vec<u8>, value: &str) {
            bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
            bytes.extend_from_slice(value.as_bytes());
        }
        fn scalar_kind(value: &Value) -> u32 {
            match value {
                Value::String(_) => 8,
                Value::Bool(_) => 7,
                Value::Number(number) if number.as_u64().is_some_and(|n| n <= u32::MAX as u64) => 4,
                Value::Number(number) if number.as_u64().is_some() => 10,
                Value::Number(number) if number.as_i64().is_some_and(|n| n < 0) => 5,
                Value::Number(_) => 12,
                Value::Array(values) => values.first().map(scalar_kind).unwrap_or(7),
                _ => panic!("unsupported test metadata value: {value}"),
            }
        }
        fn raw_value(bytes: &mut Vec<u8>, value: &Value, kind: u32) {
            match kind {
                4 => bytes.extend_from_slice(&(value.as_u64().unwrap() as u32).to_le_bytes()),
                5 => bytes.extend_from_slice(&(value.as_i64().unwrap() as i32).to_le_bytes()),
                7 => bytes.push(u8::from(value.as_bool().unwrap())),
                8 => raw_string(bytes, value.as_str().unwrap()),
                10 => bytes.extend_from_slice(&value.as_u64().unwrap().to_le_bytes()),
                12 => bytes.extend_from_slice(&value.as_f64().unwrap().to_le_bytes()),
                _ => panic!("unsupported test GGUF value kind {kind}"),
            }
        }
        let mut offset = 0u64;
        let expected = if metadata.values.get("general.architecture").and_then(Value::as_str) == Some("clip") {
            None
        } else { Some(expected_tensors(&metadata).unwrap()) };
        for (name, tensor) in &mut metadata.tensors {
            let kind = expected.as_ref().map(|expected| {
                if expected[name].contains("Q4_0") { 2 } else { 0 }
            }).unwrap_or(0);
            tensor.extents = vec![32];
            tensor.kind = kind;
            tensor.offset = offset;
            let (block_elements, block_bytes) = ggml_type_layout(kind).unwrap();
            offset += (32 / block_elements) * block_bytes;
            offset = (offset + 31) / 32 * 32;
        }
        if let Some((_, tensor)) = metadata.tensors.iter_mut().next_back() {
            let (block_elements, block_bytes) = ggml_type_layout(tensor.kind).unwrap();
            let bytes = (32 / block_elements) * block_bytes;
            offset = tensor.offset + bytes;
        }
        let mut output = b"GGUF".to_vec();
        output.extend_from_slice(&VERSION.to_le_bytes());
        output.extend_from_slice(&(metadata.tensors.len() as u64).to_le_bytes());
        output.extend_from_slice(&(metadata.values.len() as u64).to_le_bytes());
        for (key, value) in &metadata.values {
            raw_string(&mut output, key);
            let kind = scalar_kind(value);
            output.extend_from_slice(&(if matches!(value, Value::Array(_)) { 9 } else { kind }).to_le_bytes());
            if let Value::Array(values) = value {
                output.extend_from_slice(&kind.to_le_bytes());
                output.extend_from_slice(&(values.len() as u64).to_le_bytes());
                for value in values { raw_value(&mut output, value, kind); }
            } else { raw_value(&mut output, value, kind); }
        }
        for (name, tensor) in &metadata.tensors {
            raw_string(&mut output, name);
            output.extend_from_slice(&(tensor.extents.len() as u32).to_le_bytes());
            for dim in &tensor.extents { output.extend_from_slice(&dim.to_le_bytes()); }
            output.extend_from_slice(&tensor.kind.to_le_bytes());
            output.extend_from_slice(&tensor.offset.to_le_bytes());
        }
        let data_offset = (output.len() + 31) / 32 * 32;
        output.resize(data_offset + offset as usize, 0);
        output
    }

    fn config_metadata(architecture: &str, layers: u64) -> Metadata {
        let mut values = BTreeMap::from([
            ("general.architecture".into(), Value::String(architecture.into())),
            (format!("{architecture}.embedding_length"), Value::from(4096)),
            (format!("{architecture}.context_length"), Value::from(32768)),
            (format!("{architecture}.attention.head_count"), Value::from(32)),
            (format!("{architecture}.attention.head_count_kv"), Value::from(8)),
            (format!("{architecture}.attention.key_length"), Value::from(128)),
            (format!("{architecture}.block_count"), Value::from(layers)),
            ("tokenizer.ggml.tokens".into(), serde_json::json!(["a", "b", "c"])),
        ]);
        if architecture == "qwen35" || architecture == "qwen35moe" {
            values.insert(format!("{architecture}.nextn_predict_layers"), Value::from(2));
        }
        if architecture == "qwen35moe" || architecture == "qwen4exp" {
            values.insert(format!("{architecture}.expert_count"), Value::from(64));
        }
        if architecture == "flint" {
            values.extend([
                (format!("{architecture}.expert_count"), Value::from(16)),
                (format!("{architecture}.expert_gating_func"), Value::from(1)),
                (format!("{architecture}.feed_forward_length"), Value::from(14336)),
                (format!("{architecture}.expert_feed_forward_length"), Value::from(2048)),
                (format!("{architecture}.expert_shared_feed_forward_length"), Value::from(1024)),
                (format!("{architecture}.expert_used_count"), Value::from(4)),
                (format!("{architecture}.expert_weights_norm"), Value::Bool(true)),
                (format!("{architecture}.leading_dense_block_count"), Value::from(1)),
                (format!("{architecture}.attention.sliding_window"), Value::from(4096)),
                (format!("{architecture}.attention.sliding_window_pattern"), serde_json::json!([true, false, true])),
                (format!("{architecture}.attention.rope_pattern"), serde_json::json!([true, false, true])),
                ("tokenizer.ggml.eog_token_ids".into(), serde_json::json!([1, 2])),
            ]);
        }
        Metadata { values, tensors: BTreeMap::new(), data_offset: None }
    }

    fn vision_metadata() -> Metadata {
        let values = BTreeMap::from([
            ("general.architecture".into(), Value::String("clip".into())),
            ("clip.projector_type".into(), Value::String("qwen3vl_merger".into())),
            ("clip.use_gelu".into(), Value::Bool(true)),
            ("clip.vision.block_count".into(), Value::from(24)),
            ("clip.vision.embedding_length".into(), Value::from(1024)),
            ("clip.vision.attention.head_count".into(), Value::from(16)),
            ("clip.vision.feed_forward_length".into(), Value::from(4096)),
            ("clip.vision.projection_dim".into(), Value::from(3584)),
            ("clip.vision.patch_size".into(), Value::from(14)),
            ("clip.vision.spatial_merge_size".into(), Value::from(2)),
            ("clip.vision.image_size".into(), Value::from(448)),
            ("clip.vision.is_deepstack_layers".into(), serde_json::json!([])),
        ]);
        Metadata { values, tensors: BTreeMap::new(), data_offset: None }
    }

    fn loadable_qwen_metadata() -> Metadata {
        let mut metadata = config_metadata("qwen35", 3);
        metadata.values.insert("qwen35.nextn_predict_layers".into(), Value::from(1));
        metadata.values.insert("qwen35.full_attention_interval".into(), Value::from(2));
        metadata.tensors = expected_tensors(&metadata).unwrap().into_iter().map(|(name, types)| {
            let kind = if types.contains("Q4_0") { 2 } else { 0 };
            (name, Tensor { extents: vec![1, 1], kind, offset: 0 })
        }).collect();
        metadata
    }

    fn tokenizer_metadata(profile: &str) -> Metadata {
        let mut tokens = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet()
            .into_iter().collect::<Vec<_>>();
        tokens.sort_unstable();
        let mut token_strings = tokens.into_iter().map(String::from).collect::<Vec<_>>();
        token_strings.extend(["ab", "12", "123", "<|im_end|>", "<|im_start|>", "<|end_of_text|>", "<think>"]
            .into_iter().map(str::to_owned));
        let mut types = vec![Value::from(1); token_strings.len()];
        let index = |name: &str| token_strings.iter().position(|token| token == name).unwrap();
        for name in ["<|im_end|>", "<|im_start|>", "<|end_of_text|>"] { types[index(name)] = Value::from(3); }
        types[index("<think>")] = Value::from(4);
        let merges = if profile == "flint" { serde_json::json!(["1 2", "12 3"]) }
            else { serde_json::json!(["a b"]) };
        let (bos, eos) = if profile == "flint" { ("<|im_start|>", "<|end_of_text|>") }
            else { ("<|im_start|>", "<|im_end|>") };
        let values = BTreeMap::from([
            ("general.architecture".into(), Value::String(profile.into())),
            (format!("{profile}.context_length"), Value::from(4096)),
            ("tokenizer.ggml.model".into(), Value::String("gpt2".into())),
            ("tokenizer.ggml.pre".into(), Value::String(profile.into())),
            ("tokenizer.ggml.tokens".into(), Value::Array(token_strings.iter().cloned().map(Value::String).collect())),
            ("tokenizer.ggml.token_type".into(), Value::Array(types)),
            ("tokenizer.ggml.merges".into(), merges),
            ("tokenizer.ggml.bos_token_id".into(), Value::from(index(bos) as u64)),
            ("tokenizer.ggml.eos_token_id".into(), Value::from(index(eos) as u64)),
            ("tokenizer.chat_template".into(), Value::String("{% for message in messages %}{{ message.content }}{% endfor %}".into())),
        ]);
        Metadata { values, tensors: BTreeMap::new(), data_offset: None }
    }

    #[test]
    fn reads_bounded_metadata_and_tensor_descriptors() {
        let metadata = Metadata::from_reader(&fixture()[..], true).unwrap();
        assert_eq!(metadata.values["general.architecture"], "qwen35");
        assert_eq!(metadata.values["model.layers"], 32);
        assert_eq!(metadata.values["tokenizer.tokens"], serde_json::json!(["one", "two"]));
        assert_eq!(metadata.tensors["layer.0.weight"], Tensor { extents: vec![4, 8], kind: 14, offset: 16 });
        assert_eq!(Metadata::from_reader(&fixture()[..], false).unwrap().tensors.len(), 0);
    }

    #[test]
    fn local_target_directory_selects_one_model_and_refuses_ambiguous_ggufs() {
        let scratch = TempDirectory::new().unwrap();
        let directory = scratch.0.join("targets");
        fs::create_dir_all(&directory).unwrap();
        let single = directory.join("model-Q4.gguf");
        let projector = directory.join("mmproj-F16.gguf");
        fs::write(&single, []).unwrap();
        fs::write(&projector, []).unwrap();
        assert_eq!(select_local_target(&directory).unwrap(), single);

        let first = directory.join("other-Q4.gguf");
        fs::write(&first, []).unwrap();
        let error = select_local_target(&directory).unwrap_err();
        assert!(error.contains("multiple GGUF targets"));
        assert!(error.contains("model-Q4.gguf"));
        assert!(error.contains("other-Q4.gguf"));
    }

    #[test]
    fn local_dflash2_checkpoint_files_are_resolved_from_its_index() {
        let scratch = TempDirectory::new().unwrap();
        let draft = scratch.0.join("draft");
        fs::create_dir_all(&draft).unwrap();
        fs::write(draft.join("config.json"), b"{}").unwrap();
        fs::write(draft.join("model-00001-of-00002.safetensors"), b"one").unwrap();
        fs::write(draft.join("model-00002-of-00002.safetensors"), b"two").unwrap();
        fs::write(draft.join("unused.safetensors"), b"unused").unwrap();
        fs::write(draft.join("model.safetensors.index.json"), br#"{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}"#).unwrap();

        let files = crate::local_draft::files(&draft).unwrap();
        let names = files.iter().map(|(name, _)| name.as_str()).collect::<BTreeSet<_>>();
        assert_eq!(names, BTreeSet::from([
            "draft/config.json",
            "draft/model-00001-of-00002.safetensors",
            "draft/model-00002-of-00002.safetensors",
            "draft/model.safetensors.index.json",
        ]));
    }

    #[cfg(unix)]
    #[test]
    fn prepares_validated_local_gguf_and_refuses_split_without_publishing() {
        use std::os::unix::fs::PermissionsExt;
        use std::os::unix::fs::symlink;
        let scratch = TempDirectory::new().unwrap();
        let mut metadata = loadable_qwen_metadata();
        metadata.values.extend(tokenizer_metadata("qwen35").values);
        let target = scratch.0.join("model.gguf");
        fs::write(&target, encoded_gguf(metadata.clone())).unwrap();
        let engine = scratch.0.join("fake-engine");
        fs::write(&engine, "#!/bin/sh\nprintf '{\"family\":\"Qwen3.8-4B\"}\\n'\n").unwrap();
        let mut permissions = fs::metadata(&engine).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&engine, permissions).unwrap();
        let mut vision_metadata = vision_metadata();
        vision_metadata.values.insert("clip.vision.patch_size".into(), Value::from(16));
        vision_metadata.values.insert("clip.vision.image_mean".into(), serde_json::json!([0.5, 0.5, 0.5]));
        vision_metadata.values.insert("clip.vision.image_std".into(), serde_json::json!([0.5, 0.5, 0.5]));
        vision_metadata.tensors.insert("v.patch_embd.weight".into(), Tensor { extents: vec![32], kind: 0, offset: 0 });
        let projector = scratch.0.join("mmproj.gguf");
        fs::write(&projector, encoded_gguf(vision_metadata)).unwrap();
        let models = scratch.0.join("models");
        let published = prepare_local(&target, Some(&projector), &models, "local/example", &engine).unwrap();
        let record: Value = serde_json::from_slice(&fs::read(published.join("model.json")).unwrap()).unwrap();
        assert_eq!(record["target_format"], "gguf");
        assert_eq!(record["vision_format"], "gguf");
        assert_eq!(record["sources"]["target"]["repo"], fs::canonicalize(&target).unwrap().to_string_lossy().as_ref());
        assert_eq!(record["speculation"]["provider"], "none");
        assert!(record["files"].get("gguf-metadata.json").is_some());
        assert!(fs::read(published.join("config.json")).unwrap().ends_with(b"\n"));
        assert!(fs::read(published.join("gguf-metadata.json")).unwrap().ends_with(b"\n"));
        assert!(published.join("tokenizer/tokenizer.json").is_file());
        assert!(published.join("vision/mmproj.gguf").is_file());
        assert!(crate::assembly::verify(&published, &record, false).is_ok());

        let draft = scratch.0.join("draft");
        fs::create_dir_all(&draft).unwrap();
        fs::write(draft.join("config.json"), b"{\"dflash_config\":{}}\n").unwrap();
        fs::write(draft.join("model.safetensors"), b"draft weights").unwrap();
        let draft_models = scratch.0.join("draft-models");
        let drafted = prepare_local_with_draft(
            &target, Some(&projector), Some(&draft), &draft_models, "local/example-draft", &engine,
        ).unwrap();
        let drafted_record: Value = serde_json::from_slice(&fs::read(drafted.join("model.json")).unwrap()).unwrap();
        assert_eq!(drafted_record["version"], 1);
        assert_eq!(drafted_record["sources"]["draft"]["repo"], fs::canonicalize(&draft).unwrap().to_string_lossy().as_ref());
        assert!(drafted.join("draft/config.json").is_file());
        assert!(drafted.join("draft/model.safetensors").is_file());
        assert!(crate::assembly::verify(&drafted, &drafted_record, false).is_ok());

        // Hugging Face snapshots link their `.gguf` name to an extensionless
        // blob. The published assembly must retain the snapshot basename so
        // the native loader recognizes it as a GGUF.
        let blob = scratch.0.join("blob-without-extension");
        fs::write(&blob, encoded_gguf({ let mut m = loadable_qwen_metadata(); m.values.extend(tokenizer_metadata("qwen35").values); m })).unwrap();
        let snapshot = scratch.0.join("snapshot/model-alias.gguf");
        fs::create_dir_all(snapshot.parent().unwrap()).unwrap();
        symlink(&blob, &snapshot).unwrap();
        let alias_models = scratch.0.join("alias-models");
        let alias_published = prepare_local(&snapshot, None, &alias_models, "local/alias", &engine).unwrap();
        assert!(alias_published.join("target/model-alias.gguf").is_file());
        let alias_link = fs::read_link(alias_published.join("target/model-alias.gguf")).unwrap();
        assert_eq!(alias_link, fs::canonicalize(&blob).unwrap());
        let alias_record: Value = serde_json::from_slice(&fs::read(alias_published.join("model.json")).unwrap()).unwrap();
        assert_eq!(alias_record["sources"]["target"]["repo"], fs::canonicalize(&blob).unwrap().to_string_lossy().as_ref());

        let failing_engine = scratch.0.join("failing-engine");
        fs::write(&failing_engine, "#!/bin/sh\nprintf '{}\\n'\n").unwrap();
        let mut permissions = fs::metadata(&failing_engine).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&failing_engine, permissions).unwrap();
        let failed_root = scratch.0.join("failed-models");
        assert!(prepare_local(&target, None, &failed_root, "local/failed", &failing_engine).is_err());
        assert!(!failed_root.join(".resolved").exists());
        assert!(!failed_root.join(".generated").exists());

        metadata.values.insert("split.count".into(), Value::from(2));
        let split = scratch.0.join("split.gguf");
        fs::write(&split, encoded_gguf(metadata)).unwrap();
        let refused_root = scratch.0.join("refused-models");
        assert!(prepare_local(&split, None, &refused_root, "local/split", &engine).unwrap_err().contains("split metadata"));
        assert!(!refused_root.join(".resolved").exists());
        assert!(!refused_root.join(".generated").exists());
    }

    #[cfg(unix)]
    #[test]
    fn prepares_split_gguf_from_any_part_and_refuses_incomplete_sets() {
        use std::os::unix::fs::PermissionsExt;
        let scratch = TempDirectory::new().unwrap();
        let mut base = loadable_qwen_metadata();
        base.values.extend(tokenizer_metadata("qwen35").values);
        let tensor_count = base.tensors.len() as u64;
        let tensors = base.tensors.iter().map(|(name, tensor)| (name.clone(), tensor.clone())).collect::<Vec<_>>();
        let directory = scratch.0.join("split");
        fs::create_dir_all(&directory).unwrap();
        let names = (1..=2).map(|number| format!("model-Q4_K_M-{number:05}-of-00002.gguf")).collect::<Vec<_>>();
        for number in 0..2 {
            let mut part = base.clone();
            part.values.insert("split.count".into(), Value::from(2));
            part.values.insert("split.no".into(), Value::from(number));
            part.values.insert("split.tensors.count".into(), Value::from(tensor_count));
            part.tensors = tensors.iter().enumerate()
                .filter(|(index, _)| index % 2 == number as usize)
                .map(|(_, (name, tensor))| (name.clone(), tensor.clone()))
                .collect();
            fs::write(directory.join(&names[number as usize]), encoded_gguf(part)).unwrap();
        }
        let engine = scratch.0.join("fake-engine");
        fs::write(&engine, "#!/bin/sh\nprintf '{\\\"family\\\":\\\"Qwen3.8-4B\\\"}\\n'\n").unwrap();
        let mut permissions = fs::metadata(&engine).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&engine, permissions).unwrap();

        // Selecting part 2 discovers and validates the complete set, then
        // publishes every part beside one another in numeric order.
        let models = scratch.0.join("models");
        let published = prepare_local(&directory.join(&names[1]), None, &models, "local/split", &engine).unwrap();
        let record: Value = serde_json::from_slice(&fs::read(published.join("model.json")).unwrap()).unwrap();
        assert!(published.join("target").join(&names[0]).is_file());
        assert!(published.join("target").join(&names[1]).is_file());
        assert_eq!(record["files"].as_object().unwrap().keys().filter(|key| key.starts_with("target/")).count(), 2);
        assert!(crate::assembly::verify(&published, &record, true).is_ok());

        fs::remove_file(directory.join(&names[0])).unwrap();
        let incomplete_models = scratch.0.join("incomplete-models");
        let error = prepare_local(&directory.join(&names[1]), None, &incomplete_models, "local/incomplete", &engine).unwrap_err();
        assert!(error.contains("incomplete or has extra parts"));
        assert!(!incomplete_models.join(".resolved").exists());
        assert!(!incomplete_models.join(".generated").exists());
    }

    #[test]
    fn validates_tensor_layout_ranges_types_alignment_and_overlap() {
        let scratch = TempDirectory::new().unwrap();
        let file = scratch.0.join("payload.gguf");
        fs::write(&file, vec![0u8; 128]).unwrap();
        let metadata = |offset, kind| Metadata {
            values: BTreeMap::new(),
            tensors: BTreeMap::from([("weight".into(), Tensor { extents: vec![32], kind, offset })]),
            data_offset: Some(0),
        };
        assert!(validate_file_layout(&file, &metadata(0, 2)).is_ok()); // Q4_0 = 18 bytes
        assert!(validate_file_layout(&file, &metadata(1, 2)).unwrap_err().contains("misaligned"));
        assert!(validate_file_layout(&file, &metadata(128, 2)).unwrap_err().contains("past the end"));
        assert!(validate_file_layout(&file, &metadata(0, 999)).unwrap_err().contains("unknown ggml type"));

        let mut overlapping = metadata(0, 0);
        overlapping.tensors.insert("other".into(), Tensor { extents: vec![8], kind: 0, offset: 0 });
        assert!(validate_file_layout(&file, &overlapping).unwrap_err().contains("overlapping"));
        overlapping.values.insert("split.count".into(), Value::from(2));
        assert!(validate_file_layout(&file, &overlapping).unwrap_err().contains("split"));
    }

    #[test]
    fn rejects_duplicates_bad_utf8_oversized_counts_and_truncation() {
        let mut duplicate = b"GGUF".to_vec();
        duplicate.extend_from_slice(&VERSION.to_le_bytes());
        duplicate.extend_from_slice(&0u64.to_le_bytes());
        duplicate.extend_from_slice(&2u64.to_le_bytes());
        for _ in 0..2 {
            push_string(&mut duplicate, "same");
            duplicate.extend_from_slice(&8u32.to_le_bytes());
            push_string(&mut duplicate, "value");
        }
        assert!(Metadata::from_reader(&duplicate[..], false).unwrap_err().contains("duplicate"));
        let mut oversized = b"GGUF".to_vec();
        oversized.extend_from_slice(&VERSION.to_le_bytes());
        oversized.extend_from_slice(&0u64.to_le_bytes());
        oversized.extend_from_slice(&(MAX_FIELDS + 1).to_le_bytes());
        assert!(Metadata::from_reader(&oversized[..], false).unwrap_err().contains("too many"));
        assert!(Metadata::from_reader(&fixture()[..fixture().len() - 1], true).unwrap_err().contains("truncated"));
    }

    #[test]
    fn model_config_matches_python_architecture_shapes() {
        let qwen = config_metadata("qwen35", 32);
        assert_eq!(model_config(&qwen, None).unwrap(), serde_json::json!({
            "model_type":"qwen3_5",
            "text_config":{
                "hidden_size":4096,"max_position_embeddings":32768,"num_attention_heads":32,
                "num_key_value_heads":8,"head_dim":128,"num_hidden_layers":30,
                "model_type":"qwen3_5_text","vocab_size":3
            }
        }));

        let qwen_moe = config_metadata("qwen35moe", 40);
        assert_eq!(model_config(&qwen_moe, None).unwrap(), serde_json::json!({
            "model_type":"qwen3_5_moe",
            "text_config":{
                "hidden_size":4096,"max_position_embeddings":32768,"num_attention_heads":32,
                "num_key_value_heads":8,"head_dim":128,"num_experts":64,"num_hidden_layers":38,
                "model_type":"qwen3_5_moe_text","vocab_size":3
            }
        }));

        let qwen4 = config_metadata("qwen4exp", 24);
        assert_eq!(model_config(&qwen4, None).unwrap(), serde_json::json!({
            "model_type":"qwen4_exp",
            "text_config":{
                "hidden_size":4096,"max_position_embeddings":32768,"num_attention_heads":32,
                "num_key_value_heads":8,"head_dim":128,"num_experts":64,"num_hidden_layers":24,
                "model_type":"qwen4_exp_text","vocab_size":3
            }
        }));

        let flint = config_metadata("flint", 3);
        assert_eq!(model_config(&flint, None).unwrap(), serde_json::json!({
            "hidden_size":4096,"max_position_embeddings":32768,"num_attention_heads":32,
            "num_key_value_heads":8,"head_dim":128,"num_experts":16,"num_hidden_layers":3,
            "model_type":"flint","vocab_size":3,"intermediate_size":14336,
            "moe_intermediate_size":2048,"shared_expert_intermediate_size":1024,
            "num_experts_per_tok":4,"norm_topk_prob":true,"router_score_function":"softmax",
            "mlp_only_layers":[0],"sliding_window":4096,
            "layer_types":["sliding_attention","full_attention","sliding_attention"],
            "no_rope_layer_indices":[1],"eos_token_id":[1,2]
        }));
    }

    #[test]
    fn vision_config_and_invalid_architectures_match_python_rules() {
        let target = config_metadata("qwen35", 32);
        let config = model_config(&target, Some(&vision_metadata())).unwrap();
        assert_eq!(config["vision_config"], serde_json::json!({
            "depth":24,"hidden_size":1024,"num_heads":16,"intermediate_size":4096,
            "out_hidden_size":3584,"patch_size":14,"spatial_merge_size":2,
            "temporal_patch_size":2,"in_channels":3,"num_position_embeddings":1024,
            "hidden_act":"gelu_pytorch_tanh","deepstack_visual_indexes":[]
        }));
        let unknown = config_metadata("llama", 32);
        assert_eq!(model_config(&unknown, None).unwrap_err(), "unsupported GGUF model architecture: llama");
        let mut invalid = config_metadata("flint", 3);
        invalid.values.insert("flint.expert_gating_func".into(), Value::from(7));
        assert!(model_config(&invalid, None).unwrap_err().contains("unsupported GGUF expert gating function"));
    }

    #[test]
    fn loadable_tensor_screening_accepts_supported_and_refuses_missing_bad_or_unquantized() {
        let valid = loadable_qwen_metadata();
        require_loadable(&valid).unwrap();

        let mut missing = valid.clone();
        missing.tensors.remove("blk.0.attn_qkv.weight");
        assert!(require_loadable(&missing).unwrap_err().contains("missing"));

        let mut unsupported = valid.clone();
        unsupported.tensors.get_mut("output.weight").unwrap().kind = 999;
        assert!(require_loadable(&unsupported).unwrap_err().contains("type 999"));

        let mut unquantized = valid;
        for tensor in unquantized.tensors.values_mut() {
            if tensor_type_name(tensor.kind).is_some_and(|name| format_types().contains(name)) {
                tensor.kind = 0;
            }
        }
        assert!(require_loadable(&unquantized).unwrap_err().contains("unquantized"));
    }

    #[test]
    fn vision_projector_requires_python_supported_precision_and_processor() {
        let mut vision = vision_metadata();
        vision.values.insert("clip.vision.patch_size".into(), Value::from(16));
        vision.values.insert("clip.vision.image_mean".into(), serde_json::json!([0.5, 0.5, 0.5]));
        vision.values.insert("clip.vision.image_std".into(), serde_json::json!([0.5, 0.5, 0.5]));
        vision.tensors.insert("v.patch_embd.weight".into(), Tensor { extents: vec![1], kind: 8, offset: 0 });
        require_vision_loadable(&vision).unwrap();

        vision.tensors.get_mut("v.patch_embd.weight").unwrap().kind = 12;
        assert!(require_vision_loadable(&vision).unwrap_err().contains("unsupported tensor types"));
    }

    #[test]
    fn qwen_tokenizer_files_match_bpe_nfc_added_tokens_and_config_contract() {
        let metadata = tokenizer_metadata("qwen35");
        let files = tokenizer_files(&metadata).unwrap();
        assert_eq!(files.keys().map(String::as_str).collect::<Vec<_>>(), [
            "tokenizer/chat_template.jinja", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"
        ]);
        assert_eq!(files["tokenizer/chat_template.jinja"], metadata.string_value("tokenizer.chat_template").unwrap().as_bytes());
        let config: Value = serde_json::from_slice(&files["tokenizer/tokenizer_config.json"]).unwrap();
        assert_eq!(config["tokenizer_class"], "TokenizersBackend");
        assert_eq!(config["model_max_length"], 4096);
        assert_eq!(config["bos_token"], "<|im_start|>");
        assert_eq!(config["eos_token"], "<|im_end|>");
        assert!(config["pad_token"].is_null());
        assert_eq!(config["added_tokens_decoder"]["262"], serde_json::json!({
            "content":"<think>","special":false,"normalized":false,
            "single_word":false,"lstrip":false,"rstrip":false
        }));
        assert_eq!(std::str::from_utf8(&files["tokenizer/tokenizer_config.json"]).unwrap(), concat!(
            "{\n",
            "  \"added_tokens_decoder\": {\n",
            "    \"259\": {\n      \"content\": \"<|im_end|>\",\n      \"lstrip\": false,\n      \"normalized\": false,\n      \"rstrip\": false,\n      \"single_word\": false,\n      \"special\": true\n    },\n",
            "    \"260\": {\n      \"content\": \"<|im_start|>\",\n      \"lstrip\": false,\n      \"normalized\": false,\n      \"rstrip\": false,\n      \"single_word\": false,\n      \"special\": true\n    },\n",
            "    \"261\": {\n      \"content\": \"<|end_of_text|>\",\n      \"lstrip\": false,\n      \"normalized\": false,\n      \"rstrip\": false,\n      \"single_word\": false,\n      \"special\": true\n    },\n",
            "    \"262\": {\n      \"content\": \"<think>\",\n      \"lstrip\": false,\n      \"normalized\": false,\n      \"rstrip\": false,\n      \"single_word\": false,\n      \"special\": false\n    }\n",
            "  },\n",
            "  \"bos_token\": \"<|im_start|>\",\n",
            "  \"clean_up_tokenization_spaces\": false,\n",
            "  \"eos_token\": \"<|im_end|>\",\n",
            "  \"model_max_length\": 4096,\n",
            "  \"pad_token\": null,\n",
            "  \"tokenizer_class\": \"TokenizersBackend\"\n",
            "}\n"
        ));

        let tokenizer_json = &files["tokenizer/tokenizer.json"];
        let graph: Value = serde_json::from_slice(tokenizer_json).unwrap();
        assert_eq!(graph["normalizer"]["type"], "NFC");
        assert_eq!(graph["pre_tokenizer"]["type"], "Sequence");
        assert_eq!(graph["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"], r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+");
        assert_eq!(graph["post_processor"]["type"], "ByteLevel");
        assert_eq!(graph["decoder"]["type"], "ByteLevel");
        let tokenizer = tokenizers::Tokenizer::from_bytes(tokenizer_json).unwrap();
        let tokens = metadata.array("tokenizer.ggml.tokens").unwrap();
        let id = |name: &str| tokens.iter().position(|value| value.as_str() == Some(name)).unwrap() as u32;
        assert_eq!(tokenizer.encode("ab<think><|im_end|>", false).unwrap().get_ids(), [id("ab"), id("<think>"), id("<|im_end|>")]);
        assert_eq!(tokenizer.encode("e\u{301}", false).unwrap().get_ids(), tokenizer.encode("é", false).unwrap().get_ids());
        assert_eq!(tokenizer.decode(&[id("<|im_end|>"), id("<think>")], true).unwrap(), "<think>");
    }

    #[test]
    fn flint_tokenizer_files_keep_flint_splits_without_nfc() {
        let metadata = tokenizer_metadata("flint");
        let files = tokenizer_files(&metadata).unwrap();
        let graph: Value = serde_json::from_slice(&files["tokenizer/tokenizer.json"]).unwrap();
        assert!(graph["normalizer"].is_null());
        let splits = graph["pre_tokenizer"]["pretokenizers"].as_array().unwrap();
        assert_eq!(splits.len(), 4);
        let expected_patterns = [
            r"\p{N}{1,3}",
            "[一-龥぀-ゟ゠-ヿ]+",
            r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        ];
        for (actual, expected) in splits.iter().zip(expected_patterns) {
            assert_eq!(actual["type"], "Split");
            assert_eq!(actual["pattern"]["Regex"], expected);
            assert_eq!(actual["behavior"], "Isolated");
            assert_eq!(actual["invert"], false);
        }
        assert_eq!(splits[1]["pattern"]["Regex"], "[一-龥぀-ゟ゠-ヿ]+");
        let tokenizer = tokenizers::Tokenizer::from_bytes(&files["tokenizer/tokenizer.json"]).unwrap();
        assert_ne!(tokenizer.encode("e\u{301}", false).unwrap().get_ids(), tokenizer.encode("é", false).unwrap().get_ids());
        assert_eq!(files["tokenizer/chat_template.jinja"], metadata.string_value("tokenizer.chat_template").unwrap().as_bytes());
    }

    #[test]
    fn tokenizer_rejects_missing_byte_tokens_invalid_special_ids_and_automatic_bos() {
        let mut metadata = tokenizer_metadata("qwen35");
        let tokens = metadata.array("tokenizer.ggml.tokens").unwrap();
        let types = metadata.array("tokenizer.ggml.token_type").unwrap();
        let alphabet = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet();
        let remove = tokens.iter().position(|value| value.as_str().is_some_and(|token| token.chars().count() == 1 && alphabet.contains(&token.chars().next().unwrap()))).unwrap();
        let mut tokens = tokens.clone();
        let mut types = types.clone();
        tokens.remove(remove);
        types.remove(remove);
        metadata.values.insert("tokenizer.ggml.tokens".into(), Value::Array(tokens));
        metadata.values.insert("tokenizer.ggml.token_type".into(), Value::Array(types));
        assert!(tokenizer_files(&metadata).unwrap_err().contains("missing byte tokens"));

        metadata = tokenizer_metadata("qwen35");
        metadata.values.insert("tokenizer.ggml.add_bos_token".into(), Value::Bool(true));
        assert!(tokenizer_files(&metadata).unwrap_err().contains("automatic BOS/EOS insertion"));
        metadata = tokenizer_metadata("qwen35");
        metadata.values.insert("tokenizer.ggml.eos_token_id".into(), Value::from(0));
        assert!(tokenizer_files(&metadata).unwrap_err().contains("invalid GGUF special token"));
        metadata = tokenizer_metadata("qwen35");
        metadata.values.insert("tokenizer.ggml.merges".into(), serde_json::json!(["a absent"]));
        assert!(tokenizer_files(&metadata).unwrap_err().contains("invalid GGUF BPE vocabulary or merges"));
    }

    #[test]
    fn validation_only_single_file_boundary_rejects_unmerged_split_metadata() {
        let mut metadata = config_metadata("qwen35", 32);
        assert!(require_single_file(&metadata).is_ok());
        metadata.values.insert("split.count".into(), Value::from(1));
        assert!(require_single_file(&metadata).is_ok());
        metadata.values.insert("split.count".into(), Value::from(3));
        assert!(require_single_file(&metadata).unwrap_err().contains("split metadata is merged"));
        metadata.values.insert("split.count".into(), Value::Bool(true));
        assert_eq!(require_single_file(&metadata).unwrap_err(), "invalid GGUF metadata: split.count");
    }

    #[cfg(unix)]
    #[test]
    fn model_check_preflight_uses_fake_engine_for_text_then_vision() {
        use std::os::unix::fs::PermissionsExt;
        let scratch = TempDirectory::new().unwrap();
        let engine = scratch.0.join("fake-splash");
        let calls = scratch.0.join("calls");
        let quoted_calls = format!("'{}'", calls.display());
        fs::write(&engine, format!(
            "#!/bin/sh\nprintf '%s\\n' \"$3\" >> {quoted_calls}\nprintf '{{\"family\":\"Qwen3.8-4B\"}}\\n'\n"
        )).unwrap();
        let mut permissions = fs::metadata(&engine).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&engine, permissions).unwrap();

        let target = loadable_qwen_metadata();
        let mut vision = vision_metadata();
        vision.values.insert("clip.vision.patch_size".into(), Value::from(16));
        vision.values.insert("clip.vision.image_mean".into(), serde_json::json!([0.5, 0.5, 0.5]));
        vision.values.insert("clip.vision.image_std".into(), serde_json::json!([0.5, 0.5, 0.5]));
        vision.tensors.insert("v.patch_embd.weight".into(), Tensor { extents: vec![1], kind: 8, offset: 0 });
        assert_eq!(validate_with_engine(&engine, &target, Some(&vision), None).unwrap(), "Qwen3.8-4B");
        assert_eq!(fs::read_to_string(&calls).unwrap().lines().collect::<Vec<_>>(), ["none", "gguf"]);
        fs::write(&calls, "").unwrap();
        assert_eq!(validate_with_engine(&engine, &target, None, None).unwrap(), "Qwen3.8-4B");
        assert_eq!(fs::read_to_string(&calls).unwrap().lines().collect::<Vec<_>>(), ["none"]);
    }
}
