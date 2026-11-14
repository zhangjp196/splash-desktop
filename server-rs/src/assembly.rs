//! Fast startup validation for installer-produced model assemblies.
//! Content digests are recomputed only in explicit full verification mode.

use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::Value;

pub fn verify(path: &Path, record: &Value, full: bool) -> Result<(), String> {
    validate_record(record)?;
    for (name, entry) in record["files"].as_object().unwrap() {
        let linked = path.join(name);
        let link = fs::symlink_metadata(&linked)
            .map_err(|error| format!("cannot inspect assembly file {name}: {error}"))?;
        if !link.file_type().is_symlink() {
            return Err(format!("resolved model file changed: {name} (not a symlink)"));
        }
        let target = fs::read_link(&linked)
            .map_err(|error| format!("cannot read assembly link {name}: {error}"))?;
        if target.to_str() != entry["path"].as_str() {
            return Err(format!("resolved model file changed: {name} (source path mismatch)"));
        }
        let metadata = fs::metadata(&linked)
            .map_err(|error| format!("cannot inspect source file {name}: {error}"))?;
        if metadata.len() != entry["bytes"].as_u64().unwrap()
            || stat_ns(&metadata, false) != entry["mtime_ns"].as_i64()
            || stat_ns(&metadata, true) != entry["ctime_ns"].as_i64()
        {
            return Err(format!("resolved model file changed: {name}"));
        }
        if full {
            let digest = entry["digest"].as_str().unwrap();
            let actual = source_digest(&linked, digest.len())?;
            if !actual.eq_ignore_ascii_case(digest) {
                return Err(format!("source content hash mismatch: {name}"));
            }
        }
    }
    Ok(())
}

/// Create and atomically publish an immutable resolved assembly from local
/// source files. `record` is the installer-compatible metadata excluding its
/// `files` map, which is derived here. This is the filesystem publication
/// primitive used by native installers; Hub downloads and model inspection
/// remain separate concerns.
pub fn publish(
    models_root: &Path,
    mut record: Value,
    files: &std::collections::BTreeMap<String, PathBuf>,
) -> Result<PathBuf, String> {
    if files.is_empty() {
        return Err("an assembly must contain at least one file".into());
    }
    fs::create_dir_all(models_root)
        .map_err(|error| format!("cannot create models directory: {error}"))?;
    let _lock = installation_lock(models_root)?;

    // The first publication hashes each source and publishes the immutable
    // content-addressed record below. Later starts can find that record from
    // cheap file metadata and verify its links without rereading model bytes.
    let fast_key = fast_cache_key(&record, files)?;
    if let Some(destination) = fast_cached_assembly(models_root, &fast_key, &record) {
        return Ok(destination);
    }

    let mut file_records = serde_json::Map::new();
    for (name, source) in files {
        if !safe_relative_path(name) {
            return Err(format!("invalid assembly file path: {name}"));
        }
        let absolute = absolute_path(source)?;
        let metadata = fs::metadata(&absolute)
            .map_err(|error| format!("cannot inspect assembly source {}: {error}", absolute.display()))?;
        if !metadata.is_file() {
            return Err(format!("assembly source is not a file: {}", absolute.display()));
        }
        let digest = source_file_digest_cached(models_root, &absolute)?;
        file_records.insert(name.clone(), serde_json::json!({
            "path": absolute,
            "bytes": metadata.len(),
            "mtime_ns": stat_ns(&metadata, false).ok_or("source timestamps are unsupported on this platform")?,
            "ctime_ns": stat_ns(&metadata, true).ok_or("source timestamps are unsupported on this platform")?,
            "digest": digest,
        }));
    }
    let object = record.as_object_mut().ok_or("invalid resolved model record")?;
    if object.contains_key("files") {
        return Err("assembly record must not supply its own files map".into());
    }
    object.insert("files".into(), Value::Object(file_records));
    validate_record(&record)?;

    let encoded = canonical_json(&record)?;
    let identity = crate::support::sha256::hex_digest(&encoded);
    let destination = models_root.join(".resolved").join(&identity);
    if destination.exists() {
        // A matching immutable record is reusable. A damaged entry can be
        // replaced while holding the installation lock.
        match verify(&destination, &record, false) {
            Ok(()) => return Ok(destination),
            Err(_) => fs::remove_dir_all(&destination)
                .map_err(|error| format!("cannot rebuild damaged assembly: {error}"))?,
        }
    }

    let parent = destination.parent().ok_or("invalid assembly destination")?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("cannot create assembly directory: {error}"))?;
    let stage = parent.join(format!(".loading-{}-{}", std::process::id(), unique_suffix()));
    fs::create_dir(&stage).map_err(|error| format!("cannot create assembly staging directory: {error}"))?;
    let write_result = (|| {
        for (name, source) in files {
            let link = stage.join(name);
            let directory = link.parent().ok_or("invalid assembly file path")?;
            fs::create_dir_all(directory)
                .map_err(|error| format!("cannot create assembly subdirectory: {error}"))?;
            symlink_file(&absolute_path(source)?, &link)
                .map_err(|error| format!("cannot link assembly source {}: {error}", source.display()))?;
        }
        let mut model = OpenOptions::new().write(true).create_new(true).open(stage.join("model.json"))
            .map_err(|error| format!("cannot create assembly record: {error}"))?;
        model.write_all(&encoded).and_then(|_| model.sync_all())
            .map_err(|error| format!("cannot write assembly record: {error}"))?;
        fs::rename(&stage, &destination)
            .map_err(|error| format!("cannot publish assembly: {error}"))?;
        verify(&destination, &record, false)
    })();
    if write_result.is_err() {
        let _ = fs::remove_dir_all(&stage);
    }
    write_result?;
    remember_fast_cache(models_root, &fast_key, &identity)?;
    Ok(destination)
}

fn fast_cache_key(
    record: &Value,
    files: &std::collections::BTreeMap<String, PathBuf>,
) -> Result<String, String> {
    let mut file_metadata = serde_json::Map::new();
    for (name, source) in files {
        if !safe_relative_path(name) {
            return Err(format!("invalid assembly file path: {name}"));
        }
        let absolute = absolute_path(source)?;
        let metadata = fs::metadata(&absolute)
            .map_err(|error| format!("cannot inspect assembly source {}: {error}", absolute.display()))?;
        if !metadata.is_file() {
            return Err(format!("assembly source is not a file: {}", absolute.display()));
        }
        file_metadata.insert(name.clone(), serde_json::json!({
            "path": absolute,
            "source": source_metadata_identity(&metadata)?,
        }));
    }
    let key = serde_json::json!({"record": record, "files": Value::Object(file_metadata)});
    Ok(crate::support::sha256::hex_digest(&canonical_json(&key)?))
}

fn fast_cached_assembly(models_root: &Path, key: &str, expected_record: &Value) -> Option<PathBuf> {
    let cache_entry = models_root.join(".assembly-fast-cache").join(key);
    let identity = fs::read_to_string(&cache_entry).ok()?;
    let identity = identity.trim();
    if !is_hex(Some(identity), 64) {
        let _ = fs::remove_file(cache_entry);
        return None;
    }
    let destination = models_root.join(".resolved").join(identity);
    let encoded = fs::read(destination.join("model.json")).ok()?;
    let record: Value = serde_json::from_slice(&encoded).ok()?;
    let mut base_record = record.clone();
    let Some(object) = base_record.as_object_mut() else {
        let _ = fs::remove_file(cache_entry);
        return None;
    };
    if object.remove("files").is_none()
        || canonical_json(&base_record).ok()? != canonical_json(expected_record).ok()?
        || verify(&destination, &record, false).is_err()
    {
        let _ = fs::remove_file(cache_entry);
        return None;
    }
    Some(destination)
}

fn remember_fast_cache(models_root: &Path, key: &str, identity: &str) -> Result<(), String> {
    remember_cache_value(models_root, ".assembly-fast-cache", key, identity)
}

/// Persist generated assembly inputs in an immutable content-addressed cache.
/// Returned paths are stable absolute sources suitable for `publish` symlinks.
pub fn persist_generated(
    models_root: &Path,
    generated: &std::collections::BTreeMap<String, Vec<u8>>,
) -> Result<std::collections::BTreeMap<String, PathBuf>, String> {
    if generated.is_empty() { return Err("generated source set is empty".into()); }
    let mut identity = Vec::new();
    for (name, bytes) in generated {
        if !safe_relative_path(name) { return Err(format!("invalid generated file path: {name}")); }
        identity.extend_from_slice(&(name.len() as u64).to_le_bytes());
        identity.extend_from_slice(name.as_bytes());
        identity.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        identity.extend_from_slice(bytes);
    }
    let key = crate::support::sha256::hex_digest(&identity);
    let cache = models_root.join(".generated").join(key);
    fs::create_dir_all(cache.parent().ok_or("invalid generated cache path")?)
        .map_err(|error| format!("cannot create generated source cache: {error}"))?;
    let valid = || -> bool {
        generated.iter().all(|(name, expected)| fs::read(cache.join(name)).is_ok_and(|actual| actual == *expected))
    };
    if cache.exists() && valid() {
        return Ok(generated.keys().map(|name| (name.clone(), cache.join(name))).collect());
    }
    if cache.exists() { fs::remove_dir_all(&cache).map_err(|error| format!("cannot replace damaged generated source cache: {error}"))?; }
    let stage = cache.parent().unwrap().join(format!(".generated-{}-{}", std::process::id(), unique_suffix()));
    fs::create_dir(&stage).map_err(|error| format!("cannot create generated source staging directory: {error}"))?;
    let result = (|| {
        for (name, bytes) in generated {
            let destination = stage.join(name);
            fs::create_dir_all(destination.parent().ok_or("invalid generated file path")?)
                .map_err(|error| format!("cannot create generated source directory: {error}"))?;
            let mut file = OpenOptions::new().write(true).create_new(true).open(&destination)
                .map_err(|error| format!("cannot create generated source {}: {error}", destination.display()))?;
            file.write_all(bytes).and_then(|_| file.sync_all())
                .map_err(|error| format!("cannot write generated source {}: {error}", destination.display()))?;
        }
        match fs::rename(&stage, &cache) {
            Ok(()) => Ok(()),
            Err(_) if cache.exists() && valid() => { let _ = fs::remove_dir_all(&stage); Ok(()) },
            Err(error) => Err(format!("cannot publish generated source cache: {error}")),
        }
    })();
    if result.is_err() { let _ = fs::remove_dir_all(&stage); }
    result?;
    if !valid() { return Err("generated source cache failed content verification".into()); }
    Ok(generated.keys().map(|name| (name.clone(), cache.join(name))).collect())
}

/// Remove only unreferenced content-addressed generated inputs from `.generated`.
/// Model weights and published `.resolved` assemblies are never removed. A
/// published assembly's `model.json` keeps each generated cache key live.
pub fn collect_generated(models_root: &Path) -> Result<Vec<PathBuf>, String> {
    // Keep a one-day grace window for generated files between their creation
    // and the publication of the assembly record that will reference them.
    collect_generated_older_than(models_root, Duration::from_secs(24 * 60 * 60))
}

fn collect_generated_older_than(models_root: &Path, minimum_age: Duration) -> Result<Vec<PathBuf>, String> {
    fs::create_dir_all(models_root)
        .map_err(|error| format!("cannot create models directory: {error}"))?;
    let _lock = installation_lock(models_root)?;
    let generated_root = models_root.join(".generated");
    let resolved_root = models_root.join(".resolved");
    if !generated_root.exists() { return Ok(Vec::new()); }
    let generated_root_absolute = fs::canonicalize(&generated_root)
        .map_err(|error| format!("cannot resolve generated cache: {error}"))?;

    let mut referenced = BTreeSet::new();
    if resolved_root.exists() {
        for entry in fs::read_dir(&resolved_root).map_err(|error| format!("cannot scan assemblies: {error}"))? {
            let entry = entry.map_err(|error| format!("cannot scan assemblies: {error}"))?;
            let metadata = entry.file_type().map_err(|error| format!("cannot inspect assembly: {error}"))?;
            if !metadata.is_dir() { continue; }
            let record_path = entry.path().join("model.json");
            let encoded = match fs::read(&record_path) {
                Ok(encoded) => encoded,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => return Err(format!("cannot read assembly record: {error}")),
            };
            let record: Value = serde_json::from_slice(&encoded)
                .map_err(|error| format!("cannot parse assembly record {}: {error}", record_path.display()))?;
            let files = record.get("files").and_then(Value::as_object)
                .ok_or_else(|| format!("invalid assembly record: {}", record_path.display()))?;
            for file in files.values() {
                let Some(path) = file.get("path").and_then(Value::as_str) else {
                    return Err(format!("invalid assembly file source: {}", record_path.display()));
                };
                let path = Path::new(path);
                let relative = match path.strip_prefix(&generated_root_absolute) {
                    Ok(relative) => Some(relative.to_path_buf()),
                    Err(_) => fs::canonicalize(path).ok()
                        .and_then(|canonical| canonical.strip_prefix(&generated_root_absolute).ok().map(Path::to_path_buf)),
                };
                let Some(relative) = relative else { continue; };
                if let Some(key) = relative.components().next().and_then(|component| component.as_os_str().to_str()) {
                    if is_hex(Some(key), 64) { referenced.insert(key.to_owned()); }
                }
            }
        }
    }

    let mut removed = Vec::new();
    for entry in fs::read_dir(&generated_root).map_err(|error| format!("cannot scan generated cache: {error}"))? {
        let entry = entry.map_err(|error| format!("cannot scan generated cache: {error}"))?;
        let name = entry.file_name();
        let Some(key) = name.to_str() else { continue; };
        if !is_hex(Some(key), 64) || referenced.contains(key) { continue; }
        let metadata = entry.file_type().map_err(|error| format!("cannot inspect generated cache entry: {error}"))?;
        if !metadata.is_dir() { continue; }
        let path = entry.path();
        let old_enough = fs::metadata(&path).ok().and_then(|metadata| metadata.modified().ok())
            .and_then(|modified| SystemTime::now().duration_since(modified).ok())
            .is_some_and(|age| age >= minimum_age);
        if !old_enough { continue; }
        fs::remove_dir_all(&path).map_err(|error| format!("cannot remove orphan generated cache {}: {error}", path.display()))?;
        removed.push(path);
    }
    Ok(removed)
}

/// Python installer compatible key for metadata derived from one target GGUF
/// and an optional projector. Local files use the same digest scheme as their
/// assembly file records; the embedded adapter and tokenizer versions match
/// the derivation contract in install/assembly.py.
pub fn gguf_metadata_key(target: &Path, vision: Option<&Path>) -> Result<String, String> {
    let targets = [target.to_path_buf()];
    gguf_metadata_key_many(&targets, vision)
}

/// Metadata key for an ordered GGUF shard set and optional vision projector.
pub fn gguf_metadata_key_many(targets: &[PathBuf], vision: Option<&Path>) -> Result<String, String> {
    gguf_metadata_key_many_with(targets, vision, source_file_digest)
}

/// Metadata key using a digest cache scoped to the local models directory.
pub fn gguf_metadata_key_many_cached(
    models_root: &Path,
    targets: &[PathBuf],
    vision: Option<&Path>,
) -> Result<String, String> {
    gguf_metadata_key_many_with(targets, vision, |path| source_file_digest_cached(models_root, path))
}

fn gguf_metadata_key_many_with(
    targets: &[PathBuf],
    vision: Option<&Path>,
    mut digest_source: impl FnMut(&Path) -> Result<String, String>,
) -> Result<String, String> {
    if targets.is_empty() { return Err("GGUF source set is empty".into()); }
    let mut sources = Vec::new();
    for path in targets.iter().map(PathBuf::as_path).chain(vision) {
        let metadata = fs::metadata(path).map_err(|error| format!("cannot inspect GGUF source: {error}"))?;
        sources.push(serde_json::json!({
            "bytes": metadata.len(),
            "digest": digest_source(path)?,
        }));
    }
    let adapter = crate::support::sha256::hex_digest(include_bytes!("../../install/gguf.py"));
    // Keep this in sync with server-rs/Cargo.toml's tokenizers dependency.
    let identity = serde_json::json!({"sources": sources, "adapter": adapter, "tokenizers": "0.23.2"});
    Ok(crate::support::sha256::hex_digest(&canonical_json(&identity)?))
}

/// Atomically point a model selection link at an already published assembly.
/// A real directory at the selection path is never replaced.
pub fn publish_selection(link: &Path, assembly: &Path) -> Result<(), String> {
    if link.exists() && !fs::symlink_metadata(link).map_err(|error| error.to_string())?.file_type().is_symlink() {
        return Err(format!("refusing to replace non-symlink model path: {}", link.display()));
    }
    let parent = link.parent().ok_or("invalid selection link")?;
    fs::create_dir_all(parent).map_err(|error| format!("cannot create selection directory: {error}"))?;
    let staging = parent.join(format!(".prepare-{}-{}", link.file_name().and_then(|name| name.to_str()).unwrap_or("model"), unique_suffix()));
    fs::create_dir(&staging).map_err(|error| format!("cannot create selection staging directory: {error}"))?;
    let temporary = staging.join("model");
    let result = symlink_file(&absolute_path(assembly)?, &temporary)
        .and_then(|()| fs::rename(&temporary, link));
    let _ = fs::remove_dir_all(&staging);
    result.map_err(|error| format!("cannot publish model selection: {error}"))
}

fn absolute_path(path: &Path) -> Result<PathBuf, String> {
    if path.is_absolute() { Ok(path.to_path_buf()) }
    else { std::env::current_dir().map(|cwd| cwd.join(path)).map_err(|error| error.to_string()) }
}

#[cfg(test)]
thread_local! {
    static SOURCE_HASH_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn source_file_digest(path: &Path) -> Result<String, String> {
    #[cfg(test)]
    SOURCE_HASH_CALLS.with(|calls| calls.set(calls.get() + 1));
    let resolved = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    if let Some(parent) = resolved.parent() {
        if parent.file_name().and_then(|name| name.to_str()) == Some("blobs") {
            if let Some(name) = resolved.file_name().and_then(|name| name.to_str()) {
                if is_hex(Some(name), 40) || is_hex(Some(name), 64) { return Ok(name.to_ascii_lowercase()); }
            }
        }
    }
    let file = File::open(path).map_err(|error| format!("cannot hash {}: {error}", path.display()))?;
    crate::support::sha256::hash_reader(file)
        .map_err(|error| format!("cannot hash {}: {error}", path.display()))
}

fn source_metadata_identity(metadata: &fs::Metadata) -> Result<Value, String> {
    let mtime_ns = stat_ns(metadata, false).ok_or("source timestamps are unsupported on this platform")?;
    let ctime_ns = stat_ns(metadata, true).ok_or("source timestamps are unsupported on this platform")?;
    #[cfg(unix)]
    let file_id = {
        use std::os::unix::fs::MetadataExt;
        serde_json::json!({"device": metadata.dev(), "inode": metadata.ino()})
    };
    #[cfg(not(unix))]
    let file_id = Value::Null;
    Ok(serde_json::json!({
        "bytes": metadata.len(),
        "mtime_ns": mtime_ns,
        "ctime_ns": ctime_ns,
        "file_id": file_id,
    }))
}

/// Return a cached digest when a source's identity and metadata are unchanged.
/// A cache miss computes SHA-256 once and stores it for later preparations.
pub fn source_file_digest_cached(models_root: &Path, path: &Path) -> Result<String, String> {
    let absolute = absolute_path(path)?;
    let metadata = fs::metadata(&absolute)
        .map_err(|error| format!("cannot inspect source file {}: {error}", absolute.display()))?;
    if !metadata.is_file() {
        return Err(format!("assembly source is not a file: {}", absolute.display()));
    }
    let key_value = serde_json::json!({
        "path": absolute,
        "source": source_metadata_identity(&metadata)?,
    });
    let key = crate::support::sha256::hex_digest(&canonical_json(&key_value)?);
    let cache_entry = models_root.join(".source-digest-cache").join(&key);
    if let Ok(digest) = fs::read_to_string(&cache_entry) {
        let digest = digest.trim();
        if is_hex(Some(digest), 64) {
            return Ok(digest.to_ascii_lowercase());
        }
        let _ = fs::remove_file(&cache_entry);
    }
    let digest = source_file_digest(&absolute)?;
    remember_cache_value(models_root, ".source-digest-cache", &key, &digest)?;
    Ok(digest)
}

fn remember_cache_value(models_root: &Path, cache_name: &str, key: &str, value: &str) -> Result<(), String> {
    let cache_root = models_root.join(cache_name);
    fs::create_dir_all(&cache_root)
        .map_err(|error| format!("cannot create {cache_name}: {error}"))?;
    let destination = cache_root.join(key);
    let temporary = cache_root.join(format!(".cache-{}", unique_suffix()));
    let result = (|| {
        let mut file = OpenOptions::new().write(true).create_new(true).open(&temporary)
            .map_err(|error| format!("cannot create {cache_name} entry: {error}"))?;
        file.write_all(value.as_bytes()).and_then(|_| file.write_all(b"\n")).and_then(|_| file.sync_all())
            .map_err(|error| format!("cannot write {cache_name} entry: {error}"))?;
        fs::rename(&temporary, &destination)
            .map_err(|error| format!("cannot publish {cache_name} entry: {error}"))
    })();
    if result.is_err() { let _ = fs::remove_file(&temporary); }
    result
}

fn canonical_json(value: &Value) -> Result<Vec<u8>, String> {
    fn sorted(value: &Value) -> Value {
        match value {
            Value::Object(object) => {
                let mut keys: Vec<_> = object.keys().collect();
                keys.sort();
                let mut result = serde_json::Map::new();
                for key in keys { result.insert(key.clone(), sorted(&object[key])); }
                Value::Object(result)
            }
            Value::Array(array) => Value::Array(array.iter().map(sorted).collect()),
            _ => value.clone(),
        }
    }
    let mut bytes = serde_json::to_vec_pretty(&sorted(value)).map_err(|error| error.to_string())?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn unique_suffix() -> String {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let nonce = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    format!("{}-{nonce}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos())
}

fn installation_lock(models_root: &Path) -> Result<File, String> {
    let lock = OpenOptions::new().create(true).read(true).write(true).open(models_root.join(".install.lock"))
        .map_err(|error| format!("cannot open installation lock: {error}"))?;
    #[cfg(unix)]
    loop {
        // SAFETY: flock only observes the valid descriptor owned by `lock`.
        if unsafe { libc::flock(std::os::fd::AsRawFd::as_raw_fd(&lock), libc::LOCK_EX) } == 0 { break; }
        let error = std::io::Error::last_os_error();
        if error.kind() != std::io::ErrorKind::Interrupted { return Err(format!("cannot acquire installation lock: {error}")); }
    }
    Ok(lock)
}

#[cfg(unix)]
fn symlink_file(source: &Path, link: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(source, link)
}
#[cfg(not(unix))]
fn symlink_file(_: &Path, _: &Path) -> std::io::Result<()> {
    Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "model assemblies require symbolic links"))
}

#[cfg(unix)]
fn stat_ns(metadata: &fs::Metadata, changed: bool) -> Option<i64> {
    use std::os::unix::fs::MetadataExt;
    let (seconds, nanos) = if changed {
        (metadata.ctime(), metadata.ctime_nsec())
    } else {
        (metadata.mtime(), metadata.mtime_nsec())
    };
    seconds.checked_mul(1_000_000_000)?.checked_add(nanos)
}

#[cfg(not(unix))]
fn stat_ns(_: &fs::Metadata, _: bool) -> Option<i64> { None }

fn source_digest(path: &Path, length: usize) -> Result<String, String> {
    // Ordinary files use SHA-256; Hub blob files use Git's blob SHA-1.
    if length == 64 {
        let file = File::open(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        return crate::support::sha256::hash_reader(file)
            .map_err(|error| format!("cannot hash {}: {error}", path.display()));
    }
    let program = "git";
    let args = vec!["hash-object".to_owned(), path.display().to_string()];
    let output = Command::new(program).args(args).output()
        .map_err(|error| format!("full assembly verification needs {program}: {error}"))?;
    if !output.status.success() {
        return Err(format!("{program} failed while hashing {}", path.display()));
    }
    String::from_utf8_lossy(&output.stdout).split_whitespace().next()
        .map(str::to_owned)
        .ok_or_else(|| format!("{program} returned no digest for {}", path.display()))
}

fn validate_record(record: &Value) -> Result<(), String> {
    let object = record.as_object().ok_or("invalid resolved model record")?;
    let version = record["version"].as_u64().ok_or("invalid resolved model record")?;
    if version != 1 && version != 2 { return Err("invalid resolved model record".into()); }
    let target_format = record["target_format"].as_str().ok_or("invalid resolved model record")?;
    let vision_format = record["vision_format"].as_str().ok_or("invalid resolved model record")?;
    if !["mlx-affine", "gguf"].contains(&target_format)
        || !["none", "safetensors", "gguf"].contains(&vision_format)
        || record["model"].as_str().is_none() || record["family"].as_str().is_none()
    { return Err("invalid resolved model record".into()); }
    let gguf = target_format == "gguf";
    let sources = record["sources"].as_object().ok_or("invalid resolved model record")?;
    let expected_sources: BTreeSet<&str> = if version == 1
        || record.pointer("/speculation/provider").and_then(Value::as_str) != Some("none")
    { ["target", "draft"].into_iter().collect() } else { ["target"].into_iter().collect() };
    if sources.keys().map(String::as_str).collect::<BTreeSet<_>>() != expected_sources {
        return Err("invalid resolved model record".into());
    }
    for source in sources.values() {
        let source = source.as_object().ok_or("invalid resolved model record")?;
        if keys(source) != ["repo", "revision"].into_iter().collect()
            || source["repo"].as_str().is_none()
            || !(source["revision"].is_null() || is_hex(source["revision"].as_str(), 40))
        { return Err("invalid resolved model record".into()); }
    }
    let files = record["files"].as_object().ok_or("invalid resolved model record")?;
    for (name, entry) in files {
        if !safe_relative_path(name) { return Err("invalid resolved model record".into()); }
        let entry = entry.as_object().ok_or("invalid resolved model record")?;
        if keys(entry) != ["path", "bytes", "mtime_ns", "ctime_ns", "digest"].into_iter().collect()
            || entry["path"].as_str().is_none()
            || entry["bytes"].as_u64().is_none()
            || entry["mtime_ns"].as_i64().is_none()
            || entry["ctime_ns"].as_i64().is_none()
            || !(is_hex(entry["digest"].as_str(), 40) || is_hex(entry["digest"].as_str(), 64))
        { return Err("invalid resolved model record".into()); }
    }
    let mut allowed: BTreeSet<&str> = [
        "version", "model", "family", "target_format", "vision_format", "sources", "files",
    ].into_iter().collect();
    if gguf {
        allowed.insert("metadata");
        if !is_hex(record["metadata"].as_str(), 64) { return Err("invalid resolved model record".into()); }
    }
    if object.contains_key("mtp") {
        allowed.insert("mtp");
        let mtp = record["mtp"].as_object().ok_or("invalid resolved model record")?;
        if keys(mtp) != ["layers"].into_iter().collect()
            || mtp["layers"].as_i64().is_none_or(|value| value < 0)
        { return Err("invalid resolved model record".into()); }
    }
    if version == 2 {
        allowed.insert("speculation");
        validate_speculation(&record["speculation"])?;
    }
    if object.keys().any(|key| !allowed.contains(key.as_str()))
        || allowed.iter().any(|key| !object.contains_key(*key) && *key != "mtp")
    { return Err("invalid resolved model record".into()); }
    Ok(())
}

fn validate_speculation(value: &Value) -> Result<(), String> {
    let plan = value.as_object().ok_or("invalid resolved model record")?;
    let mut expected: BTreeSet<&str> = [
        "version", "provider", "format", "source", "proposal_rows", "state", "fallback",
    ].into_iter().collect();
    if plan.contains_key("alignment") { expected.insert("alignment"); }
    if keys(plan) != expected { return Err("invalid resolved model record".into()); }
    let provider = plan["provider"].as_str().ok_or("invalid resolved model record")?;
    let format = plan["format"].as_str().ok_or("invalid resolved model record")?;
    let state = plan["state"].as_str().ok_or("invalid resolved model record")?;
    let fallback = plan["fallback"].as_str().ok_or("invalid resolved model record")?;
    let source_kind = plan.get("source").and_then(|source| source.get("kind")).and_then(Value::as_str)
        .ok_or("invalid resolved model record")?;
    let rows = plan["proposal_rows"].as_u64().ok_or("invalid resolved model record")?;
    if plan["version"].as_u64() != Some(1) || !(1..=8).contains(&rows)
        || !["none", "dflash2"].contains(&provider)
        || !["none", "safetensors", "gguf"].contains(&format)
        || !["none", "draft-rings"].contains(&state)
        || !["none", "required"].contains(&fallback)
        || !["none", "builtin", "external"].contains(&source_kind)
        || (provider == "none" && !(format == "none" && source_kind == "none" && rows == 1 && state == "none"))
        || (provider == "dflash2" && !(["safetensors", "gguf"].contains(&format) && source_kind == "external" && rows == 8 && state == "draft-rings"))
    { return Err("invalid resolved model record".into()); }
    Ok(())
}

fn keys(object: &serde_json::Map<String, Value>) -> BTreeSet<&str> {
    object.keys().map(String::as_str).collect()
}

fn is_hex(value: Option<&str>, length: usize) -> bool {
    value.is_some_and(|value| value.len() == length && value.bytes().all(|byte| byte.is_ascii_hexdigit()))
}

fn safe_relative_path(value: &str) -> bool {
    !value.is_empty() && !value.starts_with('/') && !value.contains('\\')
        && !value.chars().any(char::is_control) && !value.contains(['*', '?', '[', ']'])
        && value.split('/').all(|part| !part.is_empty() && part != "." && part != "..")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use serde_json::json;


    fn fixture() -> (std::path::PathBuf, Value) {
        let root = std::env::temp_dir().join(format!("splash-assembly-{}", unique_suffix()));
        let source = root.with_extension("source");
        fs::write(&source, b"model bytes").unwrap();
        fs::create_dir_all(root.join("target")).unwrap();
        std::os::unix::fs::symlink(&source, root.join("target/weights.bin")).unwrap();
        let metadata = fs::metadata(&source).unwrap();
        let record = json!({
            "version":2,"model":"org/model","family":"Qwen3.8-4B",
            "target_format":"mlx-affine","vision_format":"none",
            "sources":{"target":{"repo":"org/model","revision":null}},
            "speculation":{"version":1,"provider":"none","format":"none","source":{"kind":"none"},"proposal_rows":1,"state":"none","fallback":"none"},
            "files":{"target/weights.bin":{"path":source.to_str().unwrap(),"bytes":metadata.len(),"mtime_ns":stat_ns(&metadata,false).unwrap(),"ctime_ns":stat_ns(&metadata,true).unwrap(),"digest":"0".repeat(64)}}
        });
        (root, record)
    }

    #[test]
    fn fast_validation_accepts_healthy_assembly_and_rejects_replaced_source() {
        let (root, record) = fixture();
        assert!(verify(&root, &record, false).is_ok());
        fs::remove_file(root.join("target/weights.bin")).unwrap();
        fs::write(root.join("target/weights.bin"), b"not a link").unwrap();
        assert!(verify(&root, &record, false).unwrap_err().contains("not a symlink"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn unsafe_paths_and_malformed_digests_are_rejected() {
        let (root, mut record) = fixture();
        record["files"]["../outside"] = record["files"]["target/weights.bin"].clone();
        assert_eq!(verify(&root, &record, false).unwrap_err(), "invalid resolved model record");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn full_validation_recomputes_source_digest() {
        let (root, mut record) = fixture();
        let source = root.with_extension("source");
        let output = Command::new("shasum").args(["-a", "256", source.to_str().unwrap()]).output();
        if let Ok(output) = output {
            if output.status.success() {
                let actual = String::from_utf8_lossy(&output.stdout).split_whitespace().next().unwrap().to_owned();
                record["files"]["target/weights.bin"]["digest"] = json!(actual);
                assert!(verify(&root, &record, true).is_ok());
                record["files"]["target/weights.bin"]["digest"] = json!("0".repeat(64));
                assert!(verify(&root, &record, true).unwrap_err().contains("hash mismatch"));
            }
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn publishes_an_immutable_assembly_and_selection_link() {
        let (root, mut record) = fixture();
        let source = root.with_extension("source");
        let models = root.join("models");
        record.as_object_mut().unwrap().remove("files");
        let files = [("target/weights.bin".to_owned(), source.clone())].into_iter().collect();
        let published = publish(&models, record.clone(), &files).unwrap();
        assert!(verify(&published, &serde_json::from_slice(&fs::read(published.join("model.json")).unwrap()).unwrap(), true).is_ok());
        assert_eq!(publish(&models, record, &files).unwrap(), published);
        let selection = models.join(".selections/model");
        publish_selection(&selection, &published).unwrap();
        assert_eq!(fs::read_link(&selection).unwrap(), published);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn publish_reuses_fast_metadata_cache_and_rehashes_after_change() {
        let (root, mut record) = fixture();
        let source = root.with_extension("source");
        let models = root.join("models");
        record.as_object_mut().unwrap().remove("files");
        let files = [("target/weights.bin".to_owned(), source.clone())].into_iter().collect();

        SOURCE_HASH_CALLS.with(|calls| calls.set(0));
        let first = publish(&models, record.clone(), &files).unwrap();
        SOURCE_HASH_CALLS.with(|calls| assert_eq!(calls.get(), 1));
        assert_eq!(fs::read_dir(models.join(".assembly-fast-cache")).unwrap().count(), 1);

        SOURCE_HASH_CALLS.with(|calls| calls.set(0));
        let second = publish(&models, record.clone(), &files).unwrap();
        SOURCE_HASH_CALLS.with(|calls| assert_eq!(calls.get(), 0));
        assert_eq!(second, first);

        fs::write(&source, b"changed weights with a different size").unwrap();
        SOURCE_HASH_CALLS.with(|calls| calls.set(0));
        let changed = publish(&models, record, &files).unwrap();
        SOURCE_HASH_CALLS.with(|calls| assert_eq!(calls.get(), 1));
        assert_ne!(changed, first);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn source_digest_cache_skips_unchanged_bytes_and_invalidates_changed_files() {
        let root = std::env::temp_dir().join(format!("splash-source-digest-{}", unique_suffix()));
        let models = root.join("models");
        let source = root.join("weights.safetensors");
        fs::create_dir_all(&root).unwrap();
        fs::write(&source, b"first weights").unwrap();

        SOURCE_HASH_CALLS.with(|calls| calls.set(0));
        let first = source_file_digest_cached(&models, &source).unwrap();
        SOURCE_HASH_CALLS.with(|calls| assert_eq!(calls.get(), 1));
        SOURCE_HASH_CALLS.with(|calls| calls.set(0));
        assert_eq!(source_file_digest_cached(&models, &source).unwrap(), first);
        SOURCE_HASH_CALLS.with(|calls| assert_eq!(calls.get(), 0));

        fs::write(&source, b"second weights changed").unwrap();
        SOURCE_HASH_CALLS.with(|calls| calls.set(0));
        let changed = source_file_digest_cached(&models, &source).unwrap();
        SOURCE_HASH_CALLS.with(|calls| assert_eq!(calls.get(), 1));
        assert_ne!(changed, first);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn generated_sources_are_stable_content_addressed_files() {
        let root = std::env::temp_dir().join(format!("splash-generated-{}", unique_suffix()));
        let generated = BTreeMap::from([
            ("config.json".to_owned(), b"{\"ok\":true}".to_vec()),
            ("tokenizer/tokenizer.json".to_owned(), b"{}".to_vec()),
        ]);
        let first = persist_generated(&root, &generated).unwrap();
        let second = persist_generated(&root, &generated).unwrap();
        assert_eq!(first, second);
        assert!(first.values().all(|path| path.is_absolute()));
        assert!(first.iter().all(|(name, path)| fs::read(path).unwrap() == generated[name]));
        let invalid = BTreeMap::from([("../escape".to_owned(), b"no".to_vec())]);
        assert!(persist_generated(&root, &invalid).unwrap_err().contains("invalid generated file path"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn generated_cache_collection_keeps_published_sources_and_removes_only_orphans() {
        let root = std::env::temp_dir().join(format!("splash-generated-gc-{}", unique_suffix()));
        let models = root.join("models");
        let live = persist_generated(&models, &BTreeMap::from([
            ("config.json".to_owned(), b"{\"live\":true}".to_vec()),
        ])).unwrap();
        let orphan = persist_generated(&models, &BTreeMap::from([
            ("config.json".to_owned(), b"{\"orphan\":true}".to_vec()),
        ])).unwrap();
        let record = json!({
            "version":2,"model":"local/example","family":"Example",
            "target_format":"mlx-affine","vision_format":"none",
            "sources":{"target":{"repo":"/tmp/model","revision":null}},
            "speculation":{"version":1,"provider":"none","format":"none","source":{"kind":"none"},"proposal_rows":1,"state":"none","fallback":"none"}
        });
        let files = BTreeMap::from([("config.json".to_owned(), live["config.json"].clone())]);
        publish(&models, record, &files).unwrap();

        let removed = collect_generated_older_than(&models, Duration::ZERO).unwrap();
        assert_eq!(removed, vec![orphan["config.json"].parent().unwrap().to_path_buf()]);
        assert!(live["config.json"].exists());
        assert!(!orphan["config.json"].parent().unwrap().exists());
        assert!(models.join(".resolved").is_dir());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn selection_publication_never_replaces_a_real_directory() {
        let root = std::env::temp_dir().join(format!("splash-selection-{}", unique_suffix()));
        let selection = root.join("models/model");
        fs::create_dir_all(&selection).unwrap();
        assert!(publish_selection(&selection, &root.join("assembly")).unwrap_err().contains("non-symlink"));
        fs::remove_dir_all(root).unwrap();
    }
}
