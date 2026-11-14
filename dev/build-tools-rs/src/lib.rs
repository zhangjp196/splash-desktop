//! Small native replacements for build_config.py and build_identity.py.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[path = "../../../server-rs/src/support/sha256.rs"]
#[allow(dead_code)]
mod sha256;

pub const SCHEMA_VERSION: u32 = 1;
const DOMAIN: &[u8] = b"splash-production-build-identity-v1\0";
const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[derive(Debug)]
pub struct ToolError(pub String);

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ToolError {}

fn tool_error(message: impl Into<String>) -> ToolError {
    ToolError(message.into())
}

pub fn default_root() -> PathBuf {
    Path::new(ROOT).parent().and_then(Path::parent).expect("crate lives under dev/").to_path_buf()
}

pub fn production_input_paths(root: &Path) -> Result<Vec<String>, ToolError> {
    let mut paths = BTreeSet::new();
    collect_production_sources(&root.join("runtime"), root, &mut paths)?;
    paths.insert("dev/build-tools-rs/Cargo.toml".to_owned());
    paths.insert("dev/build-tools-rs/src/lib.rs".to_owned());
    paths.insert("dev/build-tools-rs/src/main.rs".to_owned());
    paths.insert("server-rs/src/support/sha256.rs".to_owned());
    Ok(paths.into_iter().collect())
}

fn collect_production_sources(directory: &Path, root: &Path, paths: &mut BTreeSet<String>) -> Result<(), ToolError> {
    let entries = fs::read_dir(directory).map_err(|error| tool_error(format!("cannot scan production inputs {}: {error}", directory.display())))?;
    for entry in entries {
        let entry = entry.map_err(|error| tool_error(format!("cannot scan production inputs {}: {error}", directory.display())))?;
        let path = entry.path();
        let kind = entry.file_type().map_err(|error| tool_error(format!("cannot inspect production input {}: {error}", path.display())))?;
        if kind.is_dir() {
            collect_production_sources(&path, root, paths)?;
            continue;
        }
        if !kind.is_file() || !is_production_source(&path, root) { continue; }
        let relative = path.strip_prefix(root).map_err(|_| tool_error(format!("production input is outside the checkout: {}", path.display())))?;
        let relative = relative.to_str().ok_or_else(|| tool_error("production input path is not valid UTF-8"))?.replace('\\', "/");
        paths.insert(relative);
    }
    Ok(())
}

fn is_production_source(path: &Path, root: &Path) -> bool {
    match path.extension().and_then(|value| value.to_str()) {
        Some("c" | "cc" | "cpp" | "h" | "hpp" | "m" | "mm") => true,
        Some("metal") => path.starts_with(root.join("runtime/metal/kernels")),
        _ => false,
    }
}

fn record(hasher_input: &mut Vec<u8>, kind: &[u8], name: &[u8], value: &[u8]) {
    hasher_input.extend_from_slice(kind);
    hasher_input.extend_from_slice(&(name.len() as u64).to_be_bytes());
    hasher_input.extend_from_slice(name);
    hasher_input.extend_from_slice(&(value.len() as u64).to_be_bytes());
    hasher_input.extend_from_slice(value);
}

pub fn source_digest(root: &Path, inputs: &[String], constants: &BTreeMap<String, String>) -> Result<String, ToolError> {
    if inputs.is_empty() { return Err(tool_error("production build identity has no inputs")); }
    let mut normalized = inputs.to_vec();
    normalized.sort();
    normalized.dedup();
    if normalized.len() != inputs.len() { return Err(tool_error("production build identity inputs are duplicated")); }
    let root = root.canonicalize().map_err(|error| tool_error(format!("cannot resolve build root {}: {error}", root.display())))?;
    let mut payload = Vec::new();
    payload.extend_from_slice(DOMAIN);
    for (name, value) in constants {
        if name.is_empty() || name.contains('\0') { return Err(tool_error("build identity constant name is invalid")); }
        record(&mut payload, b"constant\0", name.as_bytes(), value.as_bytes());
    }
    for relative in normalized {
        let relative_path = Path::new(&relative);
        if relative_path.is_absolute() || relative_path.components().any(|part| matches!(part, Component::ParentDir)) {
            return Err(tool_error(format!("unsafe production input path: {relative}")));
        }
        let path = root.join(relative_path);
        let metadata = fs::symlink_metadata(&path).map_err(|error| tool_error(format!("cannot read production input {relative}: {error}")))?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(tool_error(format!("cannot read production input {relative}: input is not a regular non-symlink file")));
        }
        let content = fs::read(&path).map_err(|error| tool_error(format!("cannot read production input {relative}: {error}")))?;
        let normalized_path = relative_path.to_str().ok_or_else(|| tool_error(format!("unsafe production input path: {relative}")))?.replace('\\', "/");
        record(&mut payload, b"file\0", normalized_path.as_bytes(), &content);
    }
    Ok(sha256::hex_digest(&payload))
}

pub fn build_id(root: &Path, constants: &BTreeMap<String, String>) -> Result<String, ToolError> {
    let inputs = production_input_paths(root)?;
    Ok(format!("src-{}", source_digest(root, &inputs, constants)?))
}

pub fn header_bytes(identity: &str) -> Result<Vec<u8>, ToolError> {
    if !valid_identity(identity) { return Err(tool_error("cannot render an invalid production build id")); }
    let digest = identity.strip_prefix("src-").unwrap();
    Ok(format!("#pragma once\n#define SPLASH_BUILD_ID \"{identity}\"\n#define SPLASH_BUILD_SOURCE_SHA256 \"{digest}\"\n").into_bytes())
}

pub fn stamp_bytes(identity: &str) -> Result<Vec<u8>, ToolError> {
    if !valid_identity(identity) { return Err(tool_error("cannot render an invalid production build id")); }
    let digest = identity.strip_prefix("src-").unwrap();
    Ok(format!("{{\"build_id\":\"{identity}\",\"schema_version\":{SCHEMA_VERSION},\"source_sha256\":\"{digest}\"}}\n").into_bytes())
}

fn valid_identity(identity: &str) -> bool {
    identity.strip_prefix("src-").is_some_and(|digest| digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)))
}

pub fn update_if_changed(path: &Path, content: &[u8]) -> Result<bool, ToolError> {
    match fs::read(path) {
        Ok(previous) if previous == content => return Ok(false),
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(tool_error(format!("cannot read generated identity {}: {error}", path.display()))),
    }
    let parent = path.parent().ok_or_else(|| tool_error(format!("invalid generated identity path {}", path.display())))?;
    fs::create_dir_all(parent).map_err(|error| tool_error(format!("cannot create generated identity directory {}: {error}", parent.display())))?;
    let name = path.file_name().and_then(|value| value.to_str()).ok_or_else(|| tool_error("invalid generated identity path"))?;
    let temporary = parent.join(format!("{name}.{}.{}.tmp", std::process::id(), unique_suffix()));
    let result = (|| {
        let mut file = OpenOptions::new().write(true).create_new(true).open(&temporary)
            .map_err(|error| tool_error(format!("cannot create generated identity {}: {error}", temporary.display())))?;
        file.write_all(content).and_then(|_| file.sync_all())
            .map_err(|error| tool_error(format!("cannot write generated identity {}: {error}", temporary.display())))?;
        fs::rename(&temporary, path).map_err(|error| tool_error(format!("cannot publish generated identity {}: {error}", path.display())))
    })();
    if result.is_err() { let _ = fs::remove_file(&temporary); }
    result?;
    Ok(true)
}

fn unique_suffix() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos()
}

pub fn parse_constants(values: &[String]) -> Result<BTreeMap<String, String>, ToolError> {
    let mut constants = BTreeMap::new();
    for value in values {
        let Some((name, content)) = value.split_once('=') else { return Err(tool_error(format!("invalid or duplicate --constant: {value}"))); };
        if name.is_empty() || constants.insert(name.to_owned(), content.to_owned()).is_some() {
            return Err(tool_error(format!("invalid or duplicate --constant: {value}")));
        }
    }
    Ok(constants)
}

pub fn run_build_config_record(output: &Path, config: &str, command: &[String]) -> Result<i32, ToolError> {
    if command.is_empty() { return Err(tool_error("record requires a compiler or linker command after --")); }
    let record = record_path(output);
    if let Err(error) = fs::remove_file(&record) {
        if error.kind() != io::ErrorKind::NotFound { return Err(tool_error(error.to_string())); }
    }
    let status = Command::new(&command[0]).args(&command[1..]).status()
        .map_err(|error| tool_error(format!("cannot run {}: {error}", command[0])))?;
    if let Some(code) = status.code() { if code != 0 { return Ok(code); } }
    else {
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            return Ok(128 + status.signal().unwrap_or(0));
        }
        #[cfg(not(unix))]
        return Ok(1);
    }
    atomic_write(&record, format!("{config}\n").as_bytes())?;
    Ok(0)
}

pub fn record_path(output: &Path) -> PathBuf {
    PathBuf::from(format!("{}.config", output.to_string_lossy()))
}

pub fn config_matches(output: &Path, config: &str) -> bool {
    fs::read_to_string(record_path(output)).is_ok_and(|content| content == format!("{config}\n"))
}

pub fn stale_outputs(config: &str, outputs: &[String]) -> Vec<String> {
    outputs.iter().filter(|output| !config_matches(Path::new(output), config)).cloned().collect()
}

fn atomic_write(path: &Path, content: &[u8]) -> Result<(), ToolError> {
    let parent = path.parent().ok_or_else(|| tool_error(format!("invalid output path {}", path.display())))?;
    fs::create_dir_all(parent).map_err(|error| tool_error(error.to_string()))?;
    let name = path.file_name().and_then(|value| value.to_str()).unwrap_or("record");
    let temporary = parent.join(format!("{name}.{}.{}.tmp", std::process::id(), unique_suffix()));
    let result = (|| {
        let mut file = OpenOptions::new().write(true).create_new(true).open(&temporary).map_err(|error| tool_error(error.to_string()))?;
        file.write_all(content).and_then(|_| file.sync_all()).map_err(|error| tool_error(error.to_string()))?;
        fs::rename(&temporary, path).map_err(|error| tool_error(error.to_string()))
    })();
    if result.is_err() { let _ = fs::remove_file(&temporary); }
    result
}

pub fn write_identity(root: &Path, constants: &BTreeMap<String, String>, header: &Path, stamp: &Path) -> Result<(String, bool, bool), ToolError> {
    let identity = build_id(root, constants)?;
    let header_changed = update_if_changed(header, &header_bytes(&identity)?)?;
    let stamp_changed = update_if_changed(stamp, &stamp_bytes(&identity)?)?;
    Ok((identity, header_changed, stamp_changed))
}

pub fn check_identity(root: &Path, constants: &BTreeMap<String, String>, header: &Path, stamp: &Path, binaries: &[PathBuf]) -> Result<String, ToolError> {
    let identity = build_id(root, constants)?;
    let actual_header = fs::read(header).map_err(|error| tool_error(format!("cannot read generated build identity: {error}")))?;
    let actual_stamp = fs::read(stamp).map_err(|error| tool_error(format!("cannot read generated build identity: {error}")))?;
    if actual_header != header_bytes(&identity)? || actual_stamp != stamp_bytes(&identity)? {
        return Err(tool_error("generated build identity is stale"));
    }
    for binary in binaries {
        let bytes = fs::read(binary).map_err(|error| tool_error(format!("cannot read identity binary {}: {error}", binary.display())))?;
        let found = embedded_ids(&bytes);
        if found != BTreeSet::from([identity.clone()]) {
            return Err(tool_error(format!("binary {} embeds build identities {:?}, expected only {identity}", binary.display(), found)));
        }
    }
    Ok(identity)
}

fn embedded_ids(bytes: &[u8]) -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    const PREFIX: &[u8] = b"src-";
    let mut start = 0;
    while start + PREFIX.len() + 64 <= bytes.len() {
        let Some(relative) = bytes[start..].windows(PREFIX.len()).position(|window| window == PREFIX) else { break; };
        let position = start + relative;
        let end = position + PREFIX.len() + 64;
        let candidate = &bytes[position..end];
        if candidate[4..].iter().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte)) {
            found.insert(String::from_utf8_lossy(candidate).into_owned());
        }
        start = position + 1;
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp() -> PathBuf {
        std::env::temp_dir().join(format!("splash-build-tools-{}-{}", std::process::id(), SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()))
    }

    #[test]
    fn identity_header_and_stamp_match_python_contract() {
        let id = format!("src-{}", "a".repeat(64));
        assert_eq!(header_bytes(&id).unwrap(), format!("#pragma once\n#define SPLASH_BUILD_ID \"{id}\"\n#define SPLASH_BUILD_SOURCE_SHA256 \"{}\"\n", "a".repeat(64)).as_bytes());
        assert_eq!(stamp_bytes(&id).unwrap(), format!("{{\"build_id\":\"{id}\",\"schema_version\":1,\"source_sha256\":\"{}\"}}\n", "a".repeat(64)).as_bytes());
    }

    #[test]
    fn source_digest_is_order_independent_and_rejects_unsafe_inputs() {
        let root = temp(); fs::create_dir_all(&root).unwrap(); fs::write(root.join("a"), b"a").unwrap(); fs::write(root.join("b"), b"b").unwrap();
        let constants = BTreeMap::from([("engine".into(), "flags".into())]);
        let first = source_digest(&root, &["a".into(), "b".into()], &constants).unwrap();
        assert_eq!(first, source_digest(&root, &["b".into(), "a".into()], &constants).unwrap());
        assert!(source_digest(&root, &["../outside".into()], &constants).is_err());
        assert!(source_digest(&root, &["a".into(), "a".into()], &constants).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn config_record_only_follows_successful_command_and_stale_checks_match_python() {
        let root = temp(); fs::create_dir_all(&root).unwrap();
        let output = root.join("obj.o");
        assert_eq!(run_build_config_record(&output, "abc", &["/bin/sh".into(), "-c".into(), "exit 3".into()]).unwrap(), 3);
        assert!(!record_path(&output).exists());
        assert_eq!(run_build_config_record(&output, "abc", &["/bin/sh".into(), "-c".into(), "exit 0".into()]).unwrap(), 0);
        assert!(config_matches(&output, "abc"));
        assert_eq!(stale_outputs("abc", &[output.to_string_lossy().into_owned(), root.join("missing").to_string_lossy().into_owned()]), vec![root.join("missing").to_string_lossy().into_owned()]);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn embedded_identity_scan_requires_the_exact_set() {
        let id = format!("src-{}", "1".repeat(64));
        assert_eq!(embedded_ids(format!("prefix {id} suffix").as_bytes()), BTreeSet::from([id]));
    }
}
