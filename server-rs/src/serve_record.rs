//! Atomic lifecycle records consumed by launchers and desktop supervisors.
//! The on-disk schema deliberately contains only process/phase and safe error
//! categories. Dynamic engine/request errors are never persisted.

use serde::Serialize;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureKind { Configuration, Model, Device, Internal, Port, Upgrade, Install, Engine }

#[derive(Debug)]
pub struct NotRecorded { path: PathBuf, source: io::Error }

impl std::fmt::Display for NotRecorded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cannot write serve record {}: {}", self.path.display(), self.source)
    }
}
impl std::error::Error for NotRecorded {}

/// Record lifecycle for one server process. `starting` and `ready` are
/// mandatory checkpoints; intermediate/failure records are best-effort.
pub struct ServeRecord { path: Option<PathBuf>, pid: u32 }

impl ServeRecord {
    pub fn new(path: Option<PathBuf>) -> Self { Self { path, pid: std::process::id() } }
    pub fn starting(&self) -> Result<(), NotRecorded> { self.write_required(&Record { pid: self.pid, phase: "starting", failure: None }) }
    pub fn entered(&self, phase: &'static str) { self.write_best_effort(&Record { pid: self.pid, phase, failure: None }); }
    pub fn ready(&self) -> Result<(), NotRecorded> { self.write_required(&Record { pid: self.pid, phase: "ready", failure: None }) }
    pub fn failed(&self, kind: FailureKind, safe_message: &'static str) {
        self.write_best_effort(&Record { pid: self.pid, phase: "failed", failure: Some(Failure { kind, message: safe_message }) });
    }
    fn write_required(&self, record: &Record<'_>) -> Result<(), NotRecorded> {
        self.write(record).map_err(|source| NotRecorded { path: self.path.clone().unwrap_or_default(), source })
    }
    fn write_best_effort(&self, record: &Record<'_>) {
        if let Err(error) = self.write(record) {
            crate::diagnostics::print_error(&format!("Warning · serve record update failed ({})", error.kind()));
        }
    }
    fn write(&self, record: &Record<'_>) -> io::Result<()> {
        let Some(path) = &self.path else { return Ok(()); };
        write_atomic(path, record)
    }
}

#[derive(Serialize)]
struct Record<'a> { pid: u32, phase: &'a str, #[serde(skip_serializing_if = "Option::is_none")] failure: Option<Failure> }
#[derive(Serialize)]
struct Failure { kind: FailureKind, message: &'static str }

fn write_atomic(path: &Path, record: &impl Serialize) -> io::Result<()> {
    let sibling = path.with_file_name(format!(".{}.{}", path.file_name().and_then(|n| n.to_str()).unwrap_or("serve-record"), std::process::id()));
    let bytes = serde_json::to_vec(record).map_err(io::Error::other)?;
    let result = (|| {
        fs::write(&sibling, bytes)?;
        fs::rename(&sibling, path)
    })();
    if result.is_err() { let _ = fs::remove_file(&sibling); }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path() -> PathBuf {
        let unique = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        std::env::temp_dir().join(format!("splash-serve-record-{}-{unique}.json", std::process::id()))
    }

    #[test]
    fn record_transitions_are_atomic_and_failure_message_is_static() {
        let path = temp_path();
        let record = ServeRecord::new(Some(path.clone()));
        record.starting().unwrap();
        assert_eq!(serde_json::from_slice::<Value>(&fs::read(&path).unwrap()).unwrap()["phase"], "starting");
        record.entered("loading");
        record.ready().unwrap();
        record.failed(FailureKind::Engine, "native engine stopped");
        let value: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(value["failure"]["message"], "native engine stopped");
        assert!(!value.to_string().contains("prompt"));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn absent_record_path_is_a_noop() {
        let record = ServeRecord::new(None);
        assert!(record.starting().is_ok());
        assert!(record.ready().is_ok());
    }

    #[test]
    fn required_start_record_failure_is_returned() {
        let path = temp_path().join("missing-parent").join("serve.json");
        let record = ServeRecord::new(Some(path));
        assert!(record.starting().is_err());
    }
}
