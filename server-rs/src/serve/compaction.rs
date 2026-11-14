//! Prompt compaction policy shared by the Rust request preparation path.

use serde_json::{json, Value};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

pub const NOTE_HEADING: &str = "Earlier turns, compacted into a note:";
pub const DEFAULT_KEEP_TOKENS_CAP: u32 = 20_000;
pub const COMPACT_RESERVE_TOKENS: u32 = 1_024;
pub const COMPACT_DEFAULT_REPLY_TOKENS: u32 = 4_096;
pub const COMPACT_MAX_NOTE_TOKENS: u32 = 1_024;
pub const COMPACT_INSTRUCTION: &str = "Summarize the conversation below into a note for yourself, so you can continue the work: the goal, the constraints, what has been done, the decisions and why, the immediate next steps, and the exact names, numbers and file paths it mentions. Reply with the note alone, with no preamble and no advice to the user.";
static NEXT_TEMP: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompactAt {
    Auto,
    Fraction(f64),
}

pub fn keep_budget(context: u32, configured: Option<u32>) -> u32 {
    configured.unwrap_or_else(|| DEFAULT_KEEP_TOKENS_CAP.min((context / 4).max(1)))
}

pub fn should_compact(prompt: usize, max_new: u32, context: u32, setting: CompactAt) -> bool {
    match setting {
        CompactAt::Auto => prompt.saturating_add(max_new as usize)
            > context.saturating_sub(COMPACT_RESERVE_TOKENS) as usize,
        CompactAt::Fraction(fraction) => (prompt as f64) >= fraction * context as f64,
    }
}

pub fn leading_system_end(messages: &[Value]) -> usize {
    messages.iter().take_while(|message| message.get("role").and_then(Value::as_str) == Some("system")).count()
}

pub fn adjust_cut(messages: &[Value], mut cut: usize) -> usize {
    let start = leading_system_end(messages);
    while cut > start && messages.get(cut).and_then(|message| message.get("role")).and_then(Value::as_str) == Some("tool") {
        cut -= 1;
    }
    cut
}

pub fn choose_cut(
    messages: &[Value],
    budget: u32,
    mut count: impl FnMut(&[Value]) -> usize,
) -> usize {
    let start = leading_system_end(messages);
    if start >= messages.len() { return start; }
    let mut best = messages.len();
    let mut cut = messages.len();
    while cut > start {
        let candidate = adjust_cut(messages, cut - 1);
        if candidate <= start || candidate >= cut { break; }
        let trial: Vec<Value> = messages[..start].iter().cloned()
            .chain(messages[candidate..].iter().cloned()).collect();
        if count(&trial) > budget as usize { break; }
        best = candidate;
        cut = candidate;
    }
    if let Some(last_user) = (start..messages.len()).rev().find(|index| {
        messages[*index].get("role").and_then(Value::as_str) == Some("user")
    }) {
        if last_user < best { best = adjust_cut(messages, last_user); }
    }
    best
}

pub fn note_body(note: &str) -> String {
    let note = note.trim();
    if note.is_empty() { NOTE_HEADING.to_owned() }
    else { format!("{NOTE_HEADING}\n{note}") }
}

pub fn compacted_messages(messages: &[Value], cut: usize, note: &str) -> Vec<Value> {
    let start = leading_system_end(messages);
    let note = note_body(note);
    let mut result = Vec::with_capacity(messages.len() - cut + 1);
    if start > 0 {
        let mut system = messages[0].as_object().cloned().unwrap_or_default();
        let content = system.get("content").and_then(Value::as_str).unwrap_or("");
        system.insert("content".into(), json!(if content.is_empty() { note } else { format!("{content}\n\n{note}") }));
        result.push(Value::Object(system));
    } else {
        result.push(json!({"role": "system", "content": note}));
    }
    result.extend(messages[cut..].iter().cloned());
    result
}

pub fn transcript(messages: &[Value]) -> String {
    messages.iter().map(|message| {
        let role = message.get("role").and_then(Value::as_str).unwrap_or("");
        let mut parts = Vec::new();
        if let Some(content) = message.get("content").filter(|content| !content.is_null()) {
            parts.push(match content {
                Value::String(text) => text.clone(),
                other => other.to_string(),
            });
        }
        if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
            for call in calls {
                let name = call.pointer("/function/name").and_then(Value::as_str).unwrap_or("");
                let args = call.pointer("/function/arguments").and_then(Value::as_str).unwrap_or("");
                parts.push(format!("{name}({args})"));
            }
        }
        format!("{role}: {}", parts.join(" "))
    }).collect::<Vec<_>>().join("\n")
}

pub fn removed_digest(text: &str) -> String {
    crate::support::sha256::hex_digest(text.as_bytes())
}

pub fn conversation_key(messages: &[Value]) -> String {
    let system = messages.first()
        .filter(|message| message.get("role").and_then(Value::as_str) == Some("system"))
        .and_then(|message| message.get("content"))
        .map(value_text)
        .unwrap_or_default();
    let first_user = messages.iter()
        .find(|message| message.get("role").and_then(Value::as_str) == Some("user"))
        .and_then(|message| message.get("content"))
        .map(value_text)
        .unwrap_or_default();
    crate::support::sha256::hex_digest(format!("{system}\0{first_user}").as_bytes())
}

fn value_text(value: &Value) -> String {
    value.as_str().map(str::to_owned).unwrap_or_else(|| value.to_string())
}

pub fn read_note(directory: &Path, key: &str, digest: &str) -> Option<String> {
    let data: Value = serde_json::from_slice(&fs::read(directory.join(format!("{key}.json"))).ok()?).ok()?;
    if data.get("removed").and_then(Value::as_str) != Some(digest) { return None; }
    data.get("note").and_then(Value::as_str).map(str::to_owned)
}

/// Atomically retain a note for the same conversation and removed turns.
/// A cache write failure never fails the request.
pub fn write_note(directory: &Path, key: &str, digest: &str, note: &str) {
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    if builder.create(directory).is_err() && !directory.is_dir() { return; }
    let target = directory.join(format!("{key}.json"));
    let nonce = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
    let temporary = directory.join(format!(".{key}.{}.{}.tmp", std::process::id(), nonce));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let Ok(mut file) = options.open(&temporary) else { return; };
    let data = json!({"removed": digest, "note": note});
    if serde_json::to_writer(&mut file, &data).is_err() || file.flush().is_err() || file.sync_all().is_err() {
        let _ = fs::remove_file(&temporary);
        return;
    }
    drop(file);
    if fs::rename(&temporary, &target).is_err() {
        let _ = fs::remove_file(&temporary);
    }
}
