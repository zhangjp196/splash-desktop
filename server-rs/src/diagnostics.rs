//! Bounded console diagnostics. Call sites pass already-sanitized summaries;
//! request bodies and arbitrary engine error messages must never be logged.

use std::io::{self, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub const CONSOLE_QUEUE_LINES: usize = 1024;

enum Message {
    Line(String, bool),
    Flush(mpsc::Sender<()>),
    Close(mpsc::Sender<()>),
}

trait LineOutput: Send {
    fn write_line(&mut self, line: &str, error: bool) -> io::Result<()>;
    fn flush(&mut self) -> io::Result<()>;
}
struct SingleWriter<W>(W);
impl<W: Write + Send> LineOutput for SingleWriter<W> {
    fn write_line(&mut self, line: &str, _error: bool) -> io::Result<()> {
        self.0.write_all(line.as_bytes())?;
        self.0.flush()
    }
    fn flush(&mut self) -> io::Result<()> { self.0.flush() }
}
struct Terminal;
impl LineOutput for Terminal {
    fn write_line(&mut self, line: &str, error: bool) -> io::Result<()> {
        if error {
            let mut stderr = io::stderr();
            stderr.write_all(line.as_bytes())?;
            stderr.flush()
        } else {
            let mut stdout = io::stdout();
            stdout.write_all(line.as_bytes())?;
            stdout.flush()
        }
    }
    fn flush(&mut self) -> io::Result<()> { io::stdout().flush()?; io::stderr().flush() }
}

struct Counters {
    queued: AtomicU64,
    written: AtomicU64,
    dropped: AtomicU64,
    write_errors: AtomicU64,
}

/// A bounded single-writer console. Producers never wait for a terminal or
/// log sink; over-capacity lines are counted and dropped.
pub struct Console {
    sender: SyncSender<Message>,
    counters: Arc<Counters>,
    capacity: usize,
    worker: Mutex<Option<JoinHandle<()>>>,
}

impl Console {
    pub fn with_writer<W: Write + Send + 'static>(capacity: usize, writer: W) -> io::Result<Self> {
        Self::with_output(capacity, Box::new(SingleWriter(writer)))
    }

    fn with_output(capacity: usize, mut writer: Box<dyn LineOutput>) -> io::Result<Self> {
        if capacity == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "console queue size must be positive"));
        }
        let (sender, receiver) = mpsc::sync_channel::<Message>(capacity);
        let counters = Arc::new(Counters {
            queued: AtomicU64::new(0), written: AtomicU64::new(0), dropped: AtomicU64::new(0), write_errors: AtomicU64::new(0),
        });
        let worker_counters = counters.clone();
        let worker = thread::Builder::new().name("splash-console".into()).spawn(move || {
            while let Ok(message) = receiver.recv() {
                match message {
                    Message::Line(line, error) => {
                        worker_counters.queued.fetch_sub(1, Ordering::Relaxed);
                        if writer.write_line(&line, error).is_ok() {
                            worker_counters.written.fetch_add(1, Ordering::Relaxed);
                        } else {
                            worker_counters.write_errors.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    Message::Flush(done) => { let _ = writer.flush(); let _ = done.send(()); }
                    Message::Close(done) => { let _ = writer.flush(); let _ = done.send(()); break; }
                }
            }
        })?;
        Ok(Self { sender, counters, capacity, worker: Mutex::new(Some(worker)) })
    }

    pub fn write(&self, message: &str, error: bool) -> bool {
        let seconds = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() % 86_400;
        let stamp = format!("{:02}:{:02}:{:02}", seconds / 3_600, (seconds / 60) % 60, seconds % 60);
        let line = format!("{stamp} {} {}\n", if error { "Error" } else { "Status" }, message);
        self.counters.queued.fetch_add(1, Ordering::Relaxed);
        match self.sender.try_send(Message::Line(line, error)) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                self.counters.queued.fetch_sub(1, Ordering::Relaxed);
                self.counters.dropped.fetch_add(1, Ordering::Relaxed); false
            }
        }
    }

    pub fn stats(&self) -> ConsoleStats {
        ConsoleStats {
            queued: self.counters.queued.load(Ordering::Relaxed),
            queue_lines: self.capacity,
            written: self.counters.written.load(Ordering::Relaxed),
            dropped_lines: self.counters.dropped.load(Ordering::Relaxed),
            write_errors: self.counters.write_errors.load(Ordering::Relaxed),
        }
    }

    pub fn flush(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let (tx, rx) = mpsc::channel();
        if !self.send_control(Message::Flush(tx), deadline) { return false; }
        rx.recv_timeout(deadline.saturating_duration_since(Instant::now())).is_ok()
    }

    pub fn close(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let (tx, rx) = mpsc::channel();
        if !self.send_control(Message::Close(tx), deadline) { return false; }
        let flushed = rx.recv_timeout(deadline.saturating_duration_since(Instant::now())).is_ok();
        if let Some(worker) = self.worker.lock().expect("console worker mutex").take() {
            if flushed { let _ = worker.join(); }
        }
        flushed
    }

    fn send_control(&self, mut message: Message, deadline: Instant) -> bool {
        loop {
            match self.sender.try_send(message) {
                Ok(()) => return true,
                Err(TrySendError::Disconnected(_)) => return false,
                Err(TrySendError::Full(returned)) => {
                    if Instant::now() >= deadline { return false; }
                    message = returned;
                    thread::sleep(Duration::from_millis(1).min(deadline.saturating_duration_since(Instant::now())));
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct ConsoleStats {
    pub queued: u64,
    pub queue_lines: usize,
    pub written: u64,
    pub dropped_lines: u64,
    pub write_errors: u64,
}

static CONSOLE: OnceLock<Console> = OnceLock::new();

/// Install the process console once. A failure to create the worker is
/// returned to startup; line write failures are best-effort and counted only
/// by successful writes/drops.
pub fn start_console() -> io::Result<&'static Console> {
    if let Some(console) = CONSOLE.get() { return Ok(console); }
    let console = Console::with_output(CONSOLE_QUEUE_LINES, Box::new(Terminal))?;
    let _ = CONSOLE.set(console);
    Ok(CONSOLE.get().expect("console installed"))
}

/// Install a bounded console with a caller-provided sink. This is useful for
/// embedders that own their log destination and for end-to-end log assertions.
pub fn start_console_with_writer<W: Write + Send + 'static>(writer: W) -> io::Result<&'static Console> {
    if let Some(console) = CONSOLE.get() { return Ok(console); }
    let console = Console::with_writer(CONSOLE_QUEUE_LINES, writer)?;
    let _ = CONSOLE.set(console);
    Ok(CONSOLE.get().expect("console installed"))
}

pub fn print_status(message: &str) { print(message, false); }
pub fn print_error(message: &str) { print(message, true); }

/// Request fields that are safe to show in a Started line. No prompt text,
/// model output, or untrusted error messages belong in this record.
#[derive(Clone, Debug)]
pub struct RequestStart {
    pub prompt_tokens: u64,
    pub sent_effort: Option<String>,
    pub effective_effort: Option<String>,
    pub enable_thinking: Option<bool>,
    pub thinking: bool,
}

pub struct RequestLogContext {
    pub start: RequestStart,
    matched_tokens: std::sync::atomic::AtomicU64,
}

impl RequestLogContext {
    pub fn new(start: RequestStart) -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self {
            start,
            matched_tokens: std::sync::atomic::AtomicU64::new(0),
        })
    }

    pub fn admitted(&self, matched_tokens: u32) {
        self.matched_tokens.store(u64::from(matched_tokens), Ordering::Relaxed);
        request_started(&self.start, matched_tokens);
    }

    pub fn matched_tokens(&self) -> u32 {
        self.matched_tokens.load(Ordering::Relaxed).min(u64::from(u32::MAX)) as u32
    }
}

fn grouped_count(value: u64) -> String {
    let digits = value.to_string();
    let mut grouped = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, ch) in digits.chars().enumerate() {
        if index != 0 && (digits.len() - index).is_multiple_of(3) {
            grouped.push(',');
        }
        grouped.push(ch);
    }
    grouped
}

pub fn request_started(request: &RequestStart, cached_tokens: u32) {
    let mut parts = vec![
        "Started".to_owned(),
        format!("input {}", grouped_count(request.prompt_tokens)),
        format!("cached {}", grouped_count(u64::from(cached_tokens))),
    ];
    if let Some(effort) = request.sent_effort.as_deref() {
        parts.push(format!("effort {effort}"));
    } else if let Some(effort) = request.effective_effort.as_deref() {
        parts.push(format!("effort absent → {effort}"));
    } else {
        parts.push("effort absent".into());
    }
    if let Some(enabled) = request.enable_thinking {
        parts.push(format!("enable_thinking {}", enabled));
    }
    parts.push(format!("thinking {}", if request.thinking { "on" } else { "off" }));
    print_status(&parts.join(" · "));
}

/// Log a terminal generation record using values from the native Done event.
pub fn request_done(
    done: &crate::protocol::DoneEvent,
    matched_tokens: u32,
    first_token_batch_tokens: u64,
    tool_calls: &[(String, String)],
) {
    print_status(&format_request_done(done, matched_tokens, first_token_batch_tokens, tool_calls));
}

pub fn format_request_done(
    done: &crate::protocol::DoneEvent,
    matched_tokens: u32,
    first_token_batch_tokens: u64,
    tool_calls: &[(String, String)],
) -> String {
    let outcome = if done.reason == crate::protocol::FinishReason::Cancelled {
        "Cancelled"
    } else {
        "Done"
    };
    let mut parts = vec![
        outcome.to_owned(),
        format!("input {}", grouped_count(u64::from(done.prompt_tokens))),
        format!("cached {}", grouped_count(u64::from(matched_tokens))),
    ];
    if done.deduplicated_tokens != 0 {
        parts.push(format!("deduplicated {}", grouped_count(u64::from(done.deduplicated_tokens))));
    }
    parts.push(format!("output {}", grouped_count(u64::from(done.completion_tokens))));
    if !tool_calls.is_empty() {
        parts.push(format!(
            "tools {}·{}",
            tool_calls.len(),
            tool_call_signature(tool_calls),
        ));
    }
    if done.completion_tokens != 0 && done.wall_micros >= done.decode_micros {
        let ttft_micros = done.wall_micros - done.decode_micros;
        parts.push(format!("TTFT {:.1}s", ttft_micros as f64 / 1_000_000.0));
    }
    let decode_tokens = u64::from(done.completion_tokens).saturating_sub(first_token_batch_tokens);
    if first_token_batch_tokens != 0 && decode_tokens != 0 && done.decode_micros != 0 {
        let speed = decode_tokens as f64 * 1_000_000.0 / done.decode_micros as f64;
        if speed.is_finite() {
            parts.push(format!("{speed:.1} tok/s"));
        }
    }
    if done.draft_cycles != 0 {
        let per_cycle = 1.0 + f64::from(done.accepted_draft_tokens) / f64::from(done.draft_cycles);
        parts.push(format!("{per_cycle:.1} tok/cycle"));
    }
    parts.join(" · ")
}

/// Fingerprint only the public function names and argument key sets. Values
/// are parsed only to obtain object keys and are never serialized or logged.
pub fn tool_call_signature(tool_calls: &[(String, String)]) -> String {
    let shape: Vec<serde_json::Value> = tool_calls.iter().map(|(name, arguments)| {
        let mut keys = serde_json::from_str::<serde_json::Value>(arguments)
            .ok()
            .and_then(|value| value.as_object().cloned())
            .map(|object| object.keys().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        keys.sort();
        serde_json::json!({"name": name, "keys": keys})
    }).collect();
    let canonical = serde_json::to_vec(&shape).unwrap_or_default();
    crate::support::sha256::hex_digest(&canonical)[..8].to_owned()
}

pub fn chat_tool_call_summaries(response: &serde_json::Value) -> Vec<(String, String)> {
    response
        .pointer("/choices/0/message/tool_calls")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .map(|item| (
            item.pointer("/function/name").and_then(serde_json::Value::as_str).unwrap_or("").to_owned(),
            item.pointer("/function/arguments").and_then(serde_json::Value::as_str).unwrap_or("{}").to_owned(),
        ))
        .collect()
}

/// Errors are logged by stable code only, never by their free-form message.
pub fn request_error(code: &str) {
    let safe_code = if code.bytes().all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-') {
        code
    } else {
        "api_error"
    };
    print_error(&format!("Error · {safe_code}"));
}

fn print(message: &str, error: bool) {
    if let Some(console) = CONSOLE.get() {
        let _ = console.write(message, error);
    } else {
        // Before the async console starts, startup is single-threaded.
        let mut stream: Box<dyn Write> = if error { Box::new(io::stderr()) } else { Box::new(io::stdout()) };
        let _ = writeln!(stream, "{message}");
        let _ = stream.flush();
    }
}

pub fn console_stats() -> ConsoleStats {
    CONSOLE.get().map(Console::stats).unwrap_or(ConsoleStats {
        queued: 0, queue_lines: CONSOLE_QUEUE_LINES, written: 0, dropped_lines: 0, write_errors: 0,
    })
}

pub fn flush_console(timeout: Duration) -> bool {
    CONSOLE.get().map(|console| console.flush(timeout)).unwrap_or(true)
}

pub fn close_console() {
    if let Some(console) = CONSOLE.get() { let _ = console.close(Duration::from_secs(2)); }
}

/// Log unexpected failures without their message, which may contain prompts,
/// file paths, or credentials.
pub fn unexpected_error(error: &(dyn std::error::Error + 'static)) {
    print_error(&format!("internal_server_error · {}", std::any::type_name_of_val(error)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct Sink(Arc<Mutex<Vec<u8>>>);
    impl Write for Sink {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> { self.0.lock().unwrap().extend_from_slice(bytes); Ok(bytes.len()) }
        fn flush(&mut self) -> io::Result<()> { Ok(()) }
    }

    struct Broken;
    impl Write for Broken {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> { Err(io::Error::other("closed")) }
        fn flush(&mut self) -> io::Result<()> { Err(io::Error::other("closed")) }
    }

    #[test]
    fn console_writes_summaries_and_reports_counts() {
        let bytes = Arc::new(Mutex::new(Vec::new()));
        let console = Console::with_writer(4, Sink(bytes.clone())).unwrap();
        assert!(console.write("Done · input 12 · output 3", false));
        assert!(console.flush(Duration::from_secs(1)));
        assert_eq!(console.stats().written, 1);
        assert!(console.close(Duration::from_secs(1)));
        let output = String::from_utf8(bytes.lock().unwrap().clone()).unwrap();
        assert!(output.contains("Done · input 12 · output 3"));
    }

    #[test]
    fn zero_capacity_is_rejected() {
        let error = match Console::with_writer(0, Vec::<u8>::new()) { Ok(_) => panic!("expected invalid capacity"), Err(error) => error };
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn sink_failures_are_counted_without_panicking() {
        let console = Console::with_writer(2, Broken).unwrap();
        assert!(console.write("safe status", false));
        assert!(console.flush(Duration::from_secs(1)));
        assert_eq!(console.stats().write_errors, 1);
        assert!(console.close(Duration::from_secs(1)));
    }

    #[test]
    fn request_diagnostic_formatters_keep_only_safe_fields() {
        let started = RequestStart {
            prompt_tokens: 1_024,
            sent_effort: None,
            effective_effort: Some("high".into()),
            enable_thinking: Some(true),
            thinking: true,
        };
        let mut parts = vec![
            "Started".to_owned(),
            format!("input {}", grouped_count(started.prompt_tokens)),
            "cached 12".to_owned(),
        ];
        parts.push("effort absent → high".into());
        parts.push("enable_thinking true".into());
        parts.push("thinking on".into());
        assert_eq!(parts.join(" · "), "Started · input 1,024 · cached 12 · effort absent → high · enable_thinking true · thinking on");

        let hostile_code = "api_error\nprompt";
        assert_eq!(
            if hostile_code.bytes().all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-') { hostile_code } else { "api_error" },
            "api_error",
        );
    }

    #[test]
    fn tool_signature_uses_only_names_and_argument_keys() {
        let first = vec![
            ("search".into(), r#"{"query":"secret one","limit":2}"#.into()),
            ("save".into(), r#"{"path":"/private/a"}"#.into()),
        ];
        let same_shape = vec![
            ("search".into(), r#"{"limit":99,"query":"secret two"}"#.into()),
            ("save".into(), r#"{"path":"/private/b"}"#.into()),
        ];
        let changed_shape = vec![("search".into(), r#"{"query":"other"}"#.into())];
        let signature = tool_call_signature(&first);
        assert_eq!(signature.len(), 8);
        assert_eq!(signature, tool_call_signature(&same_shape));
        assert_ne!(signature, tool_call_signature(&changed_shape));
        assert!(!signature.contains("secret"));
        let done = crate::protocol::DoneEvent {
            request_id: 7,
            reason: crate::protocol::FinishReason::Stop,
            prompt_tokens: 1024,
            completion_tokens: 12,
            prefill_micros: 1000,
            decode_micros: 2000,
            wall_micros: 3500,
            option_logits: Vec::new(),
            deduplicated_tokens: 3,
            drafted_tokens: 0,
            accepted_draft_tokens: 0,
            draft_cycles: 0,
            bounded: false,
        };
        let line = format_request_done(&done, 512, 1, &first);
        assert!(line.contains("tools 2·"), "{line}");
        assert!(line.contains("cached 512 · deduplicated 3"), "{line}");
        assert!(!line.contains("secret one"), "{line}");
        assert!(!line.contains("/private/a"), "{line}");
    }
}
