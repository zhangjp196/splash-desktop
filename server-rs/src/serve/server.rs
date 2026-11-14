//! The serving process's network entry: a listener that answers Chat
//! completions over real TCP, complete or streamed, through the
//! pipeline.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::dialects::openai;
use crate::http::{
    authenticate, read_timed_request_with_admission, validate_headers, write_response,
    write_response_with_cors, write_sse_event, AllowedOrigin, HttpAdmission, Response,
};
use crate::metrics::prometheus_metrics;
use crate::output::StreamingToolCallProjector;
use crate::serve::preparation::{Deadline, Preparation};
use crate::serve::stream::{stream_callbacks_with_request_log, StreamSink};
use crate::serve::Pipeline;
use crate::support::errors::ApiError;

/// The server: the pipeline it serves, listening on one address.
pub struct Server {
    pipeline: Pipeline,
    loaded_model: String,
    model_names: Vec<String>,
    responses: crate::dialects::responses::ResponseStore,
    thinking_codec: Arc<crate::thinking::ThinkingCodec>,
    status_cache: Mutex<StatusCache>,
    latency: Arc<crate::metrics::LatencyMetrics>,
    preparation: OnceLock<Preparation>,
    local_security: Option<Option<String>>,
    allowed_hosts: Vec<String>,
    allowed_origins: Vec<AllowedOrigin>,
    request_timeout: Option<Duration>,
    connection_admission: Arc<HttpAdmission>,
    body_admission: Arc<HttpAdmission>,
    restart_count: Arc<AtomicU64>,
    max_request_size: usize,
    default_thinking_token_budget: Option<u32>,
    loop_guard: bool,
    compact_at: Option<crate::serve::compaction::CompactAt>,
    compact_keep: Option<u32>,
    compact_memory: Option<PathBuf>,
    webui: bool,
    next_request_id: AtomicU64,
}

const STATUS_CACHE_TTL: Duration = Duration::from_millis(500);
const STATUS_REFRESH_TIMEOUT: Duration = Duration::from_millis(50);
const STATUS_ANSWER_GRACE: Duration = Duration::from_secs(30);

#[derive(Default)]
struct StatusCache {
    snapshot: Option<Value>,
    captured_at: Option<Instant>,
    unanswered_since: Option<Instant>,
}

impl StatusCache {
    fn fresh_snapshot(&self, now: Instant) -> Option<(&Value, Duration)> {
        let snapshot = self.snapshot.as_ref()?;
        let captured_at = self.captured_at?;
        let age = now.saturating_duration_since(captured_at);
        (age < STATUS_CACHE_TTL
            && snapshot.get("ready").and_then(Value::as_bool) == Some(true))
        .then_some((snapshot, age))
    }
}

impl Server {
    pub fn new(pipeline: Pipeline) -> Server {
        let latency = Arc::new(crate::metrics::LatencyMetrics::default());
        pipeline.client.set_latency_metrics(latency.clone());
        let loaded_model = pipeline.model.clone();
        Server {
            pipeline,
            loaded_model: loaded_model.clone(),
            model_names: vec![loaded_model],
            responses: crate::dialects::responses::ResponseStore::new(),
            thinking_codec: Arc::new(
                crate::thinking::ThinkingCodec::generate()
                    .expect("system secure randomness is available"),
            ),
            status_cache: Mutex::new(StatusCache::default()),
            latency,
            preparation: OnceLock::new(),
            local_security: None,
            allowed_hosts: Vec::new(),
            allowed_origins: Vec::new(),
            request_timeout: None,
            connection_admission: Arc::new(HttpAdmission::new(72).expect("positive limit")),
            body_admission: Arc::new(
                HttpAdmission::new(1024 * 1024 * 1024).expect("positive body budget"),
            ),
            restart_count: Arc::new(AtomicU64::new(0)),
            max_request_size: crate::http::DEFAULT_MAX_REQUEST_SIZE,
            default_thinking_token_budget: None,
            loop_guard: false,
            compact_at: None,
            compact_keep: None,
            compact_memory: None,
            webui: true,
            next_request_id: AtomicU64::new(1),
        }
    }

    pub fn with_webui(mut self, enabled: bool) -> Self {
        self.webui = enabled;
        self
    }

    /// Configure additional API model names and the optional announced name.
    /// The loaded identifier remains available to `/status`.
    pub fn with_served_model_names(mut self, names: Vec<String>, announce: bool) -> Self {
        let response_model = if announce {
            names.first().cloned().unwrap_or_else(|| self.loaded_model.clone())
        } else {
            self.loaded_model.clone()
        };
        self.model_names.clear();
        for name in std::iter::once(response_model.clone())
            .chain(std::iter::once(self.loaded_model.clone()))
            .chain(names)
        {
            if !self.model_names.contains(&name) {
                self.model_names.push(name);
            }
        }
        self.pipeline.model = response_model;
        self
    }

    fn accepts_model(&self, model: Option<&str>) -> bool {
        model.map_or(true, |model| self.model_names.iter().any(|name| name == model))
    }

    /// Use the persistent Fernet key shared by Anthropic responses and
    /// subsequent requests carrying those signatures.
    pub fn with_thinking_codec(
        mut self,
        thinking_codec: Arc<crate::thinking::ThinkingCodec>,
    ) -> Self {
        self.thinking_codec = thinking_codec;
        self
    }

    /// Enforce local Host/Origin policy and optionally authenticate generation.
    pub fn with_api_key(mut self, key: Option<String>) -> Self {
        self.local_security = Some(key);
        self
    }

    /// Set the default wall-clock budget for every request. A request's
    /// own `timeout` may shorten this budget.
    pub fn with_request_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.request_timeout = timeout;
        self
    }

    /// Limit connection handlers to the request queue plus a bounded
    /// allowance for clients connecting or sending their request heads.
    pub fn with_connection_limit(mut self, limit: usize) -> Self {
        self.connection_admission = Arc::new(HttpAdmission::new(limit).expect("positive limit"));
        self
    }

    /// Share the process supervisor's cumulative engine restart count with
    /// the live status document.
    pub fn with_restart_count(mut self, restart_count: Arc<AtomicU64>) -> Self {
        self.restart_count = restart_count;
        self
    }

    /// Add host names accepted by the HTTP Host policy.
    pub fn with_allowed_hosts(mut self, hosts: Vec<String>) -> Self {
        self.allowed_hosts = hosts;
        self
    }

    /// Add origins admitted by the browser Origin policy.
    pub fn with_allowed_origins(mut self, origins: Vec<AllowedOrigin>) -> Self {
        self.allowed_origins = origins;
        self
    }

    /// Set the largest request body accepted by the listener.
    pub fn with_max_request_size(mut self, size: usize) -> Self {
        self.max_request_size = size;
        self
    }

    /// Set the default cap on thinking tokens for requests that think.
    pub fn with_thinking_token_budget(mut self, budget: Option<u32>) -> Self {
        self.default_thinking_token_budget = budget;
        self
    }

    pub fn with_loop_guard(mut self, enabled: bool) -> Self {
        self.loop_guard = enabled;
        self
    }

    fn effective_thinking_token_budget(&self) -> Option<u32> {
        self.default_thinking_token_budget.or_else(|| {
            self.loop_guard.then_some(crate::protocol::NO_THINKING_BUDGET)
        })
    }

    /// Enable context compaction at a fraction of the context, or when the
    /// prompt and requested reply would exceed the reserved context.
    pub fn with_compaction(
        mut self,
        at: Option<crate::serve::compaction::CompactAt>,
        keep: Option<u32>,
        memory: Option<PathBuf>,
    ) -> Self {
        self.compact_at = at;
        self.compact_keep = keep;
        self.compact_memory = memory;
        self
    }

    /// Serve until the process requests shutdown or the engine exits.
    pub fn serve_until(
        self,
        listener: TcpListener,
        mut stopped: impl FnMut() -> bool,
    ) -> std::io::Result<()> {
        listener.set_nonblocking(true)?;
        let server = Arc::new(self);
        while !stopped() {
            match listener.accept() {
                Ok((mut stream, _)) => {
                    stream.set_nonblocking(false)?;
                    if !server.connection_admission.acquire(1) {
                        let _ = write_response(
                            &mut stream,
                            &Response::error(
                                503,
                                "HTTP connection capacity is exhausted",
                                "frontend_overloaded",
                            ),
                        );
                        continue;
                    }
                    let slot = ConnectionSlot(server.connection_admission.clone());
                    let server = server.clone();
                    std::thread::spawn(move || {
                        let _slot = slot;
                        let _ = serve_connection(&mut stream, &server);
                    });
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error),
            }
        }
        server.pipeline.client.close();
        Ok(())
    }

    /// Serve on the listener the caller bound, until it fails. One thread
    /// per connection, as the Python's frontend runs.
    pub fn serve(self, listener: TcpListener) {
        let server = Arc::new(self);
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            if !server.connection_admission.acquire(1) {
                let _ = write_response(
                    &mut stream,
                    &Response::error(
                        503,
                        "HTTP connection capacity is exhausted",
                        "frontend_overloaded",
                    ),
                );
                continue;
            }
            let slot = ConnectionSlot(server.connection_admission.clone());
            let server = server.clone();
            std::thread::spawn(move || {
                let _slot = slot;
                let _ = serve_connection(&mut stream, &server);
            });
        }
    }

    fn preparation(&self) -> Option<&Preparation> {
        let ready = self.pipeline.client.readiness()?;
        if ready.max_concurrent_requests == 0 {
            return None;
        }
        Some(self.preparation.get_or_init(|| {
            Preparation::new(ready.max_concurrent_requests as usize, self.latency.clone())
        }))
    }
}

struct ConnectionSlot(Arc<HttpAdmission>);

impl Drop for ConnectionSlot {
    fn drop(&mut self) {
        let _ = self.0.release(1);
    }
}

/// Watch an already-read HTTP connection while a complete response waits
/// on generation. EOF or a reset cancels each engine call submitted for
/// this response. The watch also drains unexpected trailing bytes so they
/// cannot disguise a subsequent EOF.
pub struct DisconnectWatch {
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl DisconnectWatch {
    pub fn start(
        stream: &TcpStream,
        client: crate::engine::EngineClient,
        calls: Vec<Arc<crate::engine::Call>>,
    ) -> std::io::Result<Self> {
        let mut monitor = stream.try_clone()?;
        monitor.set_read_timeout(Some(Duration::from_millis(100)))?;
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = stop.clone();
        let thread = std::thread::Builder::new()
            .name("splash-http-disconnect".into())
            .spawn(move || {
                let mut buffer = [0u8; 4096];
                loop {
                    if thread_stop.load(Ordering::Relaxed) {
                        break;
                    }
                    match monitor.read(&mut buffer) {
                        Ok(0) => {
                            for call in &calls { client.cancel(call); }
                            break;
                        }
                        Ok(_) => {}
                        Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                        Err(error)
                            if matches!(error.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => continue,
                        Err(_) => {
                            for call in &calls { client.cancel(call); }
                            break;
                        }
                    }
                }
            })?;
        Ok(Self { stop, thread: Some(thread) })
    }
}

impl Drop for DisconnectWatch {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() { let _ = thread.join(); }
    }
}

/// One connection: read a request, route it, answer.
fn serve_connection(stream: &mut TcpStream, server: &Server) -> std::io::Result<()> {
    let mut latency = RequestLatencyGuard {
        metrics: &server.latency,
        started: Instant::now(),
    };
    stream.set_read_timeout(Some(
        server.request_timeout.unwrap_or(Duration::from_secs(15)),
    ))?;
    stream.set_write_timeout(Some(Duration::from_secs(15)))?;
    let request_result = match read_timed_request_with_admission(
        stream,
        Some(&server.body_admission),
        server.max_request_size,
        server.request_timeout,
    ) {
        Ok(request) => request,
        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
            return write_response(
                stream,
                &Response::error(
                    503,
                    "request body capacity is exhausted; retry shortly",
                    "frontend_overloaded",
                ),
            );
        }
        Err(error) if error.kind() == std::io::ErrorKind::FileTooLarge => {
            return write_response(
                stream,
                &Response::error(413, "request body exceeds its configured limit", "request_too_large"),
            );
        }
        Err(error) if error.kind() == std::io::ErrorKind::TimedOut => {
            return write_response(
                stream,
                &Response::error(408, "request timed out while reading", "request_timeout"),
            );
        }
        Err(error) => return Err(error),
    };
    let Some((mut request, started_at)) = request_result else {
        return Ok(());
    };
    latency.started = request.headers_received_at;
    if let Some(upload_duration) = request.upload_duration {
        server.latency.observe("upload", upload_duration);
    }
    let mut cors_origin = None;
    if let Some(key) = &server.local_security {
        let mut hosts = vec!["localhost".to_owned(), "127.0.0.1".to_owned(), "::1".to_owned()];
        hosts.extend(server.allowed_hosts.iter().cloned());
        match validate_headers(&request.headers, &hosts, &server.allowed_origins) {
            Err(error) => {
                let error = ApiError::new(error.status, error.message, error.code);
                return write_response(stream, &Response::json(error.status, &error.openai_payload()));
            }
            Ok(origin) => cors_origin = origin,
        }
        let protected = requires_api_key(&request.method, &request.path);
        if protected {
            if let Some(key) = key {
                if let Err(error) = authenticate(&request.headers, key) {
                    let error = ApiError::new(error.status, error.message, error.code);
                    return write_response_with_cors(
                        stream,
                        &Response::json(error.status, &error.openai_payload()),
                        cors_origin.as_deref(),
                    );
                }
            }
        }
    }
    let response = match (request.method.as_str(), request.path.as_str()) {
        ("POST", "/v1/chat/completions") => {
            chat_completions(
                &mut request,
                started_at,
                stream,
                server,
                cors_origin.as_deref(),
                None,
            )
        }
        ("POST", "/v1/completions") => {
            text_completions(
                &mut request,
                started_at,
                stream,
                server,
                cors_origin.as_deref(),
            )
        }
        ("POST", "/v1/judgments") => {
            judgments(&mut request, started_at, stream, server, cors_origin.as_deref())
        }
        ("POST", "/v1/systemone") => {
            systemone(&mut request, started_at, stream, server)
        }
        ("POST", "/v1/messages") => {
            anthropic_messages(
                &mut request,
                started_at,
                stream,
                server,
                cors_origin.as_deref(),
            )
        }
        ("POST", "/v1/responses") => {
            responses_create(
                &mut request,
                started_at,
                stream,
                server,
                cors_origin.as_deref(),
            )
        }
        ("GET", path) if response_id_from_path(path).is_some() => {
            let id = response_id_from_path(path).unwrap();
            match server.responses.get(id) {
                Some(stored) => Ok(Response::json(200, &stored.response)),
                None => Ok(Response::error(404, "response not found", "not_found_error")),
            }
        }
        ("DELETE", path) if response_id_from_path(path).is_some() => {
            let id = response_id_from_path(path).unwrap();
            if server.responses.delete(id) {
                Ok(Response::json(200, &json!({"id": id, "object": "response", "deleted": true})))
            } else {
                Ok(Response::error(404, "response not found", "not_found_error"))
            }
        }
        ("POST", "/v1/messages/count_tokens") => {
            let response = match request.json() {
                Ok(body) => {
                    let resolver = |signature: &str| -> Result<String, crate::chat_messages::MessageError> {
                        server.thinking_codec.decode(signature).map_err(|error| {
                            crate::chat_messages::MessageError {
                                status: error.status,
                                message: error.message,
                                code: error.code,
                            }
                        })
                    };
                    match Deadline::from_body(&body, started_at, server.request_timeout) {
                        Ok(deadline) => match crate::serve::anthropic::prepare_messages(&body, &resolver) {
                            Ok((chat_body, _)) => match prepare_prompt(server, &chat_body, deadline, None, true) {
                                Ok(prepared) => Response::json(200, &json!({"input_tokens": prepared.prompt_tokens.len()})),
                        Err(error) => anthropic_error(error.status, error.message, error.code),
                            },
                            Err(error) => anthropic_error(error.status, error.message, error.code.unwrap_or_else(|| "invalid_request_error".into())),
                        },
                        Err(error) => anthropic_error(error.status, error.message, error.code),
                    }
                }
                Err(error) => anthropic_error(400, error, "invalid_request_error".into()),
            };
            request.release_body_reservation();
            Ok(response)
        }
        ("POST", "/v1/chat/completions/count_tokens") => {
            let response = match request.json() {
                Ok(body) => match Deadline::from_body(&body, started_at, server.request_timeout)
                    .and_then(|deadline| prepare_prompt(server, &body, deadline, None, true))
                {
                    Ok(prepared) => Response::json(200, &json!({"input_tokens": prepared.prompt_tokens.len()})),
                    Err(error) => Response::json(error.status, &error.openai_payload()),
                },
                Err(error) => Response::error(400, &error, "invalid_request_error"),
            };
            request.release_body_reservation();
            Ok(response)
        }
        ("GET", "/status") => Ok(Response::json(200, &status_document(server))),
        ("GET", "/metrics") => Ok(Response {
            status: 200,
            content_type: "text/plain; version=0.0.4; charset=utf-8",
            body: prometheus_metrics(&status_document_cached(server)).into_bytes(),
        }),
        ("GET", "/health") => Ok(Response::json(200, &json!({"status": "ok"}))),
        ("GET", "/ready") => {
            let ready = server.pipeline.client.ready();
            Ok(Response::json(
                if ready { 200 } else { 503 },
                &json!({"status": if ready { "ready" } else { "unavailable" }}),
            ))
        }
        ("POST", "/tokenize") => Ok(tokenize(&request, started_at, server)),
        ("POST", "/detokenize") => Ok(detokenize(&request, started_at, server)),
        ("POST", "/apply-template") => Ok(apply_template(&request, started_at, server)),
        ("POST", "/v1/memory/release") => Ok(release_memory(&request, server)),
        ("POST", "/v1/thermal") => Ok(thermal(&request, server)),
        ("GET", "/v1/models") => Ok(Response::json(200, &json!({
            "object": "list",
            "data": server.model_names.iter().map(|name| {
                let mut document = model_document(server);
                document["id"] = json!(name);
                document
            }).collect::<Vec<_>>(),
            "models": server.model_names.iter().map(|name| json!({
                "name": name, "description": "Splash resident model", "release_date": ""
            })).collect::<Vec<_>>()
        }))),
        ("GET", path) if path.starts_with("/v1/models/") => {
            if let Some(name) = path.strip_prefix("/v1/models/").filter(|name| server.accepts_model(Some(name))) {
                let mut document = model_document(server);
                document["id"] = json!(name);
                Ok(Response::json(200, &document))
            } else {
                Ok(Response::error(404, "model not found", "model_not_found"))
            }
        }
        ("OPTIONS", _) => Ok(Response {
            status: 204,
            content_type: "application/json",
            body: Vec::new(),
        }),
        ("GET", path) if server.webui => Ok(web_asset(path).unwrap_or_else(|| Response::error(404, "no such path", "not_found"))),
        ("GET", _) => Ok(Response::error(404, "no such path", "not_found")),
        _ => Ok(Response::error(404, "no such path", "not_found")),
    };
    match response {
        Ok(response) if response.content_type == "text/event-stream" => Ok(()),
        Ok(response) => write_response_with_cors(stream, &response, cors_origin.as_deref()),
        Err(_) => write_response(
            stream,
            &Response::error(500, "the request failed", "internal_error"),
        ),
    }
}

struct RequestLatencyGuard<'a> {
    metrics: &'a crate::metrics::LatencyMetrics,
    started: Instant,
}

impl Drop for RequestLatencyGuard<'_> {
    fn drop(&mut self) {
        self.metrics.observe("http_request", self.started.elapsed());
    }
}

fn model_document(server: &Server) -> Value {
    let readiness = server.pipeline.client.readiness();
    let context = readiness.as_ref().map(|ready| ready.max_context_tokens);
    let vision = readiness.as_ref().is_some_and(|ready| ready.vision);
    json!({
        "id": server.pipeline.model, "object": "model", "created": 0,
        "owned_by": "splash", "max_model_len": context, "context_length": context,
        "vision": vision, "input_modalities": if vision { vec!["text", "image"] } else { vec!["text"] },
    })
}

fn requires_api_key(method: &str, path: &str) -> bool {
    method == "POST"
        || path == "/status"
        || path == "/metrics"
        || path == "/v1/models"
        || path.starts_with("/v1/models/")
        || path.starts_with("/v1/responses/")
}

fn response_id_from_path(path: &str) -> Option<&str> {
    let id = path.strip_prefix("/v1/responses/")?;
    if id.is_empty() || id.contains('/') || !id.starts_with("resp_")
        || !id.bytes().all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        return None;
    }
    Some(id)
}

fn scoring_error_response(error: ApiError, systemone: bool) -> Response {
    if systemone {
        let (status, _, payload) = error.systemone_answer();
        Response::json(status, &payload)
    } else {
        Response::json(error.status, &error.openai_payload())
    }
}

fn judgments(
    request: &mut crate::http::Request,
    started_at: Instant,
    stream: &mut TcpStream,
    server: &Server,
    _cors_origin: Option<&str>,
) -> std::io::Result<Response> {
    let mut body = match request.json() {
        Ok(body) => body,
        Err(error) => return Ok(Response::error(400, &error, "invalid_request_error")),
    };
    if let Err(error) = normalize_request_model(server, &mut body) {
        return Ok(Response::json(error.status, &error.openai_payload()));
    }
    let row = match crate::serve::scoring::parse_judgment(&body, &server.pipeline.model) {
        Ok(row) => row,
        Err(error) => return Ok(scoring_error_response(error, false)),
    };
    let deadline = match Deadline::from_body(&body, started_at, server.request_timeout) {
        Ok(deadline) => deadline,
        Err(error) => return Ok(scoring_error_response(error, false)),
    };
    let messages = row.messages();
    let prompt_body = json!({"messages": messages, "reasoning_effort":"none"});
    let mut prepared = match prepare_prompt(server, &prompt_body, deadline, None, true) {
        Ok(prepared) => prepared,
        Err(error) => return Ok(scoring_error_response(error, false)),
    };
    crate::serve::scoring::finalize_prompt(&server.pipeline, &mut prepared);
    let labels: Vec<String> = crate::serve::scoring::LETTERS.chars().take(row.options.len()).map(|ch| ch.to_string()).collect();
    let slots = match crate::serve::scoring::answer_slots(&server.pipeline, &prepared, &labels) {
        Ok(slots) => slots,
        Err(error) => return Ok(scoring_error_response(error, false)),
    };
    let frame_options = match crate::serve::scoring::scoring_frame_options(&body, &server.pipeline, &prepared) {
        Ok(options) => options,
        Err(error) => return Ok(Response::error(error.status, &error.message, &error.code.unwrap_or_else(|| "invalid_request_error".into()))),
    };
    request.release_body_reservation();
    let remaining = match deadline.remaining() {
        Ok(remaining) => remaining,
        Err(error) => return Ok(scoring_error_response(error, false)),
    };
    let (done, probabilities, slot_ids) = match crate::serve::scoring::run_score(
        &server.pipeline, &prepared, slots, remaining, frame_options, Some(stream),
    ) {
        Ok(result) => result,
        Err(error) => return Ok(Response::error(error.status, &error.message, &error.code.unwrap_or_else(|| "api_error".into()))),
    };
    let response = crate::serve::scoring::judgment_answer(
        &server.pipeline.model, &row, &prepared.rendered, &done, &probabilities, &slot_ids,
    );
    Ok(Response::json(200, &response))
}

fn systemone(request: &mut crate::http::Request, started_at: Instant, stream: &mut TcpStream, server: &Server) -> std::io::Result<Response> {
    let mut body = match request.json() {
        Ok(body) => body,
        Err(error) => return Ok(Response::error(400, &error, "invalid_request_error")),
    };
    if let Err(error) = normalize_request_model(server, &mut body) {
        return Ok(scoring_error_response(error, true));
    }
    let (state, entries) = match crate::serve::scoring::parse_systemone(&body, &server.pipeline.model) {
        Ok(parsed) => parsed,
        Err(error) => return Ok(scoring_error_response(error, true)),
    };
    let deadline = match Deadline::from_body(&body, started_at, server.request_timeout) {
        Ok(deadline) => deadline,
        Err(error) => return Ok(scoring_error_response(error, true)),
    };
    request.release_body_reservation();
    let supported_slots = crate::serve::scoring::tokenizer_slot_labels(&server.pipeline);
    let mut prepared = Vec::new();
    let mut total_tokens = 0usize;
    for entry in entries {
        if entry.spec.deterministic {
            prepared.push((entry, None, Vec::new()));
            continue;
        }
        if entry.spec.labels.len() > supported_slots.len() {
            let error = crate::serve::scoring::unsupported_systemone(
                &["questions", &entry.id, "criteria"],
                &format!("the served tokenizer supports {} answer slots; {} were requested", supported_slots.len(), entry.spec.labels.len()),
            );
            return Ok(scoring_error_response(error, true));
        }
        let slots = supported_slots[..entry.spec.labels.len()].to_vec();
        let prompt_body = json!({"messages": crate::serve::scoring::systemone_messages(&state, &entry.spec, &slots), "reasoning_effort":"none"});
        let mut prepared_prompt = match prepare_prompt(server, &prompt_body, deadline, None, true) {
            Ok(prompt) => prompt,
            Err(error) => return Ok(scoring_error_response(error, true)),
        };
        crate::serve::scoring::finalize_prompt(&server.pipeline, &mut prepared_prompt);
        if let Some(ready) = server.pipeline.client.readiness() {
            if prepared_prompt.prompt_tokens.len() > ready.max_context_tokens as usize {
                let error = crate::support::errors::ContextLength { input_tokens: prepared_prompt.prompt_tokens.len() as u64, maximum_input_tokens: ready.max_context_tokens as u64, image_tokens_only: false }.into_api_error();
                return Ok(scoring_error_response(error, true));
            }
        }
        total_tokens = total_tokens.saturating_add(prepared_prompt.prompt_tokens.len());
        if total_tokens > crate::serve::scoring::MAX_SYSTEMONE_TOTAL_TOKENS {
            let error = crate::serve::scoring::unsupported_systemone(&["questions", &entry.id], "total prepared question tokens exceed 1048576");
            return Ok(scoring_error_response(error, true));
        }
        let token_ids = match crate::serve::scoring::answer_slots(&server.pipeline, &prepared_prompt, &slots) {
            Ok(tokens) => tokens,
            Err(error) => return Ok(scoring_error_response(error, true)),
        };
        prepared.push((entry, Some(prepared_prompt), token_ids));
    }
    let generation_options = match prepared.iter().find_map(|(_, prompt, _)| prompt.as_ref()).map(|prompt| crate::serve::scoring::scoring_frame_options(&body, &server.pipeline, prompt)) {
        Some(Ok(options)) => options,
        Some(Err(error)) => return Ok(scoring_error_response(ApiError::new(error.status, error.message, error.code.unwrap_or_else(|| "invalid_request_error".into())), true)),
        None => crate::serve::pipeline::GenerationFrameOptions::default(),
    };
    let mut answers = serde_json::Map::new();
    let mut input_tokens = 0u64;
    for (entry, prompt, slot_ids) in prepared {
        let probabilities = if let Some(prompt) = prompt {
            let remaining = match deadline.remaining() { Ok(remaining) => remaining, Err(error) => return Ok(scoring_error_response(error, true)) };
            match crate::serve::scoring::run_score(&server.pipeline, &prompt, slot_ids, remaining, generation_options.clone(), Some(stream)) {
                Ok((done, probabilities, _)) => { input_tokens += u64::from(done.prompt_tokens); probabilities },
                Err(error) => return Ok(scoring_error_response(ApiError::new(error.status, error.message, error.code.unwrap_or_else(|| "api_error".into())), true)),
            }
        } else { vec![1.0] };
        answers.insert(entry.id, crate::serve::scoring::systemone_answer(&entry.spec, &probabilities));
    }
    Ok(Response::json(200, &json!({"model":server.pipeline.model,"answers":answers,"usage":{"input_tokens":input_tokens,"output_tokens":0}})))
}

fn responses_create(
    request: &mut crate::http::Request,
    started_at: Instant,
    stream: &mut TcpStream,
    server: &Server,
    cors_origin: Option<&str>,
) -> std::io::Result<Response> {
    let body = match request.json() {
        Ok(body) => body,
        Err(error) => return Ok(Response::error(400, &error, "invalid_request_error")),
    };
    let Some(object) = body.as_object() else {
        return Ok(Response::error(400, "request body must be an object", "invalid_request_error"));
    };
    let store = match object.get("store") {
        None | Some(Value::Null) => true,
        Some(Value::Bool(value)) => *value,
        Some(_) => return Ok(Response::error(400, "store must be a boolean", "invalid_request_error")),
    };
    let previous_response_id = match object.get("previous_response_id") {
        None | Some(Value::Null) => None,
        Some(Value::String(id)) if !id.is_empty() => Some(id.clone()),
        Some(_) => return Ok(Response::error(400, "previous_response_id must be a non-empty string", "invalid_request_error")),
    };
    let items = if let Some(id) = previous_response_id.as_deref() {
        let Some(previous) = server.responses.get(id) else {
            return Ok(Response::error(404, "previous response not found", "previous_response_not_found"));
        };
        let mut items = previous.history;
        match crate::dialects::responses::canonical_responses_input(object.get("input").unwrap_or(&Value::Null)) {
            Ok(current) => items.extend(current),
            Err(error) => return Ok(Response::error(error.status, &error.message, &error.code)),
        }
        items
    } else {
        match crate::dialects::responses::canonical_responses_input(object.get("input").unwrap_or(&Value::Null)) {
            Ok(items) => items,
            Err(error) => return Ok(Response::error(error.status, &error.message, &error.code)),
        }
    };
    let (mut chat_body, tool_metadata) = match crate::dialects::responses::responses_to_chat_body_with_tools(&body, &items) {
        Ok(chat) => chat,
        Err(error) => return Ok(Response::error(error.status, &error.message, &error.code)),
    };
    if body.get("stream").and_then(Value::as_bool) == Some(true) {
        return responses_stream(
            request,
            started_at,
            stream,
            server,
            cors_origin,
            body,
            chat_body,
            items,
            store,
            previous_response_id,
            tool_metadata,
        );
    }
    chat_body["stream"] = Value::Bool(false);
    request.body = match serde_json::to_vec(&chat_body) {
        Ok(bytes) => bytes,
        Err(_) => return Ok(Response::error(500, "request could not be normalized", "internal_error")),
    };
    // Reuse Chat's complete generation path and its deadline, admission,
    // cancellation, tool and output validation behavior.
    let chat = chat_completions(request, started_at, stream, server, cors_origin, Some(&tool_metadata))?;
    if chat.status != 200 || chat.content_type != "application/json" {
        return Ok(chat);
    }
    let chat_value: Value = match serde_json::from_slice(&chat.body) {
        Ok(value) => value,
        Err(_) => return Ok(Response::error(500, "chat generation returned invalid JSON", "internal_error")),
    };
    let public_id = chat_value.get("id").and_then(Value::as_str)
        .and_then(|id| id.strip_prefix("chatcmpl-"))
        .unwrap_or("unknown");
    let created_at = chat_value.get("created").and_then(Value::as_i64).unwrap_or(0);
    let response = match crate::dialects::responses::response_from_chat_with_tools(
        &server.pipeline.model,
        public_id,
        created_at,
        &body,
        &chat_value,
        store,
        previous_response_id.as_deref(),
        &tool_metadata,
    ) {
        Ok(response) => response,
        Err(error) => return Ok(Response::error(error.status, &error.message, &error.code)),
    };
    if store {
        let id = response.get("id").and_then(Value::as_str).unwrap_or_default().to_owned();
        server.responses.put(id, response.clone(), items);
    }
    Ok(Response::json(200, &response))
}

#[allow(clippy::too_many_arguments)]
fn responses_stream(
    request: &mut crate::http::Request,
    started_at: Instant,
    stream: &mut TcpStream,
    server: &Server,
    cors_origin: Option<&str>,
    request_body: Value,
    chat_body: Value,
    history: Vec<Value>,
    store_enabled: bool,
    previous_response_id: Option<String>,
    tool_metadata: Value,
) -> std::io::Result<Response> {
    let deadline = match Deadline::from_body(&request_body, started_at, server.request_timeout) {
        Ok(deadline) => deadline,
        Err(error) => return Ok(Response::json(error.status, &error.openai_payload())),
    };
    let prepared = match prepare_chat_with_namespaces(server, &chat_body, deadline, Some(&tool_metadata)) {
        Ok(prepared) => prepared,
        Err(error) => return Ok(Response::json(error.status, &error.openai_payload())),
    };
    request.release_body_reservation();
    let sampling = match crate::serve::pipeline::sampling_parameters(&chat_body) {
        Ok(sampling) => sampling,
        Err(error) => return Ok(Response::error(error.status, &error.message, &error.code.unwrap_or_else(|| "invalid_request_error".into()))),
    };
    let stop_sequences = match crate::serve::pipeline::stop_sequences(&chat_body) {
        Ok(stops) => stops,
        Err(error) => return Ok(Response::error(error.status, &error.message, &error.code.unwrap_or_else(|| "invalid_request_error".into()))),
    };
    let frame_options = match crate::serve::pipeline::generation_frame_options(
        &chat_body, &prepared, &server.pipeline.chat_format, server.effective_thinking_token_budget(),
    ) {
        Ok(options) => options,
        Err(error) => return Ok(Response::error(error.status, &error.message, &error.code.unwrap_or_else(|| "invalid_request_error".into()))),
    };
    let max_context = server.pipeline.client.readiness().map(|ready| ready.max_context_tokens).unwrap_or(0);
    let max_output_tokens = match crate::serve::pipeline::output_budget(&chat_body, prepared.prompt_tokens.len(), max_context) {
        Ok(tokens) => tokens,
        Err(error) => return Ok(Response::error(error.status, &error.message, &error.code.unwrap_or_else(|| "invalid_request_error".into()))),
    };
    let constraint = match server.pipeline.constraint(&prepared) {
        Ok(constraint) => constraint.map(Arc::new),
        Err(error) => return Ok(Response::error(error.status, &error.message, &error.code.unwrap_or_else(|| "server_error".into()))),
    };
    let remaining = match deadline.remaining() {
        Ok(remaining) => remaining,
        Err(error) => return Ok(Response::json(error.status, &error.openai_payload())),
    };
    let public_id = format!("req{}", server.next_request_id.fetch_add(1, Ordering::Relaxed));
    let created_at = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        .map(|since| since.as_secs() as i64).unwrap_or(0);
    let pipeline = &server.pipeline;
    let request_log = chat_request_log_context(&chat_body, pipeline, &prepared);
    let generation = pipeline.generation_request_with_frame_options(
        &prepared, max_output_tokens, constraint.clone(), remaining, sampling, frame_options,
    );

    let state = Arc::new(Mutex::new(crate::dialects::responses::ResponsesStreamState::new_with_tools(
        pipeline.model.clone(), public_id.clone(), created_at, request_body.clone(), store_enabled,
        previous_response_id, history, server.responses.clone(), tool_metadata,
    )));
    let writer = Arc::new(Mutex::new((stream.try_clone()?, false)));
    let cors = cors_origin.map(str::to_owned);
    let stream_state = state.clone();
    let event_writer = writer.clone();
    let write: Arc<Mutex<dyn FnMut(&str) -> Result<(), String> + Send>> = Arc::new(Mutex::new(move |payload: &str| {
        let events = stream_state.lock().expect("Responses stream state lock")
            .consume(payload)?;
        if events.is_empty() { return Ok(()); }
        let mut output = event_writer.lock().expect("Responses stream socket lock");
        if !output.1 {
            write_response_with_cors(&mut output.0, &Response::sse_head(), cors.as_deref())
                .map_err(|error| error.to_string())?;
            output.1 = true;
        }
        for (event, payload) in events {
            let payload = serde_json::to_string(&payload).map_err(|error| error.to_string())?;
            write!(output.0, "event: {event}\ndata: {payload}\n\n").map_err(|error| error.to_string())?;
        }
        output.0.flush().map_err(|error| error.to_string())
    }));
    let sink = Arc::new(StreamSink {
        write,
        model: pipeline.model.clone(),
        request_id: public_id,
        created_at,
        include_usage: true,
        compaction: prepared.compaction.clone(),
    });
    let projector = Arc::new(Mutex::new(match &prepared.tool_policy {
        Some(policy) => StreamingToolCallProjector::new(
            Box::new(crate::tool_schema::ToolParameterTypes::new(policy.clone())),
            1,
            &pipeline.chat_format,
            prepared.response_format.is_some(),
        ),
        None => StreamingToolCallProjector::without_policy(1, &pipeline.chat_format),
    }));
    let tokens = Arc::new(Mutex::new(Vec::new()));
    let cancel_client = pipeline.client.clone();
    let cancel = Arc::new(move |call: &crate::engine::Call| { let _ = cancel_client.cancel(call); });
    let (on_event, on_complete) = stream_callbacks_with_request_log(
        sink,
        tokens,
        pipeline.tokenizer.clone(),
        projector,
        prepared.thinking,
        pipeline.chat_format.clone(),
        prepared.tool_policy.is_some(),
        constraint,
        prepared.response_format.as_ref().map(|format| format.validator.clone()),
        stop_sequences,
        cancel,
        Some(request_log),
    );
    let call = match pipeline.client.submit(&generation, Some(on_event), Some(on_complete)) {
        Ok(call) => call,
        Err(error) => {
            crate::diagnostics::request_error("runtime_unavailable");
            return Ok(Response::error(503, &error.to_string(), "server_error"));
        }
    };
    let _disconnect_watch = DisconnectWatch::start(stream, pipeline.client.clone(), vec![call.clone()])?;
    let _ = call.result(None);
    let started = writer.lock().expect("Responses stream socket lock").1;
    if !started {
        write_response_with_cors(stream, &Response::sse_head(), cors_origin)?;
    }
    Ok(Response { status: 200, content_type: "text/event-stream", body: Vec::new() })
}

fn tokenize(request: &crate::http::Request, started_at: Instant, server: &Server) -> Response {
    let body = match request.json() {
        Ok(body) => body,
        Err(error) => return Response::error(400, &error, "invalid_request_error"),
    };
    let Some(content) = body.get("content").and_then(Value::as_str) else {
        return Response::error(400, "content must be a string", "invalid_request_error");
    };
    let add_special = match body.get("add_special") {
        None => false,
        Some(Value::Bool(value)) => *value,
        _ => return Response::error(400, "add_special must be a boolean", "invalid_request_error"),
    };
    for (field, supported) in [("parse_special", true), ("with_pieces", false)] {
        if body.get(field).and_then(Value::as_bool).is_some_and(|value| value != supported)
            || body.get(field).is_some_and(|value| !value.is_boolean())
        {
            return Response::error(400, &format!("only {field}={supported} is supported"), "invalid_request_error");
        }
    }
    let deadline = match Deadline::from_body(&body, started_at, server.request_timeout) {
        Ok(deadline) => deadline,
        Err(error) => return Response::json(error.status, &error.openai_payload()),
    };
    let Some(preparation) = server.preparation() else {
        return Response::error(503, "native process is not ready", "runtime_unavailable");
    };
    let _slot = match preparation.acquire(deadline) {
        Ok(slot) => slot,
        Err(error) => return Response::json(error.status, &error.openai_payload()),
    };
    if let Err(error) = deadline.remaining() {
        return Response::json(error.status, &error.openai_payload());
    }
    match server.pipeline.tokenizer.encode(content, add_special) {
        Ok(tokens) => Response::json(200, &json!({"tokens": tokens})),
        Err(_) => Response::error(400, "content could not be tokenized", "invalid_request_error"),
    }
}

fn anthropic_error(status: u16, message: String, code: String) -> Response {
    let error = ApiError::new(status, message, code);
    Response::json(status, &error.anthropic_payload())
}

fn detokenize(request: &crate::http::Request, started_at: Instant, server: &Server) -> Response {
    let body = match request.json() {
        Ok(body) => body,
        Err(error) => return Response::error(400, &error, "invalid_request_error"),
    };
    let Some(tokens) = body.get("tokens").and_then(Value::as_array) else {
        return Response::error(400, "tokens must be an array of token ids", "invalid_request_error");
    };
    let mut ids = Vec::with_capacity(tokens.len());
    for token in tokens {
        let Some(id) = token.as_u64().filter(|id| *id < server.pipeline.chat_format.vocabulary as u64) else {
            return Response::error(400, "a token id is outside the vocabulary", "invalid_request_error");
        };
        ids.push(id as u32);
    }
    let special = match body.get("special") {
        None => false,
        Some(Value::Bool(value)) => *value,
        _ => return Response::error(400, "special must be a boolean", "invalid_request_error"),
    };
    let deadline = match Deadline::from_body(&body, started_at, server.request_timeout) {
        Ok(deadline) => deadline,
        Err(error) => return Response::json(error.status, &error.openai_payload()),
    };
    let Some(preparation) = server.preparation() else {
        return Response::error(503, "native process is not ready", "runtime_unavailable");
    };
    let _slot = match preparation.acquire(deadline) {
        Ok(slot) => slot,
        Err(error) => return Response::json(error.status, &error.openai_payload()),
    };
    if let Err(error) = deadline.remaining() {
        return Response::json(error.status, &error.openai_payload());
    }
    match server.pipeline.tokenizer.decode(&ids, !special) {
        Ok(content) => Response::json(200, &json!({"content": content})),
        Err(_) => Response::error(400, "tokens could not be detokenized", "invalid_request_error"),
    }
}

fn apply_template(request: &crate::http::Request, started_at: Instant, server: &Server) -> Response {
    let body = match request.json() {
        Ok(body) => body,
        Err(error) => return Response::error(400, &error, "invalid_request_error"),
    };
    let add_generation_prompt = match body.get("add_generation_prompt") {
        None => true,
        Some(Value::Bool(value)) => *value,
        _ => return Response::error(400, "add_generation_prompt must be a boolean", "invalid_request_error"),
    };
    let deadline = match Deadline::from_body(&body, started_at, server.request_timeout) {
        Ok(deadline) => deadline,
        Err(error) => return Response::json(error.status, &error.openai_payload()),
    };
    match prepare_prompt(server, &body, deadline, None, add_generation_prompt) {
        Ok(prepared) => Response::json(200, &json!({"prompt": prepared.rendered})),
        Err(error) => Response::json(error.status, &json!({"error": {
            "message": error.message,
            "type": "invalid_request_error",
            "code": error.code,
        }})),
    }
}

fn release_memory(request: &crate::http::Request, server: &Server) -> Response {
    if !request.body.is_empty() {
        match request.json() {
            Ok(Value::Object(object)) if object.is_empty() => {}
            Ok(_) => return Response::error(400, "this endpoint takes no fields", "invalid_request_error"),
            Err(error) => return Response::error(400, &error, "invalid_request_error"),
        }
    }
    match server.pipeline.client.release_weights(Duration::from_secs(5)) {
        Ok(event) => {
            let snapshot: Value = serde_json::from_slice(&event.json).unwrap_or_else(|_| json!({}));
            let weights = snapshot.get("weights").cloned().unwrap_or_else(|| json!({}));
            if weights.get("released").and_then(Value::as_bool) == Some(true)
                && weights.get("restoring").and_then(Value::as_bool) == Some(false)
            {
                Response::json(200, &json!({"released": true, "weights": weights}))
            } else {
                Response::error(409, "the engine is serving requests or restoring its weights; it releases them only while idle", "engine_busy")
            }
        }
        Err(_) => Response::error(503, "the engine did not answer within 5 s; retry shortly", "engine_busy"),
    }
}

/// The heat-and-noise rung of the running engine: sets the share of wall
/// time prefill may keep the GPU busy, applied without a restart, and
/// answers with the duty the engine took. Ported from the Python's
/// `_post_thermal`.
fn thermal(request: &crate::http::Request, server: &Server) -> Response {
    let body = match request.json() {
        Ok(body) => body,
        Err(error) => return Response::error(400, &error, "invalid_request_error"),
    };
    let Some(object) = body.as_object() else {
        return Response::error(400, "prefill_duty_percent is required", "invalid_request_error");
    };
    let Some(requested) = object.get("prefill_duty_percent") else {
        return Response::error(400, "prefill_duty_percent is required", "invalid_request_error");
    };
    if let Some(unknown) = object
        .keys()
        .find(|name| name.as_str() != "prefill_duty_percent")
    {
        return Response::error(
            400,
            &format!("this endpoint takes only prefill_duty_percent; got {unknown}"),
            "invalid_request_error",
        );
    }
    let duty = match prefill_duty(requested) {
        Some(duty) => duty,
        None => {
            return Response::error(
                400,
                &format!("prefill_duty_percent {PREFILL_DUTY_RANGE}"),
                "invalid_request_error",
            )
        }
    };
    match server
        .pipeline
        .client
        .set_prefill_duty(duty as f32, Duration::from_secs(5))
    {
        Ok(event) => {
            let snapshot: Value =
                serde_json::from_slice(&event.json).unwrap_or_else(|_| json!({}));
            let took = snapshot
                .pointer("/throttle/duty_percent")
                .and_then(Value::as_f64)
                .unwrap_or(duty);
            Response::json(200, &json!({"prefill_duty_percent": took}))
        }
        Err(_) => Response::error(
            503,
            "the engine did not answer within 5 s; retry shortly",
            "engine_busy",
        ),
    }
}

/// The duty a control request names: a number, or the text of one, from the
/// minimum a person waits for to 100.
fn prefill_duty(value: &Value) -> Option<f64> {
    let duty = match value {
        Value::Number(number) => number.as_f64()?,
        Value::String(text) => text.parse::<f64>().ok()?,
        _ => return None,
    };
    (duty.is_finite() && (MINIMUM_PREFILL_DUTY..=100.0).contains(&duty)).then_some(duty)
}

const MINIMUM_PREFILL_DUTY: f64 = 10.0;
const PREFILL_DUTY_RANGE: &str = "must be a percentage from 10 to 100";

/// The status document the live board reads: the engine's own JSON — its
/// memory, cache, request, fan and throttle sections, with the rates and
/// peaks `metrics` carries — plus what the server serving it adds: its
/// readiness, the model name, the preparation slots, the transport
/// counters, and the instance identity.
fn status_document(server: &Server) -> Value {
    status_document_with_cache(server, false)
}

fn status_document_cached(server: &Server) -> Value {
    status_document_with_cache(server, true)
}

fn status_document_with_cache(server: &Server, cached: bool) -> Value {
    let now = Instant::now();
    let transport_live = server.pipeline.client.ready()
        && server.pipeline.client.fatal_error().is_none();
    if cached && transport_live {
        let cache = server.status_cache.lock().expect("status cache lock");
        if let Some((snapshot, age)) = cache.fresh_snapshot(now) {
            return decorate_status(server, snapshot.clone(), false, Some(age), None);
        }
    }

    let timeout = if cached {
        STATUS_REFRESH_TIMEOUT
    } else {
        Duration::from_secs(2)
    };
    let fresh = server
        .pipeline
        .client
        .status(timeout)
        .map_err(|error| error.to_string())
        .and_then(|event| {
            serde_json::from_slice::<Value>(&event.json)
                .map_err(|error| format!("invalid native status JSON: {error}"))
        })
        .and_then(|snapshot| {
            if snapshot.is_object() {
                Ok(snapshot)
            } else {
                Err("native status did not return an object".to_owned())
            }
        });

    match fresh {
        Ok(snapshot) => {
            let serving = server.pipeline.client.ready()
                && server.pipeline.client.fatal_error().is_none();
            {
                let mut cache = server.status_cache.lock().expect("status cache lock");
                cache.unanswered_since = None;
                if serving {
                    cache.snapshot = Some(snapshot.clone());
                    cache.captured_at = Some(Instant::now());
                }
            }
            decorate_status(server, snapshot, false, None, None)
        }
        Err(error) => {
            stale_status_document(server, error)
        }
    }
}

fn stale_status_document(server: &Server, error: String) -> Value {
    let now = Instant::now();
    let busy = error.contains("timed out");
    let (mut snapshot, age) = {
        let mut cache = server.status_cache.lock().expect("status cache lock");
        if busy && cache.unanswered_since.is_none() {
            cache.unanswered_since = Some(now);
        }
        let snapshot = cache.snapshot.clone();
        let age = cache.captured_at.map(|time| now.saturating_duration_since(time));
        if snapshot.is_none() {
            (json!({"schema_version": 8, "ready": false}), age)
        } else {
            (snapshot.unwrap(), age)
        }
    };
    let still_in_grace = busy
        && server.pipeline.client.ready()
        && server.pipeline.client.fatal_error().is_none()
        && server
            .status_cache
            .lock()
            .expect("status cache lock")
            .unanswered_since
            .is_some_and(|since| now.saturating_duration_since(since) < STATUS_ANSWER_GRACE);
    if !still_in_grace {
        snapshot["ready"] = json!(false);
    }
    let stale_age = age.unwrap_or_default();
    decorate_status(
        server,
        snapshot,
        true,
        Some(stale_age),
        Some(error),
    )
}

fn decorate_status(
    server: &Server,
    mut document: Value,
    stale: bool,
    age: Option<Duration>,
    error: Option<String>,
) -> Value {
    let transport_ready = server.pipeline.client.ready()
        && server.pipeline.client.fatal_error().is_none();
    let ready = transport_ready
        && document.get("ready").and_then(Value::as_bool).unwrap_or(true);
    document["ready"] = json!(ready);
    document["model"] = json!(server.loaded_model);
    document["latency"] = server.latency.snapshot();
    // A status request the engine did not answer leaves the readiness the
    // engine announced as the only truth: the context window the app waits
    // for comes from it rather than disappearing with the document.
    if document.get("maximum_context_tokens").is_none() {
        if let Some(readiness) = server.pipeline.client.readiness() {
            document["maximum_context_tokens"] = json!(readiness.max_context_tokens);
        }
    }
    document["frontend"] = server
        .preparation()
        .map(Preparation::status)
        .unwrap_or_else(|| json!({"preparation_capacity": 0, "active": 0, "waiting": 0}));
    let body_stats = server.body_admission.stats();
    let connection_stats = server.connection_admission.stats();
    document["transport"] = json!({
        "ready": transport_ready,
        "recovering": !transport_ready && server.pipeline.client.fatal_error().is_none(),
        "stopped": server.pipeline.client.fatal_error().is_some(),
        "pending": server.pipeline.client.pending_count(),
        "pending_limit": server
            .pipeline
            .client
            .readiness()
            .map(|ready| ready.max_concurrent_requests)
            .unwrap_or(0),
        "connections": connection_stats.active,
        "connection_limit": connection_stats.capacity,
        "request_body_bytes": body_stats.active,
        "request_body_byte_limit": body_stats.capacity,
        "restarts": server.restart_count.load(Ordering::Relaxed),
        "last_crash_trace": Value::Null,
        "status_stale": stale,
        "status_age_ms": age.map(|value| value.as_secs_f64() * 1000.0).unwrap_or(0.0),
    });
    document["console"] = serde_json::to_value(crate::diagnostics::console_stats())
        .unwrap_or_else(|_| json!({"queued": 0, "queue_lines": crate::diagnostics::CONSOLE_QUEUE_LINES, "written": 0, "dropped_lines": 0, "write_errors": 0}));
    if let Some(error) = error {
        document["transport"]["error"] = json!(error);
    }
    document["instance"] = json!({
        "pid": std::process::id(),
        "model": server.pipeline.model,
    });
    document
}

fn web_asset(path: &str) -> Option<Response> {
    let (content_type, body): (&str, &[u8]) = match path.split('?').next()? {
        "/" | "/chat.html" => ("text/html; charset=utf-8", include_bytes!("../../../server/chat.html")),
        "/favicon.ico" => ("image/svg+xml", include_bytes!("../../../server/favicon.svg")),
        "/chat-assets/github-11.12.0.min.css" => ("text/css", include_bytes!("../../../server/chat-assets/github-11.12.0.min.css")),
        "/chat-assets/github-dark-11.12.0.min.css" => ("text/css", include_bytes!("../../../server/chat-assets/github-dark-11.12.0.min.css")),
        "/chat-assets/markdown-it-15.0.2.min.js" => ("text/javascript", include_bytes!("../../../server/chat-assets/markdown-it-15.0.2.min.js")),
        "/chat-assets/highlight-11.12.0.min.js" => ("text/javascript", include_bytes!("../../../server/chat-assets/highlight-11.12.0.min.js")),
        _ => return None,
    };
    Some(Response { status: 200, content_type, body: body.to_vec() })
}

/// A Chat completion: complete or streamed, as the body's `stream` says.
fn chat_completions(
    request: &mut crate::http::Request,
    started_at: Instant,
    stream: &mut TcpStream,
    server: &Server,
    cors_origin: Option<&str>,
    tool_namespaces: Option<&Value>,
) -> std::io::Result<Response> {
    let pipeline = &server.pipeline;
    let body = match request.json() {
        Ok(body) => body,
        Err(error) => {
            return Ok(Response::error(400, &error, "invalid_request_error"));
        }
    };
    let streamed = body.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);
    let choice_count = match body.get("n").filter(|value| !value.is_null()) {
        None => 1,
        Some(value) => match value.as_u64().filter(|value| (1..=4).contains(value)) {
            Some(value) => value as usize,
            None => return Ok(Response::error(400, "n must be an integer from 1 to 4", "invalid_request_error")),
        },
    };
    if streamed && choice_count > 1 {
        return Ok(Response::error(400, "n > 1 is not supported with stream", "invalid_request_error"));
    }
    let deadline = match Deadline::from_body(&body, started_at, server.request_timeout) {
        Ok(deadline) => deadline,
        Err(error) => return Ok(Response::json(error.status, &error.openai_payload())),
    };
    let prepared = match prepare_chat_with_namespaces(server, &body, deadline, tool_namespaces) {
        Ok(prepared) => prepared,
        Err(error) => return Ok(Response::json(error.status, &error.openai_payload())),
    };
    request.release_body_reservation();
    let sampling = match crate::serve::pipeline::sampling_parameters(&body) {
        Ok(sampling) => sampling,
        Err(error) => {
            return Ok(Response::error(
                error.status,
                &error.message,
                &error.code.unwrap_or_else(|| "invalid_request_error".into()),
            ));
        }
    };
    let stop_sequences = match crate::serve::pipeline::stop_sequences(&body) {
        Ok(stops) => stops,
        Err(error) => {
            return Ok(Response::error(
                error.status,
                &error.message,
                &error.code.unwrap_or_else(|| "invalid_request_error".into()),
            ));
        }
    };
    let frame_options = match crate::serve::pipeline::generation_frame_options(
        &body,
        &prepared,
        &pipeline.chat_format,
        server.effective_thinking_token_budget(),
    ) {
        Ok(options) => options,
        Err(error) => {
            return Ok(Response::error(
                error.status,
                &error.message,
                &error.code.unwrap_or_else(|| "invalid_request_error".into()),
            ));
        }
    };
    let max_context_tokens = pipeline
        .client
        .readiness()
        .map(|ready| ready.max_context_tokens)
        .unwrap_or(0);
    let max_output_tokens = match crate::serve::pipeline::output_budget(
        &body,
        prepared.prompt_tokens.len(),
        max_context_tokens,
    ) {
        Ok(tokens) => tokens,
        Err(error) => {
            return Ok(Response::error(
                error.status,
                &error.message,
                &error.code.unwrap_or_else(|| "invalid_request_error".into()),
            ));
        }
    };
    let job = openai::JobInfo {
        public_id: format!("req{}", server.next_request_id.fetch_add(1, Ordering::Relaxed)),
        created_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|since| since.as_secs() as i64)
            .unwrap_or(0),
        reasoning_tokens: 0,
    };
    let request_log = chat_request_log_context(&body, pipeline, &prepared);
    if streamed {
        stream_answer(
            stream,
            pipeline,
            &body,
            &prepared,
            max_output_tokens,
            &job,
            deadline,
            sampling,
            stop_sequences,
            frame_options,
            cors_origin,
            Some(request.headers_received_at),
            request_log,
        )
    } else {
        complete_answer(
            pipeline,
            stream,
            &body,
            &prepared,
            max_output_tokens,
            &job,
            deadline,
            sampling,
            stop_sequences,
            choice_count,
            frame_options,
            Some(request.headers_received_at),
            request_log,
        )
    }
}

fn chat_request_log_context(
    body: &Value,
    pipeline: &Pipeline,
    prepared: &crate::serve::pipeline::Prepared,
) -> Arc<crate::diagnostics::RequestLogContext> {
    let sent_effort = body.get("reasoning_effort").and_then(Value::as_str).map(str::to_owned);
    let effective_effort = sent_effort.clone().or_else(|| pipeline.default_reasoning_effort.clone());
    let enable_thinking = body
        .get("chat_template_kwargs")
        .and_then(|kwargs| kwargs.get("enable_thinking"))
        .and_then(Value::as_bool);
    crate::diagnostics::RequestLogContext::new(crate::diagnostics::RequestStart {
        prompt_tokens: prepared.prompt_tokens.len() as u64,
        sent_effort,
        effective_effort,
        enable_thinking,
        thinking: prepared.thinking,
    })
}

fn prepare_chat(
    server: &Server,
    body: &Value,
    deadline: Deadline,
) -> Result<crate::serve::pipeline::Prepared, ApiError> {
    prepare_prompt(server, body, deadline, server.compact_at, true)
}

fn prepare_chat_with_namespaces(
    server: &Server,
    body: &Value,
    deadline: Deadline,
    tool_namespaces: Option<&Value>,
) -> Result<crate::serve::pipeline::Prepared, ApiError> {
    prepare_prompt_with_namespaces(server, body, deadline, server.compact_at, true, tool_namespaces)
}

/// Complete one non-streaming Anthropic Messages request using the same
/// admission, preparation and deadline path as Chat Completions.
fn anthropic_messages(
    request: &mut crate::http::Request,
    started_at: Instant,
    stream: &mut TcpStream,
    server: &Server,
    cors_origin: Option<&str>,
) -> std::io::Result<Response> {
    let body = match request.json() {
        Ok(body) => body,
        Err(error) => return Ok(anthropic_error(400, error, "invalid_request_error".into())),
    };
    let streaming = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let deadline = match Deadline::from_body(&body, started_at, server.request_timeout) {
        Ok(deadline) => deadline,
        Err(error) => return Ok(anthropic_error(error.status, error.message, error.code)),
    };
    let resolver = |signature: &str| -> Result<String, crate::chat_messages::MessageError> {
        server.thinking_codec.decode(signature).map_err(|error| {
            crate::chat_messages::MessageError {
                status: error.status,
                message: error.message,
                code: error.code,
            }
        })
    };
    let (chat_body, thinking_display) = match crate::serve::anthropic::prepare_messages(&body, &resolver) {
        Ok(prepared) => prepared,
        Err(error) => return Ok(anthropic_error(error.status, error.message, error.code.unwrap_or_else(|| "invalid_request_error".into()))),
    };
    let prepared = match prepare_chat(server, &chat_body, deadline) {
        Ok(prepared) => prepared,
        Err(error) => return Ok(anthropic_error(error.status, error.message, error.code)),
    };
    let request_log = chat_request_log_context(&chat_body, &server.pipeline, &prepared);
    request.release_body_reservation();
    let remaining = match deadline.remaining() {
        Ok(remaining) => remaining,
        Err(error) => return Ok(anthropic_error(error.status, error.message, error.code)),
    };
    let public_id = format!("req{}", server.next_request_id.fetch_add(1, Ordering::Relaxed));
    if streaming {
        return match crate::serve::anthropic::stream_prepared_messages(
            stream,
            cors_origin,
            &server.pipeline,
            &chat_body,
            &prepared,
            &public_id,
            remaining,
            &thinking_display,
            server.thinking_codec.clone(),
            server.effective_thinking_token_budget(),
            Some(request.headers_received_at),
            Some(request_log.clone()),
        )? {
            Ok(response) => Ok(response),
            Err(error) => Ok(anthropic_error(
                error.status,
                error.message,
                error.code.unwrap_or_else(|| "api_error".into()),
            )),
        };
    }
    match crate::serve::anthropic::complete_prepared_messages(
        &server.pipeline,
        &chat_body,
        &prepared,
        &public_id,
        remaining,
        &thinking_display,
        server.thinking_codec.as_ref(),
        Some(stream),
        Some(request.headers_received_at),
        Some(request_log),
    ) {
        Ok(response) => Ok(Response::json(200, &response)),
        Err(error) => Ok(anthropic_error(
            error.status,
            error.message,
            error.code.unwrap_or_else(|| "api_error".into()),
        )),
    }
}

/// A raw-prompt OpenAI text completion: complete or streamed, as the
/// request asks.
fn text_completions(
    request: &mut crate::http::Request,
    started_at: Instant,
    stream: &mut TcpStream,
    server: &Server,
    cors_origin: Option<&str>,
) -> std::io::Result<Response> {
    let mut body = match request.json() {
        Ok(body) => body,
        Err(error) => return Ok(Response::error(400, &error, "invalid_request_error")),
    };
    if let Err(error) = normalize_request_model(server, &mut body) {
        return Ok(Response::json(error.status, &error.openai_payload()));
    }
    let deadline = match Deadline::from_body(&body, started_at, server.request_timeout) {
        Ok(deadline) => deadline,
        Err(error) => return Ok(Response::json(error.status, &error.openai_payload())),
    };
    let Some(preparation) = server.preparation() else {
        return Ok(Response::error(503, "native process is not ready", "runtime_unavailable"));
    };
    let _slot = match preparation.acquire(deadline) {
        Ok(slot) => slot,
        Err(error) => return Ok(Response::json(error.status, &error.openai_payload())),
    };
    let prepared = match crate::serve::completions::prepare_request(
        &server.pipeline,
        &body,
        server.effective_thinking_token_budget(),
    ) {
        Ok(prepared) => prepared,
        Err(error) => {
            return Ok(Response::error(
                error.status,
                &error.message,
                &error.code.unwrap_or_else(|| "invalid_request_error".into()),
            ));
        }
    };
    if let Err(error) = deadline.remaining() {
        return Ok(Response::json(error.status, &error.openai_payload()));
    }
    request.release_body_reservation();
    let public_id = format!("req{}", server.next_request_id.fetch_add(1, Ordering::Relaxed));
    let created_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since| since.as_secs() as i64)
        .unwrap_or(0);
    let remaining = match deadline.remaining() {
        Ok(remaining) => remaining,
        Err(error) => return Ok(Response::json(error.status, &error.openai_payload())),
    };
    if prepared.streamed {
        return match crate::serve::completions::stream(
            stream,
            cors_origin,
            &server.pipeline,
            &prepared,
            &public_id,
            created_at,
            remaining,
            Some(request.headers_received_at),
        )? {
            Ok(response) => Ok(response),
            Err(error) => Ok(Response::error(
                error.status,
                &error.message,
                &error.code.unwrap_or_else(|| "invalid_request_error".into()),
            )),
        };
    }
    match crate::serve::completions::complete(
        &server.pipeline,
        &prepared,
        &public_id,
        created_at,
        remaining,
        Some(stream),
        Some(request.headers_received_at),
    ) {
        Ok(response) => Ok(Response::json(200, &response)),
        Err(error) => Ok(Response::error(
            error.status,
            &error.message,
            &error.code.unwrap_or_else(|| "api_error".into()),
        )),
    }
}

fn prepare_prompt(
    server: &Server,
    body: &Value,
    deadline: Deadline,
    compact_at: Option<crate::serve::compaction::CompactAt>,
    add_generation_prompt: bool,
) -> Result<crate::serve::pipeline::Prepared, ApiError> {
    prepare_prompt_with_namespaces(server, body, deadline, compact_at, add_generation_prompt, None)
}

fn prepare_prompt_with_namespaces(
    server: &Server,
    body: &Value,
    deadline: Deadline,
    compact_at: Option<crate::serve::compaction::CompactAt>,
    add_generation_prompt: bool,
    tool_namespaces: Option<&Value>,
) -> Result<crate::serve::pipeline::Prepared, ApiError> {
    deadline.remaining()?;
    let mut normalized_body = body.clone();
    normalize_request_model(server, &mut normalized_body)?;
    let body = &normalized_body;
    let preparation = server
        .preparation()
        .ok_or_else(|| ApiError::new(503, "native process is not ready", "runtime_unavailable"))?;
    let expanded_body = if crate::serve::pipeline::Pipeline::contains_pdf_files(body) {
        let remaining = deadline.remaining()?;
        Some(server.pipeline.expand_documents(body, remaining).map_err(|error| {
            ApiError::new(
                error.status,
                error.message,
                error.code.unwrap_or_else(|| "invalid_request_error".to_string()),
            )
        })?)
    } else {
        None
    };
    let body = expanded_body.as_ref().unwrap_or(body);
    let _slot = preparation.acquire(deadline)?;
    let _preparation_timer = server.latency.timer("preparation");
    // The deadline runs through preparation: an image decode an expired
    // request starts is a decode the answer never reads.
    let mut remaining = || deadline.remaining();
    let prepared = server.pipeline.prepare_with_compaction_and_namespaces(
        body,
        &mut remaining,
        compact_at,
        server.compact_keep,
        if compact_at.is_some() { server.compact_memory.as_deref() } else { None },
        add_generation_prompt,
        tool_namespaces,
    );
    deadline.remaining()?;
    prepared.map_err(|error| {
        ApiError::new(
            error.status,
            error.message,
            error.code.unwrap_or_else(|| "invalid_request_error".to_string()),
        )
    })
}

fn normalize_request_model(server: &Server, body: &mut Value) -> Result<(), ApiError> {
    let Some(model) = body.get("model") else { return Ok(()) };
    if model.is_null() || model.as_str() == Some("") {
        return Ok(());
    }
    let Some(model_name) = model.as_str() else {
        return Err(ApiError::new(404, "requested model not found", "model_not_found"));
    };
    if !server.accepts_model(Some(model_name)) {
        return Err(ApiError::new(404, format!("model {model_name} not found"), "model_not_found"));
    }
    if let Some(model) = body.as_object_mut().and_then(|object| object.get_mut("model")) {
        *model = json!(server.pipeline.model);
    }
    Ok(())
}

/// The complete answer: submit, collect, assemble, answer as JSON.
fn complete_answer(
    pipeline: &Pipeline,
    stream: &TcpStream,
    body: &Value,
    prepared: &crate::serve::pipeline::Prepared,
    max_output_tokens: u32,
    job: &openai::JobInfo,
    deadline: Deadline,
    sampling: crate::protocol::SamplingParameters,
    stop_sequences: Vec<String>,
    choice_count: usize,
    frame_options: crate::serve::pipeline::GenerationFrameOptions,
    http_started_at: Option<Instant>,
    request_log: Arc<crate::diagnostics::RequestLogContext>,
) -> std::io::Result<Response> {
    let mut submissions: Vec<(
        Arc<crate::engine::Call>,
        Arc<Mutex<Vec<u32>>>,
        Arc<std::sync::atomic::AtomicU64>,
    )> = Vec::with_capacity(choice_count);
    for index in 0..choice_count {
        let remaining = match deadline.remaining() {
            Ok(remaining) => remaining,
            Err(error) => {
                for (call, _, _) in &submissions {
                    pipeline.client.cancel(call);
                }
                return Ok(Response::json(error.status, &error.openai_payload()));
            }
        };
        let mut options = frame_options.clone();
        options.seed = options.seed.wrapping_add(index as u64);
        if options.logprobs && index > 0 {
            options.cache_salt = options.cache_salt.wrapping_add(index as u128);
            if options.cache_salt == 0 {
                options.cache_salt = index as u128;
            }
        }
        match pipeline.submit_with_frame_options_and_http_start_and_diagnostics(
            prepared,
            max_output_tokens,
            remaining,
            sampling,
            options,
            if index == 0 { http_started_at } else { None },
            if index == 0 { Some(request_log.clone()) } else { None },
        ) {
            Ok(submission) => submissions.push(submission),
            Err(error) => {
                crate::diagnostics::request_error(error.code.as_deref().unwrap_or("server_error"));
                for (call, _, _) in &submissions {
                    pipeline.client.cancel(call);
                }
                return Ok(Response::error(
                    error.status,
                    &error.message,
                    &error.code.unwrap_or_else(|| "server_error".to_string()),
                ));
            }
        }
    }

    let _disconnect_watch = match DisconnectWatch::start(
        stream,
        pipeline.client.clone(),
        submissions.iter().map(|(call, _, _)| call.clone()).collect(),
    ) {
        Ok(watch) => watch,
        Err(error) => {
            for (call, _, _) in &submissions { pipeline.client.cancel(call); }
            return Err(error);
        }
    };

    let mut first_response = None;
    let mut prompt_tokens = 0u64;
    let mut completion_tokens = 0u64;
    let mut choices = Vec::with_capacity(choice_count);
    for (index, (call, tokens, first_batch)) in submissions.iter().enumerate() {
        let first_token_batch_tokens = first_batch.load(std::sync::atomic::Ordering::Relaxed);
        let choice_response = match pipeline.complete_with_stops(
            body,
            prepared,
            call,
            tokens,
            first_token_batch_tokens,
            &pipeline.model,
            job,
            &stop_sequences,
        ) {
            Ok(response) => response,
            Err(error) => {
                crate::diagnostics::request_error(error.code.as_deref().unwrap_or("server_error"));
                for (other, _, _) in &submissions[index + 1..] {
                    pipeline.client.cancel(other);
                }
                return Ok(Response::error(
                    error.status,
                    &error.message,
                    &error.code.unwrap_or_else(|| "server_error".to_string()),
                ));
            }
        };
        if index == 0 {
            if let Ok(done) = call.result(Some(Duration::ZERO)) {
                let tool_calls = crate::diagnostics::chat_tool_call_summaries(&choice_response);
                crate::diagnostics::request_done(
                    &done,
                    request_log.matched_tokens(),
                    first_token_batch_tokens,
                    &tool_calls,
                );
            }
        }
        prompt_tokens = prompt_tokens.max(
            choice_response
                .pointer("/usage/prompt_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
        );
        completion_tokens = completion_tokens.saturating_add(
            choice_response
                .pointer("/usage/completion_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
        );
        if index == 0 {
            first_response = Some(choice_response.clone());
        }
        if let Some(choice) = choice_response.pointer("/choices/0") {
            let mut choice = choice.clone();
            choice["index"] = json!(index);
            choices.push(choice);
        }
    }

    let mut response = first_response.expect("choice_count is positive");
    response["choices"] = Value::Array(choices);
    response["usage"]["prompt_tokens"] = json!(prompt_tokens);
    response["usage"]["completion_tokens"] = json!(completion_tokens);
    response["usage"]["total_tokens"] = json!(prompt_tokens.saturating_add(completion_tokens));
    Ok(Response::json(200, &response))
}

/// The streamed answer: submit with the stream's callbacks writing SSE
/// onto the connection, and end with [DONE].
fn stream_answer(
    stream: &mut TcpStream,
    pipeline: &Pipeline,
    body: &Value,
    prepared: &crate::serve::pipeline::Prepared,
    max_output_tokens: u32,
    job: &openai::JobInfo,
    deadline: Deadline,
    sampling: crate::protocol::SamplingParameters,
    stop_sequences: Vec<String>,
    frame_options: crate::serve::pipeline::GenerationFrameOptions,
    cors_origin: Option<&str>,
    http_started_at: Option<Instant>,
    request_log: Arc<crate::diagnostics::RequestLogContext>,
) -> std::io::Result<Response> {
    // Compile before committing the HTTP status, so invalid schemas and
    // unavailable grammar support receive an ordinary JSON error.
    let constraint = match pipeline.constraint(prepared) {
        Ok(constraint) => constraint.map(Arc::new),
        Err(error) => {
            let code = error.code.unwrap_or_else(|| "server_error".to_string());
            return Ok(Response::error(error.status, &error.message, &code));
        }
    };
    let remaining = match deadline.remaining() {
        Ok(remaining) => remaining,
        Err(error) => return Ok(Response::json(error.status, &error.openai_payload())),
    };
    write_response_with_cors(stream, &Response::sse_head(), cors_origin)?;
    let writer = Arc::new(Mutex::new(SseWriter {
        stream: stream.try_clone()?,
    }));
    let sink = Arc::new(StreamSink {
        write: Arc::new(Mutex::new(Box::new(move |payload: &str| {
            writer.lock().expect("writer lock").write_event(payload)
        }))),
        model: pipeline.model.clone(),
        request_id: job.public_id.clone(),
        created_at: job.created_at,
        include_usage: body.pointer("/stream_options/include_usage").and_then(Value::as_bool).unwrap_or(false),
        compaction: prepared.compaction.clone(),
    });
    // The output constraint the request generates under, and the projector
    // that reads its calls as they arrive.
    let projector = Arc::new(std::sync::Mutex::new(match &prepared.tool_policy {
        Some(policy) => StreamingToolCallProjector::new(
            Box::new(crate::tool_schema::ToolParameterTypes::new(policy.clone())),
            1,
            &pipeline.chat_format,
            prepared.response_format.is_some(),
        ),
        None => StreamingToolCallProjector::without_policy(1, &pipeline.chat_format),
    }));
    let tokens = Arc::new(std::sync::Mutex::new(Vec::new()));
    let cancel_client = pipeline.client.clone();
    let cancel = Arc::new(move |call: &crate::engine::Call| {
        cancel_client.cancel(call);
    });
    let (on_event, on_complete) = stream_callbacks_with_request_log(
        sink,
        tokens,
        pipeline.tokenizer.clone(),
        projector,
        prepared.thinking,
        pipeline.chat_format.clone(),
        prepared.tool_policy.is_some(),
        constraint.clone(),
        prepared
            .response_format
            .as_ref()
            .map(|format| format.validator.clone()),
        stop_sequences,
        cancel,
        Some(request_log.clone()),
    );
    let generation = pipeline.generation_request_with_frame_options(
        prepared,
        max_output_tokens,
        constraint,
        remaining,
        sampling,
        frame_options,
    );
    let call = match pipeline
        .client
        .submit_with_http_start(&generation, Some(on_event), Some(on_complete), http_started_at)
    {
        Ok(call) => call,
        Err(error) => {
            crate::diagnostics::request_error("runtime_unavailable");
            return Err(std::io::Error::new(std::io::ErrorKind::Other, error.to_string()));
        }
    };
    let _ = call.result(None);
    Ok(Response {
        status: 200,
        content_type: "text/event-stream",
        body: Vec::new(),
    })
}

/// The SSE writer one stream holds.
struct SseWriter {
    stream: TcpStream,
}

impl SseWriter {
    fn write_event(&mut self, payload: &str) -> Result<(), String> {
        write_sse_event(&mut self.stream, payload).map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod status_cache_tests {
    use super::*;

    #[test]
    fn only_fresh_ready_engine_snapshots_are_cache_hits() {
        let now = Instant::now();
        let mut cache = StatusCache {
            snapshot: Some(json!({"ready": true, "metrics": {"decode": 1}})),
            captured_at: Some(now),
            unanswered_since: None,
        };
        assert_eq!(cache.fresh_snapshot(now).unwrap().1, Duration::ZERO);
        assert!(cache
            .fresh_snapshot(now + STATUS_CACHE_TTL - Duration::from_nanos(1))
            .is_some());
        assert!(cache.fresh_snapshot(now + STATUS_CACHE_TTL).is_none());

        cache.snapshot = Some(json!({"ready": false}));
        cache.captured_at = Some(now);
        assert!(cache.fresh_snapshot(now).is_none());
    }

    #[test]
    fn metrics_scrapes_keep_python_authentication_policy() {
        assert!(requires_api_key("GET", "/metrics"));
        assert!(requires_api_key("GET", "/status"));
        assert!(!requires_api_key("GET", "/health"));
        assert!(requires_api_key("POST", "/anything"));
    }
}
