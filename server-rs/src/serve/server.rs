//! The serving process's network entry: a listener that answers Chat
//! completions over real TCP, complete or streamed, through the
//! pipeline.

use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::dialects::openai;
use crate::http::{authenticate, read_timed_request, validate_headers, write_response, write_sse_event, Response};
use crate::output::StreamingToolCallProjector;
use crate::serve::preparation::{Deadline, Preparation};
use crate::serve::stream::{stream_callbacks, StreamSink};
use crate::serve::Pipeline;
use crate::support::errors::ApiError;

/// The server: the pipeline it serves, listening on one address.
pub struct Server {
    pipeline: Pipeline,
    preparation: OnceLock<Preparation>,
    local_security: Option<Option<String>>,
    next_request_id: AtomicU64,
}

impl Server {
    pub fn new(pipeline: Pipeline) -> Server {
        Server {
            pipeline,
            preparation: OnceLock::new(),
            local_security: None,
            next_request_id: AtomicU64::new(1),
        }
    }

    /// Enforce local Host/Origin policy and optionally authenticate generation.
    pub fn with_api_key(mut self, key: Option<String>) -> Self {
        self.local_security = Some(key);
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
                    let server = server.clone();
                    std::thread::spawn(move || {
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
            let server = server.clone();
            std::thread::spawn(move || {
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
            Preparation::new(ready.max_concurrent_requests as usize)
        }))
    }
}

/// One connection: read a request, route it, answer.
fn serve_connection(stream: &mut TcpStream, server: &Server) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(15)))?;
    stream.set_write_timeout(Some(Duration::from_secs(15)))?;
    let Some((request, started_at)) = read_timed_request(stream)? else {
        return Ok(());
    };
    if let Some(key) = &server.local_security {
        let hosts = ["localhost".to_owned(), "127.0.0.1".to_owned()];
        let validated = validate_headers(&request.headers, &hosts, &[]).and_then(|_| {
            if request.method == "POST" {
                if let Some(key) = key {
                    authenticate(&request.headers, key)?;
                }
            }
            Ok(())
        });
        if let Err(error) = validated {
            let error = ApiError::new(error.status, error.message, error.code);
            return write_response(stream, &Response::json(error.status, &error.openai_payload()));
        }
    }
    let response = match (request.method.as_str(), request.path.as_str()) {
        ("POST", "/v1/chat/completions") => chat_completions(&request, started_at, stream, server),
        ("POST", "/v1/chat/completions/count_tokens") => {
            Ok(match request.json() {
                Ok(body) => match prepare_chat(server, &body, started_at) {
                    Ok(prepared) => Response::json(200, &json!({"input_tokens": prepared.prompt_tokens.len()})),
                    Err(error) => Response::json(error.status, &error.openai_payload()),
                },
                Err(error) => Response::error(400, &error, "invalid_request_error"),
            })
        }
        ("GET", "/status") => Ok(Response::json(200, &status_document(server))),
        ("POST", "/v1/thermal") => Ok(thermal(&request, server)),
        ("GET", "/v1/models") => Ok(Response::json(
            200,
            &json!({"object": "list", "data": [{
                "id": server.pipeline.model, "object": "model", "owned_by": "splash",
                "context_length": server.pipeline.client.readiness().map(|ready| ready.max_context_tokens),
                "input_modalities": if server.pipeline.client.readiness().map(|ready| ready.vision).unwrap_or(false) {
                    vec!["text", "image"]
                } else {
                    vec!["text"]
                },
            }]}),
        )),
        ("GET", path) => Ok(web_asset(path).unwrap_or_else(|| Response::error(404, "no such path", "not_found"))),
        _ => Ok(Response::error(404, "no such path", "not_found")),
    };
    match response {
        Ok(response) if response.content_type == "text/event-stream" => Ok(()),
        Ok(response) => write_response(stream, &response),
        Err(_) => write_response(
            stream,
            &Response::error(500, "the request failed", "internal_error"),
        ),
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
    let engine = server
        .pipeline
        .client
        .status(Duration::from_secs(2))
        .ok()
        .and_then(|event| serde_json::from_slice::<Value>(&event.json).ok())
        .filter(Value::is_object);
    let ready = server.pipeline.client.ready();
    let mut document = match engine {
        Some(document) => document,
        None => json!({"schema_version": 8}),
    };
    document["ready"] = json!(ready && document.get("ready").and_then(Value::as_bool).unwrap_or(true));
    document["model"] = json!(server.pipeline.model);
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
    document["transport"] = json!({
        "ready": ready,
        "recovering": false,
        "stopped": server.pipeline.client.fatal_error().is_some(),
        "pending": server.pipeline.client.pending_count(),
        "pending_limit": server
            .pipeline
            .client
            .readiness()
            .map(|ready| ready.max_concurrent_requests)
            .unwrap_or(0),
        "restarts": 0,
        "last_crash_trace": Value::Null,
        "status_stale": false,
        "status_age_ms": 0.0,
    });
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
    request: &crate::http::Request,
    started_at: Instant,
    stream: &mut TcpStream,
    server: &Server,
) -> std::io::Result<Response> {
    let pipeline = &server.pipeline;
    let body = match request.json() {
        Ok(body) => body,
        Err(error) => {
            return Ok(Response::error(400, &error, "invalid_request_error"));
        }
    };
    let streamed = body.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);
    let prepared = match prepare_chat(server, &body, started_at) {
        Ok(prepared) => prepared,
        Err(error) => return Ok(Response::json(error.status, &error.openai_payload())),
    };
    let max_output_tokens = body
        .get("max_tokens")
        .and_then(|t| t.as_u64())
        .unwrap_or(1024) as u32;
    let job = openai::JobInfo {
        public_id: format!("req{}", server.next_request_id.fetch_add(1, Ordering::Relaxed)),
        created_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|since| since.as_secs() as i64)
            .unwrap_or(0),
        reasoning_tokens: 0,
    };
    if streamed {
        stream_answer(stream, pipeline, &body, &prepared, max_output_tokens, &job)
    } else {
        complete_answer(pipeline, &body, &prepared, max_output_tokens, &job)
    }
}

fn prepare_chat(
    server: &Server,
    body: &Value,
    started_at: Instant,
) -> Result<crate::serve::pipeline::Prepared, ApiError> {
    let deadline = Deadline::from_body(body, started_at)?;
    deadline.remaining()?;
    let preparation = server
        .preparation()
        .ok_or_else(|| ApiError::new(503, "native process is not ready", "runtime_unavailable"))?;
    let _slot = preparation.acquire(deadline)?;
    // The deadline runs through preparation: an image decode an expired
    // request starts is a decode the answer never reads.
    let mut remaining = || deadline.remaining().map(|_| ()).map_err(|error| error);
    let prepared = server.pipeline.prepare(body, &mut remaining);
    deadline.remaining()?;
    prepared.map_err(|error| {
        ApiError::new(
            error.status,
            error.message,
            error.code.unwrap_or_else(|| "invalid_request_error".to_string()),
        )
    })
}

/// The complete answer: submit, collect, assemble, answer as JSON.
fn complete_answer(
    pipeline: &Pipeline,
    body: &Value,
    prepared: &crate::serve::pipeline::Prepared,
    max_output_tokens: u32,
    job: &openai::JobInfo,
) -> std::io::Result<Response> {
    match pipeline.submit(prepared, max_output_tokens) {
        Ok((call, tokens, first_batch)) => {
            let _ = call.result(None);
            let first_token_batch_tokens =
                first_batch.load(std::sync::atomic::Ordering::Relaxed);
            match pipeline.complete(
                body,
                prepared,
                &call,
                &tokens,
                first_token_batch_tokens,
                &pipeline.model,
                job,
            ) {
                Ok(response) => Ok(Response::json(200, &response)),
                Err(error) => Ok(Response::error(
                    error.status,
                    &error.message,
                    &error.code.unwrap_or_else(|| "server_error".to_string()),
                )),
            }
        }
        Err(error) => Ok(Response::error(
                    error.status,
                    &error.message,
                    &error.code.unwrap_or_else(|| "server_error".to_string()),
                )),
    }
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
    write_response(stream, &Response::sse_head())?;
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
    let (on_event, on_complete) = stream_callbacks(
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
    );
    let generation = pipeline.generation_request(prepared, max_output_tokens, constraint);
    let call = pipeline
        .client
        .submit(&generation, Some(on_event), Some(on_complete))
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::Other, error.to_string()))?;
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
