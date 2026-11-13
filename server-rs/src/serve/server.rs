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
        ("GET", "/status") => Ok(Response::json(
            200,
            &json!({
                "ready": server.pipeline.client.ready(),
                "model": server.pipeline.model,
                "maximum_context_tokens": server.pipeline.client.readiness().map(|ready| ready.max_context_tokens),
                "frontend": server.preparation().map(Preparation::status),
            }),
        )),
        ("GET", "/v1/models") => Ok(Response::json(
            200,
            &json!({"object": "list", "data": [{
                "id": server.pipeline.model, "object": "model", "owned_by": "splash",
                "context_length": server.pipeline.client.readiness().map(|ready| ready.max_context_tokens),
                "input_modalities": ["text"],
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
    let prepared = server.pipeline.prepare(body);
    deadline.remaining()?;
    prepared.map_err(|error| {
        ApiError::new(error.status, error.message, "invalid_request_error")
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
        Ok((call, tokens)) => {
            let _ = call.result(None);
            match pipeline.complete(body, prepared, &call, &tokens, &pipeline.model, job) {
                Ok(response) => Ok(Response::json(200, &response)),
                Err(error) => Ok(Response::error(error.status, &error.message, "server_error")),
            }
        }
        Err(error) => Ok(Response::error(error.status, &error.message, "server_error")),
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
    let tokens = Arc::new(std::sync::Mutex::new(Vec::new()));
    let (on_event, on_complete) = stream_callbacks(
        sink,
        tokens.clone(),
        pipeline.tokenizer.clone(),
        Arc::new(std::sync::Mutex::new(
            StreamingToolCallProjector::without_policy(1, &pipeline.chat_format),
        )),
    );
    let request = crate::protocol::RequestFrame {
        logical_max_output_tokens: max_output_tokens,
        prompt_tokens: prepared.prompt_tokens.clone(),
        ..crate::protocol::RequestFrame::default()
    };
    let generation = crate::engine::GenerationRequest {
        frame: request,
        deadline: None,
        mask_provider: None,
    };
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
