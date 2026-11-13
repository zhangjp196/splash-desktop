//! The serving process's network entry: a listener that answers Chat
//! completions over real TCP, complete or streamed, through the
//! pipeline.

use std::net::{TcpListener, TcpStream};
use std::sync::Arc;

use serde_json::{json, Value};

use crate::dialects::openai;
use crate::http::{read_request, write_response, write_sse_event, Response};
use crate::output::StreamingToolCallProjector;
use crate::serve::stream::{stream_callbacks, StreamSink};
use crate::serve::Pipeline;

/// The server: the pipeline it serves, listening on one address.
pub struct Server {
    pipeline: Arc<Pipeline>,
}

impl Server {
    pub fn new(pipeline: Pipeline) -> Server {
        Server {
            pipeline: Arc::new(pipeline),
        }
    }

    /// Serve on the listener the caller bound, until it fails. One thread
    /// per connection, as the Python's frontend runs.
    pub fn serve(self, listener: TcpListener) {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let pipeline = self.pipeline.clone();
            std::thread::spawn(move || {
                let _ = serve_connection(&mut stream, &pipeline);
            });
        }
    }
}

/// One connection: read a request, route it, answer.
fn serve_connection(stream: &mut TcpStream, pipeline: &Pipeline) -> std::io::Result<()> {
    let Some(request) = read_request(stream)? else {
        return Ok(());
    };
    let response = match (request.method.as_str(), request.path.as_str()) {
        ("POST", "/v1/chat/completions") => chat_completions(&request, stream, pipeline),
        ("GET", "/status") => Ok(Response::json(
            200,
            &json!({"ready": pipeline.client.ready(), "model": pipeline.model}),
        )),
        _ => Ok(Response::error(404, "no such path", "not_found")),
    };
    match response {
        Ok(response) => write_response(stream, &response),
        Err(_) => write_response(
            stream,
            &Response::error(500, "the request failed", "internal_error"),
        ),
    }
}

/// A Chat completion: complete or streamed, as the body's `stream` says.
fn chat_completions(request: &crate::http::Request, stream: &mut TcpStream, pipeline: &Pipeline) -> std::io::Result<Response> {
    let body = match request.json() {
        Ok(body) => body,
        Err(error) => {
            return Ok(Response::error(400, &error, "invalid_request_error"));
        }
    };
    let streamed = body.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);
    let prepared = match pipeline.prepare(&body) {
        Ok(prepared) => prepared,
        Err(error) => {
            return Ok(Response::error(error.status, &error.message, "invalid_request_error"));
        }
    };
    let max_output_tokens = body
        .get("max_tokens")
        .and_then(|t| t.as_u64())
        .unwrap_or(1024) as u32;
    let job = openai::JobInfo {
        public_id: format!("req{}", pipeline.client.pending_count() + 1),
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
    _body: &Value,
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
        include_usage: true,
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

use std::sync::Mutex;
