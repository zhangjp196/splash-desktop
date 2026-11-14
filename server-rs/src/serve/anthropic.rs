//! Complete, non-streaming Anthropic Messages exchange over the shared
//! request conversion and generation pipeline.

use std::io;
use std::net::TcpStream;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::dialects::{anthropic as response, anthropic_request};
use crate::dialects::openai::{self, CollectedResult, JobInfo};
use crate::engine::{CompletionCallback, EventCallback};
use crate::http::{write_response_with_cors, Response};
use crate::output::{
    CallbackStreamer, ProjectorEvent, ReasoningPart, ReasoningSplitter,
    StreamingToolCallProjector,
};
use crate::protocol::EngineEvent;
use crate::serve::pipeline::{
    generation_frame_options, output_budget, sampling_parameters, stop_sequences,
    Pipeline, PipelineError, Prepared,
};
use crate::support::errors::ApiError;
use crate::tool_schema::ToolParameterTypes;

/// Convert an Anthropic Messages request, generate one completion, and
/// assemble its complete Anthropic response.
///
/// `thinking_resolver` validates/restores opaque thinking signatures from
/// prior messages. The codec signs new thinking blocks in this response.
/// `deadline` is the remaining request
/// budget at entry; `None` means no request deadline.
pub fn complete_messages(
    pipeline: &Pipeline,
    body: &Value,
    public_id: &str,
    deadline: Option<Duration>,
    thinking_resolver: &dyn Fn(&str) -> Result<String, crate::chat_messages::MessageError>,
    thinking_codec: &crate::thinking::ThinkingCodec,
) -> Result<Value, PipelineError> {
    let (chat_body, thinking_display) = prepare_messages(body, thinking_resolver)?;

    let started = Instant::now();
    let expires_at = deadline.and_then(|duration| started.checked_add(duration));
    let mut check_deadline = || {
        if let Some(expires_at) = expires_at {
            let remaining = expires_at.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(crate::support::errors::ApiError::new(
                    504,
                    "request deadline exceeded",
                    "request_timeout",
                ));
            }
            Ok(Some(remaining))
        } else {
            Ok(None)
        }
    };

    let prepared = pipeline
        .prepare_with_compaction(&chat_body, &mut check_deadline, None, None, None, true)?;
    let remaining = check_deadline()
        .map_err(|error| PipelineError {
            status: error.status,
            message: error.message,
            code: Some(error.code),
        })?
        .map(|duration| duration.max(Duration::from_millis(1)));
    complete_prepared_messages(
        pipeline,
        &chat_body,
        &prepared,
        public_id,
        remaining,
        &thinking_display,
        thinking_codec,
        None,
        None,
        None,
    )
}

/// Convert the request into the Chat body and Anthropic thinking display
/// mode, without preparing or submitting it. Server routes can use this
/// before calling their shared preparation/deadline machinery.
pub fn prepare_messages(
    body: &Value,
    thinking_resolver: &dyn Fn(&str) -> Result<String, crate::chat_messages::MessageError>,
) -> Result<(Value, String), PipelineError> {
    anthropic_request::anthropic_to_chat_body(body, thinking_resolver).map_err(PipelineError::from)
}

/// Generate and assemble a non-streaming Messages response from a request
/// already prepared by the server's shared preparation path.
pub fn complete_prepared_messages(
    pipeline: &Pipeline,
    chat_body: &Value,
    prepared: &crate::serve::pipeline::Prepared,
    public_id: &str,
    deadline: Option<Duration>,
    thinking_display: &str,
    thinking_codec: &crate::thinking::ThinkingCodec,
    disconnect_stream: Option<&std::net::TcpStream>,
    http_started_at: Option<std::time::Instant>,
    request_log: Option<Arc<crate::diagnostics::RequestLogContext>>,
) -> Result<Value, PipelineError> {
    let max_context = pipeline
        .client
        .readiness()
        .map(|ready| ready.max_context_tokens)
        .unwrap_or(u32::MAX);
    let budget = output_budget(&chat_body, prepared.prompt_tokens.len(), max_context)?;
    let sampling = sampling_parameters(&chat_body)?;
    let frame_options =
        generation_frame_options(&chat_body, &prepared, &pipeline.chat_format, None)?;
    let stops = stop_sequences(&chat_body)?;
    let (call, tokens, first_batch) = match pipeline.submit_with_frame_options_and_http_start_and_diagnostics(
        prepared,
        budget,
        deadline,
        sampling,
        frame_options,
        http_started_at,
        request_log.clone(),
    ) {
        Ok(submission) => submission,
        Err(error) => {
            if request_log.is_some() {
                crate::diagnostics::request_error(error.code.as_deref().unwrap_or("server_error"));
            }
            return Err(error);
        }
    };
    let _disconnect_watch = match disconnect_stream {
        Some(stream) => Some(
            crate::serve::server::DisconnectWatch::start(
                stream,
                pipeline.client.clone(),
                vec![call.clone()],
            )
            .map_err(|error| {
                pipeline.client.cancel(&call);
                PipelineError {
                    status: 500,
                    message: format!("could not monitor request disconnect: {error}"),
                    code: Some("internal_error".into()),
                }
            })?,
        ),
        None => None,
    };
    let first_batch = first_batch.load(std::sync::atomic::Ordering::Relaxed);
    let job = JobInfo {
        public_id: public_id.to_owned(),
        created_at: 0,
        reasoning_tokens: 0,
    };
    let chat_response = match pipeline.complete_with_stops(
        &chat_body,
        prepared,
        &call,
        &tokens,
        first_batch,
        &pipeline.model,
        &job,
        &stops,
    ) {
        Ok(response) => response,
        Err(error) => {
            if request_log.is_some() {
                crate::diagnostics::request_error(error.code.as_deref().unwrap_or("server_error"));
            }
            return Err(error);
        }
    };

    let done = call.result(None)?;
    if let Some(request) = request_log.as_ref() {
        let tool_calls = crate::diagnostics::chat_tool_call_summaries(&chat_response);
        crate::diagnostics::request_done(
            &done,
            request.matched_tokens(),
            first_batch,
            &tool_calls,
        );
    }
    let mut result = result_from_done(&done, first_batch);
    let stop_reason = chat_response
        .pointer("/choices/0/finish_reason")
        .and_then(Value::as_str)
        .unwrap_or("stop");
    if stop_reason == "stop" {
        result.reason = "stop".into();
    }
    let reasoning_tokens = chat_response
        .pointer("/usage/completion_tokens_details/reasoning_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let message = chat_response
        .pointer("/choices/0/message")
        .unwrap_or(&Value::Null);
    let mut blocks = Vec::new();
    let mut thinking_to_sign = Vec::new();
    if let Some(reasoning) = message.get("reasoning_content").and_then(Value::as_str) {
        if !reasoning.is_empty() {
            thinking_to_sign.push(reasoning.to_owned());
            blocks.push(response::OutputBlock {
                kind: "reasoning".into(),
                text: if thinking_display == "omitted" {
                    String::new()
                } else {
                    reasoning.to_owned()
                },
                call_id: None,
                name: None,
                in_progress: false,
            });
        }
    }
    if let Some(text) = message.get("content").and_then(Value::as_str) {
        if !text.is_empty() {
            blocks.push(response::OutputBlock {
                kind: "text".into(),
                text: text.to_owned(),
                call_id: None,
                name: None,
                in_progress: false,
            });
        }
    }
    if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
        for call in calls {
            let function = call.get("function").unwrap_or(&Value::Null);
            blocks.push(response::OutputBlock {
                kind: "tool_use".into(),
                text: function
                    .get("arguments")
                    .and_then(Value::as_str)
                    .unwrap_or("{}")
                    .to_owned(),
                call_id: call.get("id").and_then(Value::as_str).map(str::to_owned),
                name: function
                    .get("name")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                in_progress: false,
            });
        }
    }

    // Recover the exact sequence from the untrimmed generated tokens. The
    // assembled Chat response has already removed the matching stop text.
    let generated = tokens
        .lock()
        .expect("token lock")
        .clone();
    let raw_output = pipeline.tokenizer.decode(&generated, true).unwrap_or_default();
    let stop_sequence = if stop_reason == "stop" {
        stops
            .iter()
            .filter_map(|stop| raw_output.find(stop).map(|index| (index, stop.as_str())))
            .min_by_key(|(index, _)| *index)
            .map(|(_, stop)| stop)
    } else {
        None
    };
    let requested = chat_body
        .get("max_completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(u64::from(budget));
    let leaves = max_context.saturating_sub(done.prompt_tokens);
    let context_clamped = requested > u64::from(leaves);
    let mut response = response::anthropic_response(
        &pipeline.model,
        public_id,
        &blocks,
        &result,
        !blocks.iter().all(|block| block.kind != "tool_use"),
        stop_sequence,
        "",
        reasoning_tokens,
        context_clamped,
    );
    if let Some(content) = response.get_mut("content").and_then(Value::as_array_mut) {
        let mut signing_texts = thinking_to_sign.into_iter();
        for block in content {
            if block.get("type").and_then(Value::as_str) == Some("thinking") {
                let thinking = signing_texts.next().unwrap_or_default();
                let signature = thinking_codec.encode(&thinking).map_err(|error| PipelineError {
                    status: error.status,
                    message: error.message,
                    code: Some(error.code),
                })?;
                block["signature"] = json!(signature);
            }
        }
    }
    Ok(response)
}

/// Start a real token-by-token Anthropic Messages stream from an already
/// prepared request. Errors detected before the response headers are
/// committed are returned as `PipelineError`; after that point they are
/// emitted as an Anthropic `error` SSE event.
#[allow(clippy::too_many_arguments)]
pub fn stream_prepared_messages(
    stream: &mut TcpStream,
    cors_origin: Option<&str>,
    pipeline: &Pipeline,
    chat_body: &Value,
    prepared: &Prepared,
    public_id: &str,
    deadline: Option<Duration>,
    thinking_display: &str,
    thinking_codec: Arc<crate::thinking::ThinkingCodec>,
    default_thinking_token_budget: Option<u32>,
    http_started_at: Option<std::time::Instant>,
    request_log: Option<Arc<crate::diagnostics::RequestLogContext>>,
) -> io::Result<Result<Response, PipelineError>> {
    if prepared.thinking && !crate::thinking::ThinkingCodec::is_supported() {
        let error = crate::thinking::crypto_unavailable();
        return Ok(Err(PipelineError {
            status: error.status,
            message: error.message,
            code: Some(error.code),
        }));
    }
    let max_context = pipeline
        .client
        .readiness()
        .map(|ready| ready.max_context_tokens)
        .unwrap_or(u32::MAX);
    let budget = match output_budget(chat_body, prepared.prompt_tokens.len(), max_context) {
        Ok(budget) => budget,
        Err(error) => return Ok(Err(error)),
    };
    let sampling = match sampling_parameters(chat_body) {
        Ok(sampling) => sampling,
        Err(error) => return Ok(Err(error)),
    };
    let options = match generation_frame_options(
        chat_body,
        prepared,
        &pipeline.chat_format,
        default_thinking_token_budget,
    ) {
        Ok(options) => options,
        Err(error) => return Ok(Err(error)),
    };
    let stops = match stop_sequences(chat_body) {
        Ok(stops) => stops,
        Err(error) => return Ok(Err(error)),
    };
    let constraint = match pipeline.constraint(prepared) {
        Ok(constraint) => constraint.map(Arc::new),
        Err(error) => return Ok(Err(error)),
    };
    let generation = pipeline.generation_request_with_frame_options(
        prepared,
        budget,
        constraint.clone(),
        deadline,
        sampling,
        options,
    );
    let requested_output_budget = chat_body
        .get("max_completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(u64::from(budget));

    let writer_stream = stream.try_clone()?;
    write_response_with_cors(stream, &Response::sse_head(), cors_origin)?;
    let writer = Arc::new(Mutex::new(writer_stream));
    let state = Arc::new(Mutex::new(AnthropicStreamState::new(
        writer.clone(),
        pipeline,
        prepared,
        public_id,
        thinking_display,
        thinking_codec,
        requested_output_budget,
        &stops,
        request_log,
    )));
    let cancel_client = pipeline.client.clone();
    let on_event: EventCallback = {
        let state = state.clone();
        let cancel_client = cancel_client.clone();
        let constraint = constraint.clone();
        let tokenizer = pipeline.tokenizer.clone();
        Arc::new(move |call, event| {
            let mut stop = false;
            let outcome = {
                let mut state = state.lock().expect("Anthropic stream state lock");
                match event {
                    EngineEvent::Start(start) if start.request_id == call.request_id => {
                        state.message_start(start)
                    }
                    EngineEvent::PromptProgress(progress)
                        if progress.request_id == call.request_id =>
                    {
                        state.ping(progress.processed_tokens)
                    }
                    EngineEvent::Tokens(tokens)
                        if tokens.request_id == call.request_id =>
                    {
                        if let Some(constraint) = constraint.as_ref() {
                            if let Err(error) = constraint.commit(&tokens.tokens) {
                                call.record_callback_error(format!("constraint: {error}"));
                                return;
                            }
                        }
                        if state.first_token_batch == 0 && !tokens.tokens.is_empty() {
                            state.first_token_batch = tokens.tokens.len() as u64;
                        }
                        let streamed = state.streamer.put_tokens(&tokenizer, &tokens.tokens);
                        stop = state.streamer.stop_sequence().is_some();
                        state.stopped = stop;
                        state.publish_tokens(streamed)
                    }
                    _ => Ok(()),
                }
            };
            if stop {
                cancel_client.cancel(call);
            }
            if let Err(error) = outcome {
                call.record_callback_error(format!("Anthropic SSE write failed: {error}"));
                cancel_client.cancel(call);
            }
        })
    };
    let on_complete: CompletionCallback = {
        let state = state.clone();
        let completion_writer = writer.clone();
        let completion_tokenizer = pipeline.tokenizer.clone();
        Arc::new(move |call| {
            let done = match call.result(None) {
                Ok(done) => done,
                Err(error) => {
                    let error = PipelineError::from(error);
                    crate::diagnostics::request_error(error.code.as_deref().unwrap_or("api_error"));
                    let payload = ApiError::new(
                        error.status,
                        error.message,
                        error.code.unwrap_or_else(|| "api_error".into()),
                    )
                    .anthropic_payload();
                    let _ = write_anthropic_event(&completion_writer, "error", &payload);
                    return;
                }
            };
            let mut state = state.lock().expect("Anthropic stream state lock");
            if let Err(error) = state.finish(&done, &completion_tokenizer, max_context) {
                let payload = ApiError::new(500, error, "internal_error").anthropic_payload();
                let _ = write_anthropic_event(&completion_writer, "error", &payload);
            }
        })
    };

    let call = match pipeline
        .client
        .submit_with_http_start(&generation, Some(on_event), Some(on_complete), http_started_at)
    {
        Ok(call) => call,
        Err(error) => {
            let error = PipelineError::from(error);
            crate::diagnostics::request_error(error.code.as_deref().unwrap_or("api_error"));
            let payload = ApiError::new(
                error.status,
                error.message,
                error.code.unwrap_or_else(|| "api_error".into()),
            )
            .anthropic_payload();
            write_anthropic_event(&writer, "error", &payload)
                .map_err(|error| io::Error::new(io::ErrorKind::BrokenPipe, error))?;
            return Ok(Ok(Response {
                status: 200,
                content_type: "text/event-stream",
                body: Vec::new(),
            }));
        }
    };
    let _disconnect_watch = match crate::serve::server::DisconnectWatch::start(
        stream,
        pipeline.client.clone(),
        vec![call.clone()],
    ) {
        Ok(watch) => watch,
        Err(error) => {
            pipeline.client.cancel(&call);
            let payload = ApiError::new(
                500,
                format!("could not monitor request disconnect: {error}"),
                "internal_error",
            )
            .anthropic_payload();
            write_anthropic_event(&writer, "error", &payload)
                .map_err(|error| io::Error::new(io::ErrorKind::BrokenPipe, error))?;
            return Ok(Ok(Response {
                status: 200,
                content_type: "text/event-stream",
                body: Vec::new(),
            }));
        }
    };
    let _ = call.result(None);
    Ok(Ok(Response {
        status: 200,
        content_type: "text/event-stream",
        body: Vec::new(),
    }))
}

fn write_anthropic_event(
    writer: &Arc<Mutex<TcpStream>>,
    event: &str,
    payload: &Value,
) -> Result<(), String> {
    let data = serde_json::to_string(payload).map_err(|error| error.to_string())?;
    let mut writer = writer.lock().expect("Anthropic SSE writer lock");
    use std::io::Write;
    write!(writer, "event: {event}\ndata: {data}\n\n").map_err(|error| error.to_string())?;
    writer.flush().map_err(|error| error.to_string())
}

struct AnthropicStreamState {
    writer: Arc<Mutex<TcpStream>>,
    model: String,
    message_id: String,
    prompt_tokens: u64,
    compaction: Option<Value>,
    thinking_display: String,
    thinking_codec: Arc<crate::thinking::ThinkingCodec>,
    requested_output_budget: u64,
    streamer: CallbackStreamer,
    splitter: ReasoningSplitter,
    projector: StreamingToolCallProjector,
    active: Option<(String, usize, Option<usize>)>,
    next_block: usize,
    has_non_thinking: bool,
    reasoning_text: String,
    active_thinking: String,
    first_token_batch: u64,
    stopped: bool,
    started: bool,
    request_log: Option<Arc<crate::diagnostics::RequestLogContext>>,
}

impl AnthropicStreamState {
    fn new(
        writer: Arc<Mutex<TcpStream>>,
        pipeline: &Pipeline,
        prepared: &Prepared,
        public_id: &str,
        thinking_display: &str,
        thinking_codec: Arc<crate::thinking::ThinkingCodec>,
        requested_output_budget: u64,
        stops: &[String],
        request_log: Option<Arc<crate::diagnostics::RequestLogContext>>,
    ) -> Self {
        let stop_refs: Vec<&str> = stops.iter().map(String::as_str).collect();
        let projector = match &prepared.tool_policy {
            Some(policy) => StreamingToolCallProjector::new(
                Box::new(ToolParameterTypes::new(policy.clone())),
                1,
                &pipeline.chat_format,
                prepared.response_format.is_some(),
            ),
            None => StreamingToolCallProjector::without_policy(1, &pipeline.chat_format),
        };
        Self {
            writer,
            model: pipeline.model.clone(),
            message_id: format!("msg_{public_id}"),
            prompt_tokens: prepared.prompt_tokens.len() as u64,
            compaction: prepared.compaction.clone(),
            thinking_display: thinking_display.to_owned(),
            thinking_codec,
            requested_output_budget,
            streamer: CallbackStreamer::new(&stop_refs),
            splitter: ReasoningSplitter::new(
                prepared.thinking,
                Some(&pipeline.chat_format),
                prepared.tool_policy.is_some(),
            ),
            projector,
            active: None,
            next_block: 0,
            has_non_thinking: false,
            reasoning_text: String::new(),
            active_thinking: String::new(),
            first_token_batch: 0,
            stopped: false,
            started: false,
            request_log,
        }
    }

    fn send(&self, event: &str, payload: Value) -> Result<(), String> {
        write_anthropic_event(&self.writer, event, &payload)
    }

    fn message_start(&mut self, start: &crate::protocol::StartEvent) -> Result<(), String> {
        if self.started {
            return Ok(());
        }
        self.started = true;
        let matched = u64::from(start.matched_prompt_tokens).min(self.prompt_tokens);
        if let Some(request) = self.request_log.as_ref() {
            request.admitted(start.matched_prompt_tokens);
        }
        self.send(
            "message_start",
            json!({
                "type": "message_start",
                "message": {
                    "id": self.message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.model,
                    "content": [],
                    "stop_reason": null,
                    "stop_sequence": null,
                    "usage": {
                        "input_tokens": self.prompt_tokens.saturating_sub(matched),
                        "cache_read_input_tokens": matched,
                        "output_tokens": 0,
                    },
                }
            }),
        )
    }

    fn ping(&self, processed_tokens: u32) -> Result<(), String> {
        self.send(
            "ping",
            json!({"type": "ping", "prompt_progress": {"processed_tokens": processed_tokens}}),
        )
    }

    fn open_block(&mut self, kind: &str, block: Value, tool_index: Option<usize>) -> Result<usize, String> {
        self.close_block()?;
        let index = self.next_block;
        self.next_block += 1;
        self.active = Some((kind.to_owned(), index, tool_index));
        if kind != "thinking" {
            self.has_non_thinking = true;
        }
        self.send(
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": index,
                "content_block": block,
            }),
        )?;
        Ok(index)
    }

    fn close_block(&mut self) -> Result<(), String> {
        if let Some((kind, index, _)) = self.active.take() {
            if kind == "thinking" {
                let signature = self
                    .thinking_codec
                    .encode(&self.active_thinking)
                    .map_err(|error| error.message)?;
                self.send(
                    "content_block_delta",
                    json!({
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "signature_delta", "signature": signature},
                    }),
                )?;
                self.active_thinking.clear();
            }
            self.send(
                "content_block_stop",
                json!({"type": "content_block_stop", "index": index}),
            )?;
        }
        Ok(())
    }

    fn text_delta(&mut self, kind: &str, text: &str) -> Result<(), String> {
        if text.is_empty() {
            return Ok(());
        }
        if self.active.as_ref().map(|active| active.0.as_str()) != Some(kind) {
            let block = match kind {
                "thinking" => json!({
                    "type": "thinking",
                    "thinking": "",
                }),
                _ => json!({"type": "text", "text": ""}),
            };
            self.open_block(kind, block, None)?;
        }
        let (delta_type, field) = if kind == "thinking" {
            self.reasoning_text.push_str(text);
            self.active_thinking.push_str(text);
            if self.thinking_display == "omitted" {
                return Ok(());
            }
            ("thinking_delta", "thinking")
        } else {
            ("text_delta", "text")
        };
        let index = self.active.as_ref().expect("block opened").1;
        self.send(
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": delta_type, field: text},
            }),
        )
    }

    fn publish_parts(&mut self, parts: Vec<ReasoningPart>) -> Result<(), String> {
        for part in parts {
            match part {
                ReasoningPart::Reasoning(text) => self.text_delta("thinking", &text)?,
                ReasoningPart::Content(text) => {
                    for event in self.projector.put(&text) {
                        self.publish_projector_event(event)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn publish_projector_event(&mut self, event: ProjectorEvent) -> Result<(), String> {
        match event {
            ProjectorEvent::Content(text) => self.text_delta("text", &text),
            ProjectorEvent::ToolCall { index, id, name } => {
                self.open_block(
                    "tool_use",
                    json!({"type": "tool_use", "id": id, "name": name, "input": {}}),
                    Some(index),
                )?;
                Ok(())
            }
            ProjectorEvent::ToolDelta { index, chunk } => {
                let active_tool = self
                    .active
                    .as_ref()
                    .and_then(|active| if active.0 == "tool_use" { active.2 } else { None });
                if active_tool != Some(index) {
                    return Err("tool argument delta arrived without its tool block".into());
                }
                let block_index = self.active.as_ref().expect("tool block active").1;
                self.send(
                    "content_block_delta",
                    json!({
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "input_json_delta", "partial_json": chunk},
                    }),
                )
            }
        }
    }

    fn publish_tokens(&mut self, streamed: Vec<crate::output::streamer::Streamed>) -> Result<(), String> {
        for item in streamed {
            if let crate::output::streamer::Streamed::Text(text) = item {
                let parts = self.splitter.put(&text);
                self.publish_parts(parts)?;
            }
        }
        Ok(())
    }

    fn finish(
        &mut self,
        done: &crate::protocol::DoneEvent,
        tokenizer: &crate::tokenization::FastTokenizer,
        max_context: u32,
    ) -> Result<(), String> {
        let tail = self.streamer.end(tokenizer);
        self.stopped = self.streamer.stop_sequence().is_some();
        self.publish_tokens(tail)?;
        let tail_parts = self.splitter.finish();
        self.publish_parts(tail_parts)?;
        let incomplete = done.reason == crate::protocol::FinishReason::Length && !self.stopped;
        let (_content, calls, events) = self.projector.finish(incomplete);
        if let Some(request) = self.request_log.as_ref() {
            let tool_calls: Vec<(String, String)> = calls
                .iter()
                .map(|call| (call.name.clone(), call.arguments.clone()))
                .collect();
            crate::diagnostics::request_done(
                done,
                request.matched_tokens(),
                self.first_token_batch,
                &tool_calls,
            );
        }
        for event in events {
            self.publish_projector_event(event)?;
        }
        self.close_block()?;
        if self.next_block == 0 || !self.has_non_thinking {
            self.open_block("text", json!({"type": "text", "text": ""}), None)?;
            self.close_block()?;
        }
        let reasoning_tokens = tokenizer
            .encode(&self.reasoning_text, false)
            .map(|tokens| tokens.len() as u64)
            .unwrap_or(0);
        let mut result = result_from_done(done, self.first_token_batch);
        result.reason = if self.stopped {
            "stop".into()
        } else {
            format!("{:?}", done.reason).to_lowercase()
        };
        result.bounded = done.bounded;
        result.metrics = openai::metrics_dict(&result);
        let leaves = max_context.saturating_sub(done.prompt_tokens);
        let requested = self.requested_output_budget;
        let clamped = requested > u64::from(leaves);
        let stop_reason = response::anthropic_stop(
            &result,
            !calls.is_empty(),
            self.streamer.stop_sequence(),
            clamped,
        );
        let mut message_delta = json!({
            "type": "message_delta",
            "delta": {
                "stop_reason": stop_reason,
                "stop_sequence": self.streamer.stop_sequence(),
            },
            "usage": {
                "output_tokens": result.completion_tokens,
                "output_tokens_details": {"thinking_tokens": reasoning_tokens},
            },
            "timings": openai::timings_dict(&result),
        });
        if let Some(compaction) = &self.compaction {
            message_delta["compaction"] = compaction.clone();
        }
        self.send("message_delta", message_delta)?;
        self.send("message_stop", json!({"type": "message_stop"}))
    }
}

/// One Anthropic Messages SSE event, including its event name and complete
/// JSON data payload. The payload always carries a matching `type` field,
/// as Anthropic clients expect.
#[derive(Debug, Clone, PartialEq)]
pub struct AnthropicSseEvent {
    pub event: &'static str,
    pub payload: Value,
}

impl AnthropicSseEvent {
    /// Serialize the event data for a writer that emits `data:` lines.
    pub fn json_payload(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(&self.payload)
    }

    /// Serialize a complete SSE frame, including its `event:` field.
    pub fn sse_frame(&self) -> Result<String, serde_json::Error> {
        Ok(format!(
            "event: {}\ndata: {}\n\n",
            self.event,
            self.json_payload()?
        ))
    }
}

/// Adapt a completed Messages response into its deterministic SSE event
/// sequence. This allows the server's stream route to reuse a collected
/// completion while preserving its content, stop metadata, usage, timings,
/// and optional compaction report.
pub fn messages_sse_events(completion: &Value) -> Vec<AnthropicSseEvent> {
    let mut events = Vec::new();
    let content = completion
        .get("content")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let usage = completion.get("usage").unwrap_or(&Value::Null);
    let mut start_usage = json!({
        "input_tokens": usage.get("input_tokens").and_then(Value::as_u64).unwrap_or(0),
        "cache_read_input_tokens": usage
            .get("cache_read_input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        "output_tokens": 0,
    });
    if let Some(cache_creation) = usage.get("cache_creation_input_tokens") {
        start_usage["cache_creation_input_tokens"] = cache_creation.clone();
    }
    events.push(AnthropicSseEvent {
        event: "message_start",
        payload: json!({
            "type": "message_start",
            "message": {
                "id": completion.get("id").cloned().unwrap_or(Value::Null),
                "type": completion.get("type").cloned().unwrap_or(json!("message")),
                "role": completion.get("role").cloned().unwrap_or(json!("assistant")),
                "model": completion.get("model").cloned().unwrap_or(Value::Null),
                "content": [],
                "stop_reason": null,
                "stop_sequence": null,
                "usage": start_usage,
            }
        }),
    });

    for (index, block) in content.iter().enumerate() {
        let kind = block.get("type").and_then(Value::as_str).unwrap_or("text");
        let initial = match kind {
            "thinking" => json!({
                "type": "thinking",
                "thinking": "",
            }),
            "tool_use" => json!({
                "type": "tool_use",
                "id": block.get("id").cloned().unwrap_or(Value::Null),
                "name": block.get("name").cloned().unwrap_or(Value::Null),
                "input": {},
            }),
            _ => json!({"type": "text", "text": ""}),
        };
        events.push(AnthropicSseEvent {
            event: "content_block_start",
            payload: json!({
                "type": "content_block_start",
                "index": index,
                "content_block": initial,
            }),
        });

        let delta = match kind {
            "thinking" => {
                let thinking = block.get("thinking").and_then(Value::as_str).unwrap_or("");
                if !thinking.is_empty() {
                    events.push(AnthropicSseEvent {
                        event: "content_block_delta",
                        payload: json!({
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {"type": "thinking_delta", "thinking": thinking},
                        }),
                    });
                }
                block
                    .get("signature")
                    .and_then(Value::as_str)
                    .filter(|signature| !signature.is_empty())
                    .map(|signature| json!({"type": "signature_delta", "signature": signature}))
            }
            "tool_use" => block
                .get("input")
                .map(|input| {
                    json!({
                        "type": "input_json_delta",
                        "partial_json": input.to_string(),
                    })
                }),
            _ => block
                .get("text")
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
                .map(|text| json!({"type": "text_delta", "text": text})),
        };
        if let Some(delta) = delta {
            events.push(AnthropicSseEvent {
                event: "content_block_delta",
                payload: json!({"type": "content_block_delta", "index": index, "delta": delta}),
            });
        }
        events.push(AnthropicSseEvent {
            event: "content_block_stop",
            payload: json!({"type": "content_block_stop", "index": index}),
        });
    }

    let mut delta_payload = json!({
        "type": "message_delta",
        "delta": {
            "stop_reason": completion.get("stop_reason").cloned().unwrap_or(Value::Null),
            "stop_sequence": completion.get("stop_sequence").cloned().unwrap_or(Value::Null),
        },
        "usage": {
            "output_tokens": usage.get("output_tokens").cloned().unwrap_or(json!(0)),
        },
    });
    if let Some(details) = usage.get("output_tokens_details") {
        delta_payload["usage"]["output_tokens_details"] = details.clone();
    }
    for field in ["timings", "compaction"] {
        if let Some(value) = completion.get(field) {
            delta_payload[field] = value.clone();
        }
    }
    events.push(AnthropicSseEvent {
        event: "message_delta",
        payload: delta_payload,
    });
    events.push(AnthropicSseEvent {
        event: "message_stop",
        payload: json!({"type": "message_stop"}),
    });
    events
}

fn result_from_done(done: &crate::protocol::DoneEvent, first_batch: u64) -> CollectedResult {
    let mut result = CollectedResult {
        reason: format!("{:?}", done.reason).to_lowercase(),
        prompt_tokens: u64::from(done.prompt_tokens),
        completion_tokens: u64::from(done.completion_tokens),
        prefill_tokens: u64::from(done.prompt_tokens),
        cache_matched_tokens: u64::from(done.deduplicated_tokens),
        drafted_tokens: u64::from(done.drafted_tokens),
        accepted_draft_tokens: u64::from(done.accepted_draft_tokens),
        first_token_batch_tokens: first_batch,
        start_to_first_token_ms: done.prefill_micros as f64 / 1000.0,
        first_token_to_done_ms: done.decode_micros as f64 / 1000.0,
        request_wall_ms: done.wall_micros as f64 / 1000.0,
        stream_tokens_per_second: if done.decode_micros != 0
            && u64::from(done.completion_tokens) > first_batch
        {
            f64::from(done.completion_tokens.saturating_sub(first_batch as u32))
                * 1_000_000.0
                / done.decode_micros as f64
        } else {
            0.0
        },
        bounded: false,
        metrics: json!({}),
    };
    result.metrics = openai::metrics_dict(&result);
    result
}
