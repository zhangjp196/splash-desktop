//! OpenAI text completions over a raw token prompt.

use std::collections::HashMap;
use std::net::TcpStream;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Value};

use crate::constraints::{Grammar, TokenConstraint};
use crate::dialects::openai::{self, CollectedResult, JobInfo};
use crate::engine::{CompletionCallback, EventCallback};
use crate::http::Response;
use crate::output::streamer::Streamed;
use crate::output::CallbackStreamer;
use crate::protocol::{EngineEvent, SamplingParameters};
use crate::serve::pipeline::{
    generation_frame_options, sampling_parameters, stop_sequences, GenerationFrameOptions,
    Pipeline, PipelineError, Prepared,
};
use crate::support::errors::ApiError;

const DEFAULT_MAX_TOKENS: u64 = 16;

pub struct CompletionRequest {
    pub prepared: Prepared,
    pub max_tokens: u32,
    pub choices: usize,
    pub streamed: bool,
    pub include_usage: bool,
    pub sampling: SamplingParameters,
    pub frame_options: GenerationFrameOptions,
    pub stops: Vec<String>,
    grammar: Option<Grammar>,
}

/// Validate a text completion request and prepare its raw prompt tokens.
pub fn prepare_request(
    pipeline: &Pipeline,
    body: &Value,
    default_thinking_token_budget: Option<u32>,
) -> Result<CompletionRequest, PipelineError> {
    if let Some(model) = body.get("model") {
        let accepted = model.is_null()
            || model.as_str().is_some_and(|model| model.is_empty() || model == pipeline.model);
        if !accepted {
            return Err(err(404, format!("model {} not found", model), "model_not_found"));
        }
    }
    for field in ["suffix", "logprobs"] {
        if body.get(field).is_some() {
            return Err(err(400, format!("{field} is not supported"), "invalid_request_error"));
        }
    }
    if body.get("echo").is_some_and(|echo| echo != &Value::Bool(false)) {
        return Err(err(400, "echo is not supported", "invalid_request_error"));
    }
    if !is_integer_value(body.get("best_of").unwrap_or(&json!(1)), Some(1), Some(1)) {
        return Err(err(400, "best_of must be 1", "invalid_request_error"));
    }
    let default_choices = json!(1);
    let choices = body.get("n").unwrap_or(&default_choices);
    if !is_integer_value(choices, Some(1), Some(4)) {
        return Err(err(400, "n must be an integer from 1 to 4", "invalid_request_error"));
    }
    let choices = choices.as_u64().unwrap_or(1) as usize;
    let streamed = match body.get("stream") {
        None => false,
        Some(Value::Bool(value)) => *value,
        _ => return Err(err(400, "invalid streaming options", "invalid_request_error")),
    };
    if streamed && choices > 1 {
        return Err(err(400, "n > 1 is not supported with stream", "invalid_request_error"));
    }
    let include_usage = match body.get("stream_options") {
        None | Some(Value::Null) => false,
        Some(Value::Object(options)) => match options.get("include_usage") {
            None | Some(Value::Null) => false,
            Some(Value::Bool(value)) => *value,
            _ => return Err(err(400, "invalid streaming options", "invalid_request_error")),
        },
        _ => return Err(err(400, "invalid streaming options", "invalid_request_error")),
    };
    let return_progress = match body.get("return_progress") {
        None => false,
        Some(Value::Bool(value)) => *value,
        _ => return Err(err(400, "return_progress must be a boolean", "invalid_request_error")),
    };
    if return_progress && !streamed {
        return Err(err(
            400,
            "return_progress requires stream: true and must be a boolean",
            "invalid_request_error",
        ));
    }
    if body
        .get("thinking_token_budget")
        .is_some_and(|budget| !budget.is_null() && budget.as_i64() != Some(-1))
    {
        return Err(err(
            400,
            "thinking_token_budget is not supported for text completions",
            "invalid_request_error",
        ));
    }

    let grammar_source = body.get("grammar").filter(|value| !value.is_null());
    let regex_source = body.get("regex").filter(|value| !value.is_null());
    if grammar_source.is_some() && regex_source.is_some() {
        return Err(err(
            400,
            "grammar and regex cannot be combined",
            "invalid_request_error",
        ));
    }
    let grammar = if let Some(source) = grammar_source {
        let source = source.as_str().filter(|source| !source.trim().is_empty()).ok_or_else(|| {
            err(400, "grammar must be a nonempty string", "invalid_request_error")
        })?;
        Some(Grammar::Gbnf(source.to_owned()))
    } else if let Some(source) = regex_source {
        let source = source.as_str().filter(|source| !source.trim().is_empty()).ok_or_else(|| {
            err(400, "regex must be a nonempty string", "invalid_request_error")
        })?;
        Some(Grammar::Regex(source.to_owned()))
    } else {
        None
    };
    if grammar.is_some() && body.get("ignore_eos") == Some(&Value::Bool(true)) {
        return Err(err(
            400,
            "ignore_eos cannot be combined with grammar or regex",
            "invalid_request_error",
        ));
    }

    let prompt = body.get("prompt").ok_or_else(|| {
        err(400, "prompt must be one string or one array of token ids", "invalid_request_error")
    })?;
    let prompt_tokens = match prompt {
        Value::String(text) => pipeline
            .tokenizer
            .encode(text, true)
            .map_err(|_| err(400, "prompt could not be tokenized", "invalid_request_error"))?,
        Value::Array(ids) if ids.iter().all(|id| id.as_u64().is_some()) => {
            let mut tokens = Vec::with_capacity(ids.len());
            for id in ids {
                let id = id.as_u64().expect("validated token id");
                if id >= u64::from(pipeline.chat_format.vocabulary) {
                    return Err(err(
                        400,
                        "prompt token ids must be in the vocabulary",
                        "invalid_request_error",
                    ));
                }
                tokens.push(id as u32);
            }
            tokens
        }
        _ => {
            return Err(err(
                400,
                "prompt must be one string or one array of token ids",
                "invalid_request_error",
            ));
        }
    };
    if prompt_tokens.is_empty() {
        return Err(err(400, "prompt must not be empty", "invalid_request_error"));
    }
    let prepared = Prepared {
        rendered: String::new(),
        prompt_tokens,
        thinking: false,
        images: None,
        image_spans: Vec::new(),
        image_pixels: Vec::new(),
        tools: None,
        tool_policy: None,
        response_format: None,
        logprobs: Arc::new(Mutex::new(HashMap::new())),
        compaction: None,
    };

    let max_tokens = match body.get("max_tokens") {
        None | Some(Value::Null) => DEFAULT_MAX_TOKENS,
        Some(value) => value.as_u64().filter(|value| *value > 0).ok_or_else(|| {
            err(400, "max_tokens must be a positive integer", "invalid_request_error")
        })?,
    };
    let max_context = pipeline
        .client
        .readiness()
        .map(|ready| ready.max_context_tokens)
        .unwrap_or(u32::MAX);
    let remaining = usize::try_from(max_context)
        .unwrap_or(usize::MAX)
        .saturating_sub(prepared.prompt_tokens.len());
    if remaining == 0 {
        return Err(err(400, "prompt leaves no room for output", "context_length_exceeded"));
    }
    let max_tokens = max_tokens.min(remaining as u64).min(u64::from(u32::MAX)) as u32;
    let sampling = sampling_parameters(body)?;
    let stops = stop_sequences(body)?;
    let mut generation_body = body.clone();
    if let Some(object) = generation_body.as_object_mut() {
        // Text completions reject `logprobs` and do not read `top_logprobs`.
        object.remove("top_logprobs");
    }
    let mut frame_options = generation_frame_options(
        &generation_body,
        &prepared,
        &pipeline.chat_format,
        default_thinking_token_budget,
    )?;
    if return_progress {
        frame_options.flags |= crate::protocol::request_flags::RETURN_PROGRESS;
    }
    if let Some(grammar) = &grammar {
        if pipeline.constraints.is_none() {
            return Err(err(
                400,
                "grammar tokenizer is unavailable",
                "invalid_request_error",
            ));
        }
        create_constraint(pipeline, grammar)?;
    }

    Ok(CompletionRequest {
        prepared,
        max_tokens,
        choices,
        streamed,
        include_usage,
        sampling,
        frame_options,
        stops,
        grammar,
    })
}

fn is_integer_value(value: &Value, min: Option<u64>, max: Option<u64>) -> bool {
    value.as_u64().is_some_and(|value| {
        min.is_none_or(|min| value >= min) && max.is_none_or(|max| value <= max)
    })
}

fn err(status: u16, message: impl Into<String>, code: &str) -> PipelineError {
    PipelineError {
        status,
        message: message.into(),
        code: Some(code.to_owned()),
    }
}

fn create_constraint(
    pipeline: &Pipeline,
    grammar: &Grammar,
) -> Result<Arc<TokenConstraint>, PipelineError> {
    let factory = pipeline.constraints.as_ref().ok_or_else(|| {
        err(400, "grammar tokenizer is unavailable", "invalid_request_error")
    })?;
    factory
        .create(grammar)
        .map(Arc::new)
        .map_err(|error| err(400, error.0, "invalid_request_error"))
}

/// Run a request to completion and return OpenAI's legacy text-completion
/// envelope. Multiple choices run serially with independent constraints.
pub fn complete(
    pipeline: &Pipeline,
    request: &CompletionRequest,
    public_id: &str,
    created_at: i64,
    deadline: Option<Duration>,
    disconnect_stream: Option<&TcpStream>,
    http_started_at: Option<std::time::Instant>,
) -> Result<Value, PipelineError> {
    let expires_at = deadline.and_then(|duration| std::time::Instant::now().checked_add(duration));
    let mut choices = Vec::with_capacity(request.choices);
    let mut first_result: Option<CollectedResult> = None;
    let mut prompt_tokens = 0u64;
    let mut completion_tokens = 0u64;
    for index in 0..request.choices {
        let remaining = match expires_at {
            Some(expires_at) => {
                let remaining = expires_at.saturating_duration_since(std::time::Instant::now());
                if remaining.is_zero() {
                    return Err(err(504, "request deadline exceeded", "request_timeout"));
                }
                Some(remaining)
            }
            None => None,
        };
        let (result, content) = generate_one(
            pipeline,
            request,
            remaining,
            index,
            disconnect_stream,
            if index == 0 { http_started_at } else { None },
        )?;
        prompt_tokens = prompt_tokens.max(result.prompt_tokens);
        completion_tokens = completion_tokens.saturating_add(result.completion_tokens);
        if first_result.is_none() {
            first_result = Some(result.clone());
        }
        choices.push(json!({
            "index": index,
            "text": content,
            "logprobs": null,
            "finish_reason": result.reason,
        }));
    }
    let mut result = first_result.expect("validated n is positive");
    result.prompt_tokens = prompt_tokens;
    result.completion_tokens = completion_tokens;
    result.metrics = openai::metrics_dict(&result);
    let job = JobInfo {
        public_id: public_id.to_owned(),
        created_at,
        reasoning_tokens: 0,
    };
    Ok(json!({
        "id": format!("cmpl-{public_id}"),
        "object": "text_completion",
        "created": created_at,
        "model": pipeline.model,
        "choices": choices,
        "usage": openai::usage_dict(&result, &job),
        "metrics": result.metrics,
        "timings": openai::timings_dict(&result),
    }))
}

fn generate_one(
    pipeline: &Pipeline,
    request: &CompletionRequest,
    deadline: Option<Duration>,
    index: usize,
    disconnect_stream: Option<&TcpStream>,
    http_started_at: Option<std::time::Instant>,
) -> Result<(CollectedResult, String), PipelineError> {
    let latency = pipeline.client.latency_metrics();
    let _grammar_timer = request
        .grammar
        .as_ref()
        .and_then(|_| latency.as_ref().map(|metrics| metrics.timer("grammar")));
    let constraint = request
        .grammar
        .as_ref()
        .map(|grammar| create_constraint(pipeline, grammar))
        .transpose()?;
    let mut options = request.frame_options.clone();
    options.seed = options.seed.wrapping_add(index as u64);
    let generation = pipeline.generation_request_with_frame_options(
        &request.prepared,
        request.max_tokens,
        constraint.clone(),
        deadline,
        request.sampling,
        options,
    );
    let tokens = Arc::new(Mutex::new(Vec::new()));
    let first_batch = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let event_tokens = tokens.clone();
    let event_first_batch = first_batch.clone();
    let event_constraint = constraint.clone();
    let on_event: EventCallback = Arc::new(move |call, event| {
        if let EngineEvent::Tokens(event) = event {
            if event.request_id != call.request_id {
                return;
            }
            if let Some(constraint) = event_constraint.as_ref() {
                if let Err(error) = constraint.commit(&event.tokens) {
                    call.record_callback_error(format!("constraint: {error}"));
                    return;
                }
            }
            if event_first_batch.load(std::sync::atomic::Ordering::Relaxed) == 0
                && !event.tokens.is_empty()
            {
                event_first_batch.store(
                    event.tokens.len() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
            event_tokens.lock().expect("text completion token lock").extend_from_slice(&event.tokens);
        }
    });
    let call = pipeline.client.submit_with_http_start(
        &generation, Some(on_event), None, http_started_at,
    )?;
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
    let done = call.result(deadline)?;
    let token_ids = tokens.lock().expect("text completion token lock").clone();
    let mut content = pipeline
        .tokenizer
        .decode(&token_ids, true)
        .map_err(|error| PipelineError {
            status: 500,
            message: error.0,
            code: None,
        })?;
    let stop = truncate_at_stop(&mut content, &request.stops);
    let first_batch = first_batch.load(std::sync::atomic::Ordering::Relaxed);
    let mut result = result_from_done(&done, first_batch);
    if stop {
        result.reason = "stop".into();
    }
    result.metrics = openai::metrics_dict(&result);
    Ok((result, content))
}

fn truncate_at_stop(text: &mut String, stops: &[String]) -> bool {
    let found = stops
        .iter()
        .filter_map(|stop| text.find(stop).map(|index| (index, stop)))
        .min_by_key(|(index, _)| *index);
    if let Some((index, _)) = found {
        text.truncate(index);
        true
    } else {
        false
    }
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
        stream_tokens_per_second: if done.decode_micros > 0
            && u64::from(done.completion_tokens) > first_batch
        {
            (u64::from(done.completion_tokens) - first_batch) as f64 * 1_000_000.0
                / done.decode_micros as f64
        } else {
            0.0
        },
        bounded: done.bounded,
        metrics: json!({}),
    };
    result.metrics = openai::metrics_dict(&result);
    result
}

fn text_chunk(
    model: &str,
    request_id: &str,
    created: i64,
    text: &str,
    finish_reason: Option<String>,
    usage: Option<Value>,
    metrics: Option<Value>,
    timings: Option<Value>,
) -> Value {
    let mut chunk = json!({
        "id": format!("cmpl-{request_id}"),
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "text": text,
            "logprobs": null,
            "finish_reason": finish_reason,
        }],
    });
    if let Some(usage) = usage {
        chunk["choices"] = json!([]);
        chunk["usage"] = usage;
    }
    if let Some(metrics) = metrics {
        chunk["metrics"] = metrics;
    }
    if let Some(timings) = timings {
        chunk["timings"] = timings;
    }
    chunk
}

/// Submit a text completion and stream deltas using OpenAI's legacy
/// `text_completion` SSE chunk shape.
#[allow(clippy::too_many_arguments)]
pub fn stream(
    stream: &mut TcpStream,
    cors_origin: Option<&str>,
    pipeline: &Pipeline,
    request: &CompletionRequest,
    public_id: &str,
    created_at: i64,
    deadline: Option<Duration>,
    http_started_at: Option<std::time::Instant>,
) -> std::io::Result<Result<Response, PipelineError>> {
    use crate::http::{write_response_with_cors, Response};

    let constraint = request
        .grammar
        .as_ref()
        .map(|grammar| create_constraint(pipeline, grammar))
        .transpose();
    let constraint = match constraint {
        Ok(constraint) => constraint,
        Err(error) => return Ok(Err(error)),
    };
    let generation = pipeline.generation_request_with_frame_options(
        &request.prepared,
        request.max_tokens,
        constraint.clone(),
        deadline,
        request.sampling,
        request.frame_options.clone(),
    );
    let sink_stream = stream.try_clone()?;
    write_response_with_cors(stream, &Response::sse_head(), cors_origin)?;
    let sink = Arc::new(Mutex::new(sink_stream));
    let state = Arc::new(Mutex::new(TextStreamState {
        sink: sink.clone(),
        model: pipeline.model.clone(),
        public_id: public_id.to_owned(),
        created_at,
        streamer: CallbackStreamer::new(&request.stops.iter().map(String::as_str).collect::<Vec<_>>()),
        first_batch: 0,
        stopped: false,
        include_usage: request.include_usage,
        tokens: 0,
    }));
    let cancel_client = pipeline.client.clone();
    let on_event: EventCallback = {
        let state = state.clone();
        let cancel_client = cancel_client.clone();
        let tokenizer = pipeline.tokenizer.clone();
        let constraint = constraint.clone();
        Arc::new(move |call, event| {
            let mut stop = false;
            let outcome = {
                let mut state = state.lock().expect("text completion stream state lock");
                match event {
                    EngineEvent::Tokens(tokens) if tokens.request_id == call.request_id => {
                        if let Some(constraint) = constraint.as_ref() {
                            if let Err(error) = constraint.commit(&tokens.tokens) {
                                call.record_callback_error(format!("constraint: {error}"));
                                return;
                            }
                        }
                        if state.first_batch == 0 && !tokens.tokens.is_empty() {
                            state.first_batch = tokens.tokens.len() as u64;
                        }
                        state.tokens = state.tokens.saturating_add(tokens.tokens.len() as u64);
                        let output = state.streamer.put_tokens(&tokenizer, &tokens.tokens);
                        stop = state.streamer.stop_sequence().is_some();
                        state.stopped = stop;
                        state.publish(output)
                    }
                    EngineEvent::PromptProgress(progress)
                        if progress.request_id == call.request_id =>
                    {
                        state.progress(progress.processed_tokens)
                    }
                    _ => Ok(()),
                }
            };
            if stop {
                cancel_client.cancel(call);
            }
            if let Err(error) = outcome {
                call.record_callback_error(format!("text completion stream write failed: {error}"));
                cancel_client.cancel(call);
            }
        })
    };
    let on_complete: CompletionCallback = {
        let state = state.clone();
        let tokenizer = pipeline.tokenizer.clone();
        Arc::new(move |call| {
            let done = match call.result(None) {
                Ok(done) => done,
                Err(error) => {
                    let api_error = ApiError::new(503, error.to_string(), "engine_error");
                    let _ = state
                        .lock()
                        .expect("text completion stream state lock")
                        .error(api_error.openai_payload());
                    return;
                }
            };
            let mut state = state.lock().expect("text completion stream state lock");
            if let Err(error) = state.finish(&done, &tokenizer) {
                let _ = state.error(ApiError::new(500, error, "internal_error").openai_payload());
            }
        })
    };
    let call = match pipeline
        .client
        .submit_with_http_start(&generation, Some(on_event), Some(on_complete), http_started_at)
    {
        Ok(call) => call,
        Err(error) => {
            let payload = ApiError::new(503, error.to_string(), "engine_error").openai_payload();
            let data = serde_json::to_string(&payload).unwrap_or_else(|_| "{}".into());
            write_sse(&sink, &data)
                .map_err(|error| std::io::Error::new(std::io::ErrorKind::BrokenPipe, error))?;
            write_sse(&sink, "[DONE]")
                .map_err(|error| std::io::Error::new(std::io::ErrorKind::BrokenPipe, error))?;
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
            return Err(error);
        }
    };
    let _ = call.result(None);
    Ok(Ok(Response {
        status: 200,
        content_type: "text/event-stream",
        body: Vec::new(),
    }))
}

struct TextStreamState {
    sink: Arc<Mutex<TcpStream>>,
    model: String,
    public_id: String,
    created_at: i64,
    streamer: CallbackStreamer,
    first_batch: u64,
    stopped: bool,
    include_usage: bool,
    tokens: u64,
}

impl TextStreamState {
    fn write_chunk(&self, chunk: &Value) -> Result<(), String> {
        let data = serde_json::to_string(chunk).map_err(|error| error.to_string())?;
        write_sse(&self.sink, &data)
    }

    fn publish(&self, output: Vec<Streamed>) -> Result<(), String> {
        for output in output {
            if let Streamed::Text(text) = output {
                self.write_chunk(&text_chunk(
                    &self.model,
                    &self.public_id,
                    self.created_at,
                    &text,
                    None,
                    None,
                    None,
                    None,
                ))?;
            }
        }
        Ok(())
    }

    fn progress(&self, processed_tokens: u32) -> Result<(), String> {
        let mut chunk = text_chunk(&self.model, &self.public_id, self.created_at, "", None, None, None, None);
        chunk["prompt_progress"] = json!({"processed_tokens": processed_tokens});
        self.write_chunk(&chunk)
    }

    fn error(&self, payload: Value) -> Result<(), String> {
        self.write_chunk(&payload)?;
        write_sse(&self.sink, "[DONE]")
    }

    fn finish(
        &mut self,
        done: &crate::protocol::DoneEvent,
        tokenizer: &crate::tokenization::FastTokenizer,
    ) -> Result<(), String> {
        let tail = self.streamer.end(tokenizer);
        self.stopped = self.streamer.stop_sequence().is_some();
        self.publish(tail)?;
        let result = result_from_done(done, self.first_batch);
        let finish_reason = if self.stopped {
            "stop".to_owned()
        } else {
            result.reason.clone()
        };
        self.write_chunk(&text_chunk(
            &self.model,
            &self.public_id,
            self.created_at,
            "",
            Some(finish_reason),
            None,
            None,
            Some(openai::timings_dict(&result)),
        ))?;
        if self.include_usage {
            let job = JobInfo {
                public_id: self.public_id.clone(),
                created_at: self.created_at,
                reasoning_tokens: 0,
            };
            self.write_chunk(&text_chunk(
                &self.model,
                &self.public_id,
                self.created_at,
                "",
                None,
                Some(openai::usage_dict(&result, &job)),
                Some(result.metrics),
                None,
            ))?;
        }
        write_sse(&self.sink, "[DONE]")
    }
}

fn write_sse(sink: &Arc<Mutex<TcpStream>>, payload: &str) -> Result<(), String> {
    let mut sink = sink.lock().expect("text completion SSE writer lock");
    crate::http::write_sse_event(&mut sink, payload).map_err(|error| error.to_string())
}
