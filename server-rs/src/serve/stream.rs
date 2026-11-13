//! One Chat completion's stream: the engine's events written as OpenAI's
//! SSE chunks as they arrive, ported from `server/dialects/openai.py`'s
//! `stream`.

use std::sync::{Arc, Mutex};

use serde_json::{json, Value};

use crate::chat_format::ChatFormat;
use crate::constraints::TokenConstraint;
use crate::dialects::openai;
use crate::engine::EventCallback;
use crate::output::streamer::Streamed;
use crate::output::{
    CallbackStreamer, ProjectorEvent, ReasoningPart, ReasoningSplitter, StreamingToolCallProjector,
};
use crate::protocol::EngineEvent;
use crate::tool_schema::{validate_response_content, ResponseValidator};

/// What one stream's writer needs: the SSE sink, the model, the job's
/// ids, and whether the usage chunk is asked for.
pub struct StreamSink {
    pub write: Arc<Mutex<dyn FnMut(&str) -> Result<(), String> + Send>>,
    pub model: String,
    pub request_id: String,
    pub created_at: i64,
    pub include_usage: bool,
}

impl StreamSink {
    fn sse(&self, chunk: &Value) -> Result<(), String> {
        let payload = serde_json::to_string(chunk)
            .map_err(|error| format!("chunk serialization: {error}"))?;
        self.write.lock().expect("write lock")(&payload)
    }

    fn chunk(
        &self,
        delta: Value,
        finish_reason: Option<String>,
        usage: Option<Value>,
        metrics: Option<Value>,
        timings: Option<Value>,
    ) -> Result<(), String> {
        let chunk = openai::stream_chunk(
            &self.model,
            &self.request_id,
            self.created_at,
            delta,
            finish_reason,
            usage,
            metrics,
            timings,
        );
        self.sse(&chunk)
    }

    /// The keepalive: after the start, a chunk that adds nothing.
    #[allow(dead_code)]
    fn keepalive(&self, started: bool) -> Result<(), String> {
        if started {
            self.chunk(json!({}), None, None, None, None)
        } else {
            // An SSE comment, which clients that time out on missing data
            // events ignore.
            self.write.lock().expect("write lock")(": keepalive")
        }
    }

    /// The error a stream answers with when it has already started: the
    /// refusal as a data event, then the end, as the Python's
    /// `_sse_error` writes.
    fn error(&self, error: &crate::support::errors::ApiError) -> Result<(), String> {
        self.sse(&error.openai_payload())?;
        self.write.lock().expect("write lock")("[DONE]")
    }
}

/// One projector event as the chunk OpenAI's stream carries.
fn tool_chunk(event: &ProjectorEvent) -> Value {
    match event {
        ProjectorEvent::Content(text) => json!({"content": text}),
        ProjectorEvent::ToolCall { index, id, name } => json!({
            "tool_calls": [{
                "index": index,
                "id": id,
                "type": "function",
                "function": {"name": name},
            }],
        }),
        ProjectorEvent::ToolDelta { index, chunk } => json!({
            "tool_calls": [{
                "index": index,
                "function": {"arguments": chunk},
            }],
        }),
    }
}

/// The event callback that writes one request's stream: the role chunk at
/// the start, the output's reasoning and text as `reasoning_content` and
/// `content` deltas, the tool calls' deltas, the finish chunk with the
/// timings, the usage chunk when asked, and `[DONE]`.
pub fn stream_callbacks(
    sink: Arc<StreamSink>,
    tokens: Arc<Mutex<Vec<u32>>>,
    tokenizer: Arc<crate::tokenization::FastTokenizer>,
    projector: Arc<Mutex<StreamingToolCallProjector>>,
    thinking: bool,
    chat_format: ChatFormat,
    tool_calls: bool,
    constraint: Option<Arc<TokenConstraint>>,
    validator: Option<Arc<ResponseValidator>>,
) -> (EventCallback, crate::engine::CompletionCallback) {
    let state: Arc<Mutex<StreamState>> = Arc::new(Mutex::new(StreamState {
        started: false,
        streamer: CallbackStreamer::new(&[]),
        splitter: ReasoningSplitter::new(thinking, Some(&chat_format), tool_calls),
        first_token_batch_tokens: 0,
    }));
    let event_state = state.clone();
    let event_sink = sink.clone();
    let event_projector = projector.clone();
    let on_event: EventCallback = Arc::new(move |call, event| {
        let sink = event_sink.clone();
        let mut state = event_state.lock().expect("stream state lock");
        let outcome: Result<(), String> = match event {
            EngineEvent::Start(start) if start.request_id == call.request_id => {
                state.started = true;
                sink.chunk(
                    json!({"role": "assistant", "content": ""}),
                    None,
                    None,
                    None,
                    None,
                )
            }
            EngineEvent::Tokens(tokens_event)
                if tokens_event.request_id == call.request_id =>
            {
                if state.first_token_batch_tokens == 0 && !tokens_event.tokens.is_empty() {
                    state.first_token_batch_tokens = tokens_event.tokens.len() as u64;
                }
                // The tokens the output generates are the grammar's input:
                // each emission is committed, and the next mask consumes
                // them, as the Python's exchange does.
                if let Some(constraint) = constraint.as_ref() {
                    if let Err(error) = constraint.commit(&tokens_event.tokens) {
                        call.record_callback_error(format!("constraint: {error}"));
                        return;
                    }
                }
                tokens
                    .lock()
                    .expect("token lock")
                    .extend_from_slice(&tokens_event.tokens);
                let streamed = state.streamer.put_tokens(&tokenizer, &tokens_event.tokens);
                publish(&sink, &mut state.splitter, &event_projector, streamed)
            }
            _ => Ok(()),
        };
        if let Err(error) = outcome {
            // A write that failed ends the stream; the reader thread sees
            // the closed connection.
            eprintln!("stream write failed: {error}");
        }
    });

    // The completion callback: the reasoning the splitter still holds, the
    // finish chunk with the timings, the usage chunk when asked, and
    // [DONE].
    let complete_state = state.clone();
    let on_complete: crate::engine::CompletionCallback = Arc::new(move |call| {
        let sink = sink.clone();
        let outcome = (|| -> Result<(), String> {
            {
                let mut state = complete_state.lock().expect("stream state lock");
                let parts = state.splitter.finish();
                publish_parts(&sink, &projector, parts)?;
            }
            let done = call
                .result(None)
                .map_err(|error| error.to_string())?;
            let first_token_batch_tokens = complete_state
                .lock()
                .expect("stream state lock")
                .first_token_batch_tokens;
            // The projector's held text and calls go out before the finish
            // chunk, and an answer without calls is validated against the
            // response format only when the model ended it.
            let incomplete = done.reason == crate::protocol::FinishReason::Length;
            let (content, calls, events) = projector
                .lock()
                .expect("projector lock")
                .finish(incomplete);
            for event in &events {
                sink.chunk(tool_chunk(event), None, None, None, None)?;
            }
            let ended = done.reason != crate::protocol::FinishReason::Length;
            if ended && calls.is_empty() {
                if let Some(validator) = validator.as_ref() {
                    if let Err(error) = validate_response_content(&content, Some(validator)) {
                        sink.error(&error)?;
                        return Ok(());
                    }
                }
            }
            let mut result = openai::CollectedResult {
                reason: format!("{:?}", done.reason).to_lowercase(),
                prompt_tokens: done.prompt_tokens as u64,
                completion_tokens: done.completion_tokens as u64,
                prefill_tokens: done.prompt_tokens as u64,
                cache_matched_tokens: done.deduplicated_tokens as u64,
                drafted_tokens: done.drafted_tokens as u64,
                accepted_draft_tokens: done.accepted_draft_tokens as u64,
                first_token_batch_tokens,
                start_to_first_token_ms: done.prefill_micros as f64 / 1000.0,
                first_token_to_done_ms: done.decode_micros as f64 / 1000.0,
                request_wall_ms: done.wall_micros as f64 / 1000.0,
                stream_tokens_per_second: if done.decode_micros != 0
                    && done.completion_tokens as u64 > first_token_batch_tokens
                {
                    (done.completion_tokens as u64 - first_token_batch_tokens) as f64
                        * 1_000_000.0
                        / done.decode_micros as f64
                } else {
                    0.0
                },
                bounded: done.bounded,
                metrics: json!({}),
            };
            // The usage chunk carries the metrics the chat page reads its
            // tok/s from, as the Python's stream does.
            result.metrics = openai::metrics_dict(&result);
            let timings = openai::timings_dict(&result);
            let finish = openai::finish_reason(&result, !calls.is_empty());
            sink.chunk(json!({}), Some(finish), None, None, Some(timings))?;
            if sink.include_usage {
                sink.chunk(
                    json!({}),
                    None,
                    Some(openai::usage_dict(&result, &call_job())),
                    Some(result.metrics.clone()),
                    None,
                )?;
            }
            sink.write.lock().expect("write lock")("[DONE]")
        })();
        if let Err(error) = outcome {
            eprintln!("stream finish failed: {error}");
        }
    });

    (on_event, on_complete)
}

/// Publish one step's decoded text: the splitter keeps the reasoning apart
/// from the answer, and the answer's text goes through the projector,
/// whose events stream as the deltas they name.
fn publish(
    sink: &StreamSink,
    splitter: &mut ReasoningSplitter,
    projector: &Arc<Mutex<StreamingToolCallProjector>>,
    streamed: Vec<Streamed>,
) -> Result<(), String> {
    for decision in streamed {
        match decision {
            Streamed::Text(text) => {
                publish_parts(sink, projector, splitter.put(&text))?;
            }
            Streamed::Stopped => {}
        }
    }
    Ok(())
}

fn publish_parts(
    sink: &StreamSink,
    projector: &Arc<Mutex<StreamingToolCallProjector>>,
    parts: Vec<ReasoningPart>,
) -> Result<(), String> {
    for part in parts {
        match part {
            ReasoningPart::Reasoning(text) => {
                sink.chunk(json!({"reasoning_content": text}), None, None, None, None)?
            }
            ReasoningPart::Content(text) => {
                let events = projector.lock().expect("projector lock").put(&text);
                for event in &events {
                    sink.chunk(tool_chunk(event), None, None, None, None)?;
                }
            }
        }
    }
    Ok(())
}

fn call_job() -> openai::JobInfo {
    openai::JobInfo {
        public_id: String::new(),
        created_at: 0,
        reasoning_tokens: 0,
    }
}

/// One stream's running state.
struct StreamState {
    started: bool,
    streamer: CallbackStreamer,
    splitter: ReasoningSplitter,
    /// The tokens of the output's first emission, which precede TTFT and so
    /// stay out of the rate the usage chunk reports.
    first_token_batch_tokens: u64,
}
