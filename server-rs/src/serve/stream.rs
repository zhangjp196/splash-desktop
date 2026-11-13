//! One Chat completion's stream: the engine's events written as OpenAI's
//! SSE chunks as they arrive, ported from `server/dialects/openai.py`'s
//! `stream`.

use std::sync::{Arc, Mutex};

use serde_json::{json, Value};

use crate::dialects::openai;
use crate::engine::EventCallback;
use crate::output::streamer::Streamed;
use crate::output::{CallbackStreamer, StreamingToolCallProjector};
use crate::protocol::EngineEvent;

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
        timings: Option<Value>,
    ) -> Result<(), String> {
        let chunk = openai::stream_chunk(
            &self.model,
            &self.request_id,
            self.created_at,
            delta,
            finish_reason,
            usage,
            metrics_of(&timings),
            timings,
        );
        self.sse(&chunk)
    }

    /// The keepalive: after the start, a chunk that adds nothing.
    #[allow(dead_code)]
    fn keepalive(&self, started: bool) -> Result<(), String> {
        if started {
            self.chunk(json!({}), None, None, None)
        } else {
            // An SSE comment, which clients that time out on missing data
            // events ignore.
            self.write.lock().expect("write lock")(": keepalive")
        }
    }
}

fn metrics_of(_timings: &Option<Value>) -> Option<Value> {
    None
}

/// The event callback that writes one request's stream: the role chunk at
/// the start, the output's text as content deltas, the tool calls'
/// deltas, the finish chunk with the timings, the usage chunk when asked,
/// and `[DONE]`.
pub fn stream_callbacks(
    sink: Arc<StreamSink>,
    tokens: Arc<Mutex<Vec<u32>>>,
    tokenizer: Arc<crate::tokenization::FastTokenizer>,
    _projector: Arc<Mutex<StreamingToolCallProjector>>,
) -> (EventCallback, crate::engine::CompletionCallback) {
    let state: Arc<Mutex<StreamState>> = Arc::new(Mutex::new(StreamState {
        started: false,
        streamer: CallbackStreamer::new(&[]),
    }));
    let event_sink = sink.clone();
    let on_event: EventCallback = Arc::new(move |call, event| {
        let sink = event_sink.clone();
        let mut state = state.lock().expect("stream state lock");
        let outcome: Result<(), String> = match event {
            EngineEvent::Start(start) if start.request_id == call.request_id => {
                state.started = true;
                sink.chunk(
                    json!({"role": "assistant", "content": ""}),
                    None,
                    None,
                    None,
                )
            }
            EngineEvent::Tokens(tokens_event)
                if tokens_event.request_id == call.request_id =>
            {
                tokens
                    .lock()
                    .expect("token lock")
                    .extend_from_slice(&tokens_event.tokens);
                let whole = tokenizer
                    .decode(&tokens_event_tokens(&tokens), true)
                    .unwrap_or_default();
                let streamed = state.streamer.put_decoded(&whole);
                let results: Vec<Result<(), String>> = streamed
                    .iter()
                    .map(|decision| match decision {
                        Streamed::Text(text) => {
                            sink.chunk(json!({"content": text}), None, None, None)
                        }
                        Streamed::Stopped => Ok(()),
                    })
                    .collect();
                results.into_iter().collect()
            }
            _ => Ok(()),
        };
        if let Err(error) = outcome {
            // A write that failed ends the stream; the reader thread sees
            // the closed connection.
            eprintln!("stream write failed: {error}");
        }
    });

    // The completion callback: the finish chunk with the timings, the
    // usage chunk when asked, and [DONE].
    let on_complete: crate::engine::CompletionCallback = Arc::new(move |call| {
        let outcome = (|| -> Result<(), String> {
            let done = call
                .result(None)
                .map_err(|error| error.to_string())?;
            let result = openai::CollectedResult {
                reason: format!("{:?}", done.reason).to_lowercase(),
                prompt_tokens: done.prompt_tokens as u64,
                completion_tokens: done.completion_tokens as u64,
                prefill_tokens: done.prompt_tokens as u64,
                cache_matched_tokens: done.deduplicated_tokens as u64,
                drafted_tokens: done.drafted_tokens as u64,
                accepted_draft_tokens: done.accepted_draft_tokens as u64,
                first_token_batch_tokens: 0,
                start_to_first_token_ms: done.prefill_micros as f64 / 1000.0,
                first_token_to_done_ms: done.decode_micros as f64 / 1000.0,
                request_wall_ms: done.wall_micros as f64 / 1000.0,
                stream_tokens_per_second: if done.decode_micros != 0 {
                    done.completion_tokens as f64 * 1_000_000.0 / done.decode_micros as f64
                } else {
                    0.0
                },
                metrics: json!({}),
            };
            let timings = openai::timings_dict(&result);
            let finish = openai::finish_reason(&result, false);
            sink.chunk(json!({}), Some(finish), None, Some(timings))?;
            if sink.include_usage {
                sink.chunk(
                    json!({}),
                    None,
                    Some(openai::usage_dict(&result, &call_job())),
                    None,
                )?;
            }
            sink.sse(&json!("[DONE]"))
        })();
        if let Err(error) = outcome {
            eprintln!("stream finish failed: {error}");
        }
    });

    (on_event, on_complete)
}

fn tokens_event_tokens(tokens: &Arc<Mutex<Vec<u32>>>) -> Vec<u32> {
    tokens.lock().expect("token lock").clone()
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
}

