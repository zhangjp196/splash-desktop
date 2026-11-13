//! One request's path: normalize the messages, render the chat template,
//! tokenize the prompt, submit the wire frame to the engine, and read the
//! output through the streamer and the projector.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::Arc;

use serde_json::{json, Value};

use crate::chat_format::ChatFormat;
use crate::chat_messages::normalize_messages;
use crate::dialects::openai;
use crate::engine::{EngineClient, EngineError, GenerationRequest};
use crate::output::StreamingToolCallProjector;
use crate::templates::{ChatTemplateRenderer, Templates};
use crate::tokenization::{FastTokenizer, PromptTokenizer};

/// The prepared pieces of one request.
pub struct Prepared {
    pub rendered: String,
    pub prompt_tokens: Vec<u32>,
}

/// One error the pipeline answers with.
#[derive(Debug, Clone)]
pub struct PipelineError {
    pub status: u16,
    pub message: String,
}

impl From<crate::chat_messages::MessageError> for PipelineError {
    fn from(error: crate::chat_messages::MessageError) -> PipelineError {
        PipelineError {
            status: error.status,
            message: error.message,
        }
    }
}

impl From<crate::templates::ChatTemplateError> for PipelineError {
    fn from(error: crate::templates::ChatTemplateError) -> PipelineError {
        PipelineError {
            status: 500,
            message: error.0,
        }
    }
}

impl From<EngineError> for PipelineError {
    fn from(error: EngineError) -> PipelineError {
        PipelineError {
            status: 503,
            message: error.to_string(),
        }
    }
}

/// The serving pipeline: the model's tokenizer, templates and chat
/// format, and the engine client it submits to.
pub struct Pipeline {
    pub tokenizer: Arc<FastTokenizer>,
    pub prompt_tokenizer: PromptTokenizer,
    pub templates: Templates,
    pub chat_format: ChatFormat,
    pub client: EngineClient,
    pub model: String,
}

impl Pipeline {
    /// Prepare one Chat request's prompt: the messages normalized, the
    /// template rendered, the prompt tokenized.
    pub fn prepare(&self, body: &Value) -> Result<Prepared, PipelineError> {
        let messages = body
            .get("messages")
            .ok_or_else(|| PipelineError { status: 400, message: "messages is required".into() })?;
        let normalized = normalize_messages(messages, false)?;
        let templated = crate::chat_messages::template_messages(&normalized);
        let messages_value = Value::Array(templated);
        let source = self.templates.select(None, body.get("tools"))?;
        let renderer = ChatTemplateRenderer::new(&source)
            .map_err(|error| PipelineError { status: 500, message: error.0 })?;
        let tools = body.get("tools").cloned();
        let kwargs: BTreeMap<String, Value> = BTreeMap::new();
        let rendered = renderer
            .render(&messages_value, tools.as_ref(), None, true, &kwargs)
            .map_err(|error| PipelineError { status: 500, message: error.0 })?;
        let prompt_tokens = self.prompt_tokenizer.encode(&rendered);
        Ok(Prepared {
            rendered,
            prompt_tokens,
        })
    }

    /// Submit one prepared request to the engine, collecting the output's
    /// tokens through the call's events, as the Python's exchange does.
    pub fn submit(
        &self,
        prepared: &Prepared,
        max_output_tokens: u32,
    ) -> Result<(Arc<crate::engine::Call>, Arc<Mutex<Vec<u32>>>), PipelineError> {
        let tokens: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));
        let collector = tokens.clone();
        let on_event: crate::engine::EventCallback = Arc::new(move |call, event| {
            if let crate::protocol::EngineEvent::Tokens(tokens_event) = event {
                if call.request_id == tokens_event.request_id {
                    collector
                        .lock()
                        .expect("token lock")
                        .extend_from_slice(&tokens_event.tokens);
                }
            }
        });
        let request = GenerationRequest {
            frame: crate::protocol::RequestFrame {
                logical_max_output_tokens: max_output_tokens,
                prompt_tokens: prepared.prompt_tokens.clone(),
                ..crate::protocol::RequestFrame::default()
            },
            deadline: None,
            mask_provider: None,
        };
        let call = self.client.submit(&request, Some(on_event), None)?;
        Ok((call, tokens))
    }

    /// One complete (non-streaming) Chat response: the engine's output
    /// collected and assembled as OpenAI's dialect answers.
    pub fn complete(
        &self,
        body: &Value,
        prepared: &Prepared,
        call: &Arc<crate::engine::Call>,
        tokens: &Arc<Mutex<Vec<u32>>>,
        model: &str,
        job: &openai::JobInfo,
    ) -> Result<Value, PipelineError> {
        let done = call.result(None)?;
        let tokens = tokens.lock().expect("token lock").clone();
        let text = self
            .tokenizer
            .decode(&tokens, true)
            .map_err(|error| PipelineError { status: 500, message: error.0 })?;
        let mut projector = StreamingToolCallProjector::without_policy(1, &self.chat_format);
        projector.put(&text);
        let (content, calls, _owed) = projector.finish(false);
        let collected = openai::Collected {
            content,
            reasoning: None,
            tool_calls: calls
                .iter()
                .map(|call| {
                    json!({
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments},
                    })
                })
                .collect(),
            result: Some(collected_result(&done)),
            called: !calls.is_empty(),
        };
        let choice = openai::chat_choice(&collected, job, 0);
        let response = openai::completion_response(
            model,
            job,
            collected.result.as_ref().expect("set above"),
            vec![choice],
        );
        let _ = body;
        let _ = prepared;
        Ok(response)
    }
}

/// The collected result from a Done event.
fn collected_result(done: &crate::protocol::DoneEvent) -> openai::CollectedResult {
    openai::CollectedResult {
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
        bounded: false,
        metrics: json!({}),
    }
}
