//! One request's path: normalize the messages, render the chat template,
//! tokenize the prompt, submit the wire frame to the engine, and read the
//! output through the streamer and the projector.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::Arc;

use serde_json::{json, Value};

use crate::chat_format::{ChatFormat, THINK_END, THINK_START};
use crate::chat_messages::normalize_messages;
use crate::dialects::openai;
use crate::engine::{EngineClient, EngineError, GenerationRequest};
use crate::output::{ReasoningPart, ReasoningSplitter, StreamingToolCallProjector};
use crate::templates::{ChatTemplateRenderer, Templates};
use crate::tokenization::{FastTokenizer, PromptTokenizer};

/// The prepared pieces of one request.
pub struct Prepared {
    pub rendered: String,
    pub prompt_tokens: Vec<u32>,
    /// Whether the generation prompt opened a think block, so the output
    /// starts in the reasoning: the Python's `RenderedPrompt.thinking`,
    /// which is the last think marker the prompt's end carries.
    pub thinking: bool,
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

/// The template variables a request's reasoning selects, as the Python's
/// `template_options` builds them: whether the model thinks follows the
/// request's own `enable_thinking`, else its effort, and both switches the
/// templates read say so. A request's other `chat_template_kwargs` outrank
/// them, and may not set what preparation owns.
fn template_kwargs(body: &Value) -> Result<BTreeMap<String, Value>, PipelineError> {
    let other = match body.get("chat_template_kwargs") {
        None | Some(Value::Null) => None,
        Some(Value::Object(kwargs)) => Some(kwargs),
        Some(_) => return Err(invalid("chat_template_kwargs must be an object")),
    };
    if let Some(other) = other {
        if let Some(name) = other
            .keys()
            .find(|name| RESERVED_TEMPLATE_KWARGS.contains(&name.as_str()))
        {
            return Err(invalid(format!("chat_template_kwargs cannot set {name}")));
        }
    }
    let reasoning_effort = match body.get("reasoning_effort") {
        None | Some(Value::Null) => None,
        Some(Value::String(effort)) if REASONING_EFFORTS.contains(&effort.as_str()) => {
            Some(effort.as_str())
        }
        Some(_) => return Err(invalid("invalid reasoning_effort")),
    };
    let mut options = BTreeMap::new();
    let enable_thinking = match other.and_then(|kwargs| kwargs.get("enable_thinking")) {
        None | Some(Value::Null) => reasoning_effort.map(|effort| effort != "none"),
        Some(Value::Bool(enabled)) => Some(*enabled),
        Some(_) => {
            return Err(invalid(
                "chat_template_kwargs enable_thinking must be a boolean",
            ))
        }
    };
    if let Some(enabled) = enable_thinking {
        options.insert("enable_thinking".into(), Value::Bool(enabled));
        if !enabled {
            options.insert("reasoning_effort".into(), json!("none"));
        } else if let Some(effort) = reasoning_effort.filter(|effort| *effort != "none") {
            options.insert("reasoning_effort".into(), json!(effort));
        }
    }
    if let Some(other) = other {
        for (name, value) in other {
            // The request's own switch is already in place above.
            if name != "enable_thinking" {
                options.insert(name.clone(), value.clone());
            }
        }
    }
    Ok(options)
}

fn invalid(message: impl Into<String>) -> PipelineError {
    PipelineError {
        status: 400,
        message: message.into(),
    }
}

/// The reasoning efforts `reasoning_effort` accepts, as the Python's
/// `REASONING_EFFORTS`.
const REASONING_EFFORTS: &[&str] = &["none", "minimal", "low", "medium", "high", "xhigh", "max"];

/// Template arguments request preparation sets itself, and the tokenizer's
/// own controls: a request's `chat_template_kwargs` cannot set them.
const RESERVED_TEMPLATE_KWARGS: &[&str] = &[
    "add_generation_prompt",
    "chat_template",
    "continue_final_message",
    "conversation",
    "documents",
    "messages",
    "return_dict",
    "tokenize",
    "tools",
];

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
        let kwargs = template_kwargs(body)?;
        let rendered = renderer
            .render(&messages_value, tools.as_ref(), None, true, &kwargs)
            .map_err(|error| PipelineError { status: 500, message: error.0 })?;
        let prompt_tokens = self.prompt_tokenizer.encode(&rendered);
        // The generation prompt opens the think block where its text ends
        // inside one: the same question `server/frontend.py`'s
        // `_generation_prompt` asks of the probed prompt, read from the
        // rendered text's own last markers.
        let thinking = rendered.rfind(THINK_START) > rendered.rfind(THINK_END);
        Ok(Prepared {
            rendered,
            prompt_tokens,
            thinking,
        })
    }

    /// Submit one prepared request to the engine, collecting the output's
    /// tokens through the call's events, as the Python's exchange does. The
    /// tokens of the output's first emission travel beside them: that batch
    /// precedes TTFT, so the stream's rate leaves it out.
    pub fn submit(
        &self,
        prepared: &Prepared,
        max_output_tokens: u32,
    ) -> Result<
        (
            Arc<crate::engine::Call>,
            Arc<Mutex<Vec<u32>>>,
            Arc<std::sync::atomic::AtomicU64>,
        ),
        PipelineError,
    > {
        let tokens: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));
        let first_batch = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let collector = tokens.clone();
        let first_batch_seen = first_batch.clone();
        let on_event: crate::engine::EventCallback = Arc::new(move |call, event| {
            if let crate::protocol::EngineEvent::Tokens(tokens_event) = event {
                if call.request_id == tokens_event.request_id {
                    if first_batch_seen.load(std::sync::atomic::Ordering::Relaxed) == 0
                        && !tokens_event.tokens.is_empty()
                    {
                        first_batch_seen.store(
                            tokens_event.tokens.len() as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    }
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
        Ok((call, tokens, first_batch))
    }

    /// One complete (non-streaming) Chat response: the engine's output
    /// collected and assembled as OpenAI's dialect answers.
    pub fn complete(
        &self,
        body: &Value,
        prepared: &Prepared,
        call: &Arc<crate::engine::Call>,
        tokens: &Arc<Mutex<Vec<u32>>>,
        first_token_batch_tokens: u64,
        model: &str,
        job: &openai::JobInfo,
    ) -> Result<Value, PipelineError> {
        let done = call.result(None)?;
        let tokens = tokens.lock().expect("token lock").clone();
        let text = self
            .tokenizer
            .decode(&tokens, true)
            .map_err(|error| PipelineError { status: 500, message: error.0 })?;
        // The output's reasoning is split from its answer before the
        // projector reads it, as the Python's exchange does: the answer is
        // what a call's tags are read from, and the reasoning is what the
        // message reports as `reasoning_content`.
        let mut splitter = ReasoningSplitter::new(
            prepared.thinking,
            Some(&self.chat_format),
            body.get("tools").is_some(),
        );
        let mut reasoning = String::new();
        let mut answer = String::new();
        let mut parts = splitter.put(&text);
        parts.extend(splitter.finish());
        for part in parts {
            match part {
                ReasoningPart::Reasoning(text) => reasoning.push_str(&text),
                ReasoningPart::Content(text) => answer.push_str(&text),
            }
        }
        let mut projector = StreamingToolCallProjector::without_policy(1, &self.chat_format);
        projector.put(&answer);
        let (content, calls, _owed) = projector.finish(false);
        let mut result = collected_result(&done, first_token_batch_tokens);
        result.metrics = openai::metrics_dict(&result);
        let collected = openai::Collected {
            content,
            reasoning: if reasoning.is_empty() { None } else { Some(reasoning) },
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
            result: Some(result),
            called: !calls.is_empty(),
        };
        let choice = openai::chat_choice(&collected, job, 0);
        let response = openai::completion_response(
            model,
            job,
            collected.result.as_ref().expect("set above"),
            vec![choice],
        );
        Ok(response)
    }
}

/// The collected result from a Done event.
fn collected_result(
    done: &crate::protocol::DoneEvent,
    first_token_batch_tokens: u64,
) -> openai::CollectedResult {
    openai::CollectedResult {
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
            (done.completion_tokens as u64 - first_token_batch_tokens) as f64 * 1_000_000.0
                / done.decode_micros as f64
        } else {
            0.0
        },
        bounded: false,
        metrics: json!({}),
    }
}
