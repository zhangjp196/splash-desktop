//! One request's path: normalize the messages, render the chat template,
//! tokenize the prompt, submit the wire frame to the engine, and read the
//! output through the streamer and the projector.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::Arc;

use serde_json::{json, Value};

use crate::chat_format::{ChatFormat, THINK_END, THINK_START};
use crate::chat_messages::normalize_messages;
use crate::constraints::{ConstraintFactory, Grammar, TokenConstraint};
use crate::dialects::openai;
use crate::engine::{EngineClient, EngineError, GenerationRequest};
use crate::images::prompts::{
    expand_image_pads, image_pad_token_id, prepare_images, render_image_tokens,
};
use crate::images::{ImageCache, RequestImages};
use crate::output::{ReasoningPart, ReasoningSplitter, StreamingToolCallProjector};
use crate::protocol::{ConstraintMode, ImageSpan};
use crate::templates::{ChatTemplateRenderer, Templates};
use crate::tokenization::{FastTokenizer, PromptTokenizer};
use crate::tool_schema::policy::ResponseFormat;
use crate::tool_schema::{
    json_grammar, normalize_response_format, normalize_tools, tool_grammar, ToolPolicy,
};

/// The prepared pieces of one request.
pub struct Prepared {
    pub rendered: String,
    pub prompt_tokens: Vec<u32>,
    /// Whether the generation prompt opened a think block, so the output
    /// starts in the reasoning: the Python's `RenderedPrompt.thinking`,
    /// which is the last think marker the prompt's end carries.
    pub thinking: bool,
    /// The request's prepared images, held so their pixels outlive
    /// preparation: the Python's `image_owner`, which keeps the batch's
    /// charge on the image cache until the request ends.
    pub images: Option<RequestImages>,
    /// The image spans the engine injects into, and their pixels in span
    /// order; both empty for a text-only request.
    pub image_spans: Vec<ImageSpan>,
    pub image_pixels: Vec<u8>,
    /// The tools the prompt declares, as the Python's `Prompt.tools` —
    /// every offered tool, each naming its function.
    pub tools: Option<Vec<Value>>,
    /// How the request's tool calls are generated and read; `None`
    /// without tools.
    pub tool_policy: Option<Arc<ToolPolicy>>,
    /// The response format a structured output generates and validates
    /// to; `None` without one.
    pub response_format: Option<ResponseFormat>,
}

impl Prepared {
    /// Whether the output generates under a grammar: constrained tool
    /// calls, or a response schema beside or without them.
    pub fn constrained(&self) -> bool {
        self.tool_policy
            .as_ref()
            .map(|policy| policy.constrained)
            .unwrap_or(false)
            || self.response_format.is_some()
    }
}

/// One error the pipeline answers with.
#[derive(Debug, Clone)]
pub struct PipelineError {
    pub status: u16,
    pub message: String,
    /// The machine code a refusal carries where the Python's error names
    /// one; `None` for the plain server error.
    pub code: Option<String>,
}

impl PipelineError {
    fn new(status: u16, message: impl Into<String>) -> PipelineError {
        PipelineError {
            status,
            message: message.into(),
            code: None,
        }
    }

    fn with_code(status: u16, message: impl Into<String>, code: &str) -> PipelineError {
        PipelineError {
            status,
            message: message.into(),
            code: Some(code.to_string()),
        }
    }
}

impl From<crate::chat_messages::MessageError> for PipelineError {
    fn from(error: crate::chat_messages::MessageError) -> PipelineError {
        PipelineError {
            status: error.status,
            message: error.message,
            code: None,
        }
    }
}

impl From<crate::templates::ChatTemplateError> for PipelineError {
    fn from(error: crate::templates::ChatTemplateError) -> PipelineError {
        PipelineError {
            status: 500,
            message: error.0,
            code: None,
        }
    }
}

impl From<EngineError> for PipelineError {
    fn from(error: EngineError) -> PipelineError {
        PipelineError {
            status: 503,
            message: error.to_string(),
            code: None,
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
    /// The image cache the request images prepare through, and the pixel
    /// budget one image resizes within: the Python's `images` and
    /// `max_image_pixels`.
    pub images: ImageCache,
    pub max_image_pixels: u32,
    /// The compiled output constraints, over the model's grammar
    /// tokenizer; `None` where the model's tokenizer directory is
    /// unknown, which a constrained request refuses.
    pub constraints: Option<ConstraintFactory>,
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
        code: None,
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
    /// The engine's readiness: whether the served model takes images, and
    /// the context window a request's images count against.
    fn readiness(&self) -> Option<crate::protocol::ReadyEvent> {
        self.client.readiness()
    }

    /// Prepare one Chat request's prompt: the messages normalized, the
    /// template rendered, the prompt tokenized, the images prepared and
    /// their placeholders widened.
    pub fn prepare(
        &self,
        body: &Value,
        deadline: &mut dyn FnMut() -> Result<(), crate::support::errors::ApiError>,
    ) -> Result<Prepared, PipelineError> {
        let messages = body
            .get("messages")
            .ok_or_else(|| PipelineError::new(400, "messages is required"))?;
        let ready = self.readiness();
        let vision = ready.map(|ready| ready.vision).unwrap_or(false);
        let max_context = ready
            .map(|ready| u64::from(ready.max_context_tokens))
            .unwrap_or(u64::MAX);
        let normalized = normalize_messages(messages, vision)?;
        let templated = crate::chat_messages::template_messages(&normalized);
        let messages_value = Value::Array(templated.clone());
        // The tools and the response format read before the prompt
        // renders, as the Python's `_prepare_prompt` runs: a refusal here
        // never renders documents or builds validators.
        let (offered, tool_policy) = normalize_tools(
            body.get("tools"),
            body.get("tool_choice"),
            body.get("parallel_tool_calls"),
            None,
        )
        .map_err(|error| PipelineError {
            status: error.status,
            message: error.message,
            code: None,
        })?;
        let response_format = normalize_response_format(body.get("response_format"))
            .map_err(|error| PipelineError {
                status: error.status,
                message: error.message,
                code: None,
            })?;
        let source = self.templates.select(None, body.get("tools"))?;
        let tools = offered.clone();
        let kwargs = template_kwargs(body)?;
        // The images prepare before the template renders, as the Python's
        // `_render_prompt` runs: an expired request starts no more decodes.
        let images = if vision {
            let mut check_deadline = || deadline();
            Some(
                prepare_images(
                    &self.images,
                    &templated,
                    self.max_image_pixels,
                    max_context,
                    true,
                    &mut check_deadline,
                )
                .map_err(|error| PipelineError { status: error.status, message: error.message, code: None })?,
            )
        } else {
            None
        };
        let has_images = images.as_ref().map(|batch| !batch.is_empty()).unwrap_or(false);
        let render = |template_source: &str| -> Result<String, crate::templates::ChatTemplateError> {
            ChatTemplateRenderer::new(template_source)?
                .render(
                &messages_value,
                tools.as_ref().map(|tools| Value::Array(tools.clone())).as_ref(),
                None,
                true,
                &kwargs,
            )
        };
        let (rendered, prompt_tokens, positions) = if has_images {
            let (tokens, positions, rendered) = render_image_tokens(
                &source,
                &render,
                &|text| self.tokenizer.encode_with_offsets(text, false),
            )
            .map_err(|error| PipelineError { status: error.status, message: error.message, code: None })?;
            (rendered, tokens, Some(positions))
        } else {
            let rendered = render(&source)
                .map_err(|error| PipelineError { status: 500, message: error.0, code: None })?;
            let prompt_tokens = self.prompt_tokenizer.encode(&rendered);
            (rendered, prompt_tokens, None)
        };
        // The generation prompt opens the think block where its text ends
        // inside one: the same question `server/frontend.py`'s
        // `_generation_prompt` asks of the probed prompt, read from the
        // rendered text's own last markers.
        let thinking = rendered.rfind(THINK_START) > rendered.rfind(THINK_END);
        // The placeholders widen to the images' merged token counts, and
        // the spans name where the engine injects.
        let (prompt_tokens, image_spans, image_pixels) = match (&images, positions) {
            (Some(batch), Some(positions)) if !batch.is_empty() => {
                let pad_id = image_pad_token_id(&self.tokenizer)
                    .map_err(|error| PipelineError { status: error.status, message: error.message, code: None })?;
                let (expanded, spans, pixels) = expand_image_pads(
                    &prompt_tokens,
                    batch.images(),
                    &positions,
                    pad_id,
                    max_context,
                    true,
                )
                .map_err(|error| PipelineError { status: error.status, message: error.message, code: None })?;
                (expanded, spans, pixels)
            }
            _ => (prompt_tokens, Vec::new(), Vec::new()),
        };
        Ok(Prepared {
            rendered,
            prompt_tokens,
            thinking,
            images,
            image_spans,
            image_pixels,
            tools,
            tool_policy: tool_policy.map(Arc::new),
            response_format,
        })
    }

    /// The engine request one prepared request submits: its frame, with
    /// the image spans and the constraint mode the request carries, and
    /// the mask provider a constrained request answers with.
    pub fn generation_request(
        &self,
        prepared: &Prepared,
        max_output_tokens: u32,
        constraint: Option<Arc<TokenConstraint>>,
    ) -> GenerationRequest {
        GenerationRequest {
            frame: crate::protocol::RequestFrame {
                logical_max_output_tokens: max_output_tokens,
                prompt_tokens: prepared.prompt_tokens.clone(),
                image_spans: prepared.image_spans.clone(),
                image_pixels: prepared.image_pixels.clone(),
                constraint: if constraint.is_some() {
                    ConstraintMode::TokenMask
                } else {
                    ConstraintMode::None
                },
                ..crate::protocol::RequestFrame::default()
            },
            deadline: None,
            mask_provider: constraint.map(|constraint| -> crate::engine::MaskProvider {
                Arc::new(move |event| {
                    constraint.masks(&event.simulation_tokens).map_err(|error| {
                        crate::engine::EngineError::MaskComputationFailed {
                            message: error,
                            retryable: false,
                        }
                    })
                })
            }),
        }
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
        let constraint = self.constraint(prepared)?;
        let constraint: Option<Arc<TokenConstraint>> =
            constraint.map(|constraint| Arc::new(constraint));
        // The tokens the output generates are the grammar's input: each
        // emission is committed, and the next mask consumes them, as the
        // Python's exchange does.
        let committing = constraint.clone();
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
                    if let Some(constraint) = committing.as_ref() {
                        if let Err(error) = constraint.commit(&tokens_event.tokens) {
                            call.record_callback_error(format!("constraint: {error}"));
                            return;
                        }
                    }
                    collector
                        .lock()
                        .expect("token lock")
                        .extend_from_slice(&tokens_event.tokens);
                }
            }
        });
        let request = self.generation_request(prepared, max_output_tokens, constraint);
        let call = self.client.submit(&request, Some(on_event), None)?;
        Ok((call, tokens, first_batch))
    }

    /// The output constraint one prepared request generates under: the
    /// grammar its tool policy or response format builds, compiled over
    /// the model's grammar tokenizer. `None` for an unconstrained output.
    pub fn constraint(
        &self,
        prepared: &Prepared,
    ) -> Result<Option<TokenConstraint>, PipelineError> {
        if !prepared.constrained() {
            return Ok(None);
        }
        let factory = self.constraints.as_ref().ok_or_else(|| {
            PipelineError::new(
                500,
                "the request is constrained but the model's grammar tokenizer is unavailable",
            )
        })?;
        let grammar = match (&prepared.tool_policy, &prepared.response_format) {
            (Some(policy), response_schema) => {
                let response_schema = response_schema.as_ref().map(|format| &format.schema);
                tool_grammar(policy, prepared.thinking, response_schema, &self.chat_format)
                    .map_err(|error| PipelineError {
                        status: error.status,
                        message: error.message,
                        code: None,
                    })?
            }
            (None, Some(format)) => json_grammar(&format.schema, prepared.thinking, &self.chat_format),
            (None, None) => return Ok(None),
        };
        let constraint = factory
            .create(&Grammar::List(grammar))
            .map_err(|error| PipelineError::with_code(400, error.0, "invalid_request_error"))?;
        Ok(Some(constraint))
    }

    /// One complete (non-streaming) Chat response: the engine's output
    /// collected and assembled as OpenAI's dialect answers.
    pub fn complete(
        &self,
        _body: &Value,
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
            .map_err(|error| PipelineError { status: 500, message: error.0, code: None })?;
        // The output's reasoning is split from its answer before the
        // projector reads it, as the Python's exchange does: the answer is
        // what a call's tags are read from, and the reasoning is what the
        // message reports as `reasoning_content`.
        let mut splitter = ReasoningSplitter::new(
            prepared.thinking,
            Some(&self.chat_format),
            prepared.tool_policy.is_some(),
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
        // A chat answer is parsed as it arrives whether it streams or not:
        // its calls where a tool policy stands, and in any answer, the
        // reasoning the model writes after its own ended is dropped.
        let mut projector = match &prepared.tool_policy {
            Some(policy) => StreamingToolCallProjector::new(
                Box::new(crate::tool_schema::ToolParameterTypes::new(policy.clone())),
                1,
                &self.chat_format,
                prepared.response_format.is_some(),
            ),
            None => StreamingToolCallProjector::without_policy(1, &self.chat_format),
        };
        projector.put(&answer);
        let (content, calls, _owed) = projector.finish(false);
        // An answer without calls is validated against the response format
        // only when the model ended it.
        let ended = done.reason != crate::protocol::FinishReason::Length;
        if ended && calls.is_empty() {
            if let Some(format) = &prepared.response_format {
                crate::tool_schema::validate_response_content(&content, Some(&format.validator))
                    .map_err(|error| {
                        PipelineError::with_code(error.status, error.message, &error.code)
                    })?;
            }
        }
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
