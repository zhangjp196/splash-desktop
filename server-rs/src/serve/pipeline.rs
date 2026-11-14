//! One request's path: normalize the messages, render the chat template,
//! tokenize the prompt, submit the wire frame to the engine, and read the
//! output through the streamer and the projector.

use std::collections::{BTreeMap, HashMap};
use std::io::Read;
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
use crate::output::{StopSearch, MAX_STOP_CHARACTERS};
use crate::protocol::{ConstraintMode, ImageSpan, SamplingParameters};
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
    /// Per-request log probabilities collected from engine events.
    pub logprobs: Arc<Mutex<HashMap<u64, Vec<crate::protocol::LogprobValues>>>>,
    pub compaction: Option<Value>,
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

/// Generation fields stored in the request frame outside of sampling.
#[derive(Debug, Clone)]
pub struct GenerationFrameOptions {
    pub seed: u64,
    pub priority: crate::protocol::RequestPriority,
    pub cache_salt: u128,
    pub flags: u32,
    pub bias_tokens: Vec<u32>,
    pub bias_values: Vec<f32>,
    pub thinking_budget: Option<crate::protocol::ThinkingBudget>,
    pub logprobs: bool,
    pub top_logprobs: u32,
}

impl Default for GenerationFrameOptions {
    fn default() -> Self {
        Self {
            seed: 0,
            priority: crate::protocol::RequestPriority::Foreground,
            cache_salt: 0,
            flags: 0,
            bias_tokens: Vec::new(),
            bias_values: Vec::new(),
            thinking_budget: None,
            logprobs: false,
            top_logprobs: 0,
        }
    }
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
        if let EngineError::RequestFailed(failure) = &error {
            if failure.machine_code() == "deadline_exceeded" {
                return PipelineError {
                    status: 504,
                    message: failure.text(),
                    code: Some("request_timeout".into()),
                };
            }
        }
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
    /// Server-wide template effort used when a request leaves it unset.
    pub default_reasoning_effort: Option<String>,
    /// Token sequence used to close a thinking block when its budget ends.
    pub thinking_end_tokens: Option<Vec<u32>>,
    /// Isolated PDF renderer shared by requests when PDFium is configured.
    pub documents: Option<Arc<crate::documents::DocumentService>>,
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
fn template_kwargs(
    body: &Value,
    default_reasoning_effort: Option<&str>,
) -> Result<BTreeMap<String, Value>, PipelineError> {
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
        None | Some(Value::Null) => default_reasoning_effort,
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

/// Parse the Chat request's sampling controls with the same defaults and
/// numeric bounds as the Python generation options.
pub fn sampling_parameters(body: &Value) -> Result<SamplingParameters, PipelineError> {
    let number =
        |name: &str, default: f64, min: f64, max: f64, range: &str, min_positive: bool| {
            let value = match body.get(name) {
                None => default,
                Some(value) => value.as_f64().ok_or_else(|| {
                    PipelineError::with_code(
                        400,
                        format!("{name} must be a number in {range}"),
                        "invalid_request_error",
                    )
                })?,
            };
            if !value.is_finite()
                || value < min
                || value > max
                || (min_positive && value <= 0.0)
            {
                return Err(PipelineError::with_code(
                    400,
                    format!("{name} must be a number in {range}"),
                    "invalid_request_error",
                ));
            }
            let rounded = value as f32;
            if !rounded.is_finite() || (min_positive && rounded == 0.0) {
                return Err(PipelineError::with_code(
                    400,
                    format!("{name} must be a number in {range}"),
                    "invalid_request_error",
                ));
            }
            Ok(rounded)
        };
    let temperature = number("temperature", 1.0, 0.0, 2.0, "[0, 2]", false)?;
    let temperature = if temperature > 0.0 && temperature < 0.01 {
        0.01
    } else {
        temperature
    };
    let top_p = number("top_p", 0.95, f32::from_bits(1) as f64, 1.0, "(0, 1]", true)?;
    let presence_penalty = number("presence_penalty", 0.0, -2.0, 2.0, "[-2, 2]", false)?;
    let frequency_penalty = number("frequency_penalty", 0.0, -2.0, 2.0, "[-2, 2]", false)?;
    let repetition_penalty = number(
        "repetition_penalty",
        1.0,
        f32::from_bits(1) as f64,
        f32::MAX as f64,
        "a positive number",
        true,
    )?;
    let min_p = number("min_p", 0.0, 0.0, 1.0, "[0, 1]", false)?;
    let top_k = match body.get("top_k") {
        None => 20,
        Some(value) => match value.as_i64() {
            Some(-1) => 0,
            Some(value) if value >= 0 => value.min(u32::MAX as i64) as u32,
            _ => match value.as_u64() {
                Some(value) => value.min(u32::MAX as u64) as u32,
                None => {
                    return Err(PipelineError::with_code(
                        400,
                        "top_k must be 0 or -1 (disabled) or a positive integer",
                        "invalid_request_error",
                    ));
                }
            },
        },
    };
    Ok(SamplingParameters {
        temperature,
        top_p,
        top_k,
        presence_penalty,
        frequency_penalty,
        repetition_penalty,
        min_p,
    })
}

/// The output budget is limited by the context the rendered prompt leaves,
/// while still rejecting zero and non-integer request values.
pub fn output_budget(
    body: &Value,
    prompt_tokens: usize,
    max_context_tokens: u32,
) -> Result<u32, PipelineError> {
    let maximum = max_context_tokens as usize;
    if prompt_tokens >= maximum {
        return Err(PipelineError::with_code(
            400,
            "prompt leaves no room for output",
            "context_length_exceeded",
        ));
    }
    let remaining = maximum - prompt_tokens;
    let requested = body
        .get("max_completion_tokens")
        .filter(|value| !value.is_null())
        .or_else(|| body.get("max_tokens").filter(|value| !value.is_null()));
    let Some(requested) = requested else {
        return Ok(remaining.min(u32::MAX as usize) as u32);
    };
    let requested = requested.as_u64().filter(|value| *value > 0).ok_or_else(|| {
        let field = if body.get("max_completion_tokens").is_some_and(|value| !value.is_null()) {
            "max_completion_tokens"
        } else {
            "max_tokens"
        };
        PipelineError::with_code(
            400,
            format!("{field} must be a positive integer"),
            "invalid_request_error",
        )
    })?;
    Ok(requested.min(remaining as u64).min(u32::MAX as u64) as u32)
}

/// Parse OpenAI's optional stop string or list of up to four stop strings.
pub fn stop_sequences(body: &Value) -> Result<Vec<String>, PipelineError> {
    let Some(stop) = body.get("stop") else {
        return Ok(Vec::new());
    };
    if stop.is_null() || stop.as_array().is_some_and(Vec::is_empty) {
        return Ok(Vec::new());
    }
    let stops = match stop {
        Value::String(value) if !value.is_empty() => vec![value.clone()],
        Value::Array(values)
            if (1..=4).contains(&values.len())
                && values.iter().all(|value| {
                    value.as_str().is_some_and(|value| !value.is_empty())
                }) =>
        {
            values
                .iter()
                .map(|value| value.as_str().expect("validated string").to_owned())
                .collect()
        }
        _ => {
            return Err(PipelineError::with_code(
                400,
                "stop must be a string or up to four strings",
                "invalid_request_error",
            ));
        }
    };
    if stops
        .iter()
        .any(|stop| stop.chars().count() > MAX_STOP_CHARACTERS)
    {
        return Err(PipelineError::with_code(
            400,
            format!("each of stop must be at most {MAX_STOP_CHARACTERS} characters"),
            "invalid_request_error",
        ));
    }
    Ok(stops)
}

/// Read per-request engine controls which are independent of sampling.
pub fn generation_frame_options(
    body: &Value,
    prepared: &Prepared,
    chat_format: &ChatFormat,
    default_thinking_token_budget: Option<u32>,
) -> Result<GenerationFrameOptions, PipelineError> {
    use crate::protocol::{request_flags, RequestPriority, ThinkingBudget};

    let logprobs = match body.get("logprobs") {
        None => false,
        Some(Value::Bool(value)) => *value,
        _ => return Err(PipelineError::with_code(
            400,
            "logprobs must be a boolean",
            "invalid_request_error",
        )),
    };
    let top_logprobs = match body.get("top_logprobs") {
        None => 0,
        Some(value) => value.as_u64().filter(|count| *count <= 20).ok_or_else(|| {
            PipelineError::with_code(
                400,
                "top_logprobs must be an integer from 0 to 20",
                "invalid_request_error",
            )
        })? as u32,
    };
    if top_logprobs > 0 && !logprobs {
        return Err(PipelineError::with_code(
            400,
            "top_logprobs requires logprobs",
            "invalid_request_error",
        ));
    }
    if logprobs {
        if body.get("stream").and_then(Value::as_bool).unwrap_or(false) {
            return Err(PipelineError::with_code(
                400,
                "streaming logprobs are not supported",
                "invalid_request_error",
            ));
        }
        if prepared.thinking {
            return Err(PipelineError::with_code(
                400,
                "logprobs require reasoning off; set reasoning_effort to none",
                "invalid_request_error",
            ));
        }
        if prepared.constrained() || prepared.tools.is_some() {
            return Err(PipelineError::with_code(
                400,
                "logprobs cannot be combined with tools, structured output, grammar or regex",
                "invalid_request_error",
            ));
        }
    }

    let seed = match body.get("seed") {
        None => random_seed()?,
        Some(value) => match value.as_i64() {
            Some(-1) => random_seed()?,
            Some(value) => value as u64,
            None => value.as_u64().ok_or_else(|| {
                PipelineError::with_code(400, "seed must be a 64-bit integer", "invalid_request_error")
            })?,
        },
    };
    let priority = match body.get("priority") {
        None => RequestPriority::Normal,
        Some(Value::String(value)) if value == "foreground" => RequestPriority::Foreground,
        Some(Value::String(value)) if value == "normal" => RequestPriority::Normal,
        Some(Value::String(value)) if value == "background" => RequestPriority::Background,
        _ => {
            return Err(PipelineError::with_code(
                400,
                "priority must be foreground, normal, or background",
                "invalid_request_error",
            ));
        }
    };
    let mut cache_salt = match body.get("cache_salt").filter(|value| !value.is_null()) {
        None => 0,
        Some(Value::String(value)) if !value.is_empty() => stable_cache_salt(value),
        _ => {
            return Err(PipelineError::with_code(
                400,
                "cache_salt must be a non-empty string",
                "invalid_request_error",
            ));
        }
    };
    // A logprobs request needs the logits row from its own cold prefill;
    // a shared cached prefix does not retain that row.
    if logprobs && cache_salt == 0 {
        cache_salt = (u128::from(random_seed()?) << 64) | u128::from(random_seed()?);
        if cache_salt == 0 { cache_salt = 1; }
    }
    let ignore_eos = match body.get("ignore_eos") {
        None => false,
        Some(Value::Bool(value)) => *value,
        _ => {
            return Err(PipelineError::with_code(
                400,
                "ignore_eos must be a boolean",
                "invalid_request_error",
            ));
        }
    };
    if ignore_eos && prepared.constrained() {
        return Err(PipelineError::with_code(
            400,
            "ignore_eos cannot be combined with constrained output",
            "invalid_request_error",
        ));
    }
    let mut bias = BTreeMap::new();
    if let Some(raw) = body.get("logit_bias").filter(|value| !value.is_null()) {
        let Some(object) = raw.as_object() else {
            return Err(PipelineError::with_code(
                400,
                "logit_bias must be an object",
                "invalid_request_error",
            ));
        };
        if object.len() > 256 {
            return Err(PipelineError::with_code(
                400,
                "logit_bias takes at most 256 entries",
                "invalid_request_error",
            ));
        }
        for (key, value) in object {
            let token = key.parse::<u32>().ok().filter(|token| token.to_string() == *key)
                .filter(|token| *token < chat_format.vocabulary)
                .ok_or_else(|| PipelineError::with_code(
                    400,
                    format!("logit_bias token {key} is outside the vocabulary"),
                    "invalid_request_error",
                ))?;
            let bias_value = value.as_f64().filter(|value| value.is_finite() && (-100.0..=100.0).contains(value))
                .ok_or_else(|| PipelineError::with_code(
                    400,
                    "logit_bias values must be numbers in [-100, 100]",
                    "invalid_request_error",
                ))? as f32;
            bias.insert(token, bias_value);
        }
    }
    let requested_budget = match body.get("thinking_token_budget") {
        None | Some(Value::Null) => default_thinking_token_budget.map(u64::from),
        Some(value) => match value.as_i64() {
            Some(-1) => default_thinking_token_budget
                .filter(|budget| *budget == crate::protocol::NO_THINKING_BUDGET)
                .map(u64::from),
            Some(value) if value >= 0 => Some(value as u64),
            Some(_) => {
                return Err(PipelineError::with_code(
                    400,
                    "thinking_token_budget must be a non-negative integer or -1 for no cap",
                    "invalid_request_error",
                ));
            }
            None => Some(value.as_u64().ok_or_else(|| PipelineError::with_code(
                400,
                "thinking_token_budget must be a non-negative integer or -1 for no cap",
                "invalid_request_error",
            ))?),
        },
    };
    let thinking_budget = match requested_budget {
        Some(budget) if budget > u32::MAX as u64 => {
            return Err(PipelineError::with_code(
                400,
                "thinking_token_budget is too large",
                "invalid_request_error",
            ));
        }
        Some(budget) if prepared.thinking => Some(ThinkingBudget {
            tokens: budget as u32,
            end: vec![chat_format.think_end_id],
            also_ends: if prepared.tool_policy.is_some() && chat_format.calls_end_reasoning() {
                chat_format.call_open_id
            } else {
                0
            },
        }),
        _ => None,
    };
    Ok(GenerationFrameOptions {
        seed,
        priority,
        cache_salt,
        flags: if ignore_eos { request_flags::IGNORE_END_OF_SEQUENCE } else { 0 },
        bias_tokens: bias.keys().copied().collect(),
        bias_values: bias.values().copied().collect(),
        thinking_budget,
        logprobs,
        top_logprobs,
    })
}

/// Match `hashlib.blake2b(value, digest_size=16, person=b"splash-salt")`.
fn stable_cache_salt(value: &str) -> u128 {
    const IV: [u64; 8] = [
        0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
        0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
        0x510e527fade682d1, 0x9b05688c2b3e6c1f,
        0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
    ];
    const SIGMA: [[usize; 16]; 12] = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
        [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
        [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
        [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
        [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
        [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
        [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
        [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
        [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    ];
    fn mix(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
        v[d] = (v[d] ^ v[a]).rotate_right(32);
        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(24);
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
        v[d] = (v[d] ^ v[a]).rotate_right(16);
        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(63);
    }
    fn compress(h: &mut [u64; 8], block: &[u8; 128], count: u128, last: bool) {
        let mut m = [0u64; 16];
        for (word, bytes) in m.iter_mut().zip(block.chunks_exact(8)) {
            *word = u64::from_le_bytes(bytes.try_into().expect("eight byte word"));
        }
        let mut v = [0u64; 16];
        v[..8].copy_from_slice(h);
        v[8..].copy_from_slice(&IV);
        v[12] ^= count as u64;
        v[13] ^= (count >> 64) as u64;
        if last { v[14] = !v[14]; }
        for schedule in SIGMA {
            mix(&mut v, 0, 4, 8, 12, m[schedule[0]], m[schedule[1]]);
            mix(&mut v, 1, 5, 9, 13, m[schedule[2]], m[schedule[3]]);
            mix(&mut v, 2, 6, 10, 14, m[schedule[4]], m[schedule[5]]);
            mix(&mut v, 3, 7, 11, 15, m[schedule[6]], m[schedule[7]]);
            mix(&mut v, 0, 5, 10, 15, m[schedule[8]], m[schedule[9]]);
            mix(&mut v, 1, 6, 11, 12, m[schedule[10]], m[schedule[11]]);
            mix(&mut v, 2, 7, 8, 13, m[schedule[12]], m[schedule[13]]);
            mix(&mut v, 3, 4, 9, 14, m[schedule[14]], m[schedule[15]]);
        }
        for index in 0..8 { h[index] ^= v[index] ^ v[index + 8]; }
    }
    let mut h = IV;
    h[0] ^= 0x0101_0000 ^ 16; // digest length 16, no key, fanout 1, depth 1
    h[6] ^= u64::from_le_bytes(*b"splash-s");
    h[7] ^= u64::from_le_bytes([b'a', b'l', b't', 0, 0, 0, 0, 0]);
    let bytes = value.as_bytes();
    let mut consumed = 0usize;
    while bytes.len().saturating_sub(consumed) > 128 {
        let block: &[u8; 128] = bytes[consumed..consumed + 128].try_into().expect("full block");
        compress(&mut h, block, (consumed + 128) as u128, false);
        consumed += 128;
    }
    let tail = &bytes[consumed..];
    let mut final_block = [0u8; 128];
    final_block[..tail.len()].copy_from_slice(tail);
    compress(&mut h, &final_block, bytes.len() as u128, true);
    let mut digest = [0u8; 16];
    digest[..8].copy_from_slice(&h[0].to_le_bytes());
    digest[8..].copy_from_slice(&h[1].to_le_bytes());
    u128::from_le_bytes(digest)
}

fn random_seed() -> Result<u64, PipelineError> {
    let mut bytes = [0u8; 8];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut source| source.read_exact(&mut bytes))
        .map_err(|error| PipelineError {
            status: 500,
            message: format!("could not read random seed: {error}"),
            code: None,
        })?;
    Ok(u64::from_le_bytes(bytes))
}

/// Truncate decoded output before its earliest stop sequence.
fn truncate_at_stop(text: &mut String, stop_sequences: &[String]) -> bool {
    if stop_sequences.is_empty() {
        return false;
    }
    let stops: Vec<&str> = stop_sequences.iter().map(String::as_str).collect();
    let mut search = StopSearch::new(&stops);
    let Some(found) = search.search(text) else {
        return false;
    };
    let keep = text.chars().count().saturating_sub(found.back);
    *text = text.chars().take(keep).collect();
    true
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

    pub fn contains_pdf_files(body: &Value) -> bool {
        body.get("messages").and_then(Value::as_array).is_some_and(|messages| messages.iter().any(|message| {
            message.pointer("/content").and_then(Value::as_array).is_some_and(|parts| {
                parts.iter().any(|part| part.get("type").and_then(Value::as_str) == Some("file"))
            })
        }))
    }

    /// Expand PDF parts ahead of prompt-preparation admission. PDF rendering
    /// uses the request deadline but does not hold a preparation slot.
    pub fn expand_documents(&self, body: &Value, timeout: Option<std::time::Duration>) -> Result<Value, PipelineError> {
        let incoming_messages = body.get("messages")
            .ok_or_else(|| PipelineError::new(400, "messages is required"))?;
        let documents = self.documents.as_ref().ok_or_else(|| PipelineError {
            status: 503,
            message: "PDF support is unavailable; start with --pdfium PATH".into(),
            code: Some("document_unavailable".into()),
        })?;
        let vision = self.readiness().map(|ready| ready.vision).unwrap_or(false);
        let mut budget = crate::documents::DocumentBudget::from_timeout(timeout);
        let expanded = documents.expand_messages(incoming_messages, &mut budget, vision).map_err(|error| PipelineError {
            status: error.status, message: error.message, code: Some(error.code),
        })?;
        let mut body = body.clone();
        body["messages"] = expanded;
        Ok(body)
    }

    fn prepare_request_images(
        &self,
        messages: &[Value],
        max_context: u64,
        check_context: bool,
        deadline: &mut dyn FnMut() -> Result<Option<std::time::Duration>, crate::support::errors::ApiError>,
    ) -> Result<Option<RequestImages>, PipelineError> {
        if !self.readiness().map(|ready| ready.vision).unwrap_or(false) {
            return Ok(None);
        }
        let latency = self.client.latency_metrics();
        let _timer = latency.as_ref().map(|metrics| metrics.timer("images"));
        let mut check_deadline = || deadline().map(|_| ());
        prepare_images(
            &self.images, messages, self.max_image_pixels, max_context,
            check_context, &mut check_deadline,
        ).map(Some).map_err(|error| PipelineError {
            status: error.status, message: error.message, code: Some(error.code),
        })
    }

    fn render_prompt_messages(
        &self,
        source: &str,
        messages: &Value,
        tools: Option<&Vec<Value>>,
        kwargs: &BTreeMap<String, Value>,
        add_generation_prompt: bool,
        template_error_status: u16,
    ) -> Result<(String, Vec<u32>, Option<Vec<usize>>), PipelineError> {
        let latency = self.client.latency_metrics();
        let tools_value = tools.map(|tools| Value::Array(tools.clone()));
        let render = |template_source: &str| -> Result<String, crate::templates::ChatTemplateError> {
            let _timer = latency.as_ref().map(|metrics| metrics.timer("template"));
            ChatTemplateRenderer::new(template_source)?.render(
                messages, tools_value.as_ref(), None, add_generation_prompt, kwargs,
            )
        };
        let carries_images = messages.as_array().is_some_and(|messages| messages.iter().any(|message| {
            message.get("content").and_then(Value::as_array).is_some_and(|parts| {
                parts.iter().any(|part| part.get("type").and_then(Value::as_str) == Some("image_url"))
            })
        }));
        if carries_images && self.readiness().map(|ready| ready.vision).unwrap_or(false) {
            let tokenizer = self.tokenizer.clone();
            let tokenization_latency = latency.clone();
            let (tokens, positions, rendered) = render_image_tokens(
                source, &render, &|text| {
                    let _timer = tokenization_latency.as_ref().map(|metrics| metrics.timer("tokenization"));
                    tokenizer.encode_with_offsets(text, false)
                },
            ).map_err(|error| PipelineError { status: error.status, message: error.message, code: Some(error.code) })?;
            Ok((rendered, tokens, Some(positions)))
        } else {
            let rendered = render(source).map_err(|error| PipelineError { status: template_error_status, message: error.to_string(), code: None })?;
            let _timer = latency.as_ref().map(|metrics| metrics.timer("tokenization"));
            let tokens = self.prompt_tokenizer.encode(&rendered);
            Ok((rendered, tokens, None))
        }
    }

    /// Prepare one Chat request's prompt: the messages normalized, the
    /// template rendered, the prompt tokenized, the images prepared and
    /// their placeholders widened.
    pub fn prepare(
        &self,
        body: &Value,
        deadline: &mut dyn FnMut() -> Result<(), crate::support::errors::ApiError>,
    ) -> Result<Prepared, PipelineError> {
        let mut remaining = || {
            deadline()?;
            Ok(None)
        };
        self.prepare_with_compaction(body, &mut remaining, None, None, None, true)
    }

    /// Prepare a Chat prompt and compact older turns when configured.
    pub fn prepare_with_compaction(
        &self,
        body: &Value,
        deadline: &mut dyn FnMut() -> Result<Option<std::time::Duration>, crate::support::errors::ApiError>,
        compact_at: Option<crate::serve::compaction::CompactAt>,
        compact_keep: Option<u32>,
        compact_memory: Option<&std::path::Path>,
        add_generation_prompt: bool,
    ) -> Result<Prepared, PipelineError> {
        self.prepare_with_compaction_and_namespaces(body, deadline, compact_at, compact_keep, compact_memory, add_generation_prompt, None)
    }

    pub fn prepare_with_compaction_and_namespaces(
        &self,
        body: &Value,
        deadline: &mut dyn FnMut() -> Result<Option<std::time::Duration>, crate::support::errors::ApiError>,
        compact_at: Option<crate::serve::compaction::CompactAt>,
        compact_keep: Option<u32>,
        compact_memory: Option<&std::path::Path>,
        add_generation_prompt: bool,
        tool_namespaces: Option<&Value>,
    ) -> Result<Prepared, PipelineError> {
        let expanded_body = if Self::contains_pdf_files(body) {
            let timeout = deadline().map_err(|error| PipelineError {
                status: error.status, message: error.message, code: Some(error.code),
            })?;
            Some(self.expand_documents(body, timeout)?)
        } else { None };
        let body = expanded_body.as_ref().unwrap_or(body);
        let incoming_messages = body
            .get("messages")
            .ok_or_else(|| PipelineError::new(400, "messages is required"))?;
        let ready = self.readiness();
        let vision = ready.map(|ready| ready.vision).unwrap_or(false);
        let max_context = ready
            .map(|ready| u64::from(ready.max_context_tokens))
            .unwrap_or(u64::MAX);
        let normalized = normalize_messages(incoming_messages, vision)?;
        let mut templated = crate::chat_messages::template_messages(&normalized);
        let messages_value = Value::Array(templated.clone());
        // The tools and the response format read before the prompt
        // renders, as the Python's `_prepare_prompt` runs: a refusal here
        // never renders documents or builds validators.
        let (offered, tool_policy) = normalize_tools(
            body.get("tools"),
            body.get("tool_choice"),
            body.get("parallel_tool_calls"),
            tool_namespaces,
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
        let kwargs = template_kwargs(body, self.default_reasoning_effort.as_deref())?;
        // Images belong to the message batch selected after compaction.
        // Candidate renders prepare temporary batches that drop immediately;
        // the final batch remains owned by `Prepared` through inference.
        let mut images = self.prepare_request_images(&templated, max_context, true, deadline)?;
        let has_images = images.as_ref().is_some_and(|batch| !batch.is_empty());
        let mut rendered_prompt = self.render_prompt_messages(&source, &messages_value, tools.as_ref(), &kwargs, add_generation_prompt, 500)?;
        let mut rendered = rendered_prompt.0;
        let mut prompt_tokens = rendered_prompt.1;
        let mut positions = rendered_prompt.2;
        let mut compaction_report = None;
        if let Some(setting) = compact_at {
            let context = u32::try_from(max_context).unwrap_or(u32::MAX);
            let reply = body
                .get("max_completion_tokens")
                .filter(|value| !value.is_null())
                .or_else(|| body.get("max_tokens").filter(|value| !value.is_null()))
                .and_then(Value::as_u64)
                .filter(|value| *value > 0)
                .map(|value| value.min(u32::MAX as u64) as u32)
                .unwrap_or(crate::serve::compaction::COMPACT_DEFAULT_REPLY_TOKENS);
            if crate::serve::compaction::should_compact(prompt_tokens.len(), reply, context, setting) {
                let keep = crate::serve::compaction::keep_budget(context, compact_keep);
                let original_had_images = has_images;
                if original_had_images { drop(images.take()); }
                let mut render_error = None;
                let mut count_candidate = |candidate: &[Value]| -> usize {
                    if let Err(error) = deadline() {
                        render_error = Some(PipelineError::new(error.status, error.message));
                        return usize::MAX;
                    }
                    let candidate_value = Value::Array(candidate.to_vec());
                    let mut trial_images = match self.prepare_request_images(candidate, max_context, false, deadline) {
                        Ok(images) => images,
                        Err(error) => { render_error = Some(error); return usize::MAX; }
                    };
                    let result = self.render_prompt_messages(&source, &candidate_value, tools.as_ref(), &kwargs, add_generation_prompt, 400);
                    let count = match result {
                        Ok((_text, tokens, _positions)) => tokens.len(),
                        Err(error) => { render_error = Some(error); usize::MAX }
                    };
                    drop(trial_images.take());
                    count
                };
                let cut = crate::serve::compaction::choose_cut(&templated, keep, &mut count_candidate);
                drop(count_candidate);
                if let Some(error) = render_error { return Err(error); }
                let system_end = crate::serve::compaction::leading_system_end(&templated);
                if cut > system_end && cut < templated.len() {
                    let prompt_tokens_before = prompt_tokens.len();
                    let removed_messages = cut - system_end;
                    let transcript = crate::serve::compaction::transcript(&templated[system_end..cut]);
                    let digest = crate::serve::compaction::removed_digest(&transcript);
                    let key = crate::serve::compaction::conversation_key(&templated);
                    let note = match compact_memory.and_then(|directory| crate::serve::compaction::read_note(directory, &key, &digest)) {
                        Some(note) => note,
                        None => {
                            let note = self.summarize_compaction(&transcript, deadline, context)?;
                            if let Some(directory) = compact_memory { crate::serve::compaction::write_note(directory, &key, &digest, &note); }
                            note
                        }
                    };
                    templated = crate::serve::compaction::compacted_messages(&templated, cut, &note);
                    let compacted_value = Value::Array(templated.clone());
                    images = self.prepare_request_images(&templated, max_context, true, deadline)?;
                    rendered_prompt = self.render_prompt_messages(&source, &compacted_value, tools.as_ref(), &kwargs, add_generation_prompt, 400)?;
                    rendered = rendered_prompt.0;
                    prompt_tokens = rendered_prompt.1;
                    positions = rendered_prompt.2;
                    compaction_report = Some(json!({"note":note,"removed_messages":removed_messages,"prompt_tokens_before":prompt_tokens_before,"prompt_tokens_after":prompt_tokens.len()}));
                } else if original_had_images {
                    // The selected cut can be the identity. Reacquire the
                    // original owner after trial batches have been released.
                    images = self.prepare_request_images(&templated, max_context, true, deadline)?;
                }
            }
        }
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
            logprobs: Arc::new(Mutex::new(HashMap::new())),
            compaction: compaction_report,
        })
    }

    fn summarize_compaction(
        &self,
        transcript: &str,
        deadline: &mut dyn FnMut() -> Result<Option<std::time::Duration>, crate::support::errors::ApiError>,
        context: u32,
    ) -> Result<String, PipelineError> {
        let mut transcript = transcript.to_owned();
        let fit_limit = context
            .saturating_sub(crate::serve::compaction::COMPACT_RESERVE_TOKENS)
            .saturating_sub(crate::serve::compaction::COMPACT_MAX_NOTE_TOKENS)
            .max(256) as usize;
        for _ in 0..8 {
            let candidate = format!("{}\n\n{transcript}", crate::serve::compaction::COMPACT_INSTRUCTION);
            let tokens = self.tokenizer.encode(&candidate, false)
                .map_err(|error| PipelineError::new(500, error.to_string()))?;
            if tokens.len() <= fit_limit { break; }
            let keep = ((transcript.chars().count() as f64 * fit_limit as f64
                / tokens.len().max(1) as f64 * 0.9) as usize).max(2);
            let head_count = (keep * 3 / 5).max(1);
            let tail_count = keep.saturating_sub(head_count).max(1);
            let head: String = transcript.chars().take(head_count).collect();
            let tail: String = transcript.chars().rev().take(tail_count).collect::<String>().chars().rev().collect();
            transcript = format!("{head}\n\n[...]\n\n{tail}");
        }

        let content = format!("{}\n\n{transcript}", crate::serve::compaction::COMPACT_INSTRUCTION);
        let messages = json!([{"role": "user", "content": content}]);
        let summary_template = self.templates.select(None, None)
            .map_err(|error| PipelineError::new(400, error.to_string()))?;
        let summary_kwargs = BTreeMap::from([
            ("enable_thinking".to_owned(), Value::Bool(false)),
            ("reasoning_effort".to_owned(), json!("none")),
        ]);
        let prompt = ChatTemplateRenderer::new(&summary_template)
            .and_then(|renderer| renderer.render(&messages, None, None, true, &summary_kwargs))
            .map_err(|error| PipelineError::new(400, error.to_string()))?;
        let prompt_tokens = self.prompt_tokenizer.encode(&prompt);
        let max_new = crate::serve::compaction::COMPACT_MAX_NOTE_TOKENS.min(
            context.saturating_sub(prompt_tokens.len().min(u32::MAX as usize) as u32)
                .saturating_sub(crate::serve::compaction::COMPACT_RESERVE_TOKENS).max(1),
        );
        let remaining = deadline().map_err(|error| PipelineError::new(error.status, error.message))?;
        let output = Arc::new(Mutex::new(Vec::new()));
        let output_callback = output.clone();
        let on_event: crate::engine::EventCallback = Arc::new(move |call, event| {
            if let crate::protocol::EngineEvent::Tokens(tokens) = event {
                if call.request_id == tokens.request_id {
                    output_callback.lock().expect("compaction token lock")
                        .extend_from_slice(&tokens.tokens);
                }
            }
        });
        let request = GenerationRequest {
            frame: crate::protocol::RequestFrame {
                priority: crate::protocol::RequestPriority::Background,
                logical_max_output_tokens: max_new,
                prompt_tokens,
                sampling: SamplingParameters { temperature: 0.0, ..SamplingParameters::default() },
                seed: 0,
                ..crate::protocol::RequestFrame::default()
            },
            deadline: remaining,
            mask_provider: None,
        };
        let call = self.client.submit(&request, Some(on_event), None)?;
        if let Err(error) = call.result(remaining) {
            self.client.cancel(&call);
            return Err(PipelineError::from(error));
        }
        deadline().map_err(|error| PipelineError::new(error.status, error.message))?;
        let generated = output.lock().expect("compaction token lock").clone();
        self.tokenizer.decode(&generated, true)
            .map(|note| note.trim().to_owned())
            .map_err(|error| PipelineError::new(500, error.to_string()))
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
        self.generation_request_until(prepared, max_output_tokens, constraint, None)
    }

    /// Build a generation request carrying the remaining wall budget from
    /// the HTTP request's absolute deadline.
    pub fn generation_request_until(
        &self,
        prepared: &Prepared,
        max_output_tokens: u32,
        constraint: Option<Arc<TokenConstraint>>,
        deadline: Option<std::time::Duration>,
    ) -> GenerationRequest {
        self.generation_request_with_sampling(
            prepared,
            max_output_tokens,
            constraint,
            deadline,
            SamplingParameters::default(),
        )
    }

    /// Build a generation request with caller-selected sampling parameters.
    pub fn generation_request_with_sampling(
        &self,
        prepared: &Prepared,
        max_output_tokens: u32,
        constraint: Option<Arc<TokenConstraint>>,
        deadline: Option<std::time::Duration>,
        sampling: SamplingParameters,
    ) -> GenerationRequest {
        self.generation_request_with_frame_options(
            prepared,
            max_output_tokens,
            constraint,
            deadline,
            sampling,
            GenerationFrameOptions::default(),
        )
    }

    /// Build a generation request with all parsed engine request fields.
    pub fn generation_request_with_frame_options(
        &self,
        prepared: &Prepared,
        max_output_tokens: u32,
        constraint: Option<Arc<TokenConstraint>>,
        deadline: Option<std::time::Duration>,
        sampling: SamplingParameters,
        options: GenerationFrameOptions,
    ) -> GenerationRequest {
        let mut thinking_budget = options.thinking_budget;
        if let (Some(budget), Some(end)) =
            (thinking_budget.as_mut(), self.thinking_end_tokens.as_ref())
        {
            budget.end = end.clone();
        }
        GenerationRequest {
            frame: crate::protocol::RequestFrame {
                priority: options.priority,
                logical_max_output_tokens: max_output_tokens,
                prompt_tokens: prepared.prompt_tokens.clone(),
                sampling,
                seed: options.seed,
                cache_salt: options.cache_salt,
                flags: options.flags | if options.logprobs {
                    crate::protocol::request_flags::LOGPROBS
                } else {
                    0
                },
                top_logprobs: options.top_logprobs,
                bias_tokens: options.bias_tokens,
                bias_values: options.bias_values,
                thinking_budget,
                image_spans: prepared.image_spans.clone(),
                image_pixels: prepared.image_pixels.clone(),
                constraint: if constraint.is_some() {
                    ConstraintMode::TokenMask
                } else {
                    ConstraintMode::None
                },
                ..crate::protocol::RequestFrame::default()
            },
            deadline,
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
        self.submit_until(prepared, max_output_tokens, None)
    }

    /// Submit a prepared request with the remaining HTTP deadline.
    pub fn submit_until(
        &self,
        prepared: &Prepared,
        max_output_tokens: u32,
        deadline: Option<std::time::Duration>,
    ) -> Result<
        (
            Arc<crate::engine::Call>,
            Arc<Mutex<Vec<u32>>>,
            Arc<std::sync::atomic::AtomicU64>,
        ),
        PipelineError,
    > {
        self.submit_with_sampling(
            prepared,
            max_output_tokens,
            deadline,
            SamplingParameters::default(),
        )
    }

    /// Submit with the Chat request's sampling options and remaining deadline.
    pub fn submit_with_sampling(
        &self,
        prepared: &Prepared,
        max_output_tokens: u32,
        deadline: Option<std::time::Duration>,
        sampling: SamplingParameters,
    ) -> Result<
        (
            Arc<crate::engine::Call>,
            Arc<Mutex<Vec<u32>>>,
            Arc<std::sync::atomic::AtomicU64>,
        ),
        PipelineError,
    > {
        self.submit_with_frame_options(
            prepared,
            max_output_tokens,
            deadline,
            sampling,
            GenerationFrameOptions::default(),
        )
    }

    pub fn submit_with_frame_options(
        &self,
        prepared: &Prepared,
        max_output_tokens: u32,
        deadline: Option<std::time::Duration>,
        sampling: SamplingParameters,
        options: GenerationFrameOptions,
    ) -> Result<
        (
            Arc<crate::engine::Call>,
            Arc<Mutex<Vec<u32>>>,
            Arc<std::sync::atomic::AtomicU64>,
        ),
        PipelineError,
    > {
        self.submit_with_frame_options_and_http_start(
            prepared, max_output_tokens, deadline, sampling, options, None,
        )
    }

    pub fn submit_with_frame_options_and_http_start(
        &self,
        prepared: &Prepared,
        max_output_tokens: u32,
        deadline: Option<std::time::Duration>,
        sampling: SamplingParameters,
        options: GenerationFrameOptions,
        http_started_at: Option<std::time::Instant>,
    ) -> Result<
        (
            Arc<crate::engine::Call>,
            Arc<Mutex<Vec<u32>>>,
            Arc<std::sync::atomic::AtomicU64>,
        ),
        PipelineError,
    > {
        self.submit_with_frame_options_and_http_start_and_diagnostics(
            prepared, max_output_tokens, deadline, sampling, options, http_started_at, None,
        )
    }

    /// Submit a Chat choice and report its native admission event to the
    /// request diagnostics callback. Other dialects use the plain method so
    /// their log format can remain independently compatible.
    pub fn submit_with_frame_options_and_http_start_and_diagnostics(
        &self,
        prepared: &Prepared,
        max_output_tokens: u32,
        deadline: Option<std::time::Duration>,
        sampling: SamplingParameters,
        options: GenerationFrameOptions,
        http_started_at: Option<std::time::Instant>,
        request_start: Option<Arc<crate::diagnostics::RequestLogContext>>,
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
        let collected_logprobs = prepared.logprobs.clone();
        let log_request_start = request_start.clone();
        let constraint = self.constraint(prepared)?;
        let constraint: Option<Arc<TokenConstraint>> =
            constraint.map(|constraint| Arc::new(constraint));
        // The tokens the output generates are the grammar's input: each
        // emission is committed, and the next mask consumes them, as the
        // Python's exchange does.
        let committing = constraint.clone();
        let on_event: crate::engine::EventCallback = Arc::new(move |call, event| {
            match event {
                crate::protocol::EngineEvent::Start(start)
                    if call.request_id == start.request_id =>
                {
                    if let Some(request) = log_request_start.as_ref() {
                        request.admitted(start.matched_prompt_tokens);
                    }
                }
                crate::protocol::EngineEvent::Logprobs(logprobs_event)
                    if call.request_id == logprobs_event.request_id =>
                {
                    collected_logprobs.lock().expect("logprobs lock")
                        .entry(call.request_id).or_default()
                        .extend_from_slice(&logprobs_event.values);
                }
                crate::protocol::EngineEvent::Tokens(tokens_event)
                    if call.request_id == tokens_event.request_id =>
                {
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
                _ => {}
            }
        });
        let request = self.generation_request_with_frame_options(
            prepared,
            max_output_tokens,
            constraint,
            deadline,
            sampling,
            options,
        );
        let call = self.client.submit_with_http_start(
            &request, Some(on_event), None, http_started_at,
        )?;
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
        let latency = self.client.latency_metrics();
        let _timer = latency.as_ref().map(|metrics| metrics.timer("grammar"));
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
        self.complete_with_stops(
            _body,
            prepared,
            call,
            tokens,
            first_token_batch_tokens,
            model,
            job,
            &[],
        )
    }

    /// Assemble a complete response, stopping its visible output at the
    /// first requested stop sequence.
    pub fn complete_with_stops(
        &self,
        _body: &Value,
        prepared: &Prepared,
        call: &Arc<crate::engine::Call>,
        tokens: &Arc<Mutex<Vec<u32>>>,
        first_token_batch_tokens: u64,
        model: &str,
        job: &openai::JobInfo,
        stop_sequences: &[String],
    ) -> Result<Value, PipelineError> {
        let done = call.result(None)?;
        let tokens = tokens.lock().expect("token lock").clone();
        let mut text = self
            .tokenizer
            .decode(&tokens, true)
            .map_err(|error| PipelineError { status: 500, message: error.0, code: None })?;
        let stopped = truncate_at_stop(&mut text, stop_sequences);
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
        let ended = stopped || done.reason != crate::protocol::FinishReason::Length;
        if ended && calls.is_empty() {
            if let Some(format) = &prepared.response_format {
                crate::tool_schema::validate_response_content(&content, Some(&format.validator))
                    .map_err(|error| {
                        PipelineError::with_code(error.status, error.message, &error.code)
                    })?;
            }
        }
        let mut result = collected_result(&done, first_token_batch_tokens);
        if stopped {
            result.reason = "stop".into();
        }
        result.metrics = openai::metrics_dict(&result);
        let logprobs = if _body.get("logprobs").and_then(Value::as_bool).unwrap_or(false) {
            let values = prepared.logprobs.lock().expect("logprobs lock")
                .remove(&call.request_id).unwrap_or_default();
            Some(chat_logprobs(&self.tokenizer, &values, &content)?)
        } else {
            None
        };
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
            logprobs,
        };
        let choice = openai::chat_choice(&collected, job, 0);
        let mut response = openai::completion_response(
            model,
            job,
            collected.result.as_ref().expect("set above"),
            vec![choice],
        );
        if let Some(compaction) = &prepared.compaction {
            response["compaction"] = compaction.clone();
        }
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

/// Build OpenAI Chat's token-level probability object and trim entries to
/// the decoded answer after stop-sequence processing.
fn chat_logprobs(
    tokenizer: &FastTokenizer,
    values: &[crate::protocol::LogprobValues],
    content: &str,
) -> Result<Value, PipelineError> {
    let mut entries = Vec::new();
    for value in values {
        let token = tokenizer.decode(&[value.token], true).map_err(|error| {
            PipelineError::new(500, format!("could not decode logprob token: {error}"))
        })?;
        if token.is_empty() { continue; }
        let mut top = Vec::new();
        for (top_token, logprob) in &value.top {
            let text = tokenizer.decode(&[*top_token], false).map_err(|error| {
                PipelineError::new(500, format!("could not decode top logprob token: {error}"))
            })?;
            top.push(json!({
                "token": text,
                "logprob": logprob,
                "bytes": text.as_bytes(),
            }));
        }
        entries.push(json!({
            "token": token,
            "logprob": value.logprob,
            "bytes": token.as_bytes(),
            "top_logprobs": top,
        }));
    }
    let mut joined = String::new();
    let mut kept = Vec::new();
    for entry in entries {
        let token = entry.get("token").and_then(Value::as_str).unwrap_or("");
        let longer = format!("{joined}{token}");
        if !content.starts_with(&longer) { break; }
        joined = longer;
        kept.push(entry);
    }
    Ok(json!({"content": kept, "refusal": null}))
}
