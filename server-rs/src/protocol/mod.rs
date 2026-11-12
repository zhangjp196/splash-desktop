//! The server's side of the Splash native protocol, ported from
//! `server/protocol.py`. It encodes client frames and decodes engine
//! events; `runtime/engine/Protocol.*` does the reverse, and
//! `dev/tests/engine/protocol_golden.txt` pins the bytes both sides agree
//! on. Every integer is little-endian; a frame is its 16-byte header (the
//! magic, the protocol version, the frame type and the payload length),
//! then the payload.

use std::fmt;

pub const PROTOCOL_VERSION: u16 = 13;
pub const FRAME_HEADER_BYTES: usize = 16;
pub const STATUS_SCHEMA_VERSION: u32 = 8;
/// Score-only requests carry 2..255 distinct option token ids and produce
/// no generated tokens; a successful score DoneEvent returns one raw
/// final-position logit per requested token, in request order.
pub const MIN_SCORE_TOKENS: usize = 2;
pub const MAX_SCORE_TOKENS: usize = 255;
/// A thinking budget's end holds 1 to this many tokens.
pub const MAX_THINKING_END_TOKENS: usize = 64;
/// Patches per image, the most the vision encoder takes; the server's pixel
/// cap may allow fewer.
pub const MAX_IMAGE_PATCHES: usize = 16384;
/// Image pixels travel inside the request frame; a multi-image agent turn
/// can carry well over 64 MiB of resized RGB bytes.
pub const MAX_FRAME_PAYLOAD_BYTES: usize = 256 * 1024 * 1024;
pub const MAX_STATUS_JSON_BYTES: usize = 32 * 1024 * 1024;
pub const MAX_ERROR_STRING_BYTES: usize = 1024 * 1024;
pub const MAX_PROMPT_TOKENS: usize = 1 << 20;
pub const MAX_LOGICAL_OUTPUT_TOKENS: usize = 1 << 20;
pub const MAX_TOKEN_BATCH: usize = 4096;
pub const MAX_SIMULATION_TOKENS: usize = 32;
pub const MAX_LOGIT_BIAS: usize = 256;
pub const MAX_IMAGE_SPANS: usize = 64;
pub const MAX_MASK_WORDS: usize = 1 << 20;

pub const MAGIC: [u8; 4] = *b"SPLH";
/// The request frame's fixed part, in its wire order.
pub const REQUEST_FIXED_BYTES: usize = 122;
pub const IMAGE_SPAN_BYTES: usize = 32;

/// The request frame's thinking-budget tokens without one.
pub const NO_THINKING_BUDGET: u32 = 0xFFFF_FFFF;

pub mod decode;
pub mod encode;

pub use decode::{decode_frame, FrameParser};
pub use encode::{refresh_request_deadline, serialize_frame, serialize_message};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum FrameType {
    Request = 0x0001,
    Cancel = 0x0002,
    MaskResponse = 0x0003,
    StatusRequest = 0x0004,
    ReleaseRequest = 0x0005,
    PrefillDutyRequest = 0x0006,
    Ready = 0x0100,
    Start = 0x0101,
    Tokens = 0x0102,
    MaskRequest = 0x0103,
    Done = 0x0104,
    Error = 0x0105,
    StatusJson = 0x0107,
    PromptProgress = 0x0108,
    Logprobs = 0x0109,
}

impl FrameType {
    pub fn from_raw(value: u16) -> Option<FrameType> {
        Some(match value {
            0x0001 => FrameType::Request,
            0x0002 => FrameType::Cancel,
            0x0003 => FrameType::MaskResponse,
            0x0004 => FrameType::StatusRequest,
            0x0005 => FrameType::ReleaseRequest,
            0x0006 => FrameType::PrefillDutyRequest,
            0x0100 => FrameType::Ready,
            0x0101 => FrameType::Start,
            0x0102 => FrameType::Tokens,
            0x0103 => FrameType::MaskRequest,
            0x0104 => FrameType::Done,
            0x0105 => FrameType::Error,
            0x0107 => FrameType::StatusJson,
            0x0108 => FrameType::PromptProgress,
            0x0109 => FrameType::Logprobs,
            _ => return None,
        })
    }

    /// The frame types the engine sends; the others are the server's.
    pub fn is_event(self) -> bool {
        matches!(
            self,
            FrameType::Ready
                | FrameType::Start
                | FrameType::Tokens
                | FrameType::Logprobs
                | FrameType::MaskRequest
                | FrameType::Done
                | FrameType::Error
                | FrameType::StatusJson
                | FrameType::PromptProgress
        )
    }

    pub fn name(self) -> &'static str {
        match self {
            FrameType::Request => "request",
            FrameType::Cancel => "cancel",
            FrameType::MaskResponse => "mask_response",
            FrameType::StatusRequest => "status_request",
            FrameType::ReleaseRequest => "release_request",
            FrameType::PrefillDutyRequest => "prefill_duty_request",
            FrameType::Ready => "ready",
            FrameType::Start => "start",
            FrameType::Tokens => "tokens",
            FrameType::MaskRequest => "mask_request",
            FrameType::Done => "done",
            FrameType::Error => "error",
            FrameType::StatusJson => "status_json",
            FrameType::PromptProgress => "prompt_progress",
            FrameType::Logprobs => "logprobs",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FailureClass {
    RequestError = 1,
    EngineUnhealthy = 2,
    ProtocolFatal = 3,
}

impl FailureClass {
    pub fn from_raw(value: u8) -> Option<FailureClass> {
        Some(match value {
            1 => FailureClass::RequestError,
            2 => FailureClass::EngineUnhealthy,
            3 => FailureClass::ProtocolFatal,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            FailureClass::RequestError => "request_error",
            FailureClass::EngineUnhealthy => "engine_unhealthy",
            FailureClass::ProtocolFatal => "protocol_fatal",
        }
    }
}

/// Issue codes travel by name; the values are runtime/engine/Protocol.hpp's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum IssueCode {
    None = 0,
    BadMagic = 1,
    UnsupportedVersion = 2,
    UnknownFrameType = 3,
    FrameTooLarge = 4,
    InvalidPayloadLength = 5,
    TruncatedFrame = 6,
    InvalidRequestId = 7,
    InvalidEnumValue = 8,
    InvalidDeadline = 9,
    InvalidSampling = 10,
    InvalidCount = 11,
    InvalidConstraint = 12,
    InvalidErrorClassification = 13,
    LimitExceeded = 14,
    IntegerOverflow = 15,
    AllocationFailure = 16,
}

impl IssueCode {
    pub fn name(self) -> &'static str {
        match self {
            IssueCode::None => "none",
            IssueCode::BadMagic => "bad_magic",
            IssueCode::UnsupportedVersion => "unsupported_version",
            IssueCode::UnknownFrameType => "unknown_frame_type",
            IssueCode::FrameTooLarge => "frame_too_large",
            IssueCode::InvalidPayloadLength => "invalid_payload_length",
            IssueCode::TruncatedFrame => "truncated_frame",
            IssueCode::InvalidRequestId => "invalid_request_id",
            IssueCode::InvalidEnumValue => "invalid_enum_value",
            IssueCode::InvalidDeadline => "invalid_deadline",
            IssueCode::InvalidSampling => "invalid_sampling",
            IssueCode::InvalidCount => "invalid_count",
            IssueCode::InvalidConstraint => "invalid_constraint",
            IssueCode::InvalidErrorClassification => "invalid_error_classification",
            IssueCode::LimitExceeded => "limit_exceeded",
            IssueCode::IntegerOverflow => "integer_overflow",
            IssueCode::AllocationFailure => "allocation_failure",
        }
    }
}

/// A protocol failure: its classification, its issue code, the request it
/// names (zero for a stream failure) and a message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProtocolIssue {
    pub failure_class: FailureClass,
    pub code: IssueCode,
    pub request_id: u64,
    pub message: String,
}

impl ProtocolIssue {
    pub fn new(
        failure_class: FailureClass,
        code: IssueCode,
        request_id: u64,
        message: impl Into<String>,
    ) -> ProtocolIssue {
        ProtocolIssue {
            failure_class,
            code,
            request_id,
            message: message.into(),
        }
    }
}

impl fmt::Display for ProtocolIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.failure_class.name(), self.code.name())?;
        if self.request_id != 0 {
            write!(f, " request={}", self.request_id)?;
        }
        if !self.message.is_empty() {
            write!(f, ": {}", self.message)?;
        }
        Ok(())
    }
}

/// A protocol failure, raised where the Python raises `ProtocolError`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProtocolError(pub ProtocolIssue);

impl fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for ProtocolError {}

/// The failure a client frame's rule break carries: the request it names
/// fails, and one that names none fails the stream, as the engine
/// classifies it (Protocol.cpp requestIssue).
fn request_issue(code: IssueCode, message: impl Into<String>, request_id: u64) -> ProtocolError {
    let class = if request_id != 0 {
        FailureClass::RequestError
    } else {
        FailureClass::ProtocolFatal
    };
    ProtocolError(ProtocolIssue::new(class, code, request_id, message))
}

/// An engine event that breaks the protocol: the stream is not trusted.
fn fatal(code: IssueCode, message: impl Into<String>, request_id: u64) -> ProtocolError {
    ProtocolError(ProtocolIssue::new(
        FailureClass::ProtocolFatal,
        code,
        request_id,
        message,
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum RequestPriority {
    Foreground = 0,
    Normal = 1,
    Background = 2,
}

impl RequestPriority {
    pub fn from_raw(value: u8) -> Option<RequestPriority> {
        Some(match value {
            0 => RequestPriority::Foreground,
            1 => RequestPriority::Normal,
            2 => RequestPriority::Background,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ConstraintMode {
    None = 0,
    TokenMask = 1,
}

impl ConstraintMode {
    pub fn from_raw(value: u8) -> Option<ConstraintMode> {
        Some(match value {
            0 => ConstraintMode::None,
            1 => ConstraintMode::TokenMask,
            _ => return None,
        })
    }
}

pub mod request_flags {
    /// Never select the model's stop tokens, so generation runs to its
    /// output limit. Only unconstrained generation can carry it.
    pub const IGNORE_END_OF_SEQUENCE: u32 = 1 << 0;
    /// Report the prompt's progress while it prefills (PromptProgressEvent).
    pub const RETURN_PROGRESS: u32 = 1 << 1;
    /// Return each generated token's log probability (LogprobsEvent); the
    /// request frame's top_logprobs names the tokens above it.
    pub const LOGPROBS: u32 = 1 << 2;

    /// A request with any other bit set is a request error.
    pub const KNOWN_BITS: u32 =
        IGNORE_END_OF_SEQUENCE | RETURN_PROGRESS | LOGPROBS;
}

pub use request_flags as flags;

/// The sampling options a request names, in the frame's order. The defaults
/// are greedy selection with nothing changing the logits, which score
/// requests require. A top_k of 0, or one past the vocabulary, keeps every
/// token; the default penalties change nothing, and a min_p of 0 drops no
/// token.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParameters {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub repetition_penalty: f32,
    pub min_p: f32,
}

impl Default for SamplingParameters {
    fn default() -> SamplingParameters {
        SamplingParameters {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            min_p: 0.0,
        }
    }
}

/// One image in the prompt: its placeholder token run, the patch grid of
/// the resized pixels, and a 128-bit digest of that content. Placeholder
/// token ids are identical for every image, so cache identity keys on
/// spans.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageSpan {
    pub offset: u32,
    pub tokens: u32,
    pub grid_height: u32,
    pub grid_width: u32,
    pub digest_lo: u64,
    pub digest_hi: u64,
}

impl ImageSpan {
    pub fn pixel_bytes(&self) -> usize {
        self.grid_height as usize * 16 * self.grid_width as usize * 16 * 3
    }
}

/// A request that generates inside a thinking block may select at most
/// `tokens` thinking tokens; the engine then selects the `end` tokens in
/// order, the last of which closes the block. Thinking also ends where the
/// request selects that close or `also_ends` first, such as a tool call's
/// opening that follows thinking directly; no other end token is either.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThinkingBudget {
    pub tokens: u32,
    pub end: Vec<u32>,
    pub also_ends: u32,
}

/// The request frame, as the server builds it and the engine reads it.
#[derive(Debug, Clone, PartialEq)]
pub struct RequestFrame {
    pub request_id: u64,
    pub priority: RequestPriority,
    pub absolute_deadline_unix_micros: u64,
    pub remaining_deadline_micros: u64,
    pub logical_max_output_tokens: u32,
    pub prompt_tokens: Vec<u32>,
    pub sampling: SamplingParameters,
    pub seed: u64,
    pub constraint: ConstraintMode,
    /// Sorted, non-overlapping image spans and their resized uint8 RGB
    /// pixels concatenated in span order; both empty for text-only
    /// requests.
    pub image_spans: Vec<ImageSpan>,
    pub image_pixels: Vec<u8>,
    /// Option token ids for score-only requests; empty means ordinary
    /// generation. Score tokens serialize after the image pixel bytes, and
    /// a thinking budget's end tokens after them.
    pub score_tokens: Vec<u32>,
    /// Trailing prompt tokens of the chat template's generation prompt;
    /// zero when unknown. It must leave at least one prompt token.
    pub generation_prompt_tokens: u32,
    pub flags: u32,
    /// The request's cache_salt hashed to 128 bits (Frontend._cache_salt);
    /// zero without one.
    pub cache_salt: u128,
    /// Tokens above each generated token in its logprobs, 0..20; requires
    /// the LOGPROBS flag.
    pub top_logprobs: u32,
    /// Raw-logit adjustments (OpenAI logit_bias), tokens then values, equal
    /// lengths, at most MAX_LOGIT_BIAS.
    pub bias_tokens: Vec<u32>,
    pub bias_values: Vec<f32>,
    /// Only a request that generates inside a thinking block has one; a
    /// score request never does.
    pub thinking_budget: Option<ThinkingBudget>,
}

impl Default for RequestFrame {
    fn default() -> RequestFrame {
        RequestFrame {
            request_id: 0,
            priority: RequestPriority::Foreground,
            absolute_deadline_unix_micros: 0,
            remaining_deadline_micros: 0,
            logical_max_output_tokens: 0,
            prompt_tokens: Vec::new(),
            sampling: SamplingParameters::default(),
            seed: 0,
            constraint: ConstraintMode::None,
            image_spans: Vec::new(),
            image_pixels: Vec::new(),
            score_tokens: Vec::new(),
            generation_prompt_tokens: 0,
            flags: 0,
            cache_salt: 0,
            top_logprobs: 0,
            bias_tokens: Vec::new(),
            bias_values: Vec::new(),
            thinking_budget: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CancelFrame {
    pub request_id: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MaskResponseFrame {
    pub request_id: u64,
    pub mask_request_id: u64,
    /// Little-endian uint32 words, words_per_mask per mask row.
    pub mask_words: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StatusRequestFrame {
    pub correlation_id: u64,
}

/// Give the weights back now, as the idle release does, when the engine
/// holds no request and restores none, with the idle release on or off;
/// answered by a status JSON with the same correlation id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReleaseRequestFrame {
    pub correlation_id: u64,
}

/// Change a running engine's prefill duty, as --prefill-duty does at
/// startup: the share of wall time prefill may run. The engine clamps it
/// and answers nothing; the next status document's throttle section is the
/// truth.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrefillDutyRequestFrame {
    pub duty_percent: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadyEvent {
    pub max_concurrent_requests: u32,
    pub max_context_tokens: u32,
    /// Requests may carry image spans; false for a model serving without
    /// vision.
    pub vision: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StartEvent {
    pub request_id: u64,
    pub lane: u32,
    /// The cached prefix the request starts from; zero for a cold start.
    pub matched_prompt_tokens: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromptProgressEvent {
    pub request_id: u64,
    pub processed_tokens: u32,
    pub elapsed_micros: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TokensEvent {
    pub request_id: u64,
    pub sequence_offset: u32,
    pub tokens: Vec<u32>,
}

/// One generated token's log probability and the top (token, logprob)
/// pairs above it.
#[derive(Debug, Clone, PartialEq)]
pub struct LogprobValues {
    pub token: u32,
    pub logprob: f32,
    pub top: Vec<(u32, f32)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogprobsEvent {
    pub request_id: u64,
    pub sequence_offset: u32,
    pub values: Vec<LogprobValues>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaskRequestEvent {
    pub request_id: u64,
    pub mask_request_id: u64,
    pub words_per_mask: u32,
    pub simulation_tokens: Vec<u32>,
}

impl MaskRequestEvent {
    /// Rows before and after each simulated context token.
    ///
    /// The initial request carries no tokens and therefore has one row. A
    /// verify request carries `(pending_anchor, *draft_proposals)`; its
    /// target-logit rows consume masks beginning at row one.
    pub fn mask_rows(&self) -> usize {
        self.simulation_tokens.len() + 1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FinishReason {
    Stop = 0,
    Length = 1,
    Cancelled = 2,
}

impl FinishReason {
    pub fn from_raw(value: u8) -> Option<FinishReason> {
        Some(match value {
            0 => FinishReason::Stop,
            1 => FinishReason::Length,
            2 => FinishReason::Cancelled,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DoneEvent {
    pub request_id: u64,
    pub reason: FinishReason,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub prefill_micros: u64,
    pub decode_micros: u64,
    pub wall_micros: u64,
    /// Raw final-position logits for a score-only request, in requested
    /// token order; empty for generation and for cancelled or failed
    /// scoring.
    pub option_logits: Vec<f32>,
    /// Tokens whose KV the request read from cached blocks another request
    /// computed, past its cached prefix.
    pub deduplicated_tokens: u32,
    pub drafted_tokens: u32,
    pub accepted_draft_tokens: u32,
    pub draft_cycles: u32,
    /// A LENGTH end short of the request's logical maximum, at a bound of
    /// the engine's own: the SSD cache's quota, or memory a lone lane
    /// could not grow past. It ends the response as its context would.
    pub bounded: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorEvent {
    pub failure_class: FailureClass,
    pub request_id: u64,
    pub retryable: bool,
    pub code: Vec<u8>,
    pub message: Vec<u8>,
    /// For a retryable request error, the seconds after which a retry may
    /// succeed, as when the SSD cache's write budget refills; 0 for at
    /// once.
    pub retry_after_seconds: u32,
}

/// The JSON is opaque to the transport: the document carries its own
/// schema_version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatusJsonEvent {
    pub correlation_id: u64,
    pub json: Vec<u8>,
}

/// A client message: what the server sends the engine.
#[derive(Debug, Clone, PartialEq)]
pub enum ClientMessage {
    Request(RequestFrame),
    Cancel(CancelFrame),
    MaskResponse(MaskResponseFrame),
    StatusRequest(StatusRequestFrame),
    ReleaseRequest(ReleaseRequestFrame),
    PrefillDutyRequest(PrefillDutyRequestFrame),
}

/// An engine event: what the server reads from the engine.
#[derive(Debug, Clone, PartialEq)]
pub enum EngineEvent {
    Ready(ReadyEvent),
    Start(StartEvent),
    PromptProgress(PromptProgressEvent),
    Tokens(TokensEvent),
    Logprobs(LogprobsEvent),
    MaskRequest(MaskRequestEvent),
    Done(DoneEvent),
    Error(ErrorEvent),
    StatusJson(StatusJsonEvent),
}

/// A parsed frame: its type and payload, the transport's own currency.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub frame_type: FrameType,
    pub payload: Vec<u8>,
}

/// What one `FrameParser::consume` step used and produced.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseStep {
    pub consumed_bytes: usize,
    pub frame: Option<Frame>,
    pub issue: Option<ProtocolIssue>,
}

impl ParseStep {
    fn consumed(consumed_bytes: usize) -> ParseStep {
        ParseStep {
            consumed_bytes,
            frame: None,
            issue: None,
        }
    }
}
