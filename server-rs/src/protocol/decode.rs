//! Engine-event decoding and the incremental frame parser, ported from
//! `server/protocol.py`'s `decode_frame` and `FrameParser`. Every event is
//! strictly validated; a failure is either the request's or the stream's,
//! as its classification says.

use super::*;
use crate::protocol::encode::payload_length_issue;

fn read_u64(payload: &[u8], offset: usize) -> u64 {
    let bytes: [u8; 8] = payload[offset..offset + 8].try_into().expect("fixed field");
    u64::from_le_bytes(bytes)
}

fn read_u32(payload: &[u8], offset: usize) -> u32 {
    let bytes: [u8; 4] = payload[offset..offset + 4].try_into().expect("fixed field");
    u32::from_le_bytes(bytes)
}

fn read_f32(payload: &[u8], offset: usize) -> f32 {
    let bytes: [u8; 4] = payload[offset..offset + 4].try_into().expect("fixed field");
    f32::from_le_bytes(bytes)
}

/// The `count` words (u32) or floats (f32) that end an event payload.
fn tail_u32(
    payload: &[u8],
    offset: usize,
    count: usize,
    label: &str,
    request_id: u64,
) -> Result<Vec<u32>, ProtocolError> {
    if payload.len() - offset != count * 4 {
        return Err(fatal(
            IssueCode::InvalidPayloadLength,
            format!("{label} count does not match the binary event payload"),
            request_id,
        ));
    }
    Ok((0..count)
        .map(|index| read_u32(payload, offset + index * 4))
        .collect())
}

fn tail_f32(
    payload: &[u8],
    offset: usize,
    count: usize,
    label: &str,
    request_id: u64,
) -> Result<Vec<f32>, ProtocolError> {
    if payload.len() - offset != count * 4 {
        return Err(fatal(
            IssueCode::InvalidPayloadLength,
            format!("{label} count does not match the binary event payload"),
            request_id,
        ));
    }
    Ok((0..count)
        .map(|index| read_f32(payload, offset + index * 4))
        .collect())
}

/// Decode and strictly validate one engine event.
pub fn decode_frame(frame: &Frame) -> Result<EngineEvent, ProtocolError> {
    if let Some(issue) = payload_length_issue(frame.frame_type, frame.payload.len()) {
        return Err(ProtocolError(issue));
    }
    let payload = &frame.payload;
    match frame.frame_type {
        FrameType::Ready => {
            let concurrent = read_u32(payload, 0);
            let context = read_u32(payload, 4);
            let vision = payload[8];
            if vision > 1 {
                return Err(fatal(
                    IssueCode::InvalidEnumValue,
                    "ready vision must be a boolean",
                    0,
                ));
            }
            if concurrent == 0 || context == 0 {
                return Err(fatal(
                    IssueCode::InvalidCount,
                    "ready event capacities must be non-zero",
                    0,
                ));
            }
            Ok(EngineEvent::Ready(ReadyEvent {
                max_concurrent_requests: concurrent,
                max_context_tokens: context,
                vision: vision == 1,
            }))
        }
        FrameType::Start => {
            let request_id = read_u64(payload, 0);
            let lane = read_u32(payload, 8);
            let matched = read_u32(payload, 12);
            if request_id == 0 {
                return Err(fatal(
                    IssueCode::InvalidRequestId,
                    "start event request id must be non-zero",
                    request_id,
                ));
            }
            Ok(EngineEvent::Start(StartEvent {
                request_id,
                lane,
                matched_prompt_tokens: matched,
            }))
        }
        FrameType::PromptProgress => {
            let request_id = read_u64(payload, 0);
            let processed = read_u32(payload, 8);
            let elapsed = read_u64(payload, 12);
            if request_id == 0 || processed as usize > MAX_PROMPT_TOKENS {
                return Err(fatal(
                    IssueCode::InvalidCount,
                    "prompt progress id or token count is invalid",
                    request_id,
                ));
            }
            Ok(EngineEvent::PromptProgress(PromptProgressEvent {
                request_id,
                processed_tokens: processed,
                elapsed_micros: elapsed,
            }))
        }
        FrameType::Tokens => {
            let request_id = read_u64(payload, 0);
            let offset = read_u32(payload, 8);
            let count = read_u32(payload, 12) as usize;
            let tokens = tail_u32(payload, 16, count, "token", request_id)?;
            if request_id == 0 {
                return Err(fatal(
                    IssueCode::InvalidRequestId,
                    "tokens event request id must be non-zero",
                    request_id,
                ));
            }
            if tokens.is_empty() {
                return Err(fatal(
                    IssueCode::LimitExceeded,
                    "tokens event carries no token",
                    request_id,
                ));
            }
            if offset as u64 + tokens.len() as u64 > u32::MAX as u64 {
                return Err(fatal(
                    IssueCode::IntegerOverflow,
                    "tokens event sequence range overflows uint32",
                    request_id,
                ));
            }
            Ok(EngineEvent::Tokens(TokensEvent {
                request_id,
                sequence_offset: offset,
                tokens,
            }))
        }
        FrameType::Logprobs => {
            let request_id = read_u64(payload, 0);
            let offset = read_u32(payload, 8);
            let count = read_u32(payload, 12) as usize;
            if request_id == 0 {
                return Err(fatal(
                    IssueCode::InvalidRequestId,
                    "logprobs event request id must be non-zero",
                    request_id,
                ));
            }
            if !(1..=MAX_TOKEN_BATCH).contains(&count) {
                return Err(fatal(
                    IssueCode::LimitExceeded,
                    "logprobs event batch size exceeds its limit",
                    request_id,
                ));
            }
            let mut values = Vec::with_capacity(count);
            let mut position = 16usize;
            for _ in 0..count {
                let token = read_u32(payload, position);
                let logprob = read_f32(payload, position + 4);
                let top_count = read_u32(payload, position + 8) as usize;
                position += 12;
                if top_count > 20 || position + top_count * 8 > payload.len() {
                    return Err(fatal(
                        IssueCode::LimitExceeded,
                        "logprobs event top count exceeds its limit",
                        request_id,
                    ));
                }
                let mut top = Vec::with_capacity(top_count);
                for _ in 0..top_count {
                    let top_token = read_u32(payload, position);
                    let top_logprob = read_f32(payload, position + 4);
                    position += 8;
                    top.push((top_token, top_logprob));
                }
                values.push(LogprobValues {
                    token,
                    logprob,
                    top,
                });
            }
            if position != payload.len() {
                return Err(fatal(
                    IssueCode::InvalidPayloadLength,
                    "logprobs event payload does not match its counts",
                    request_id,
                ));
            }
            if offset as u64 + values.len() as u64 > u32::MAX as u64 {
                return Err(fatal(
                    IssueCode::IntegerOverflow,
                    "logprobs event sequence range overflows uint32",
                    request_id,
                ));
            }
            Ok(EngineEvent::Logprobs(LogprobsEvent {
                request_id,
                sequence_offset: offset,
                values,
            }))
        }
        FrameType::MaskRequest => {
            let request_id = read_u64(payload, 0);
            let mask_request_id = read_u64(payload, 8);
            let words_per_mask = read_u32(payload, 16);
            let count = read_u32(payload, 20) as usize;
            let tokens = tail_u32(payload, 24, count, "simulation token", request_id)?;
            if request_id == 0 || mask_request_id == 0 {
                return Err(fatal(
                    IssueCode::InvalidRequestId,
                    "mask request ids must be non-zero",
                    request_id,
                ));
            }
            if words_per_mask == 0 {
                return Err(fatal(
                    IssueCode::InvalidCount,
                    "mask request dimensions are invalid",
                    request_id,
                ));
            }
            if words_per_mask as usize * (tokens.len() + 1) > MAX_MASK_WORDS {
                return Err(fatal(
                    IssueCode::LimitExceeded,
                    "mask request output would exceed the mask limit",
                    request_id,
                ));
            }
            Ok(EngineEvent::MaskRequest(MaskRequestEvent {
                request_id,
                mask_request_id,
                words_per_mask,
                simulation_tokens: tokens,
            }))
        }
        FrameType::Done => {
            let request_id = read_u64(payload, 0);
            let reason_raw = payload[8];
            let bounded = payload[9];
            let prompt_tokens = read_u32(payload, 10);
            let completion_tokens = read_u32(payload, 14);
            let prefill_micros = read_u64(payload, 18);
            let decode_micros = read_u64(payload, 26);
            let wall_micros = read_u64(payload, 34);
            let deduplicated = read_u32(payload, 42);
            let drafted = read_u32(payload, 46);
            let accepted = read_u32(payload, 50);
            let cycles = read_u32(payload, 54);
            let logit_count = read_u32(payload, 58) as usize;
            let logits = tail_f32(payload, 62, logit_count, "option logit", request_id)?;
            let reason = FinishReason::from_raw(reason_raw).ok_or_else(|| {
                fatal(
                    IssueCode::InvalidEnumValue,
                    "done finish reason is invalid",
                    request_id,
                )
            })?;
            if request_id == 0 {
                return Err(fatal(
                    IssueCode::InvalidRequestId,
                    "done event request id must be non-zero",
                    request_id,
                ));
            }
            if bounded > 1 || (bounded != 0 && reason != FinishReason::Length) {
                return Err(fatal(
                    IssueCode::InvalidEnumValue,
                    "only a length-finished done event is bounded",
                    request_id,
                ));
            }
            // Each cycle verified at least one drafted token and accepted at
            // most all.
            if accepted > drafted || cycles > drafted || (drafted != 0 && cycles == 0) {
                return Err(fatal(
                    IssueCode::InvalidCount,
                    "done speculative counts are inconsistent",
                    request_id,
                ));
            }
            if !logits.is_empty()
                && (logits.len() < MIN_SCORE_TOKENS
                    || logits.iter().any(|value| !value.is_finite())
                    || reason != FinishReason::Stop
                    || completion_tokens != 0
                    || decode_micros != 0
                    || drafted != 0)
            {
                return Err(fatal(
                    IssueCode::InvalidCount,
                    "scored done events need 2..255 finite logits, stop, and no \
                     completion or decode activity",
                    request_id,
                ));
            }
            Ok(EngineEvent::Done(DoneEvent {
                request_id,
                reason,
                prompt_tokens,
                completion_tokens,
                prefill_micros,
                decode_micros,
                wall_micros,
                option_logits: logits,
                deduplicated_tokens: deduplicated,
                drafted_tokens: drafted,
                accepted_draft_tokens: accepted,
                draft_cycles: cycles,
                bounded: bounded != 0,
            }))
        }
        FrameType::Error => {
            let failure_raw = payload[0];
            let retryable = payload[1];
            let retry_after = read_u32(payload, 2);
            let request_id = read_u64(payload, 6);
            let code_bytes = read_u32(payload, 14) as usize;
            let message_bytes = read_u32(payload, 18) as usize;
            if retryable > 1 || code_bytes + message_bytes != payload.len() - 22 {
                return Err(fatal(
                    IssueCode::InvalidPayloadLength,
                    "error string lengths do not match the frame payload",
                    request_id,
                ));
            }
            let classification = FailureClass::from_raw(failure_raw).ok_or_else(|| {
                fatal(
                    IssueCode::InvalidErrorClassification,
                    "error event classification is invalid",
                    request_id,
                )
            })?;
            if (classification == FailureClass::RequestError) != (request_id != 0) {
                return Err(fatal(
                    IssueCode::InvalidErrorClassification,
                    "only request errors may carry a non-zero request id",
                    request_id,
                ));
            }
            if code_bytes == 0
                || code_bytes > MAX_ERROR_STRING_BYTES
                || message_bytes > MAX_ERROR_STRING_BYTES
            {
                return Err(fatal(
                    IssueCode::LimitExceeded,
                    "error code or message exceeds its safe limit",
                    request_id,
                ));
            }
            if retry_after != 0 && retryable == 0 {
                return Err(fatal(
                    IssueCode::InvalidErrorClassification,
                    "only a retryable error says when to retry",
                    request_id,
                ));
            }
            Ok(EngineEvent::Error(ErrorEvent {
                failure_class: classification,
                request_id,
                retryable: retryable == 1,
                code: payload[22..22 + code_bytes].to_vec(),
                message: payload[22 + code_bytes..].to_vec(),
                retry_after_seconds: retry_after,
            }))
        }
        FrameType::StatusJson => {
            let correlation_id = read_u64(payload, 0);
            let json = &payload[8..];
            if json.is_empty() {
                return Err(fatal(IssueCode::LimitExceeded, "status JSON is empty", 0));
            }
            Ok(EngineEvent::StatusJson(StatusJsonEvent {
                correlation_id,
                json: json.to_vec(),
            }))
        }
        _ => Err(fatal(
            IssueCode::UnknownFrameType,
            "frame type is not a native protocol event",
            0,
        )),
    }
}

/// Incremental, one-frame-at-a-time parser of engine events. A client frame
/// type is an unknown frame type.
#[derive(Debug)]
pub struct FrameParser {
    header: Vec<u8>,
    reading_payload: bool,
    current_type: FrameType,
    expected_payload_bytes: usize,
    payload: Vec<u8>,
    terminal_issue: Option<ProtocolIssue>,
}

impl Default for FrameParser {
    fn default() -> FrameParser {
        FrameParser {
            header: Vec::new(),
            reading_payload: false,
            current_type: FrameType::Ready,
            expected_payload_bytes: 0,
            payload: Vec::new(),
            terminal_issue: None,
        }
    }
}

impl FrameParser {
    pub fn new() -> FrameParser {
        FrameParser::default()
    }

    fn fail(&mut self, consumed: usize, issue: ProtocolIssue) -> ParseStep {
        self.terminal_issue = Some(issue.clone());
        ParseStep {
            consumed_bytes: consumed,
            frame: None,
            issue: Some(issue),
        }
    }

    fn reset_current_frame(&mut self) {
        self.header.clear();
        self.reading_payload = false;
        self.expected_payload_bytes = 0;
        self.payload.clear();
    }

    fn parse_header(&mut self) -> Option<ProtocolIssue> {
        let magic: [u8; 4] = self.header[..4].try_into().expect("header magic");
        let version = u16::from_le_bytes([self.header[4], self.header[5]]);
        let raw_type = u16::from_le_bytes([self.header[6], self.header[7]]);
        let payload_bytes =
            u64::from_le_bytes(self.header[8..16].try_into().expect("header length")) as usize;
        if magic != MAGIC {
            return Some(ProtocolIssue::new(
                FailureClass::ProtocolFatal,
                IssueCode::BadMagic,
                0,
                "frame magic is not SPLH",
            ));
        }
        if version != PROTOCOL_VERSION {
            return Some(ProtocolIssue::new(
                FailureClass::ProtocolFatal,
                IssueCode::UnsupportedVersion,
                0,
                "unsupported native protocol version",
            ));
        }
        let frame_type = FrameType::from_raw(raw_type).filter(|t| t.is_event());
        let frame_type = match frame_type {
            Some(frame_type) => frame_type,
            None => {
                return Some(ProtocolIssue::new(
                    FailureClass::ProtocolFatal,
                    IssueCode::UnknownFrameType,
                    0,
                    "frame type is not a native protocol event",
                ))
            }
        };
        self.current_type = frame_type;
        if let Some(issue) = payload_length_issue(frame_type, payload_bytes) {
            return Some(issue);
        }
        self.expected_payload_bytes = payload_bytes;
        self.payload.clear();
        self.reading_payload = true;
        None
    }

    /// Consume up to one frame and report exactly how many bytes were used.
    pub fn consume(&mut self, data: &[u8]) -> ParseStep {
        if let Some(issue) = &self.terminal_issue {
            panic!("frame parser used after it failed: {issue}");
        }
        let mut consumed = 0usize;
        while consumed < data.len() {
            if !self.reading_payload {
                let count = (FRAME_HEADER_BYTES - self.header.len()).min(data.len() - consumed);
                self.header.extend_from_slice(&data[consumed..consumed + count]);
                consumed += count;
                if self.header.len() < FRAME_HEADER_BYTES {
                    return ParseStep::consumed(consumed);
                }
                if let Some(issue) = self.parse_header() {
                    return self.fail(consumed, issue);
                }
                if self.expected_payload_bytes == 0 {
                    let frame = Frame {
                        frame_type: self.current_type,
                        payload: Vec::new(),
                    };
                    self.reset_current_frame();
                    return ParseStep {
                        consumed_bytes: consumed,
                        frame: Some(frame),
                        issue: None,
                    };
                }
            }
            let needed = self.expected_payload_bytes - self.payload.len();
            let count = needed.min(data.len() - consumed);
            self.payload
                .extend_from_slice(&data[consumed..consumed + count]);
            consumed += count;
            if self.payload.len() == self.expected_payload_bytes {
                let frame = Frame {
                    frame_type: self.current_type,
                    payload: std::mem::take(&mut self.payload),
                };
                self.reset_current_frame();
                return ParseStep {
                    consumed_bytes: consumed,
                    frame: Some(frame),
                    issue: None,
                };
            }
        }
        ParseStep::consumed(consumed)
    }

    /// Finish EOF processing, making any partial frame terminally fatal.
    pub fn finish(&mut self) -> Option<ProtocolIssue> {
        if self.terminal_issue.is_some() {
            panic!("frame parser used after it failed");
        }
        if self.header.is_empty() && !self.reading_payload {
            return None;
        }
        let message = if !self.reading_payload {
            format!(
                "stream ended after {} of {FRAME_HEADER_BYTES} frame-header bytes",
                self.header.len()
            )
        } else {
            format!(
                "stream ended after {} of {} {} payload bytes",
                self.payload.len(),
                self.expected_payload_bytes,
                self.current_type.name()
            )
        };
        let issue = ProtocolIssue::new(
            FailureClass::ProtocolFatal,
            IssueCode::TruncatedFrame,
            0,
            message,
        );
        self.terminal_issue = Some(issue.clone());
        Some(issue)
    }
}
