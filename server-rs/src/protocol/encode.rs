//! Client-frame serialization, ported from `server/protocol.py`'s
//! `serialize_message` and its validators. Every rule the Python enforces
//! is here, in the same order, with the same issue classification.

use super::*;

/// The sampling block as the frame carries it, in its wire order: the
/// values' types are the wire's, so the writer gives each its own width.
type SamplingBlock = (f32, f32, u32, f32, f32, f32, f32);

fn sampling_values(sampling: &SamplingParameters) -> SamplingBlock {
    (
        sampling.temperature,
        sampling.top_p,
        sampling.top_k,
        sampling.presence_penalty,
        sampling.frequency_penalty,
        sampling.repetition_penalty,
        sampling.min_p,
    )
}

/// A sampling temperature is 0 or a normal float32: the kernels divide by
/// it, and Metal flushes a subnormal one to zero.
const FLOAT32_MIN: f32 = 1.1754943508222875e-38; // 0x1p-126

fn neutral_sampling() -> SamplingBlock {
    sampling_values(&SamplingParameters::default())
}

/// The payload bounds of each frame type: the minimum its fixed part needs
/// and the maximum its variable parts may reach, both capped by the frame
/// limit.
fn payload_bounds(frame_type: FrameType) -> (usize, usize) {
    let (minimum, maximum) = match frame_type {
        FrameType::Request => (REQUEST_FIXED_BYTES, MAX_FRAME_PAYLOAD_BYTES),
        FrameType::Cancel => (8, 8),
        FrameType::MaskResponse => (20, 20 + MAX_MASK_WORDS * 4),
        FrameType::StatusRequest => (8, 8),
        FrameType::ReleaseRequest => (8, 8),
        FrameType::PrefillDutyRequest => (4, 4),
        FrameType::Ready => (9, 9),
        FrameType::PromptProgress => (20, 20),
        FrameType::Start => (16, 16),
        FrameType::Tokens => (16, 16 + MAX_TOKEN_BATCH * 4),
        FrameType::Logprobs => (16, 16 + MAX_TOKEN_BATCH * (12 + 20 * 8)),
        FrameType::MaskRequest => (24, 24 + MAX_SIMULATION_TOKENS * 4),
        FrameType::Done => (62, 62 + MAX_SCORE_TOKENS * 4),
        FrameType::Error => (22, 22 + MAX_ERROR_STRING_BYTES * 2),
        FrameType::StatusJson => (8, 8 + MAX_STATUS_JSON_BYTES),
    };
    (minimum, maximum.min(MAX_FRAME_PAYLOAD_BYTES))
}

/// The issue a payload length breaks the frame type's bounds with, if any.
pub fn payload_length_issue(frame_type: FrameType, payload_bytes: usize) -> Option<ProtocolIssue> {
    let (minimum, maximum) = payload_bounds(frame_type);
    if payload_bytes > maximum {
        return Some(ProtocolIssue::new(
            FailureClass::ProtocolFatal,
            IssueCode::FrameTooLarge,
            0,
            format!(
                "{} payload length {payload_bytes} exceeds its safe limit {maximum}",
                frame_type.name()
            ),
        ));
    }
    if payload_bytes < minimum {
        return Some(ProtocolIssue::new(
            FailureClass::ProtocolFatal,
            IssueCode::InvalidPayloadLength,
            0,
            format!(
                "{} payload length {payload_bytes} is below its required minimum {minimum}",
                frame_type.name()
            ),
        ));
    }
    None
}

fn put_u16(payload: &mut Vec<u8>, value: u16) {
    payload.extend_from_slice(&value.to_le_bytes());
}

fn put_u32(payload: &mut Vec<u8>, value: u32) {
    payload.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(payload: &mut Vec<u8>, value: u64) {
    payload.extend_from_slice(&value.to_le_bytes());
}

fn put_f32(payload: &mut Vec<u8>, value: f32) {
    payload.extend_from_slice(&value.to_le_bytes());
}

fn put_words(payload: &mut Vec<u8>, words: &[u32]) {
    for word in words {
        put_u32(payload, *word);
    }
}

fn header(frame_type: FrameType, payload_bytes: usize) -> Vec<u8> {
    let mut frame = Vec::with_capacity(FRAME_HEADER_BYTES + payload_bytes);
    frame.extend_from_slice(&MAGIC);
    put_u16(&mut frame, PROTOCOL_VERSION);
    put_u16(&mut frame, frame_type as u16);
    put_u64(&mut frame, payload_bytes as u64);
    frame
}

/// The image spans' rules: even-sided grids within the patch limit, token
/// runs that match the merged grid, sorted and non-overlapping inside the
/// prompt, and pixels that match the grids.
fn image_spans_check(
    spans: &[ImageSpan],
    pixels: &[u8],
    prompt_tokens: usize,
) -> Result<(), String> {
    let mut previous_end: u32 = 0;
    let mut pixel_bytes = 0usize;
    for span in spans {
        if span.grid_height < 2
            || span.grid_width < 2
            || span.grid_height % 2 != 0
            || span.grid_width % 2 != 0
            || span.grid_height as usize * span.grid_width as usize > MAX_IMAGE_PATCHES
        {
            return Err("image grid must be even-sided and within the patch limit".into());
        }
        if span.tokens != (span.grid_height / 2) * (span.grid_width / 2) {
            return Err("image span tokens must equal the merged grid size".into());
        }
        if span.offset < previous_end || span.offset as usize + span.tokens as usize > prompt_tokens
        {
            return Err(
                "image spans must be sorted, non-overlapping runs inside the prompt".into(),
            );
        }
        previous_end = span.offset + span.tokens;
        pixel_bytes += span.pixel_bytes();
    }
    if pixels.len() != pixel_bytes {
        return Err("image pixels do not match the image grids".into());
    }
    Ok(())
}

/// The request's rules, in the order the Python checks them. Returns the
/// sampling block as the frame carries it, or the failure the first broken
/// rule raises.
fn validated_request(
    request: &RequestFrame,
) -> Result<(SamplingBlock, Vec<u32>, Vec<u32>, Vec<u32>, Vec<f32>), ProtocolError> {
    let request_id = request.request_id;
    if request_id == 0 {
        return Err(request_issue(
            IssueCode::InvalidRequestId,
            "request id must be non-zero",
            request_id,
        ));
    }
    let flags = request.flags;
    if flags & !flags::KNOWN_BITS != 0 {
        return Err(request_issue(
            IssueCode::InvalidEnumValue,
            "request flags are not defined by native protocol",
            request_id,
        ));
    }
    if request.top_logprobs > 20 {
        return Err(request_issue(
            IssueCode::InvalidEnumValue,
            "top logprobs count exceeds its limit",
            request_id,
        ));
    }
    if request.top_logprobs != 0 && flags & flags::LOGPROBS == 0 {
        return Err(request_issue(
            IssueCode::InvalidEnumValue,
            "top logprobs requires the logprobs flag",
            request_id,
        ));
    }
    if flags & flags::LOGPROBS != 0 && request.constraint != ConstraintMode::None {
        return Err(request_issue(
            IssueCode::InvalidEnumValue,
            "constrained requests do not take logprobs",
            request_id,
        ));
    }
    if request.absolute_deadline_unix_micros == 0 || request.remaining_deadline_micros == 0 {
        return Err(request_issue(
            IssueCode::InvalidDeadline,
            "absolute and remaining deadlines must be non-zero",
            request_id,
        ));
    }
    let output_tokens = request.logical_max_output_tokens;
    let prompt = &request.prompt_tokens;
    let scores = &request.score_tokens;
    if !scores.is_empty() {
        if output_tokens != 0 {
            return Err(request_issue(
                IssueCode::InvalidCount,
                "score requests must not generate output tokens",
                request_id,
            ));
        }
    } else if output_tokens == 0 || output_tokens as usize > MAX_LOGICAL_OUTPUT_TOKENS {
        return Err(request_issue(
            IssueCode::LimitExceeded,
            "logical max output token count exceeds its limit",
            request_id,
        ));
    }
    if prompt.is_empty() || prompt.len() > MAX_PROMPT_TOKENS {
        return Err(request_issue(
            IssueCode::LimitExceeded,
            "prompt token count exceeds its limit",
            request_id,
        ));
    }
    if request.image_spans.len() > MAX_IMAGE_SPANS {
        return Err(request_issue(
            IssueCode::LimitExceeded,
            "image span count exceeds its limit",
            request_id,
        ));
    }
    if let Err(message) = image_spans_check(&request.image_spans, &request.image_pixels, prompt.len())
    {
        return Err(request_issue(IssueCode::InvalidCount, message, request_id));
    }
    if request.generation_prompt_tokens as usize >= prompt.len() {
        return Err(request_issue(
            IssueCode::InvalidCount,
            "generation prompt must leave a prompt token",
            request_id,
        ));
    }
    if !scores.is_empty() && (!request.image_spans.is_empty() || !request.image_pixels.is_empty())
    {
        return Err(request_issue(
            IssueCode::InvalidCount,
            "score requests are text-only",
            request_id,
        ));
    }
    if !scores.is_empty()
        && (scores.len() < MIN_SCORE_TOKENS
            || scores.len() > MAX_SCORE_TOKENS
            || {
                let mut sorted = scores.clone();
                sorted.sort_unstable();
                sorted.dedup();
                sorted.len() != scores.len()
            })
    {
        return Err(request_issue(
            IssueCode::InvalidCount,
            format!(
                "score requests need {MIN_SCORE_TOKENS}..{MAX_SCORE_TOKENS} distinct option tokens"
            ),
            request_id,
        ));
    }
    let sampling = sampling_values(&request.sampling);
    let (temperature, top_p, _, presence, frequency, repetition, min_p) = sampling;
    if !(temperature == 0.0 || (FLOAT32_MIN..f32::INFINITY).contains(&temperature))
        || !top_p.is_finite()
        || !(0.0 < top_p && top_p <= 1.0)
        || !(0.0..=1.0).contains(&min_p)
    {
        return Err(request_issue(
            IssueCode::InvalidSampling,
            "sampling requires temperature 0 or at least FLT_MIN, top_p in (0,1] and min_p in [0,1]",
            request_id,
        ));
    }
    if presence.abs() > 2.0
        || frequency.abs() > 2.0
        || !repetition.is_finite()
        || repetition <= 0.0
    {
        return Err(request_issue(
            IssueCode::InvalidSampling,
            "sampling requires presence and frequency penalties in [-2,2] and a positive repetition penalty",
            request_id,
        ));
    }
    if !scores.is_empty() && sampling != neutral_sampling() {
        return Err(request_issue(
            IssueCode::InvalidSampling,
            "score requests require default greedy sampling",
            request_id,
        ));
    }
    let mut end: Vec<u32> = Vec::new();
    if let Some(thinking) = &request.thinking_budget {
        if !scores.is_empty() {
            return Err(request_issue(
                IssueCode::InvalidCount,
                "score requests take no thinking budget",
                request_id,
            ));
        }
        // --loop-guard: an uncapped budget (the sentinel) is a closable
        // thinking block the engine's guard may end early.
        end = thinking.end.clone();
        let also_ends = thinking.also_ends;
        if end.is_empty() || end.len() > MAX_THINKING_END_TOKENS {
            return Err(request_issue(
                IssueCode::InvalidCount,
                format!(
                    "thinking end token count must be in [1, {MAX_THINKING_END_TOKENS}]"
                ),
                request_id,
            ));
        }
        // Thinking ends where its end does, and nowhere inside it.
        let last = end[end.len() - 1];
        if end[..end.len() - 1].contains(&last) || end[..end.len() - 1].contains(&also_ends) {
            return Err(request_issue(
                IssueCode::InvalidCount,
                "only the last thinking end token may end thinking",
                request_id,
            ));
        }
    }
    if !scores.is_empty() && request.constraint != ConstraintMode::None {
        return Err(request_issue(
            IssueCode::InvalidConstraint,
            "score requests do not accept output constraints",
            request_id,
        ));
    }
    // A grammar decides where constrained output ends.
    if flags & flags::IGNORE_END_OF_SEQUENCE != 0
        && (!scores.is_empty() || request.constraint != ConstraintMode::None)
    {
        return Err(request_issue(
            IssueCode::InvalidConstraint,
            "only unconstrained generation can ignore end-of-sequence",
            request_id,
        ));
    }
    if request.bias_tokens.len() != request.bias_values.len() {
        return Err(request_issue(
            IssueCode::InvalidCount,
            "logit bias tokens and values differ in length",
            request_id,
        ));
    }
    if request.bias_tokens.len() > MAX_LOGIT_BIAS {
        return Err(request_issue(
            IssueCode::InvalidCount,
            "logit bias count exceeds its limit",
            request_id,
        ));
    }
    {
        let mut sorted = request.bias_tokens.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != request.bias_tokens.len() {
            return Err(request_issue(
                IssueCode::InvalidCount,
                "logit bias tokens must be distinct",
                request_id,
            ));
        }
    }
    if request
        .bias_values
        .iter()
        .any(|value| !value.is_finite() || value.abs() > 100.0)
    {
        return Err(request_issue(
            IssueCode::InvalidCount,
            "logit bias values must be finite in [-100, 100]",
            request_id,
        ));
    }
    Ok((
        sampling,
        prompt.clone(),
        scores.clone(),
        end,
        request.bias_values.clone(),
    ))
}

fn request_frame(request: &RequestFrame) -> Result<Vec<u8>, ProtocolError> {
    let (sampling, prompt, scores, end, bias_values) = validated_request(request)?;
    let bias_tokens = &request.bias_tokens;
    let prompt_offset = FRAME_HEADER_BYTES + REQUEST_FIXED_BYTES;
    let spans_offset = prompt_offset + 4 * prompt.len();
    let pixels_offset = spans_offset + IMAGE_SPAN_BYTES * request.image_spans.len();
    let scores_offset = pixels_offset + request.image_pixels.len();
    let end_offset = scores_offset + 4 * scores.len();
    let bias_tokens_offset = end_offset + 4 * end.len();
    let bias_values_offset = bias_tokens_offset + 4 * bias_tokens.len();
    let payload_bytes = bias_values_offset + 4 * bias_values.len() - FRAME_HEADER_BYTES;
    if let Some(issue) = payload_length_issue(FrameType::Request, payload_bytes) {
        return Err(request_issue(issue.code, issue.message, request.request_id));
    }
    let mut frame = header(FrameType::Request, payload_bytes);
    let (thinking_tokens, thinking_end_count, thinking_also_ends) = match &request.thinking_budget {
        None => (NO_THINKING_BUDGET, 0, 0),
        Some(budget) => (budget.tokens, end.len() as u32, budget.also_ends),
    };
    put_u64(&mut frame, request.request_id);
    frame.push(request.priority as u8);
    frame.push(request.constraint as u8);
    put_u64(&mut frame, request.absolute_deadline_unix_micros);
    put_u64(&mut frame, request.remaining_deadline_micros);
    put_u32(&mut frame, request.logical_max_output_tokens);
    put_u32(&mut frame, prompt.len() as u32);
    put_u32(&mut frame, request.image_spans.len() as u32);
    put_f32(&mut frame, sampling.0);
    put_f32(&mut frame, sampling.1);
    put_u32(&mut frame, sampling.2);
    put_f32(&mut frame, sampling.3);
    put_f32(&mut frame, sampling.4);
    put_f32(&mut frame, sampling.5);
    put_f32(&mut frame, sampling.6);
    put_u64(&mut frame, request.seed);
    put_u32(&mut frame, scores.len() as u32);
    put_u32(&mut frame, request.generation_prompt_tokens);
    put_u32(&mut frame, request.flags);
    put_u32(&mut frame, request.top_logprobs);
    put_u32(&mut frame, bias_tokens.len() as u32);
    put_u64(&mut frame, (request.cache_salt & 0xFFFF_FFFF_FFFF_FFFF) as u64);
    put_u64(&mut frame, (request.cache_salt >> 64) as u64);
    put_u32(&mut frame, thinking_tokens);
    put_u32(&mut frame, thinking_end_count);
    put_u32(&mut frame, thinking_also_ends);
    debug_assert_eq!(frame.len(), prompt_offset);
    put_words(&mut frame, &prompt);
    debug_assert_eq!(frame.len(), spans_offset);
    for span in &request.image_spans {
        put_u32(&mut frame, span.offset);
        put_u32(&mut frame, span.tokens);
        put_u32(&mut frame, span.grid_height);
        put_u32(&mut frame, span.grid_width);
        put_u64(&mut frame, span.digest_lo);
        put_u64(&mut frame, span.digest_hi);
    }
    debug_assert_eq!(frame.len(), pixels_offset);
    frame.extend_from_slice(&request.image_pixels);
    put_words(&mut frame, &scores);
    put_words(&mut frame, &end);
    put_words(&mut frame, bias_tokens);
    for value in bias_values {
        put_f32(&mut frame, value);
    }
    Ok(frame)
}

fn cancel_frame(cancel: &CancelFrame) -> Result<Vec<u8>, ProtocolError> {
    if cancel.request_id == 0 {
        return Err(request_issue(
            IssueCode::InvalidRequestId,
            "cancel request id must be non-zero",
            cancel.request_id,
        ));
    }
    let mut frame = header(FrameType::Cancel, 8);
    put_u64(&mut frame, cancel.request_id);
    Ok(frame)
}

fn mask_response_frame(response: &MaskResponseFrame) -> Result<Vec<u8>, ProtocolError> {
    if response.request_id == 0 || response.mask_request_id == 0 {
        return Err(request_issue(
            IssueCode::InvalidRequestId,
            "mask response request ids must be non-zero",
            response.request_id,
        ));
    }
    if response.mask_words.len() % 4 != 0 {
        return Err(request_issue(
            IssueCode::LimitExceeded,
            "mask bytes must contain complete uint32 words",
            response.request_id,
        ));
    }
    let words = response.mask_words.len() / 4;
    if words == 0 || words > MAX_MASK_WORDS {
        return Err(request_issue(
            IssueCode::LimitExceeded,
            "mask response word count exceeds its limit",
            response.request_id,
        ));
    }
    let mut frame = header(FrameType::MaskResponse, 20 + response.mask_words.len());
    put_u64(&mut frame, response.request_id);
    put_u64(&mut frame, response.mask_request_id);
    put_u32(&mut frame, words as u32);
    frame.extend_from_slice(&response.mask_words);
    Ok(frame)
}

fn status_request_frame(request: &StatusRequestFrame) -> Result<Vec<u8>, ProtocolError> {
    let mut frame = header(FrameType::StatusRequest, 8);
    put_u64(&mut frame, request.correlation_id);
    Ok(frame)
}

fn release_request_frame(request: &ReleaseRequestFrame) -> Result<Vec<u8>, ProtocolError> {
    let mut frame = header(FrameType::ReleaseRequest, 8);
    put_u64(&mut frame, request.correlation_id);
    Ok(frame)
}

fn prefill_duty_request_frame(request: &PrefillDutyRequestFrame) -> Result<Vec<u8>, ProtocolError> {
    if !request.duty_percent.is_finite() {
        return Err(ProtocolError(ProtocolIssue::new(
            FailureClass::ProtocolFatal,
            IssueCode::InvalidSampling,
            0,
            "prefill duty must be finite",
        )));
    }
    let mut frame = header(FrameType::PrefillDutyRequest, 4);
    put_f32(&mut frame, request.duty_percent);
    Ok(frame)
}

/// Validate and serialize one client message as a whole frame, built in one
/// buffer: a request's image pixels are copied once.
pub fn serialize_message(message: &ClientMessage) -> Result<Vec<u8>, ProtocolError> {
    match message {
        ClientMessage::Request(request) => request_frame(request),
        ClientMessage::Cancel(cancel) => cancel_frame(cancel),
        ClientMessage::MaskResponse(response) => mask_response_frame(response),
        ClientMessage::StatusRequest(request) => status_request_frame(request),
        ClientMessage::ReleaseRequest(request) => release_request_frame(request),
        ClientMessage::PrefillDutyRequest(request) => prefill_duty_request_frame(request),
    }
}

/// Serialize a validated low-level frame with a strict runtime header.
pub fn serialize_frame(frame: &Frame) -> Result<Vec<u8>, ProtocolError> {
    if let Some(issue) = payload_length_issue(frame.frame_type, frame.payload.len()) {
        return Err(ProtocolError(issue));
    }
    let mut bytes = header(frame.frame_type, frame.payload.len());
    bytes.extend_from_slice(&frame.payload);
    Ok(bytes)
}

/// Re-stamp a REQUEST frame's absolute deadline as `now` plus its budget.
///
/// Every other byte, and every other frame type, is left as it is. Only the
/// integer head is rewritten: decoding and repacking a signaling NaN would
/// change its bits. Invalid and incomplete frames may be the reason for the
/// trace, so no validator runs and short fixed prefixes pass through.
pub fn refresh_request_deadline(frame: &[u8], now_unix_micros: u64) -> Vec<u8> {
    if frame.len() < FRAME_HEADER_BYTES + REQUEST_FIXED_BYTES {
        return frame.to_vec();
    }
    let raw_type = u16::from_le_bytes([frame[6], frame[7]]);
    if FrameType::from_raw(raw_type) != Some(FrameType::Request) {
        return frame.to_vec();
    }
    let head = &frame[FRAME_HEADER_BYTES..];
    let read_u64 = |offset: usize| -> u64 {
        let bytes: [u8; 8] = head[offset..offset + 8].try_into().expect("fixed head");
        u64::from_le_bytes(bytes)
    };
    let request_id = read_u64(0);
    let priority = head[8];
    let constraint = head[9];
    let remaining = read_u64(18);
    let absolute = now_unix_micros.saturating_add(remaining);
    let mut out = Vec::with_capacity(frame.len());
    out.extend_from_slice(&frame[..FRAME_HEADER_BYTES]);
    put_u64(&mut out, request_id);
    out.push(priority);
    out.push(constraint);
    put_u64(&mut out, absolute);
    put_u64(&mut out, remaining);
    out.extend_from_slice(&frame[FRAME_HEADER_BYTES + 26..]);
    out
}
