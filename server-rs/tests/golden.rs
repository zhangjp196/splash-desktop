//! The wire protocol's golden test, ported from
//! `dev/tests/server/test_protocol_python.py`: the messages the Python
//! builds must serialize to exactly the bytes
//! `dev/tests/engine/protocol_golden.txt` pins, and the events those bytes
//! carry must decode back to the same values.

use std::path::PathBuf;

use splash_server::protocol as p;

fn f32(value: f64) -> f32 {
    value as f32
}

fn example_request() -> p::RequestFrame {
    p::RequestFrame {
        request_id: 0x0123456789ABCDEF,
        priority: p::RequestPriority::Foreground,
        absolute_deadline_unix_micros: 1_800_000_000_000_000,
        remaining_deadline_micros: 45_000_000,
        logical_max_output_tokens: 32_768,
        prompt_tokens: vec![0, 1, 42, 0x80000000, 0xFFFFFFFF],
        sampling: p::SamplingParameters {
            temperature: f32(0.8),
            top_p: f32(0.95),
            top_k: 32,
            presence_penalty: 1.5,
            frequency_penalty: -0.25,
            repetition_penalty: f32(1.1),
            min_p: f32(0.05),
        },
        seed: 0xFEDCBA9876543210,
        constraint: p::ConstraintMode::TokenMask,
        generation_prompt_tokens: 2,
        cache_salt: 0x00112233445566778899AABBCCDDEEFF,
        thinking_budget: Some(p::ThinkingBudget {
            tokens: 4096,
            end: vec![198, 248069],
            also_ends: 248058,
        }),
        ..p::RequestFrame::default()
    }
}

fn example_image_request() -> p::RequestFrame {
    let span = p::ImageSpan {
        offset: 1,
        tokens: 1,
        grid_height: 2,
        grid_width: 2,
        digest_lo: 0x1111222233334444,
        digest_hi: 0x5555666677778888,
    };
    p::RequestFrame {
        prompt_tokens: vec![7, 3, 9],
        image_spans: vec![span],
        image_pixels: (0..span.pixel_bytes())
            .map(|i| ((i * 7 + 1) & 0xFF) as u8)
            .collect(),
        ..example_request()
    }
}

fn example_score_request() -> p::RequestFrame {
    p::RequestFrame {
        logical_max_output_tokens: 0,
        prompt_tokens: vec![5, 6, 7],
        sampling: p::SamplingParameters::default(),
        constraint: p::ConstraintMode::None,
        score_tokens: vec![101, 202, 303],
        thinking_budget: None,
        ..example_request()
    }
}

fn example_ignore_eos_request() -> p::RequestFrame {
    p::RequestFrame {
        constraint: p::ConstraintMode::None,
        flags: p::flags::IGNORE_END_OF_SEQUENCE,
        ..example_request()
    }
}

fn example_progress_request() -> p::RequestFrame {
    p::RequestFrame {
        flags: p::flags::RETURN_PROGRESS,
        ..example_request()
    }
}

fn client_messages() -> Vec<(&'static str, p::ClientMessage)> {
    vec![
        ("request", p::ClientMessage::Request(example_request())),
        (
            "request_image",
            p::ClientMessage::Request(example_image_request()),
        ),
        (
            "request_score",
            p::ClientMessage::Request(example_score_request()),
        ),
        (
            "request_ignore_eos",
            p::ClientMessage::Request(example_ignore_eos_request()),
        ),
        (
            "request_progress",
            p::ClientMessage::Request(example_progress_request()),
        ),
        (
            "cancel",
            p::ClientMessage::Cancel(p::CancelFrame { request_id: 91 }),
        ),
        (
            "mask_response",
            p::ClientMessage::MaskResponse(p::MaskResponseFrame {
                request_id: 91,
                mask_request_id: 7,
                mask_words: [
                    0xFFFFFFFFu32.to_le_bytes(),
                    0u32.to_le_bytes(),
                    0xA5A5A5A5u32.to_le_bytes(),
                ]
                .concat(),
            }),
        ),
        (
            "status_request",
            p::ClientMessage::StatusRequest(p::StatusRequestFrame { correlation_id: 808 }),
        ),
        (
            "release_request",
            p::ClientMessage::ReleaseRequest(p::ReleaseRequestFrame { correlation_id: 808 }),
        ),
        (
            "prefill_duty_request",
            p::ClientMessage::PrefillDutyRequest(p::PrefillDutyRequestFrame { duty_percent: 60.0 }),
        ),
    ]
}

fn engine_events() -> Vec<(&'static str, p::EngineEvent)> {
    vec![
        (
            "ready",
            p::EngineEvent::Ready(p::ReadyEvent {
                max_concurrent_requests: 4,
                max_context_tokens: 524_288,
                vision: false,
            }),
        ),
        (
            "ready_vision",
            p::EngineEvent::Ready(p::ReadyEvent {
                max_concurrent_requests: 4,
                max_context_tokens: 524_288,
                vision: true,
            }),
        ),
        (
            "start",
            p::EngineEvent::Start(p::StartEvent {
                request_id: 91,
                lane: 2,
                matched_prompt_tokens: 4096,
            }),
        ),
        (
            "prompt_progress",
            p::EngineEvent::PromptProgress(p::PromptProgressEvent {
                request_id: 91,
                processed_tokens: 2048,
                elapsed_micros: 123456,
            }),
        ),
        (
            "tokens",
            p::EngineEvent::Tokens(p::TokensEvent {
                request_id: 91,
                sequence_offset: 17,
                tokens: vec![10, 11, 12],
            }),
        ),
        (
            "logprobs",
            p::EngineEvent::Logprobs(p::LogprobsEvent {
                request_id: 91,
                sequence_offset: 17,
                values: vec![p::LogprobValues {
                    token: 10,
                    logprob: f32(-0.5),
                    top: vec![(11, f32(-1.0)), (12, f32(-2.0))],
                }],
            }),
        ),
        (
            "mask_request_initial",
            p::EngineEvent::MaskRequest(p::MaskRequestEvent {
                request_id: 91,
                mask_request_id: 6,
                words_per_mask: 4,
                simulation_tokens: vec![],
            }),
        ),
        (
            "mask_request_verify",
            p::EngineEvent::MaskRequest(p::MaskRequestEvent {
                request_id: 91,
                mask_request_id: 7,
                words_per_mask: 4,
                simulation_tokens: vec![101, 102, 103],
            }),
        ),
        (
            "done",
            p::EngineEvent::Done(p::DoneEvent {
                request_id: 91,
                reason: p::FinishReason::Stop,
                prompt_tokens: 4096,
                completion_tokens: 512,
                prefill_micros: 1000,
                decode_micros: 2000,
                wall_micros: 3500,
                option_logits: vec![],
                deduplicated_tokens: 64,
                drafted_tokens: 350,
                accepted_draft_tokens: 280,
                draft_cycles: 50,
                bounded: false,
            }),
        ),
        (
            "done_scored",
            p::EngineEvent::Done(p::DoneEvent {
                request_id: 91,
                reason: p::FinishReason::Stop,
                prompt_tokens: 4096,
                completion_tokens: 0,
                prefill_micros: 1000,
                decode_micros: 0,
                wall_micros: 3500,
                option_logits: vec![f32(1.5), f32(-2.25), f32(0.5)],
                deduplicated_tokens: 32,
                drafted_tokens: 0,
                accepted_draft_tokens: 0,
                draft_cycles: 0,
                bounded: false,
            }),
        ),
        (
            "done_bounded",
            p::EngineEvent::Done(p::DoneEvent {
                request_id: 91,
                reason: p::FinishReason::Length,
                prompt_tokens: 4096,
                completion_tokens: 63,
                prefill_micros: 1000,
                decode_micros: 2000,
                wall_micros: 3500,
                option_logits: vec![],
                deduplicated_tokens: 0,
                drafted_tokens: 0,
                accepted_draft_tokens: 0,
                draft_cycles: 0,
                bounded: true,
            }),
        ),
        (
            "error_request",
            p::EngineEvent::Error(p::ErrorEvent {
                failure_class: p::FailureClass::RequestError,
                request_id: 91,
                retryable: true,
                code: b"deadline_exceeded".to_vec(),
                message: b"request deadline expired".to_vec(),
                retry_after_seconds: 0,
            }),
        ),
        (
            "error_retry_after",
            p::EngineEvent::Error(p::ErrorEvent {
                failure_class: p::FailureClass::RequestError,
                request_id: 91,
                retryable: true,
                code: b"kv_write_budget".to_vec(),
                message: b"the SSD cache's write budget is spent".to_vec(),
                retry_after_seconds: 21_600,
            }),
        ),
        (
            "error_engine",
            p::EngineEvent::Error(p::ErrorEvent {
                failure_class: p::FailureClass::EngineUnhealthy,
                request_id: 0,
                retryable: false,
                code: b"gpu_fault".to_vec(),
                message: b"Metal command buffer failed".to_vec(),
                retry_after_seconds: 0,
            }),
        ),
        (
            "error_protocol",
            p::EngineEvent::Error(p::ErrorEvent {
                failure_class: p::FailureClass::ProtocolFatal,
                request_id: 0,
                retryable: false,
                code: b"bad_frame".to_vec(),
                message: b"stream framing cannot be trusted".to_vec(),
                retry_after_seconds: 0,
            }),
        ),
        (
            "status_json",
            p::EngineEvent::StatusJson(p::StatusJsonEvent {
                correlation_id: 808,
                json: b"{\n  \"schema_version\": 6, \"ready\": true\n}".to_vec(),
            }),
        ),
    ]
}

fn golden_path() -> PathBuf {
    // The crate sits at server-rs/; the repository root is its parent.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate parent")
        .join("dev/tests/engine/protocol_golden.txt")
}

fn read_golden() -> Vec<(&'static str, Vec<u8>)> {
    let text = std::fs::read_to_string(golden_path()).expect("golden file");
    let mut names: Vec<&'static str> = Vec::new();
    let mut bytes: Vec<Vec<u8>> = Vec::new();
    for line in text.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (name, data) = line.split_once(' ').expect("golden line");
        names.push(Box::leak(name.to_string().into_boxed_str()));
        bytes.push(hex::decode(data));
    }
    names.into_iter().zip(bytes).collect()
}

/// Minimal hex decoding, so the crate keeps no dependencies.
mod hex {
    pub fn decode(text: &str) -> Vec<u8> {
        let text = text.trim();
        assert!(text.len() % 2 == 0, "odd hex: {text}");
        (0..text.len() / 2)
            .map(|i| u8::from_str_radix(&text[i * 2..i * 2 + 2], 16).expect("hex digit"))
            .collect()
    }
}

/// Feed the whole stream through the incremental parser, one byte at a
/// time, and collect the frames in order.
fn parse_all(data: &[u8]) -> Vec<p::Frame> {
    let mut parser = p::FrameParser::new();
    let mut frames = Vec::new();
    let mut offset = 0usize;
    while offset < data.len() {
        let step = parser.consume(&data[offset..offset + 1]);
        assert!(step.issue.is_none(), "unexpected parser issue: {:?}", step.issue);
        offset += step.consumed_bytes;
        if let Some(frame) = step.frame {
            frames.push(frame);
        }
    }
    assert!(parser.finish().is_none(), "parser finished with an issue");
    frames
}

#[test]
fn client_frames_match_the_golden_bytes() {
    let golden = read_golden();
    for (name, message) in client_messages() {
        let expected = golden
            .iter()
            .find(|(golden_name, _)| *golden_name == name)
            .unwrap_or_else(|| panic!("golden has no {name}"))
            .1
            .clone();
        let actual = p::serialize_message(&message).unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(
            actual,
            expected,
            "{name} does not match the golden bytes"
        );
    }
}

#[test]
fn engine_events_match_the_golden_bytes() {
    let golden = read_golden();
    for (name, event) in engine_events() {
        let expected = golden
            .iter()
            .find(|(golden_name, _)| *golden_name == name)
            .unwrap_or_else(|| panic!("golden has no {name}"))
            .1
            .clone();
        // The event's own serialization is the engine's side; here the
        // bytes are decoded and the values compared, then the frame is
        // re-serialized through the transport's frame serializer.
        let frame = p::Frame {
            frame_type: match &event {
                p::EngineEvent::Ready(_) => p::FrameType::Ready,
                p::EngineEvent::Start(_) => p::FrameType::Start,
                p::EngineEvent::PromptProgress(_) => p::FrameType::PromptProgress,
                p::EngineEvent::Tokens(_) => p::FrameType::Tokens,
                p::EngineEvent::Logprobs(_) => p::FrameType::Logprobs,
                p::EngineEvent::MaskRequest(_) => p::FrameType::MaskRequest,
                p::EngineEvent::Done(_) => p::FrameType::Done,
                p::EngineEvent::Error(_) => p::FrameType::Error,
                p::EngineEvent::StatusJson(_) => p::FrameType::StatusJson,
            },
            payload: expected[16..].to_vec(),
        };
        let decoded = p::decode_frame(&frame).unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(decoded, event, "{name} did not decode to its values");
        let serialized = p::serialize_frame(&frame).unwrap();
        assert_eq!(serialized, expected, "{name} does not re-serialize exactly");
    }
}

#[test]
fn the_whole_golden_stream_parses_incrementally() {
    let golden = read_golden();
    // The engine events only: the incremental parser reads the engine's
    // stream, and a client frame type is an unknown frame type to it.
    let events: Vec<&(&'static str, Vec<u8>)> = golden
        .iter()
        .filter(|(name, _)| EVENT_NAMES.contains(&name))
        .collect();
    let mut stream = Vec::new();
    for (_, bytes) in &events {
        stream.extend_from_slice(bytes);
    }
    let frames = parse_all(&stream);
    assert_eq!(frames.len(), events.len(), "frame count");
    for (frame, (name, bytes)) in frames.iter().zip(&events) {
        assert_eq!(
            p::serialize_frame(frame).unwrap(),
            *bytes,
            "{name} did not survive the incremental parse"
        );
    }
}

/// The engine event names the golden file carries.
const EVENT_NAMES: &[&str] = &[
    "ready",
    "ready_vision",
    "start",
    "prompt_progress",
    "tokens",
    "logprobs",
    "mask_request_initial",
    "mask_request_verify",
    "done",
    "done_scored",
    "done_bounded",
    "error_request",
    "error_retry_after",
    "error_engine",
    "error_protocol",
    "status_json",
];

#[test]
fn request_deadlines_refresh_without_touching_other_bytes() {
    let golden = read_golden();
    let request = golden.iter().find(|(n, _)| *n == "request").unwrap().1.clone();
    let now: u64 = 1_000_000;
    let refreshed = p::refresh_request_deadline(&request, now);
    assert_eq!(refreshed.len(), request.len());
    // The absolute deadline becomes now + remaining; the remaining deadline
    // and everything past the integer head stay as they were.
    let remaining = {
        let bytes: [u8; 8] = request[16 + 18..16 + 26].try_into().unwrap();
        u64::from_le_bytes(bytes)
    };
    let absolute = {
        let bytes: [u8; 8] = refreshed[16 + 10..16 + 18].try_into().unwrap();
        u64::from_le_bytes(bytes)
    };
    assert_eq!(absolute, now + remaining);
    // Everything past the integer head (request_id, priority, constraint,
    // absolute, remaining) stays as it was.
    assert_eq!(&refreshed[16 + 26..], &request[16 + 26..]);
    // A short frame passes through unchanged.
    assert_eq!(p::refresh_request_deadline(&request[..20], 0), request[..20]);
}
