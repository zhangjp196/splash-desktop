//! The engine client's tests, against a fake engine that speaks the wire
//! protocol over a socket pair, as `dev/tests`' fake engines do for the
//! Python client.

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use splash_server::engine::{EngineClient, EngineError, GenerationRequest};
use splash_server::protocol as wire;

/// One client frame the fake engine reads.
#[derive(Debug, Clone)]
struct ClientFrame {
    frame_type: wire::FrameType,
    payload: Vec<u8>,
}

fn read_client_frame(stream: &mut UnixStream) -> Option<ClientFrame> {
    let mut header = [0u8; 16];
    if stream.read_exact(&mut header).is_err() {
        return None; // the client hung up
    }
    assert_eq!(&header[..4], b"SPLH");
    let raw_type = u16::from_le_bytes([header[6], header[7]]);
    let payload_bytes =
        u64::from_le_bytes(header[8..16].try_into().unwrap()) as usize;
    let mut payload = vec![0u8; payload_bytes];
    stream.read_exact(&mut payload).expect("engine reads payload");
    Some(ClientFrame {
        frame_type: wire::FrameType::from_raw(raw_type).expect("client frame type"),
        payload,
    })
}

fn event_frame(event: &wire::EngineEvent) -> Vec<u8> {
    let frame = match event {
        wire::EngineEvent::Ready(ready) => wire::Frame {
            frame_type: wire::FrameType::Ready,
            payload: {
                let mut payload = Vec::new();
                payload.extend_from_slice(&ready.max_concurrent_requests.to_le_bytes());
                payload.extend_from_slice(&ready.max_context_tokens.to_le_bytes());
                payload.push(ready.vision as u8);
                payload
            }
        },
        wire::EngineEvent::Start(start) => wire::Frame {
            frame_type: wire::FrameType::Start,
            payload: {
                let mut payload = Vec::new();
                payload.extend_from_slice(&start.request_id.to_le_bytes());
                payload.extend_from_slice(&start.lane.to_le_bytes());
                payload.extend_from_slice(&start.matched_prompt_tokens.to_le_bytes());
                payload
            }
        },
        wire::EngineEvent::Tokens(tokens) => wire::Frame {
            frame_type: wire::FrameType::Tokens,
            payload: {
                let mut payload = Vec::new();
                payload.extend_from_slice(&tokens.request_id.to_le_bytes());
                payload.extend_from_slice(&tokens.sequence_offset.to_le_bytes());
                payload.extend_from_slice(&(tokens.tokens.len() as u32).to_le_bytes());
                for token in &tokens.tokens {
                    payload.extend_from_slice(&token.to_le_bytes());
                }
                payload
            }
        },
        wire::EngineEvent::Done(done) => wire::Frame {
            frame_type: wire::FrameType::Done,
            payload: {
                let mut payload = Vec::new();
                payload.extend_from_slice(&done.request_id.to_le_bytes());
                payload.push(done.reason as u8);
                payload.push(done.bounded as u8);
                payload.extend_from_slice(&done.prompt_tokens.to_le_bytes());
                payload.extend_from_slice(&done.completion_tokens.to_le_bytes());
                payload.extend_from_slice(&done.prefill_micros.to_le_bytes());
                payload.extend_from_slice(&done.decode_micros.to_le_bytes());
                payload.extend_from_slice(&done.wall_micros.to_le_bytes());
                payload.extend_from_slice(&done.deduplicated_tokens.to_le_bytes());
                payload.extend_from_slice(&done.drafted_tokens.to_le_bytes());
                payload.extend_from_slice(&done.accepted_draft_tokens.to_le_bytes());
                payload.extend_from_slice(&done.draft_cycles.to_le_bytes());
                payload.extend_from_slice(&(done.option_logits.len() as u32).to_le_bytes());
                payload
            }
        },
        wire::EngineEvent::Error(error) => wire::Frame {
            frame_type: wire::FrameType::Error,
            payload: {
                let mut payload = Vec::new();
                payload.push(error.failure_class as u8);
                payload.push(error.retryable as u8);
                payload.extend_from_slice(&error.retry_after_seconds.to_le_bytes());
                payload.extend_from_slice(&error.request_id.to_le_bytes());
                payload.extend_from_slice(&(error.code.len() as u32).to_le_bytes());
                payload.extend_from_slice(&(error.message.len() as u32).to_le_bytes());
                payload.extend_from_slice(&error.code);
                payload.extend_from_slice(&error.message);
                payload
            }
        },
        wire::EngineEvent::StatusJson(status) => wire::Frame {
            frame_type: wire::FrameType::StatusJson,
            payload: {
                let mut payload = Vec::new();
                payload.extend_from_slice(&status.correlation_id.to_le_bytes());
                payload.extend_from_slice(&status.json);
                payload
            }
        },
        _ => unreachable!("test events only"),
    };
    wire::serialize_frame(&frame).expect("event frame")
}

/// A fake engine: it answers a script of events per request it reads.
fn fake_engine(
    stream: UnixStream,
    script: Arc<Mutex<Vec<Vec<wire::EngineEvent>>>>,
    seen: Arc<Mutex<Vec<ClientFrame>>>,
) {
    std::thread::spawn(move || {
        let mut stream = stream;
        // The engine announces itself as it starts, before any frame.
        stream
            .write_all(&event_frame(&wire::EngineEvent::Ready(wire::ReadyEvent {
                max_concurrent_requests: 4,
                max_context_tokens: 524_288,
                vision: false,
            })))
            .expect("engine writes ready");
        loop {
            let frame = match read_client_frame(&mut stream) {
                Some(frame) => frame,
                None => return,
            };
            seen.lock().expect("seen lock").push(frame.clone());
            match frame.frame_type {
                wire::FrameType::StatusRequest | wire::FrameType::ReleaseRequest => {
                    let correlation = u64::from_le_bytes(
                        frame.payload[..8].try_into().unwrap(),
                    );
                    let answer = wire::StatusJsonEvent {
                        correlation_id: correlation,
                        json: br#"{"schema_version": 8, "ready": true}"#.to_vec(),
                    };
                    stream
                        .write_all(&event_frame(&wire::EngineEvent::StatusJson(answer)))
                        .expect("engine writes status");
                }
                wire::FrameType::Request => {
                    let request_id = u64::from_le_bytes(
                        frame.payload[..8].try_into().unwrap(),
                    );
                    let prompt_count = u32::from_le_bytes(
                        frame.payload[30..34].try_into().unwrap(),
                    );
                    let mut script = script.lock().expect("script lock");
                    let events = if script.is_empty() {
                        // The default script: start, one token, done.
                        vec![
                            wire::EngineEvent::Start(wire::StartEvent {
                                request_id,
                                lane: 0,
                                matched_prompt_tokens: 0,
                            }),
                            wire::EngineEvent::Tokens(wire::TokensEvent {
                                request_id,
                                sequence_offset: 0,
                                tokens: vec![42],
                            }),
                            wire::EngineEvent::Done(wire::DoneEvent {
                                request_id,
                                reason: wire::FinishReason::Stop,
                                prompt_tokens: prompt_count,
                                completion_tokens: 1,
                                prefill_micros: 0,
                                decode_micros: 0,
                                wall_micros: 0,
                                option_logits: vec![],
                                deduplicated_tokens: 0,
                                drafted_tokens: 0,
                                accepted_draft_tokens: 0,
                                draft_cycles: 0,
                                bounded: false,
                            }),
                        ]
                    } else {
                        script.remove(0)
                    };
                    drop(script);
                    for event in events {
                        stream.write_all(&event_frame(&event)).expect("engine writes");
                    }
                }
                _ => {}
            }
        }
    });
}

fn paired_client(script: Vec<Vec<wire::EngineEvent>>) -> (EngineClient, SeenFrames) {
    let (engine_side, client_side) = UnixStream::pair().expect("socket pair");
    let client_read = client_side.try_clone().expect("client read clone");
    let client_write = client_side.try_clone().expect("client write clone");
    let seen: SeenFrames = Arc::new(Mutex::new(Vec::new()));
    fake_engine(engine_side, Arc::new(Mutex::new(script)), seen.clone());
    (
        EngineClient::over_streams(Box::new(client_read), Box::new(client_write), 8),
        seen,
    )
}

/// The client frames the fake engine has read.
type SeenFrames = Arc<Mutex<Vec<ClientFrame>>>;

fn generation_request() -> GenerationRequest {
    GenerationRequest {
        frame: wire::RequestFrame {
            logical_max_output_tokens: 8,
            prompt_tokens: vec![1, 2, 3],
            ..wire::RequestFrame::default()
        },
        deadline: None,
        mask_provider: None,
    }
}

#[test]
fn a_request_runs_start_tokens_done() {
    let (client, _engine) = paired_client(vec![]);
    assert!(client.wait_ready(Some(Duration::from_secs(5))));
    let call = client
        .submit(&generation_request(), None, None)
        .expect("submit");
    let done = call.result(Some(Duration::from_secs(5))).expect("done");
    assert_eq!(done.completion_tokens, 1);
    assert_eq!(done.reason, wire::FinishReason::Stop);
    assert_eq!(client.pending_count(), 0);
}

#[test]
fn submit_before_ready_is_refused() {
    let (client, _engine) = paired_client(vec![]);
    let error = client
        .submit(&generation_request(), None, None)
        .expect_err("refused before ready");
    assert!(matches!(error, EngineError::EngineUnhealthy(_)));
}

#[test]
fn a_request_error_ends_its_call() {
    let script = vec![vec![wire::EngineEvent::Error(wire::ErrorEvent {
        failure_class: wire::FailureClass::RequestError,
        request_id: 1,
        retryable: true,
        code: b"deadline_exceeded".to_vec(),
        message: b"request deadline expired".to_vec(),
        retry_after_seconds: 30,
    })]];
    let (client, _engine) = paired_client(script);
    assert!(client.wait_ready(Some(Duration::from_secs(5))));
    let call = client.submit(&generation_request(), None, None).expect("submit");
    let error = call.result(Some(Duration::from_secs(5))).expect_err("failed");
    match error {
        EngineError::RequestFailed(failure) => {
            assert_eq!(failure.machine_code(), "deadline_exceeded");
            assert!(failure.retryable);
            assert_eq!(failure.retry_after_seconds, 30);
        }
        other => panic!("expected a request failure, got {other}"),
    }
    assert_eq!(client.pending_count(), 0);
}

#[test]
fn an_event_for_an_unknown_request_fails_the_generation() {
    let script = vec![vec![
        wire::EngineEvent::Start(wire::StartEvent {
            request_id: 99,
            lane: 0,
            matched_prompt_tokens: 0,
        }),
    ]];
    let (client, _engine) = paired_client(script);
    assert!(client.wait_ready(Some(Duration::from_secs(5))));
    let call = client.submit(&generation_request(), None, None).expect("submit");
    let error = call.result(Some(Duration::from_secs(5))).expect_err("failed");
    assert!(matches!(error, EngineError::ProtocolFatal(_)));
    assert!(client.fatal_error().is_some());
    // The client refuses further work.
    assert!(client.submit(&generation_request(), None, None).is_err());
}

#[test]
fn status_correlates_its_answer() {
    let (client, _engine) = paired_client(vec![]);
    assert!(client.wait_ready(Some(Duration::from_secs(5))));
    let status = client.status(Duration::from_secs(5)).expect("status");
    assert!(status.json.windows(b"ready".len()).any(|w| w == b"ready"));
}

#[test]
fn the_pending_limit_refuses_the_next_request() {
    // A script whose first request never finishes, so the slot stays held.
    let (client, _engine) = paired_client(vec![vec![]]);
    assert!(client.wait_ready(Some(Duration::from_secs(5))));
    let first = client.submit(&generation_request(), None, None).expect("first");
    for _ in 1..8 {
        client.submit(&generation_request(), None, None).expect("fill");
    }
    let error = client
        .submit(&generation_request(), None, None)
        .expect_err("limit full");
    assert!(matches!(error, EngineError::PendingLimitExceeded(_)));
    // Cancelling the stuck request frees its slot.
    assert!(client.cancel(&first));
    let _ = first.result(Some(Duration::from_secs(5)));
    let next = client.submit(&generation_request(), None, None);
    assert!(next.is_ok(), "the cancelled slot was not released");
}

#[test]
fn close_fails_every_pending_call() {
    let (client, _engine) = paired_client(vec![vec![]]);
    assert!(client.wait_ready(Some(Duration::from_secs(5))));
    let call = client.submit(&generation_request(), None, None).expect("submit");
    client.close();
    let error = call.result(Some(Duration::from_secs(5))).expect_err("closed");
    assert!(matches!(error, EngineError::RuntimeClosed(_)));
    assert!(client.submit(&generation_request(), None, None).is_err());
}

#[test]
fn a_cancel_frame_reaches_the_engine() {
    let (client, seen) = paired_client(vec![vec![]]);
    assert!(client.wait_ready(Some(Duration::from_secs(5))));
    let call = client.submit(&generation_request(), None, None).expect("submit");
    assert!(client.cancel(&call));
    // The engine read the request, then the cancel.
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        let count = seen.lock().expect("seen lock").len();
        if count >= 2 || std::time::Instant::now() >= deadline {
            assert_eq!(count, 2, "request then cancel");
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let seen = seen.lock().expect("seen lock");
    assert_eq!(seen.len(), 2, "request then cancel");
    assert_eq!(seen[0].frame_type, wire::FrameType::Request);
    assert_eq!(seen[1].frame_type, wire::FrameType::Cancel);
    let request_id = u64::from_le_bytes(seen[1].payload[..8].try_into().unwrap());
    assert_eq!(request_id, call.request_id);
    drop(seen);
    // A second cancel is refused.
    assert!(!client.cancel(&call));
}
