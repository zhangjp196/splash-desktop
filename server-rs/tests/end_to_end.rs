//! The end-to-end path's test: a Chat request from HTTP through the
//! pipeline to the engine and its answer back, against a fake engine that
//! speaks the wire protocol.

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use splash_server::chat_format::qwen;
use splash_server::chat_messages::normalize_messages;
use splash_server::dialects::openai;
use splash_server::engine::EngineClient;
use splash_server::output::StreamingToolCallProjector;
use splash_server::serve::Pipeline;
use splash_server::templates::load_chat_templates;
use splash_server::tokenization::{FastTokenizer, PromptTokenizer};

#[derive(Clone)]
struct CapturedConsole(Arc<Mutex<Vec<u8>>>);

impl Write for CapturedConsole {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
}

fn qwen_directory() -> Option<PathBuf> {
    let base = std::env::var("HOME").ok()?;
    let dir: PathBuf = base.parse().ok()?;
    let snapshots = dir.join(
        ".cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots",
    );
    let snapshot = std::fs::read_dir(&snapshots).ok()?.next()?.ok()?;
    Some(snapshot.path())
}

/// The fake engine: Ready at once, then per request a start, the tokens
/// of a fixed answer, and a done.
fn fake_engine(mut stream: UnixStream, answer: &'static str) {
    let ready = wire_event(&json!({
        "frame_type": 0x0100,
        "max_concurrent_requests": 4,
        "max_context_tokens": 524288,
        "vision": false,
    }));
    let _ = stream.write_all(&ready);
    // The prefill duty the engine currently holds, which a control request
    // changes and the status document reports back.
    let mut duty_percent = 100.0f64;
    loop {
        let mut header = [0u8; 16];
        if stream.read_exact(&mut header).is_err() {
            return;
        }
        let raw_type = u16::from_le_bytes([header[6], header[7]]);
        let payload_bytes =
            u64::from_le_bytes(header[8..16].try_into().unwrap()) as usize;
        let mut payload = vec![0u8; payload_bytes];
        if payload_bytes > 0 {
            stream.read_exact(&mut payload).expect("engine reads payload");
        }
        if raw_type == 0x0006 {
            // A prefill duty request: the engine takes it and answers
            // nothing; the status request written after it reads it back.
            duty_percent = f32::from_le_bytes(payload[..4].try_into().unwrap()) as f64;
            continue;
        }
        if raw_type == 0x0004 {
            // A status request: answer with the document the live board
            // reads, as the engine composes it.
            let correlation_id = u64::from_le_bytes(payload[..8].try_into().unwrap());
            let status = json!({
                "schema_version": 8,
                "ready": true,
                "maximum_context_tokens": 524288,
                "metrics": {
                    "decode_tokens_per_second": 12.5,
                    "prefill_tokens_per_second": 100.0,
                },
                "requests": {
                    "submitted": 3, "completed": 2, "failed": 0, "cancelled": 0,
                },
                "speculation": {"provider": "dflash2", "proposal_rows": 4},
                "throttle": {"duty_percent": duty_percent},
            });
            let _ = stream.write_all(&status_event(correlation_id, &status));
            continue;
        }
        if raw_type != 0x0001 {
            continue;
        }
        let request_id = u64::from_le_bytes(payload[..8].try_into().unwrap());
        let prompt_count = u32::from_le_bytes(payload[30..34].try_into().unwrap());
        for event in [
            json!({"frame_type": 0x0101, "request_id": request_id, "lane": 0,
                   "matched_prompt_tokens": 0}),
            // Two emissions: a lone first token, then the rest, so the
            // first batch is smaller than the answer and the rate exists.
            json!({"frame_type": 0x0102, "request_id": request_id, "sequence_offset": 0,
                   "tokens": answer_tokens(answer)[..1].to_vec()}),
            json!({"frame_type": 0x0102, "request_id": request_id, "sequence_offset": 1,
                   "tokens": answer_tokens(answer)[1..].to_vec()}),
            json!({"frame_type": 0x0104, "request_id": request_id, "reason": 0,
                   "bounded": false, "prompt_tokens": prompt_count,
                   "completion_tokens": answer_tokens(answer).len(),
                   "prefill_micros": 1000, "decode_micros": 2000,
                   "wall_micros": 3500, "deduplicated_tokens": 0,
                   "drafted_tokens": 0, "accepted_draft_tokens": 0,
                   "draft_cycles": 0, "logit_count": 0}),
        ] {
            let _ = stream.write_all(&wire_event(&event));
        }
    }
}

/// A status JSON event: the correlation id it answers and the document.
fn status_event(correlation_id: u64, status: &Value) -> Vec<u8> {
    let body = serde_json::to_vec(status).expect("status JSON");
    let mut payload = Vec::new();
    payload.extend_from_slice(&correlation_id.to_le_bytes());
    payload.extend_from_slice(&body);
    let mut frame = Vec::new();
    frame.extend_from_slice(b"SPLH");
    frame.extend_from_slice(&13u16.to_le_bytes());
    frame.extend_from_slice(&0x0107u16.to_le_bytes());
    frame.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    frame.extend_from_slice(&payload);
    frame
}

/// The stream's `field` deltas, joined in arrival order: a reply may arrive
/// as several chunks, and its text is what they spell together.
fn streamed_text(lines: &str, field: &str) -> String {
    let mut text = String::new();
    for line in lines.lines() {
        let Some(payload) = line.strip_prefix("data: ") else {
            continue;
        };
        if payload == "[DONE]" {
            continue;
        }
        let Ok(chunk) = serde_json::from_str::<Value>(payload) else {
            continue;
        };
        if let Some(part) = chunk
            .pointer(&format!("/choices/0/delta/{field}"))
            .and_then(Value::as_str)
        {
            text.push_str(part);
        }
    }
    text
}

/// The answer's tokens, as the tokenizers crate encodes it.
fn answer_tokens(answer: &str) -> Vec<u32> {
    let directory = qwen_directory().expect("qwen tokenizer");
    let tokenizer = FastTokenizer::from_file(&directory).expect("loads");
    tokenizer.encode(answer, false).expect("encodes")
}

/// One wire frame carrying one event, as the engine writes it.
fn wire_event(event: &Value) -> Vec<u8> {
    let frame_type = event["frame_type"].as_u64().unwrap() as u16;
    let payload = match frame_type {
        0x0100 => {
            // The wire's READY is <IIB>: u32, u32, one byte.
            let mut payload = Vec::new();
            payload.extend_from_slice(
                &(event["max_concurrent_requests"].as_u64().unwrap() as u32).to_le_bytes(),
            );
            payload
                .extend_from_slice(&(event["max_context_tokens"].as_u64().unwrap() as u32).to_le_bytes());
            payload.push(event["vision"].as_bool().unwrap() as u8);
            payload
        }
        0x0101 => {
            let mut payload = Vec::new();
            payload.extend_from_slice(&event["request_id"].as_u64().unwrap().to_le_bytes());
            payload.extend_from_slice(&(event["lane"].as_u64().unwrap() as u32).to_le_bytes());
            payload.extend_from_slice(
                &(event["matched_prompt_tokens"].as_u64().unwrap() as u32).to_le_bytes(),
            );
            payload
        }
        0x0102 => {
            let mut payload = Vec::new();
            payload.extend_from_slice(&event["request_id"].as_u64().unwrap().to_le_bytes());
            payload.extend_from_slice(
                &(event["sequence_offset"].as_u64().unwrap() as u32).to_le_bytes(),
            );
            let tokens: Vec<u32> = event["tokens"]
                .as_array()
                .expect("tokens")
                .iter()
                .map(|t| t.as_u64().unwrap() as u32)
                .collect();
            payload.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
            for token in tokens {
                payload.extend_from_slice(&token.to_le_bytes());
            }
            payload
        }
        0x0104 => {
            let mut payload = Vec::new();
            payload.extend_from_slice(&event["request_id"].as_u64().unwrap().to_le_bytes());
            payload.push(event["reason"].as_u64().unwrap() as u8);
            payload.push(event["bounded"].as_bool().unwrap() as u8);
            payload.extend_from_slice(&(event["prompt_tokens"].as_u64().unwrap() as u32).to_le_bytes());
            payload.extend_from_slice(
                &(event["completion_tokens"].as_u64().unwrap() as u32).to_le_bytes(),
            );
            payload.extend_from_slice(&event["prefill_micros"].as_u64().unwrap().to_le_bytes());
            payload.extend_from_slice(&event["decode_micros"].as_u64().unwrap().to_le_bytes());
            payload.extend_from_slice(&event["wall_micros"].as_u64().unwrap().to_le_bytes());
            payload.extend_from_slice(
                &(event["deduplicated_tokens"].as_u64().unwrap() as u32).to_le_bytes(),
            );
            payload.extend_from_slice(&(event["drafted_tokens"].as_u64().unwrap() as u32).to_le_bytes());
            payload.extend_from_slice(
                &(event["accepted_draft_tokens"].as_u64().unwrap() as u32).to_le_bytes(),
            );
            payload.extend_from_slice(&(event["draft_cycles"].as_u64().unwrap() as u32).to_le_bytes());
            let option_logits = event["option_logits"].as_array().cloned().unwrap_or_default();
            let logit_count = event["logit_count"].as_u64().unwrap_or(option_logits.len() as u64) as u32;
            payload.extend_from_slice(&logit_count.to_le_bytes());
            for logit in option_logits { payload.extend_from_slice(&(logit.as_f64().unwrap() as f32).to_le_bytes()); }
            payload
        }
        _ => unreachable!("test events only"),
    };
    let mut frame = Vec::new();
    frame.extend_from_slice(b"SPLH");
    frame.extend_from_slice(&13u16.to_le_bytes());
    frame.extend_from_slice(&frame_type.to_le_bytes());
    frame.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    frame.extend_from_slice(&payload);
    frame
}

#[test]
fn a_chat_request_runs_the_whole_pipeline() {
    let Some(directory) = qwen_directory() else {
        eprintln!("no local Qwen tokenizer; skipped");
        return;
    };
    // The engine side of a socket pair, with the answer it writes. The
    // template opens the think block, so the output starts in the reasoning
    // and the answer follows the marker that closes it.
    let (engine_side, client_read) = UnixStream::pair().expect("socket pair");
    let client_write = client_read.try_clone().expect("clone");
    std::thread::spawn(move || {
        fake_engine(
            engine_side,
            "Let me think.\u{003c}\u{002f}\u{0074}\u{0068}\u{0069}\u{006e}\u{006b}\u{003e}The answer is 42.",
        )
    });
    let client = EngineClient::over_streams(Box::new(client_read), Box::new(client_write), 8);

    let format = qwen();
    let tokenizer = Arc::new(FastTokenizer::from_file(&directory).expect("loads"));
    let prompt_tokenizer =
        PromptTokenizer::from_directory(tokenizer.clone(), &directory, format.message_boundary);
    let templates = load_chat_templates(&directory).expect("loads");
    assert!(!templates.is_empty(), "the model has a chat template");
    let pipeline = Pipeline {
        tokenizer,
        prompt_tokenizer,
        templates,
        chat_format: format,
        client,
        model: "Qwen".into(),
        images: splash_server::images::ImageCache::new(),
        max_image_pixels: splash_server::images::MAX_PIXELS,
        default_reasoning_effort: None,
        thinking_end_tokens: None,
        documents: None,
        constraints: None,
    };

    // The request: normalize, render, tokenize, submit, collect, assemble.
    let body = json!({
        "model": "Qwen",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is the answer?"},
        ],
        "max_tokens": 64,
    });
    let normalized = normalize_messages(&body["messages"], false).expect("normalizes");
    let templated = crate_chat_messages_template(&normalized);
    let source = pipeline
        .templates
        .select(None, None)
        .expect("the model has a chat template");
    let renderer =
        splash_server::templates::ChatTemplateRenderer::new(&source).expect("compiles");
    let rendered = renderer
        .render(&templated, None, None, true, &Default::default())
        .expect("renders");
    assert!(!rendered.is_empty());
    let prompt_tokens = pipeline.prompt_tokenizer.encode(&rendered);
    assert!(!prompt_tokens.is_empty());

    let prepared = pipeline.prepare(&body, &mut || Ok(())).expect("prepares");
    let (call, tokens, first_batch) = pipeline.submit(&prepared, 64).expect("submits");
    let done = call.result(None).expect("done");
    assert_eq!(done.prompt_tokens as usize, prompt_tokens.len());
    let job = openai::JobInfo {
        public_id: "test1".into(),
        created_at: 1_700_000_000,
        reasoning_tokens: 0,
    };
    let response = pipeline
        .complete(
            &body,
            &prepared,
            &call,
            &tokens,
            first_batch.load(std::sync::atomic::Ordering::Relaxed),
            "Qwen",
            &job,
        )
        .expect("assembles");
    // The response is OpenAI's shape, with the answer the engine wrote and
    // the reasoning that preceded it kept apart from it.
    assert_eq!(response["object"], json!("chat.completion"));
    assert_eq!(response["model"], json!("Qwen"));
    assert!(prepared.thinking, "the template opens the think block");
    assert_eq!(response["choices"][0]["message"]["content"], json!("The answer is 42."));
    assert_eq!(
        response["choices"][0]["message"]["reasoning_content"],
        json!("Let me think.")
    );
    assert!(!response.to_string().contains("</think>"), "the close is dropped");
    assert_eq!(response["choices"][0]["finish_reason"], json!("stop"));
    assert_eq!(response["usage"]["prompt_tokens"], json!(prompt_tokens.len() as u64));
    // The metrics object carries the decode rate the chat page reads.
    assert!(
        response["metrics"]["request_latency"]["stream_tokens_per_second"].is_number(),
        "{response}"
    );
}

/// The request's reasoning switch reaches the prompt: an effort of `none`
/// renders the think block closed, an effort renders it open, and the
/// request's own `enable_thinking` outranks the effort.
#[test]
fn a_reasoning_effort_decides_whether_the_prompt_thinks() {
    let Some(directory) = qwen_directory() else {
        eprintln!("no local Qwen tokenizer; skipped");
        return;
    };
    let format = qwen();
    let tokenizer = Arc::new(FastTokenizer::from_file(&directory).expect("loads"));
    let (engine_side, client_read) = UnixStream::pair().expect("socket pair");
    let client_write = client_read.try_clone().expect("clone");
    drop(engine_side);
    let client = EngineClient::over_streams(Box::new(client_read), Box::new(client_write), 8);
    let pipeline = pipeline_of(&tokenizer, &directory, &format, client);
    let mut body = json!({
        "messages": [{"role": "user", "content": "What is the answer?"}],
        "max_tokens": 64,
    });
    assert!(pipeline.prepare(&body, &mut || Ok(())).expect("prepares").thinking,
        "the template opens the think block by default");
    body["reasoning_effort"] = json!("none");
    assert!(!pipeline.prepare(&body, &mut || Ok(())).expect("prepares").thinking,
        "reasoning off closes it");
    body["reasoning_effort"] = json!("medium");
    assert!(pipeline.prepare(&body, &mut || Ok(())).expect("prepares").thinking,
        "an effort opens it");
    body["chat_template_kwargs"] = json!({"enable_thinking": false});
    assert!(!pipeline.prepare(&body, &mut || Ok(())).expect("prepares").thinking,
        "the request's own switch outranks the effort");
    body["chat_template_kwargs"] = json!({"messages": []});
    let refused = pipeline.prepare(&body, &mut || Ok(())).err().expect("refused");
    assert_eq!(refused.status, 400);
}

/// The template messages step, as the pipeline's prepare runs it.
fn crate_chat_messages_template(normalized: &[Value]) -> Value {
    Value::Array(splash_server::chat_messages::template_messages(normalized))
}

/// The streaming path: the engine's events written as OpenAI's SSE chunks
/// as they arrive, read back off a socket pair.
#[test]
fn a_chat_request_streams_as_openai_sse() {
    use splash_server::serve::stream::{stream_callbacks, StreamSink};
    
    let Some(directory) = qwen_directory() else {
        eprintln!("no local Qwen tokenizer; skipped");
        return;
    };
    let (engine_side, client_read) = UnixStream::pair().expect("socket pair");
    let client_write = client_read.try_clone().expect("clone");
    std::thread::spawn(move || fake_engine(engine_side, "The answer is 42."));
    let client = EngineClient::over_streams(Box::new(client_read), Box::new(client_write), 8);

    // The SSE sink writes onto its own socket pair; the test reads it.
    let (sink_write_side, mut sink_read_side) = UnixStream::pair().expect("socket pair");
    let writer_stream = Mutex::new(sink_write_side.try_clone().expect("clone"));
    let sink = Arc::new(StreamSink {
        write: Arc::new(Mutex::new(Box::new(move |payload: &str| {
            writer_stream
                .lock()
                .unwrap()
                .write_all(format!("data: {payload}\n\n").as_bytes())
                .map_err(|error| error.to_string())
        }))),
        model: "Qwen".into(),
        request_id: "req1".into(),
        created_at: 1_700_000_000,
        include_usage: true,
        compaction: None,
    });

    let format = qwen();
    let tokenizer = Arc::new(FastTokenizer::from_file(&directory).expect("loads"));
    let body = json!({
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is the answer?"},
        ],
        "max_tokens": 64,
    });
    let normalized = normalize_messages(&body["messages"], false).expect("normalizes");
    let templated = crate_chat_messages_template(&normalized);
    let rendered = {
        let source = pipeline_source(&pipeline_of(&tokenizer, &directory, &format, client.clone()));
        let renderer =
            splash_server::templates::ChatTemplateRenderer::new(&source).expect("compiles");
        renderer
            .render(&templated, None, None, true, &Default::default())
            .expect("renders")
    };
    let prompt_tokens = {
        let pipeline = pipeline_of(&tokenizer, &directory, &format, client.clone());
        pipeline.prompt_tokenizer.encode(&rendered)
    };

    let tokens: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));
    let (on_event, on_complete) = stream_callbacks(
        sink.clone(),
        tokens.clone(),
        tokenizer.clone(),
        Arc::new(Mutex::new(
            StreamingToolCallProjector::without_policy(1, &format),
        )),
        false,
        format.clone(),
        false,
        None,
        None,
    );
    let request = splash_server::engine::GenerationRequest {
        frame: splash_server::protocol::RequestFrame {
            logical_max_output_tokens: 64,
            prompt_tokens: prompt_tokens.clone(),
            ..splash_server::protocol::RequestFrame::default()
        },
        deadline: None,
        mask_provider: None,
    };
    let call = client
        .submit(&request, Some(on_event), Some(on_complete))
        .expect("submits");
    let _done = call.result(None).expect("done");

    // The SSE stream: the role chunk, the content deltas, the finish
    // chunk, the usage chunk, and [DONE].
    let mut lines = String::new();
    sink_read_side
        .set_read_timeout(Some(std::time::Duration::from_millis(200)))
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let mut buffer = [0u8; 65536];
    while std::time::Instant::now() < deadline {
        match sink_read_side.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => lines.push_str(&String::from_utf8_lossy(&buffer[..count])),
            Err(_) => continue, // a read timeout: more may follow
        }
        if lines.contains("[DONE]") {
            break;
        }
    }
    assert!(lines.contains("\"chat.completion.chunk\""), "{lines}");
    assert!(lines.contains("\"role\":\"assistant\""), "{lines}");
    assert_eq!(streamed_text(&lines, "content"), "The answer is 42.", "{lines}");
    assert!(lines.contains("\"finish_reason\":\"stop\""), "{lines}");
    assert!(lines.contains("\"usage\""), "{lines}");
    // The usage chunk carries the metrics whose rate the chat page shows.
    assert!(lines.contains("\"stream_tokens_per_second\""), "{lines}");
    assert!(lines.contains("[DONE]"), "{lines}");
}

/// The streaming path splits the reasoning from the answer: the reasoning
/// goes out as `reasoning_content` deltas, the answer as `content` deltas,
/// and the marker that closes the reasoning never reaches the client.
#[test]
fn a_chat_stream_splits_reasoning_from_the_answer() {
    use splash_server::serve::stream::{stream_callbacks, StreamSink};

    let Some(directory) = qwen_directory() else {
        eprintln!("no local Qwen tokenizer; skipped");
        return;
    };
    let (engine_side, client_read) = UnixStream::pair().expect("socket pair");
    let client_write = client_read.try_clone().expect("clone");
    std::thread::spawn(move || {
        fake_engine(
            engine_side,
            "Let me think.\u{003c}\u{002f}\u{0074}\u{0068}\u{0069}\u{006e}\u{006b}\u{003e}The answer is 42.",
        )
    });
    let client = EngineClient::over_streams(Box::new(client_read), Box::new(client_write), 8);

    let (sink_write_side, mut sink_read_side) = UnixStream::pair().expect("socket pair");
    let writer_stream = Mutex::new(sink_write_side.try_clone().expect("clone"));
    let sink = Arc::new(StreamSink {
        write: Arc::new(Mutex::new(Box::new(move |payload: &str| {
            writer_stream
                .lock()
                .unwrap()
                .write_all(format!("data: {payload}\n\n").as_bytes())
                .map_err(|error| error.to_string())
        }))),
        model: "Qwen".into(),
        request_id: "req2".into(),
        created_at: 1_700_000_000,
        include_usage: false,
        compaction: None,
    });

    let format = qwen();
    let tokenizer = Arc::new(FastTokenizer::from_file(&directory).expect("loads"));
    let prompt_tokens = {
        let pipeline = pipeline_of(&tokenizer, &directory, &format, client.clone());
        pipeline.prompt_tokenizer.encode("hello")
    };
    let tokens: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));
    let (on_event, on_complete) = stream_callbacks(
        sink.clone(),
        tokens.clone(),
        tokenizer.clone(),
        Arc::new(Mutex::new(
            StreamingToolCallProjector::without_policy(1, &format),
        )),
        true,
        format.clone(),
        false,
        None,
        None,
    );
    let request = splash_server::engine::GenerationRequest {
        frame: splash_server::protocol::RequestFrame {
            logical_max_output_tokens: 64,
            prompt_tokens,
            ..splash_server::protocol::RequestFrame::default()
        },
        deadline: None,
        mask_provider: None,
    };
    let call = client
        .submit(&request, Some(on_event), Some(on_complete))
        .expect("submits");
    let _done = call.result(None).expect("done");

    let mut lines = String::new();
    sink_read_side
        .set_read_timeout(Some(std::time::Duration::from_millis(200)))
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let mut buffer = [0u8; 65536];
    while std::time::Instant::now() < deadline {
        match sink_read_side.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => lines.push_str(&String::from_utf8_lossy(&buffer[..count])),
            Err(_) => continue,
        }
        if lines.contains("[DONE]") {
            break;
        }
    }
    assert_eq!(streamed_text(&lines, "reasoning_content"), "Let me think.", "{lines}");
    assert_eq!(streamed_text(&lines, "content"), "The answer is 42.", "{lines}");
    assert!(!lines.contains("</think>"), "the close never reaches the client: {lines}");
}

fn pipeline_of(
    tokenizer: &Arc<FastTokenizer>,
    directory: &PathBuf,
    format: &splash_server::chat_format::ChatFormat,
    client: EngineClient,
) -> Pipeline {
    let prompt_tokenizer =
        PromptTokenizer::from_directory(tokenizer.clone(), directory, format.message_boundary);
    let templates = load_chat_templates(directory).expect("loads");
    Pipeline {
        tokenizer: tokenizer.clone(),
        prompt_tokenizer,
        templates,
        chat_format: format.clone(),
        client,
        model: "Qwen".into(),
        images: splash_server::images::ImageCache::new(),
        max_image_pixels: splash_server::images::MAX_PIXELS,
        default_reasoning_effort: None,
        thinking_end_tokens: None,
        documents: None,
        constraints: None,
    }
}

fn pipeline_source(pipeline: &Pipeline) -> String {
    pipeline
        .templates
        .select(None, None)
        .expect("the model has a chat template")
}

fn fake_score_engine(mut stream: UnixStream) {
    let ready = wire_event(&json!({"frame_type":0x0100,"max_concurrent_requests":4,"max_context_tokens":524288,"vision":false}));
    let _ = stream.write_all(&ready);
    loop {
        let mut header = [0u8;16];
        if stream.read_exact(&mut header).is_err() { return; }
        let kind = u16::from_le_bytes([header[6],header[7]]);
        let count = u64::from_le_bytes(header[8..16].try_into().unwrap()) as usize;
        let mut payload = vec![0u8;count];
        if stream.read_exact(&mut payload).is_err() { return; }
        if kind != 0x0001 { continue; }
        let request_id = u64::from_le_bytes(payload[..8].try_into().unwrap());
        let prompt_count = u32::from_le_bytes(payload[30..34].try_into().unwrap());
        let score_count = u32::from_le_bytes(payload[74..78].try_into().unwrap()) as usize;
        let start = wire_event(&json!({"frame_type":0x0101,"request_id":request_id,"lane":0,"matched_prompt_tokens":0}));
        let done = wire_event(&json!({
            "frame_type":0x0104,"request_id":request_id,"reason":0,"bounded":false,
            "prompt_tokens":prompt_count,"completion_tokens":0,"prefill_micros":1000,
            "decode_micros":0,"wall_micros":1300,"deduplicated_tokens":0,
            "drafted_tokens":0,"accepted_draft_tokens":0,"draft_cycles":0,
            "option_logits":(0..score_count).map(|index| index as f64).collect::<Vec<_>>()
        }));
        let _ = stream.write_all(&start);
        let _ = stream.write_all(&done);
    }
}

#[test]
fn judgments_and_systemone_score_through_the_native_engine_route() {
    use splash_server::serve::Server;
    let Some(directory) = qwen_directory() else { eprintln!("no local Qwen tokenizer; skipped"); return; };
    let (engine_side, client_read) = UnixStream::pair().expect("engine socket");
    let client_write = client_read.try_clone().expect("clone engine socket");
    std::thread::spawn(move || fake_score_engine(engine_side));
    let client = EngineClient::over_streams(Box::new(client_read), Box::new(client_write), 8);
    let format = qwen();
    let tokenizer = Arc::new(FastTokenizer::from_file(&directory).expect("loads tokenizer"));
    let pipeline = pipeline_of(&tokenizer, &directory, &format, client);
    let server = Server::new(pipeline);
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("binds");
    let address = listener.local_addr().unwrap().to_string();
    std::thread::spawn(move || server.serve(listener));

    let call = |path: &str, body: Value| -> String {
        let bytes = serde_json::to_vec(&body).unwrap();
        let mut stream = std::net::TcpStream::connect(&address).expect("connects");
        write!(stream, "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", bytes.len()).unwrap();
        stream.write_all(&bytes).unwrap();
        stream.set_read_timeout(Some(std::time::Duration::from_secs(10))).unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        response
    };
    let judgment = call("/v1/judgments", json!({"model":"Qwen","id":"r1","state":"evidence","question":"pick","options":[{"id":"a","description":"A"},{"id":"b","description":"B"}]}));
    assert!(judgment.starts_with("HTTP/1.1 200"), "{judgment}");
    assert!(judgment.contains("\"answer_token_ids\":["), "{judgment}");
    let judgment_json: Value = serde_json::from_str(judgment.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(judgment_json["id"], "r1");
    assert!(judgment_json["probabilities"][1].as_f64().unwrap() > judgment_json["probabilities"][0].as_f64().unwrap());

    let systemone = call("/v1/systemone", json!({"model":"Qwen","state":"evidence","questions":{
        "n":{"type":"noul"},
        "c":{"type":"choice","criteria":{"left":"L","right":"R"}},
        "s":{"type":"score","criteria":["low","mid","high"]},
        "d":{"type":"choice","criteria":{"only":"one"}}
    }}));
    assert!(systemone.starts_with("HTTP/1.1 200"), "{systemone}");
    let result: Value = serde_json::from_str(systemone.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(result["answers"]["c"]["choice"], "right");
    assert!(result["answers"]["n"]["noul"].as_f64().unwrap() < 0.5);
    assert!(result["answers"]["s"]["score"].as_f64().unwrap() > 1.0);
    assert_eq!(result["answers"]["d"]["choice"], "only");
    assert_eq!(result["usage"]["output_tokens"], 0);
}

/// The server over real TCP: a Chat completion answered complete, and one
/// answered streamed, both read back off the socket.
#[test]
fn the_server_answers_over_real_tcp() {
    use splash_server::serve::Server;

    let captured_logs = Arc::new(Mutex::new(Vec::new()));
    splash_server::diagnostics::start_console_with_writer(CapturedConsole(captured_logs.clone()))
        .expect("install captured console");

    let Some(directory) = qwen_directory() else {
        eprintln!("no local Qwen tokenizer; skipped");
        return;
    };
    let (engine_side, client_read) = UnixStream::pair().expect("socket pair");
    let client_write = client_read.try_clone().expect("clone");
    std::thread::spawn(move || fake_engine(engine_side, "The answer is 42."));
    let client = EngineClient::over_streams(Box::new(client_read), Box::new(client_write), 8);

    let format = qwen();
    let tokenizer = Arc::new(FastTokenizer::from_file(&directory).expect("loads"));
    let prompt_tokenizer =
        PromptTokenizer::from_directory(tokenizer.clone(), &directory, format.message_boundary);
    let templates = load_chat_templates(&directory).expect("loads");
    let pipeline = Pipeline {
        tokenizer,
        prompt_tokenizer,
        templates,
        chat_format: format,
        client,
        model: "Qwen".into(),
        images: splash_server::images::ImageCache::new(),
        max_image_pixels: splash_server::images::MAX_PIXELS,
        default_reasoning_effort: None,
        thinking_end_tokens: None,
        documents: None,
        constraints: None,
    };
    let thinking_codec = Arc::new(
        splash_server::thinking::ThinkingCodec::new(
            b"MDEyMzQ1Njc4OWFiY2RlZmZlZGNiYTk4NzY1NDMyMTA=",
        )
        .expect("Fernet key"),
    );
    let server = Server::new(pipeline).with_thinking_codec(thinking_codec.clone());

    // The listener on an ephemeral port.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("binds");
    let address = listener.local_addr().expect("address").to_string();
    std::thread::spawn(move || server.serve(listener));

    for (path, expected_status, expected_body) in [
        ("/v1/models", "200", "\"models\""),
        ("/v1/models/Qwen", "200", "\"id\":\"Qwen\""),
        ("/v1/models/missing", "404", "model_not_found"),
    ] {
        let mut stream = std::net::TcpStream::connect(&address).expect("connects");
        write!(stream, "GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
        stream.set_read_timeout(Some(std::time::Duration::from_secs(10))).unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).expect("reads");
        assert!(response.starts_with(&format!("HTTP/1.1 {expected_status}")), "{response}");
        assert!(response.contains(expected_body), "{response}");
    }

    let body = json!({
        "model": "Qwen",
        "reasoning_effort": "high",
        "messages": [
            {"role": "user", "content": "What is the answer?"},
        ],
        "max_tokens": 64,
    });

    // The complete answer.
    let mut stream = std::net::TcpStream::connect(&address).expect("connects");
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        serde_json::to_string(&body).unwrap().len(),
        serde_json::to_string(&body).unwrap()
    );
    stream.write_all(request.as_bytes()).expect("writes");
    let mut response = String::new();
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(10)))
        .unwrap();
    let _ = stream.read_to_string(&mut response);
    assert!(response.contains("\"chat.completion\""), "{response}");
    assert!(response.contains("The answer is 42."), "{response}");
    assert!(response.contains("\"usage\""), "{response}");

    // The streamed answer.
    let mut stream = std::net::TcpStream::connect(&address).expect("connects");
    let streamed_body = json!({
        "model": "Qwen",
        "messages": [
            {"role": "user", "content": "What is the answer?"},
        ],
        "max_tokens": 64,
        "stream": true,
    });
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        serde_json::to_string(&streamed_body).unwrap().len(),
        serde_json::to_string(&streamed_body).unwrap()
    );
    stream.write_all(request.as_bytes()).expect("writes");
    let mut response = String::new();
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(10)))
        .unwrap();
    let _ = stream.read_to_string(&mut response);
    assert!(response.contains("text/event-stream"), "{response}");
    assert!(response.contains("\"chat.completion.chunk\""), "{response}");
    let (_, streamed) = response.split_once("\r\n\r\n").expect("HTTP response");
    assert_eq!(streamed_text(streamed, "reasoning_content"), "The answer is 42.", "{response}");
    assert!(response.contains("[DONE]"), "{response}");

    splash_server::diagnostics::flush_console(std::time::Duration::from_secs(1));
    let log = String::from_utf8(captured_logs.lock().unwrap().clone()).expect("UTF-8 diagnostics");
    assert!(log.contains("Started · input "), "{log}");
    assert!(log.contains("cached 0 · effort high · thinking "), "{log}");
    assert!(log.contains("Done · input "), "{log}");
    assert!(log.contains("TTFT 0.0s"), "{log}");
    assert!(log.contains("tok/s"), "{log}");
    assert!(!log.contains("What is the answer?"), "request body leaked to diagnostics: {log}");

    // Responses returns a distinct complete-response envelope and keeps
    // the input history available for retrieval and continuation.
    let responses_body = json!({"input": "What is the answer?"});
    let payload = responses_body.to_string();
    let mut stream = std::net::TcpStream::connect(&address).expect("connects");
    stream.set_read_timeout(Some(std::time::Duration::from_secs(10))).unwrap();
    write!(stream,
        "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        payload.len(), payload,
    ).unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).expect("reads Responses response");
    assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
    let first: Value = response.split_once("\r\n\r\n")
        .and_then(|(_, body)| serde_json::from_str(body).ok()).expect("Responses JSON");
    assert_eq!(first["object"], "response");
    assert_eq!(first["status"], "completed");
    assert!(first.to_string().contains("The answer is 42."));
    let id = first["id"].as_str().expect("stored response id");

    // Custom and namespace tool declarations are normalized into the Chat
    // tool policy and make it through the actual Responses route.
    let custom_tool_body = json!({
        "input":"What is the answer?",
        "reasoning":{"effort":"none"},
        "tools":[
            {"type":"custom","name":"shell","description":"Run a command","format":{"type":"text"}},
            {"type":"namespace","name":"files","tools":[{"type":"function","name":"search","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}]},
        ],
        "tool_choice":"none",
    }).to_string();
    let mut stream = std::net::TcpStream::connect(&address).expect("connects");
    stream.set_read_timeout(Some(std::time::Duration::from_secs(10))).unwrap();
    write!(stream,
        "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        custom_tool_body.len(), custom_tool_body,
    ).unwrap();
    let mut custom_response = String::new();
    stream.read_to_string(&mut custom_response).expect("reads custom-tool response");
    assert!(custom_response.starts_with("HTTP/1.1 200 OK"), "{custom_response}");
    let custom_response: Value = custom_response.split_once("\r\n\r\n")
        .and_then(|(_, body)| serde_json::from_str(body).ok()).expect("custom-tool Responses JSON");
    assert_eq!(custom_response["status"], "completed");
    assert!(custom_response.to_string().contains("The answer is 42."));

    let mut stream = std::net::TcpStream::connect(&address).expect("connects");
    stream.set_read_timeout(Some(std::time::Duration::from_secs(10))).unwrap();
    write!(stream, "GET /v1/responses/{id} HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
    let mut fetched = String::new();
    stream.read_to_string(&mut fetched).expect("reads stored response");
    assert!(fetched.starts_with("HTTP/1.1 200 OK"), "{fetched}");
    let fetched: Value = fetched.split_once("\r\n\r\n")
        .and_then(|(_, body)| serde_json::from_str(body).ok()).expect("stored JSON");
    assert_eq!(fetched, first);

    let next_body = json!({"previous_response_id": id, "input": "Continue"}).to_string();
    let mut stream = std::net::TcpStream::connect(&address).expect("connects");
    stream.set_read_timeout(Some(std::time::Duration::from_secs(10))).unwrap();
    write!(stream,
        "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        next_body.len(), next_body,
    ).unwrap();
    let mut continued = String::new();
    stream.read_to_string(&mut continued).expect("reads continued response");
    assert!(continued.starts_with("HTTP/1.1 200 OK"), "{continued}");
    let continued: Value = continued.split_once("\r\n\r\n")
        .and_then(|(_, body)| serde_json::from_str(body).ok()).expect("continued JSON");
    assert_eq!(continued["previous_response_id"], id);

    let responses_streamed_body = json!({
        "input":"What is the answer?",
        "reasoning":{"effort":"none"},
        "stream":true,
    }).to_string();
    let mut stream = std::net::TcpStream::connect(&address).expect("connects");
    stream.set_read_timeout(Some(std::time::Duration::from_secs(10))).unwrap();
    write!(stream,
        "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        responses_streamed_body.len(), responses_streamed_body,
    ).unwrap();
    let mut streamed = String::new();
    stream.read_to_string(&mut streamed).expect("reads Responses SSE");
    assert!(streamed.starts_with("HTTP/1.1 200 OK"), "{streamed}");
    assert!(streamed.contains("Content-Type: text/event-stream"), "{streamed}");
    let (_, frames) = streamed.split_once("\r\n\r\n").expect("HTTP response");
    let frames: Vec<_> = frames.split("\n\n").filter(|frame| !frame.trim().is_empty()).collect();
    let names: Vec<_> = frames.iter().filter_map(|frame| frame.lines()
        .find_map(|line| line.strip_prefix("event: "))).collect();
    assert_eq!(&names[..2], &["response.created", "response.in_progress"]);
    assert!(names.contains(&"response.output_text.delta"), "{names:?}");
    assert_eq!(names.last(), Some(&"response.completed"));
    let sequence: Vec<u64> = frames.iter().filter_map(|frame| frame.lines()
        .find_map(|line| line.strip_prefix("data: "))
        .and_then(|data| serde_json::from_str::<Value>(data).ok())
        .and_then(|payload| payload.get("sequence_number").and_then(Value::as_u64))).collect();
    assert_eq!(sequence, (0..sequence.len() as u64).collect::<Vec<_>>());

    // Anthropic Messages uses its own incremental event sequence. A stop
    // emitted text ends before the stop string and the final delta reports
    // that exact match.
    let anthropic_body = json!({
        "model": "Qwen",
        "messages": [{"role": "user", "content": "What is the answer?"}],
        "max_tokens": 64,
        "stream": true,
        "stop_sequences": ["42."],
    });
    let payload = anthropic_body.to_string();
    let mut stream = std::net::TcpStream::connect(&address).expect("connects");
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(10)))
        .unwrap();
    write!(
        stream,
        "POST /v1/messages HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        payload.len(),
        payload
    )
    .unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).expect("reads stream");
    assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
    assert!(
        response.contains("Content-Type: text/event-stream"),
        "{response}"
    );
    let (_, event_stream) = response.split_once("\r\n\r\n").expect("HTTP response");
    let mut event_names = Vec::new();
    let mut text_deltas = String::new();
    let mut text_delta_count = 0;
    let mut stop_reason = None;
    let mut stop_sequence = None;
    for frame in event_stream
        .split("\n\n")
        .filter(|frame| !frame.trim().is_empty())
    {
        let event = frame
            .lines()
            .find_map(|line| line.strip_prefix("event: "))
            .expect("named Anthropic event")
            .to_owned();
        let data = frame
            .lines()
            .find_map(|line| line.strip_prefix("data: "))
            .expect("JSON event data");
        let payload: Value = serde_json::from_str(data).expect("valid SSE JSON");
        assert_eq!(payload["type"], json!(event));
        event_names.push(event.clone());
        if event == "content_block_delta"
            && payload.pointer("/delta/type").and_then(Value::as_str) == Some("text_delta")
        {
            text_deltas.push_str(
                payload
                    .pointer("/delta/text")
                    .and_then(Value::as_str)
                    .unwrap_or(""),
            );
            text_delta_count += 1;
        }
        if event == "message_delta" {
            stop_reason = payload
                .pointer("/delta/stop_reason")
                .and_then(Value::as_str)
                .map(str::to_owned);
            stop_sequence = payload
                .pointer("/delta/stop_sequence")
                .and_then(Value::as_str)
                .map(str::to_owned);
        }
    }
    assert_eq!(event_names.first().map(String::as_str), Some("message_start"));
    assert_eq!(event_names.last().map(String::as_str), Some("message_stop"));
    let block_start = event_names
        .iter()
        .position(|event| event == "content_block_start")
        .unwrap();
    let first_delta = event_names
        .iter()
        .position(|event| event == "content_block_delta")
        .unwrap();
    let block_stop = event_names
        .iter()
        .position(|event| event == "content_block_stop")
        .unwrap();
    let message_delta = event_names
        .iter()
        .position(|event| event == "message_delta")
        .unwrap();
    assert!(
        block_start < first_delta
            && first_delta < block_stop
            && block_stop < message_delta,
        "{event_names:?}"
    );
    assert!(
        text_delta_count >= 2,
        "expected multiple incremental text deltas: {event_stream}"
    );
    assert_eq!(text_deltas, "The answer is ");
    assert_eq!(stop_reason.as_deref(), Some("stop_sequence"));
    assert_eq!(stop_sequence.as_deref(), Some("42."));

    // Omitted thinking stays hidden in the live SSE body, while its trailing
    // signature roundtrips through a later Messages request.
    let omitted_body = json!({
        "model": "Qwen",
        "messages": [{"role": "user", "content": "Think privately."}],
        "max_tokens": 64,
        "stream": true,
        "thinking": {"type": "enabled", "display": "omitted"},
    });
    let payload = omitted_body.to_string();
    let mut stream = std::net::TcpStream::connect(&address).expect("connects");
    stream.set_read_timeout(Some(std::time::Duration::from_secs(10))).unwrap();
    write!(
        stream,
        "POST /v1/messages HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        payload.len(),
        payload
    )
    .unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).expect("reads SSE");
    assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
    let (_, event_stream) = response.split_once("\r\n\r\n").expect("HTTP response");
    let mut signature = None;
    let mut hidden_text_was_sent = false;
    for frame in event_stream.split("\n\n").filter(|frame| !frame.trim().is_empty()) {
        let event_name = frame.lines().find_map(|line| line.strip_prefix("event: ")).unwrap_or("");
        let data = frame.lines().find_map(|line| line.strip_prefix("data: ")).unwrap_or("");
        let event: Value = serde_json::from_str(data).expect("SSE JSON");
        if event["type"] == "content_block_delta"
            && event.pointer("/delta/type").and_then(Value::as_str) == Some("thinking_delta")
        {
            hidden_text_was_sent = true;
        }
        if event["type"] == "content_block_delta"
            && event.pointer("/delta/type").and_then(Value::as_str) == Some("signature_delta")
        {
            signature = event.pointer("/delta/signature").and_then(Value::as_str).map(str::to_owned);
        }
        assert_eq!(event["type"], json!(event_name));
    }
    assert!(!hidden_text_was_sent, "omitted thinking leaked: {event_stream}");
    let signature = signature.expect("thinking signature delta");
    assert_eq!(thinking_codec.decode(&signature).expect("valid signature"), "The answer is 42.");

    let continuation = json!({
        "model": "Qwen",
        "messages": [
            {"role": "user", "content": "Think privately."},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "", "signature": signature},
                {"type": "text", "text": "visible answer"}
            ]},
            {"role": "user", "content": "Continue."}
        ],
        "max_tokens": 64,
        "thinking": {"type": "enabled", "display": "omitted"},
    });
    let payload = continuation.to_string();
    let mut stream = std::net::TcpStream::connect(&address).expect("connects");
    stream.set_read_timeout(Some(std::time::Duration::from_secs(10))).unwrap();
    write!(
        stream,
        "POST /v1/messages HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        payload.len(),
        payload
    )
    .unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).expect("reads continuation");
    assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
    assert!(response.contains("signature"), "{response}");

    // The legacy text-completion route tokenizes its raw prompt, returns the
    // `text_completion` envelope, and streams text chunks with stop handling.
    let completion_body = json!({
        "model": "Qwen",
        "prompt": "Complete this sentence:",
        "max_tokens": 64,
    });
    let payload = completion_body.to_string();
    let mut stream = std::net::TcpStream::connect(&address).expect("connects");
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(10)))
        .unwrap();
    write!(
        stream,
        "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        payload.len(),
        payload
    )
    .unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).expect("reads completion");
    assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
    let (_, payload) = response.split_once("\r\n\r\n").expect("HTTP response");
    let completion: Value = serde_json::from_str(payload).expect("JSON completion");
    assert_eq!(completion["object"], json!("text_completion"));
    assert_eq!(completion["choices"][0]["text"], json!("The answer is 42."));
    assert_eq!(completion["choices"][0]["finish_reason"], json!("stop"));
    assert!(completion["usage"]["prompt_tokens"].as_u64().unwrap() > 0);

    let streamed_completion_body = json!({
        "model": "Qwen",
        "prompt": "Complete this sentence:",
        "max_tokens": 64,
        "stream": true,
        "stop": "42.",
    });
    let payload = streamed_completion_body.to_string();
    let mut stream = std::net::TcpStream::connect(&address).expect("connects");
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(10)))
        .unwrap();
    write!(
        stream,
        "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        payload.len(),
        payload
    )
    .unwrap();
    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .expect("reads completion stream");
    assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
    assert!(response.contains("Content-Type: text/event-stream"), "{response}");
    let (_, event_stream) = response.split_once("\r\n\r\n").expect("HTTP response");
    let mut streamed_text = String::new();
    let mut chunks = 0;
    let mut finish = None;
    for line in event_stream.lines() {
        let Some(data) = line.strip_prefix("data: ") else { continue };
        if data == "[DONE]" {
            continue;
        }
        let chunk: Value = serde_json::from_str(data).expect("valid completion chunk");
        assert_eq!(chunk["object"], json!("text_completion"));
        if let Some(choice) = chunk["choices"].as_array().and_then(|choices| choices.first()) {
            streamed_text.push_str(choice["text"].as_str().unwrap_or(""));
            if let Some(reason) = choice["finish_reason"].as_str() {
                finish = Some(reason.to_owned());
            }
            if choice["text"].as_str().is_some_and(|text| !text.is_empty()) {
                chunks += 1;
            }
        }
    }
    assert!(chunks >= 2, "expected incremental text chunks: {event_stream}");
    assert_eq!(streamed_text, "The answer is ");
    assert_eq!(finish.as_deref(), Some("stop"));

    // A constraint that cannot be compiled must fail before SSE headers,
    // so the client receives a single usable HTTP error response.
    let mut constrained_body = streamed_body;
    constrained_body["response_format"] = json!({"type": "json_object"});
    let payload = constrained_body.to_string();
    let mut stream = std::net::TcpStream::connect(&address).expect("connects");
    stream.set_read_timeout(Some(std::time::Duration::from_secs(10))).unwrap();
    write!(stream, "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}", payload.len(), payload).unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).expect("reads");
    assert!(response.starts_with("HTTP/1.1 500"), "{response}");
    assert!(response.contains("grammar tokenizer is unavailable"), "{response}");
    assert!(!response.contains("text/event-stream"), "{response}");

    // An elapsed request budget returns the OpenAI timeout before generation.
    let mut expired_body = body;
    expired_body["timeout"] = json!(1e-12);
    let mut stream = std::net::TcpStream::connect(&address).expect("connects");
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        expired_body.to_string().len(),
        expired_body
    );
    stream.write_all(request.as_bytes()).expect("writes");
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(10)))
        .unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).expect("reads");
    assert!(response.starts_with("HTTP/1.1 504 Gateway Timeout\r\n"), "{response}");
    assert!(response.contains("\"code\":\"request_timeout\""), "{response}");

    let mut stream = std::net::TcpStream::connect(&address).expect("connects");
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(10)))
        .unwrap();
    stream.write_all(b"GET /status HTTP/1.1\r\n\r\n").expect("writes");
    let mut response = String::new();
    stream.read_to_string(&mut response).expect("reads");
    let (_, payload) = response.split_once("\r\n\r\n").expect("HTTP response");
    let status: Value = serde_json::from_str(payload).expect("JSON status");
    assert_eq!(status["frontend"], json!({"preparation_capacity": 4, "active": 0, "waiting": 0}));
    // The engine's own document is what the live board reads: its
    // readiness, the context window, the rates, the counters and the
    // speculation plan, with the server's transport accounting beside it.
    assert_eq!(status["ready"], json!(true));
    assert_eq!(status["model"], json!("Qwen"));
    assert_eq!(status["maximum_context_tokens"], json!(524288));
    assert_eq!(status["metrics"]["decode_tokens_per_second"], json!(12.5));
    assert_eq!(status["requests"]["completed"], json!(2));
    assert_eq!(status["speculation"]["provider"], json!("dflash2"));
    assert_eq!(status["transport"]["pending_limit"], json!(4));
    assert_eq!(status["transport"]["ready"], json!(true));

    // The heat-and-noise rung: the duty the engine took comes back.
    let mut stream = std::net::TcpStream::connect(&address).expect("connects");
    let duty = json!({"prefill_duty_percent": 55});
    let request = format!(
        "POST /v1/thermal HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        serde_json::to_string(&duty).unwrap().len(),
        serde_json::to_string(&duty).unwrap()
    );
    stream.write_all(request.as_bytes()).expect("writes");
    let mut response = String::new();
    stream.read_to_string(&mut response).expect("reads");
    let (_, payload) = response.split_once("\r\n\r\n").expect("HTTP response");
    let answer: Value = serde_json::from_str(payload).expect("JSON answer");
    assert_eq!(answer["prefill_duty_percent"], json!(55.0));

    // A duty outside the range is refused before the engine sees it.
    let mut stream = std::net::TcpStream::connect(&address).expect("connects");
    let duty = json!({"prefill_duty_percent": 2});
    let request = format!(
        "POST /v1/thermal HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        serde_json::to_string(&duty).unwrap().len(),
        serde_json::to_string(&duty).unwrap()
    );
    stream.write_all(request.as_bytes()).expect("writes");
    let mut response = String::new();
    stream.read_to_string(&mut response).expect("reads");
    assert!(response.starts_with("HTTP/1.1 400 Bad Request"), "{response}");

    splash_server::diagnostics::flush_console(std::time::Duration::from_secs(1));
    let route_logs = String::from_utf8(captured_logs.lock().unwrap().clone()).expect("UTF-8 diagnostics");
    assert!(route_logs.matches("Started · input ").count() >= 9, "{route_logs}");
    assert!(route_logs.matches("Done · input ").count() >= 9, "{route_logs}");
    assert!(route_logs.contains("effort none"), "Responses/Anthropic converted effort missing: {route_logs}");
    assert!(!route_logs.contains("What is the answer?"), "request content leaked: {route_logs}");
}

/// The facts of one request frame the engine reads: the prompt's token
/// count, the image spans it carries and their pixel bytes.
struct RequestFacts {
    prompt_count: u32,
    span_count: u32,
    spans: Vec<(u32, u32, u32, u32, u64, u64)>,
    pixel_bytes: usize,
}

/// The fake engine of a vision model: Ready with vision on, then per
/// request a start, the tokens of a fixed answer, and a done — reporting
/// each request's image facts to the test.
fn fake_engine_vision(mut stream: UnixStream, answer: &'static str, report: std::sync::mpsc::Sender<RequestFacts>) {
    let ready = wire_event(&json!({
        "frame_type": 0x0100,
        "max_concurrent_requests": 4,
        "max_context_tokens": 524288,
        "vision": true,
    }));
    let _ = stream.write_all(&ready);
    loop {
        let mut header = [0u8; 16];
        if stream.read_exact(&mut header).is_err() {
            return;
        }
        let raw_type = u16::from_le_bytes([header[6], header[7]]);
        let payload_bytes = u64::from_le_bytes(header[8..16].try_into().unwrap()) as usize;
        let mut payload = vec![0u8; payload_bytes];
        if payload_bytes > 0 {
            stream.read_exact(&mut payload).expect("engine reads payload");
        }
        if raw_type != 0x0001 {
            continue;
        }
        let request_id = u64::from_le_bytes(payload[..8].try_into().unwrap());
        let prompt_count = u32::from_le_bytes(payload[30..34].try_into().unwrap());
        let span_count = u32::from_le_bytes(payload[34..38].try_into().unwrap());
        // The fixed part is 122 bytes; the prompt tokens, the spans and the
        // pixels follow in that order.
        let spans_offset = 122 + 4 * prompt_count as usize;
        let mut spans = Vec::new();
        for index in 0..span_count as usize {
            let base = spans_offset + index * 32;
            let read_u32 = |offset: usize| u32::from_le_bytes(payload[offset..offset + 4].try_into().unwrap());
            let read_u64 = |offset: usize| u64::from_le_bytes(payload[offset..offset + 8].try_into().unwrap());
            spans.push((
                read_u32(base),
                read_u32(base + 4),
                read_u32(base + 8),
                read_u32(base + 12),
                read_u64(base + 16),
                read_u64(base + 24),
            ));
        }
        let pixel_bytes = payload.len() - spans_offset - 32 * span_count as usize;
        let _ = report.send(RequestFacts { prompt_count, span_count, spans, pixel_bytes });
        for event in [
            json!({"frame_type": 0x0101, "request_id": request_id, "lane": 0,
                   "matched_prompt_tokens": 0}),
            json!({"frame_type": 0x0102, "request_id": request_id, "sequence_offset": 0,
                   "tokens": answer_tokens(answer)}),
            json!({"frame_type": 0x0104, "request_id": request_id, "reason": 0,
                   "bounded": false, "prompt_tokens": prompt_count,
                   "completion_tokens": answer_tokens(answer).len(),
                   "prefill_micros": 1000, "decode_micros": 2000,
                   "wall_micros": 3500, "deduplicated_tokens": 0,
                   "drafted_tokens": 0, "accepted_draft_tokens": 0,
                   "draft_cycles": 0, "logit_count": 0}),
        ] {
            let _ = stream.write_all(&wire_event(&event));
        }
    }
}

/// A solid-color PNG the image crate encodes, as the images tests' helper.
fn png_bytes(width: u32, height: u32, color: [u8; 3]) -> Vec<u8> {
    let image = image::RgbImage::from_pixel(width, height, image::Rgb(color));
    let mut buffer = std::io::Cursor::new(Vec::new());
    use image::ImageEncoder as _;
    image::codecs::png::PngEncoder::new(&mut buffer)
        .write_image(image.as_raw(), width, height, image::ExtendedColorType::Rgb8)
        .expect("png encodes");
    buffer.into_inner()
}

/// An image request prepares its placeholders wide and carries the spans
/// and the pixels to the engine, as the Python's exchange does.
#[test]
fn an_image_request_carries_its_spans_and_pixels_to_the_engine() {
    let Some(directory) = qwen_directory() else {
        eprintln!("no local Qwen tokenizer; skipped");
        return;
    };
    let (engine_side, client_read) = UnixStream::pair().expect("socket pair");
    let client_write = client_read.try_clone().expect("clone");
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || fake_engine_vision(engine_side, "The answer is 42.", tx));
    let client = EngineClient::over_streams(Box::new(client_read), Box::new(client_write), 8);

    let format = qwen();
    let tokenizer = Arc::new(FastTokenizer::from_file(&directory).expect("loads"));
    let prompt_tokenizer =
        PromptTokenizer::from_directory(tokenizer.clone(), &directory, format.message_boundary);
    let templates = load_chat_templates(&directory).expect("loads");
    let pipeline = Pipeline {
        tokenizer: tokenizer.clone(),
        prompt_tokenizer,
        templates,
        chat_format: format,
        client,
        model: "Qwen".into(),
        images: splash_server::images::ImageCache::new(),
        max_image_pixels: splash_server::images::MAX_PIXELS,
        default_reasoning_effort: None,
        thinking_end_tokens: None,
        documents: None,
        constraints: None,
    };

    // A 32x32 image: one patch grid of 2x2, one merged token.
    let payload = png_bytes(32, 32, [200, 30, 30]);
    use base64::Engine as _;
    let encoded = base64::engine::general_purpose::STANDARD.encode(&payload);
    let url = format!("data:image/png;base64,{encoded}");
    let body = json!({
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": "What is in this image?"},
        ]}],
    });
    let prepared = pipeline.prepare(&body, &mut || Ok(())).expect("prepares");
    assert_eq!(prepared.image_spans.len(), 1, "one image, one span");
    // The 32x32 source upscales to the 65536-pixel minimum: 256x256.
    assert_eq!(prepared.image_pixels.len(), 256 * 256 * 3, "resized RGB bytes");
    let span = &prepared.image_spans[0];
    assert_eq!((span.grid_height, span.grid_width), (16, 16));
    assert_eq!(span.tokens, (16 / 2) * (16 / 2));

    // The prompt the engine reads is the expanded one, and the frame
    // carries the spans and the pixels the preparation built.
    let (call, _tokens, _first_batch) = pipeline.submit(&prepared, 64).expect("submits");
    call.result(None).expect("done");
    let facts = rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("the engine read the request");
    assert_eq!(facts.prompt_count as usize, prepared.prompt_tokens.len());
    assert_eq!(facts.span_count, 1);
    assert_eq!(facts.pixel_bytes, prepared.image_pixels.len());
    assert_eq!(facts.spans.len(), 1);
    let (offset, tokens, grid_height, grid_width, digest_lo, digest_hi) = facts.spans[0];
    assert_eq!((tokens, grid_height, grid_width), (span.tokens, span.grid_height, span.grid_width));
    assert_eq!((digest_lo, digest_hi), (span.digest_lo, span.digest_hi));
    assert!(offset < prepared.prompt_tokens.len() as u32);
    // The span's run holds the pad token, widened from the placeholder.
    let pad_id = splash_server::images::prompts::image_pad_token_id(&tokenizer).expect("pad token");
    assert_eq!(prepared.prompt_tokens[offset as usize], pad_id);
}

/// A text-only request against a vision engine carries no image facts.
#[test]
fn a_text_request_on_a_vision_engine_carries_no_images() {
    let Some(directory) = qwen_directory() else {
        eprintln!("no local Qwen tokenizer; skipped");
        return;
    };
    let (engine_side, client_read) = UnixStream::pair().expect("socket pair");
    let client_write = client_read.try_clone().expect("clone");
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || fake_engine_vision(engine_side, "The answer is 42.", tx));
    let client = EngineClient::over_streams(Box::new(client_read), Box::new(client_write), 8);

    let format = qwen();
    let tokenizer = Arc::new(FastTokenizer::from_file(&directory).expect("loads"));
    let prompt_tokenizer =
        PromptTokenizer::from_directory(tokenizer.clone(), &directory, format.message_boundary);
    let templates = load_chat_templates(&directory).expect("loads");
    let pipeline = Pipeline {
        tokenizer,
        prompt_tokenizer,
        templates,
        chat_format: format,
        client,
        model: "Qwen".into(),
        images: splash_server::images::ImageCache::new(),
        max_image_pixels: splash_server::images::MAX_PIXELS,
        default_reasoning_effort: None,
        thinking_end_tokens: None,
        documents: None,
        constraints: None,
    };
    let body = json!({"messages": [{"role": "user", "content": "Hello"}]});
    let prepared = pipeline.prepare(&body, &mut || Ok(())).expect("prepares");
    assert!(prepared.image_spans.is_empty() && prepared.image_pixels.is_empty());
    let (call, _tokens, _first_batch) = pipeline.submit(&prepared, 64).expect("submits");
    call.result(None).expect("done");
    let facts = rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("the engine read the request");
    assert_eq!(facts.span_count, 0);
    assert_eq!(facts.pixel_bytes, 0);
}

#[test]
fn compaction_keeps_the_final_image_batch_in_message_order() {
    let Some(directory) = qwen_directory() else { eprintln!("no local Qwen tokenizer; skipped"); return; };
    let (engine_side, client_read) = UnixStream::pair().expect("engine socket");
    let client_write = client_read.try_clone().expect("clone engine socket");
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || fake_engine_vision(engine_side, "summary", tx));
    let client = EngineClient::over_streams(Box::new(client_read), Box::new(client_write), 8);
    let format = qwen();
    let tokenizer = Arc::new(FastTokenizer::from_file(&directory).expect("loads tokenizer"));
    let pipeline = pipeline_of(&tokenizer, &directory, &format, client);
    let red = png_bytes(32, 32, [220, 10, 10]);
    let blue = png_bytes(32, 32, [10, 10, 220]);
    use base64::Engine as _;
    let data_url = |bytes: &[u8]| format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(bytes));
    let body = json!({"messages":[
        {"role":"system","content":"Inspect the evidence."},
        {"role":"user","content":[{"type":"text","text":"old image"},{"type":"image_url","image_url":{"url":data_url(&red)}}]},
        {"role":"assistant","content":"The older image was red."},
        {"role":"user","content":[{"type":"text","text":"Now inspect this one"},{"type":"image_url","image_url":{"url":data_url(&blue)}}]}
    ]});
    let mut deadline = || Ok(None);
    let prepared = pipeline.prepare_with_compaction(
        &body, &mut deadline, Some(splash_server::serve::compaction::CompactAt::Fraction(0.0)), Some(1), None, true,
    ).expect("image-aware compaction prepares");
    assert!(prepared.compaction.is_some(), "the older turn was compacted");
    assert_eq!(prepared.image_spans.len(), 1, "only the latest retained image remains");
    let expected = splash_server::images::images::prepare(&blue, splash_server::images::MAX_PIXELS).unwrap();
    assert_eq!(prepared.image_pixels, expected.pixels, "the retained pixels belong to the newest message");
    assert_eq!(prepared.image_spans[0].digest_lo, expected.digest_lo);
    assert_eq!(prepared.image_spans[0].digest_hi, expected.digest_hi);
    assert_eq!(pipeline.images.stats().request_bytes, expected.pixels.len(), "Prepared owns exactly the final batch");
    let summary_facts = rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
    assert_eq!(summary_facts.span_count, 0, "the note request is text only");
    drop(prepared);
    assert_eq!(pipeline.images.stats().request_bytes, 0, "dropping Prepared releases the final image budget");
}

#[test]
fn image_only_compaction_keeps_the_latest_image() {
    let Some(directory) = qwen_directory() else { eprintln!("no local Qwen tokenizer; skipped"); return; };
    let (engine_side, client_read) = UnixStream::pair().expect("engine socket");
    let client_write = client_read.try_clone().expect("clone engine socket");
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || fake_engine_vision(engine_side, "summary", tx));
    let client = EngineClient::over_streams(Box::new(client_read), Box::new(client_write), 8);
    let format = qwen();
    let tokenizer = Arc::new(FastTokenizer::from_file(&directory).expect("loads tokenizer"));
    let pipeline = pipeline_of(&tokenizer, &directory, &format, client);
    let old_image = png_bytes(32, 32, [220, 10, 10]);
    let latest_image = png_bytes(32, 32, [10, 10, 220]);
    use base64::Engine as _;
    let data_url = |bytes: &[u8]| format!("data:image/png;base64,{}", base64::engine::general_purpose::STANDARD.encode(bytes));
    let body = json!({"messages":[
        {"role":"system","content":"Compare the images."},
        {"role":"user","content":[{"type":"image_url","image_url":{"url":data_url(&old_image)}}]},
        {"role":"assistant","content":"The earlier image was red."},
        {"role":"user","content":[{"type":"image_url","image_url":{"url":data_url(&latest_image)}}]}
    ]});
    let mut deadline = || Ok(None);
    let prepared = pipeline.prepare_with_compaction(
        &body, &mut deadline, Some(splash_server::serve::compaction::CompactAt::Fraction(0.0)), Some(1), None, true,
    ).expect("text compaction prepares");
    assert!(prepared.compaction.is_some());
    assert_eq!(prepared.image_spans.len(), 1);
    let expected = splash_server::images::images::prepare(&latest_image, splash_server::images::MAX_PIXELS).unwrap();
    assert_eq!(prepared.image_pixels, expected.pixels);
    assert_eq!(prepared.image_spans[0].digest_lo, expected.digest_lo);
    assert_eq!(prepared.image_spans[0].digest_hi, expected.digest_hi);
    assert_eq!(pipeline.images.stats().request_bytes, expected.pixels.len());
    let summary_facts = rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
    assert_eq!(summary_facts.span_count, 0);
    drop(prepared);
    assert_eq!(pipeline.images.stats().request_bytes, 0);
}

/// The tool-call answer a Qwen model writes in its template's XML layout.
const TOOL_CALL_ANSWER: &str = concat!(
    "\n",
    "\u{003c}\u{0074}\u{006f}\u{006f}\u{006c}\u{005f}\u{0063}\u{0061}\u{006c}\u{006c}\u{003e}",
    "\n\u{003c}function=get_weather\u{003e}\n",
    "\u{003c}parameter=city\u{003e}\n",
    "Beijing\n",
    "\u{003c}\u{002f}parameter\u{003e}\n",
    "\u{003c}\u{002f}function\u{003e}\n",
    "\u{003c}\u{002f}\u{0074}\u{006f}\u{006f}\u{006c}\u{005f}\u{0063}\u{0061}\u{006c}\u{006c}\u{003e}",
);

/// The weather tool a request offers.
fn weather_tools() -> Value {
    json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Read the weather of a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }])
}

/// A pipeline with the constraint factory built over the model's tokenizer.
fn constrained_pipeline_of(
    tokenizer: &Arc<FastTokenizer>,
    directory: &PathBuf,
    format: &splash_server::chat_format::ChatFormat,
    client: EngineClient,
) -> Pipeline {
    let env = splash_server::constraints::ConstraintTokenizer::from_directory(
        directory,
        format.vocabulary,
        &format.stop_ids(),
    )
    .expect("env builds");
    let constraints =
        splash_server::constraints::ConstraintFactory::new(&env, format).expect("factory builds");
    let pipeline = pipeline_of(tokenizer, directory, format, client);
    Pipeline {
        constraints: Some(constraints),
        ..pipeline
    }
}

/// A tool call the model writes is projected into the response's
/// tool_calls, with the finish reason that names them.
#[test]
fn a_tool_call_answer_is_projected_into_tool_calls() {
    let Some(directory) = qwen_directory() else {
        eprintln!("no local Qwen tokenizer; skipped");
        return;
    };
    let (engine_side, client_read) = UnixStream::pair().expect("socket pair");
    let client_write = client_read.try_clone().expect("clone");
    std::thread::spawn(move || fake_engine(engine_side, TOOL_CALL_ANSWER));
    let client = EngineClient::over_streams(Box::new(client_read), Box::new(client_write), 8);
    let format = qwen();
    let tokenizer = Arc::new(FastTokenizer::from_file(&directory).expect("loads"));
    let pipeline = constrained_pipeline_of(&tokenizer, &directory, &format, client);
    let body = json!({
        "messages": [{"role": "user", "content": "Weather in Beijing?"}],
        "tools": weather_tools(),
    });
    let prepared = pipeline.prepare(&body, &mut || Ok(())).expect("prepares");
    let (call, tokens, first_batch) = pipeline.submit(&prepared, 256).expect("submits");
    call.result(None).expect("done");
    let first_token_batch_tokens = first_batch.load(std::sync::atomic::Ordering::Relaxed);
    let job = openai::JobInfo { public_id: "req1".into(), created_at: 1, reasoning_tokens: 0 };
    let response = pipeline
        .complete(&body, &prepared, &call, &tokens, first_token_batch_tokens, "Qwen", &job)
        .expect("completes");
    let choice = &response["choices"][0];
    assert_eq!(choice["finish_reason"], json!("tool_calls"));
    let tool_calls = choice["message"]["tool_calls"].as_array().expect("tool calls");
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0]["type"], json!("function"));
    assert_eq!(tool_calls[0]["function"]["name"], json!("get_weather"));
    let arguments: Value =
        serde_json::from_str(tool_calls[0]["function"]["arguments"].as_str().expect("arguments"))
            .expect("arguments JSON");
    assert_eq!(arguments, json!({"city": "Beijing"}));
}

/// A streamed tool call arrives as its header and argument deltas.
#[test]
fn a_tool_call_streams_as_deltas() {
    let Some(directory) = qwen_directory() else {
        eprintln!("no local Qwen tokenizer; skipped");
        return;
    };
    let (engine_side, client_read) = UnixStream::pair().expect("socket pair");
    let client_write = client_read.try_clone().expect("clone");
    std::thread::spawn(move || fake_engine(engine_side, TOOL_CALL_ANSWER));
    let client = EngineClient::over_streams(Box::new(client_read), Box::new(client_write), 8);
    let format = qwen();
    let tokenizer = Arc::new(FastTokenizer::from_file(&directory).expect("loads"));
    let pipeline = constrained_pipeline_of(&tokenizer, &directory, &format, client);
    let body = json!({
        "messages": [{"role": "user", "content": "Weather in Beijing?"}],
        "tools": weather_tools(),
        "stream": true,
    });
    let prepared = pipeline.prepare(&body, &mut || Ok(())).expect("prepares");
    // The SSE sink writes onto its own socket pair; the test reads it.
    use splash_server::serve::stream::StreamSink;
    let (sink_write_side, mut sink_read_side) = UnixStream::pair().expect("socket pair");
    let writer_stream = Mutex::new(sink_write_side.try_clone().expect("clone"));
    let sink = Arc::new(StreamSink {
        write: Arc::new(Mutex::new(Box::new(move |payload: &str| {
            writer_stream
                .lock()
                .unwrap()
                .write_all(format!("data: {payload}\n\n").as_bytes())
                .map_err(|error| error.to_string())
        }))),
        model: "Qwen".into(),
        request_id: "req1".into(),
        created_at: 1_700_000_000,
        include_usage: false,
        compaction: None,
    });
    let constraint = pipeline
        .constraint(&prepared)
        .expect("constraint reads")
        .map(std::sync::Arc::new);
    let projector = Arc::new(Mutex::new(StreamingToolCallProjector::new(
        Box::new(splash_server::tool_schema::ToolParameterTypes::new(
            prepared.tool_policy.clone().expect("policy"),
        )),
        1,
        &format,
        false,
    )));
    let tokens: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));
    let (on_event, on_complete) = splash_server::serve::stream::stream_callbacks(
        sink,
        tokens,
        tokenizer,
        projector,
        prepared.thinking,
        format.clone(),
        true,
        constraint,
        None,
    );
    let generation = pipeline.generation_request(&prepared, 256, None);
    let call = pipeline
        .client
        .submit(&generation, Some(on_event), Some(on_complete))
        .expect("submits");
    call.result(None).expect("done");
    // The stream's events, read off the socket until the end.
    let mut text = String::new();
    sink_read_side
        .set_read_timeout(Some(std::time::Duration::from_millis(200)))
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let mut buffer = [0u8; 65536];
    while std::time::Instant::now() < deadline {
        match sink_read_side.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => text.push_str(&String::from_utf8_lossy(&buffer[..count])),
            Err(_) => continue,
        }
        if text.contains("[DONE]") {
            break;
        }
    }
    // The call's header and its arguments streamed as tool_calls deltas.
    let mut names = String::new();
    let mut arguments = String::new();
    for line in text.lines() {
        let Some(payload) = line.strip_prefix("data: ") else { continue };
        if payload == "[DONE]" { continue; }
        let Ok(chunk) = serde_json::from_str::<Value>(payload) else { continue };
        if let Some(calls) = chunk.pointer("/choices/0/delta/tool_calls").and_then(Value::as_array) {
            for call in calls {
                if let Some(name) = call.pointer("/function/name").and_then(Value::as_str) {
                    names.push_str(name);
                }
                if let Some(piece) = call.pointer("/function/arguments").and_then(Value::as_str) {
                    arguments.push_str(piece);
                }
            }
        }
    }
    assert_eq!(names, "get_weather", "{text}");
    let assembled: Value = serde_json::from_str(&arguments).expect("arguments JSON");
    assert_eq!(assembled, json!({"city": "Beijing"}), "{text}");
    // The finish chunk names the calls.
    assert!(text.contains("\"finish_reason\":\"tool_calls\""), "{text}");
}

/// A structured output the model ended is validated against the response
/// format; an invalid one is refused.
#[test]
fn a_structured_output_is_validated_against_the_response_format() {
    let Some(directory) = qwen_directory() else {
        eprintln!("no local Qwen tokenizer; skipped");
        return;
    };
    let (engine_side, client_read) = UnixStream::pair().expect("socket pair");
    let client_write = client_read.try_clone().expect("clone");
    std::thread::spawn(move || {
        fake_engine(
            engine_side,
            "Let me think.\u{003c}\u{002f}\u{0074}\u{0068}\u{0069}\u{006e}\u{006b}\u{003e}{\"answer\": 42}",
        )
    });
    let client = EngineClient::over_streams(Box::new(client_read), Box::new(client_write), 8);
    let format = qwen();
    let tokenizer = Arc::new(FastTokenizer::from_file(&directory).expect("loads"));
    let pipeline = constrained_pipeline_of(&tokenizer, &directory, &format, client);
    let body = json!({
        "messages": [{"role": "user", "content": "Give me a number"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "integer"}},
                    "required": ["answer"],
                    "additionalProperties": false,
                },
            },
        },
    });
    let prepared = pipeline.prepare(&body, &mut || Ok(())).expect("prepares");
    assert!(prepared.constrained(), "a response format constrains");
    let (call, tokens, first_batch) = pipeline.submit(&prepared, 256).expect("submits");
    call.result(None).expect("done");
    let first_token_batch_tokens = first_batch.load(std::sync::atomic::Ordering::Relaxed);
    let job = openai::JobInfo { public_id: "req1".into(), created_at: 1, reasoning_tokens: 0 };
    let response = pipeline
        .complete(&body, &prepared, &call, &tokens, first_token_batch_tokens, "Qwen", &job)
        .expect("completes");
    let content = response["choices"][0]["message"]["content"]
        .as_str()
        .expect("content");
    let parsed: Value = serde_json::from_str(content).expect("content JSON");
    assert_eq!(parsed, json!({"answer": 42}));
}

/// A structured output that breaks the response format is refused with the
/// model-output code.
#[test]
fn an_invalid_structured_output_is_refused() {
    let Some(directory) = qwen_directory() else {
        eprintln!("no local Qwen tokenizer; skipped");
        return;
    };
    let (engine_side, client_read) = UnixStream::pair().expect("socket pair");
    let client_write = client_read.try_clone().expect("clone");
    std::thread::spawn(move || fake_engine(engine_side, "{\"answer\": \"not a number\"}"));
    let client = EngineClient::over_streams(Box::new(client_read), Box::new(client_write), 8);
    let format = qwen();
    let tokenizer = Arc::new(FastTokenizer::from_file(&directory).expect("loads"));
    let pipeline = constrained_pipeline_of(&tokenizer, &directory, &format, client);
    let body = json!({
        "messages": [{"role": "user", "content": "Give me a number"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "integer"}},
                    "required": ["answer"],
                    "additionalProperties": false,
                },
            },
        },
    });
    let prepared = pipeline.prepare(&body, &mut || Ok(())).expect("prepares");
    let (call, tokens, first_batch) = pipeline.submit(&prepared, 256).expect("submits");
    call.result(None).expect("done");
    let first_token_batch_tokens = first_batch.load(std::sync::atomic::Ordering::Relaxed);
    let job = openai::JobInfo { public_id: "req1".into(), created_at: 1, reasoning_tokens: 0 };
    let error = pipeline
        .complete(&body, &prepared, &call, &tokens, first_token_batch_tokens, "Qwen", &job)
        .expect_err("refused");
    assert_eq!(error.status, 500);
    assert_eq!(error.code.as_deref(), Some("invalid_model_output"));
}

/// The facts of one constrained request the engine reads: the constraint
/// mode byte, and the mask response the mask request drew.
struct MaskFacts {
    constraint_mode: u8,
    mask_request_id: u64,
    mask_bytes: Vec<u8>,
}

/// The fake engine of a constrained request: it reads the request's
/// constraint mode, asks for one mask over the call's opening token, reads
/// the mask response back, and then answers as the plain fake does.
fn fake_engine_constrained(
    mut stream: UnixStream,
    answer: &'static str,
    words_per_mask: u32,
    report: std::sync::mpsc::Sender<MaskFacts>,
) {
    let ready = wire_event(&json!({
        "frame_type": 0x0100,
        "max_concurrent_requests": 4,
        "max_context_tokens": 524288,
        "vision": false,
    }));
    let _ = stream.write_all(&ready);
    loop {
        let mut header = [0u8; 16];
        if stream.read_exact(&mut header).is_err() {
            return;
        }
        let raw_type = u16::from_le_bytes([header[6], header[7]]);
        let payload_bytes = u64::from_le_bytes(header[8..16].try_into().unwrap()) as usize;
        let mut payload = vec![0u8; payload_bytes];
        if payload_bytes > 0 {
            stream.read_exact(&mut payload).expect("engine reads payload");
        }
        if raw_type != 0x0001 {
            continue;
        }
        let request_id = u64::from_le_bytes(payload[..8].try_into().unwrap());
        let constraint_mode = payload[9];
        let prompt_count = u32::from_le_bytes(payload[30..34].try_into().unwrap());
        // One mask over the call's opening token, before any output.
        let mask_request_id = request_id;
        let mut mask_payload = Vec::new();
        mask_payload.extend_from_slice(&request_id.to_le_bytes());
        mask_payload.extend_from_slice(&mask_request_id.to_le_bytes());
        mask_payload.extend_from_slice(&words_per_mask.to_le_bytes());
        let call_open = splash_server::chat_format::qwen().call_open_id;
        mask_payload.extend_from_slice(&1u32.to_le_bytes());
        mask_payload.extend_from_slice(&call_open.to_le_bytes());
        let mut frame = Vec::new();
        frame.extend_from_slice(b"SPLH");
        frame.extend_from_slice(&13u16.to_le_bytes());
        frame.extend_from_slice(&0x0103u16.to_le_bytes());
        frame.extend_from_slice(&(mask_payload.len() as u64).to_le_bytes());
        frame.extend_from_slice(&mask_payload);
        let _ = stream.write_all(&frame);
        // The mask response: its header, then the words.
        let mut response_header = [0u8; 16];
        stream.read_exact(&mut response_header).expect("engine reads mask response");
        let response_type = u16::from_le_bytes([response_header[6], response_header[7]]);
        let response_bytes =
            u64::from_le_bytes(response_header[8..16].try_into().unwrap()) as usize;
        let mut response = vec![0u8; response_bytes];
        stream.read_exact(&mut response).expect("engine reads mask words");
        assert_eq!(response_type, 0x0003, "the mask response frame type");
        let response_request_id = u64::from_le_bytes(response[..8].try_into().unwrap());
        let response_mask_id = u64::from_le_bytes(response[8..16].try_into().unwrap());
        assert_eq!(response_request_id, request_id);
        assert_eq!(response_mask_id, mask_request_id);
        let _ = report.send(MaskFacts {
            constraint_mode,
            mask_request_id,
            mask_bytes: response[20..].to_vec(),
        });
        for event in [
            json!({"frame_type": 0x0101, "request_id": request_id, "lane": 0,
                   "matched_prompt_tokens": 0}),
            json!({"frame_type": 0x0102, "request_id": request_id, "sequence_offset": 0,
                   "tokens": answer_tokens(answer)}),
            json!({"frame_type": 0x0104, "request_id": request_id, "reason": 0,
                   "bounded": false, "prompt_tokens": prompt_count,
                   "completion_tokens": answer_tokens(answer).len(),
                   "prefill_micros": 1000, "decode_micros": 2000,
                   "wall_micros": 3500, "deduplicated_tokens": 0,
                   "drafted_tokens": 0, "accepted_draft_tokens": 0,
                   "draft_cycles": 0, "logit_count": 0}),
        ] {
            let _ = stream.write_all(&wire_event(&event));
        }
    }
}

/// A constrained request reaches the engine under a token mask, and the
/// mask the grammar produces holds the call's opening and refuses what
/// cannot follow it.
#[test]
fn a_constrained_request_generates_under_a_token_mask() {
    let Some(directory) = qwen_directory() else {
        eprintln!("no local Qwen tokenizer; skipped");
        return;
    };
    let (engine_side, client_read) = UnixStream::pair().expect("socket pair");
    let client_write = client_read.try_clone().expect("clone");
    let (tx, rx) = std::sync::mpsc::channel();
    let format = qwen();
    let words_per_mask = (format.vocabulary + 31) / 32;
    std::thread::spawn(move || {
        fake_engine_constrained(engine_side, "The answer is 42.", words_per_mask, tx)
    });
    let client = EngineClient::over_streams(Box::new(client_read), Box::new(client_write), 8);
    let tokenizer = Arc::new(FastTokenizer::from_file(&directory).expect("loads"));
    let pipeline = constrained_pipeline_of(&tokenizer, &directory, &format, client);
    let body = json!({
        "messages": [{"role": "user", "content": "Weather in Beijing?"}],
        "tools": weather_tools(),
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        "reasoning_effort": "none",
    });
    let prepared = pipeline.prepare(&body, &mut || Ok(())).expect("prepares");
    assert!(prepared.constrained(), "a required choice constrains");
    let (call, _tokens, _first_batch) = pipeline.submit(&prepared, 256).expect("submits");
    call.result(None).expect("done");
    let facts = rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("the engine read the request");
    assert_eq!(facts.constraint_mode, 1, "the request carries a token mask");
    // Two rows: one before the simulated token, one after it.
    let words_per_row = (format.vocabulary as usize + 31) / 32;
    assert_eq!(facts.mask_bytes.len(), 2 * words_per_row * 4);
    let allowed = |row: usize, token: u32| {
        let (word_index, bit) = ((token / 32) as usize, token % 32);
        let word = u32::from_le_bytes(
            facts.mask_bytes[row * words_per_row * 4 + word_index * 4
                ..row * words_per_row * 4 + word_index * 4 + 4]
                .try_into()
                .expect("mask word"),
        );
        word & (1 << bit) != 0
    };
    // The call's opening may start the output, and may follow itself only
    // as the grammar's layout writes it: the name's opening text.
    let call_open = format.call_open_id;
    assert!(allowed(0, call_open), "the call may start the output");
    // A plain text token cannot follow the call's opening.
    let the = tokenizer.encode("The", false).expect("encodes")[0];
    assert!(!allowed(1, the), "plain text cannot follow the call's opening");
}
