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
            payload.extend_from_slice(&(event["logit_count"].as_u64().unwrap() as u32).to_le_bytes());
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

    let prepared = pipeline.prepare(&body).expect("prepares");
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
    assert!(pipeline.prepare(&body).expect("prepares").thinking,
        "the template opens the think block by default");
    body["reasoning_effort"] = json!("none");
    assert!(!pipeline.prepare(&body).expect("prepares").thinking,
        "reasoning off closes it");
    body["reasoning_effort"] = json!("medium");
    assert!(pipeline.prepare(&body).expect("prepares").thinking,
        "an effort opens it");
    body["chat_template_kwargs"] = json!({"enable_thinking": false});
    assert!(!pipeline.prepare(&body).expect("prepares").thinking,
        "the request's own switch outranks the effort");
    body["chat_template_kwargs"] = json!({"messages": []});
    let refused = pipeline.prepare(&body).err().expect("refused");
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
    }
}

fn pipeline_source(pipeline: &Pipeline) -> String {
    pipeline
        .templates
        .select(None, None)
        .expect("the model has a chat template")
}

/// The server over real TCP: a Chat completion answered complete, and one
/// answered streamed, both read back off the socket.
#[test]
fn the_server_answers_over_real_tcp() {
    use splash_server::serve::Server;

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
    };
    let server = Server::new(pipeline);

    // The listener on an ephemeral port.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("binds");
    let address = listener.local_addr().expect("address").to_string();
    std::thread::spawn(move || server.serve(listener));

    let body = json!({
        "model": "Qwen",
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
}
