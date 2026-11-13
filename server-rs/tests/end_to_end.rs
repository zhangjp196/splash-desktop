//! The end-to-end path's test: a Chat request from HTTP through the
//! pipeline to the engine and its answer back, against a fake engine that
//! speaks the wire protocol.

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::sync::Arc;

use serde_json::{json, Value};
use splash_server::chat_format::qwen;
use splash_server::chat_messages::normalize_messages;
use splash_server::dialects::openai;
use splash_server::engine::EngineClient;
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
        if raw_type != 0x0001 {
            continue;
        }
        let request_id = u64::from_le_bytes(payload[..8].try_into().unwrap());
        let prompt_count = u32::from_le_bytes(payload[30..34].try_into().unwrap());
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
    // The engine side of a socket pair, with the answer it writes.
    let (engine_side, client_read) = UnixStream::pair().expect("socket pair");
    let client_write = client_read.try_clone().expect("clone");
    std::thread::spawn(move || fake_engine(engine_side, "The answer is 42."));
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
    let (call, tokens) = pipeline.submit(&prepared, 64).expect("submits");
    let done = call.result(None).expect("done");
    assert_eq!(done.prompt_tokens as usize, prompt_tokens.len());
    let job = openai::JobInfo {
        public_id: "test1".into(),
        created_at: 1_700_000_000,
        reasoning_tokens: 0,
    };
    let response = pipeline
        .complete(&body, &prepared, &call, &tokens, "Qwen", &job)
        .expect("assembles");
    // The response is OpenAI's shape, with the answer the engine wrote.
    assert_eq!(response["object"], json!("chat.completion"));
    assert_eq!(response["model"], json!("Qwen"));
    assert_eq!(response["choices"][0]["message"]["content"], json!("The answer is 42."));
    assert_eq!(response["choices"][0]["finish_reason"], json!("stop"));
    assert_eq!(response["usage"]["prompt_tokens"], json!(prompt_tokens.len() as u64));
}

/// The template messages step, as the pipeline's prepare runs it.
fn crate_chat_messages_template(normalized: &[Value]) -> Value {
    Value::Array(splash_server::chat_messages::template_messages(normalized))
}
