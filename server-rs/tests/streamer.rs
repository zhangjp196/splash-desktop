//! The streamer's tests, ported from the Python streamer's behavior: the
//! incremental decode agrees with the whole decode, the stop sequences
//! hold their may-complete ends, and a stop ends the stream.

use splash_server::output::streamer::Streamed;
use splash_server::output::CallbackStreamer;
use splash_server::tokenization::FastTokenizer;
use std::path::PathBuf;
use std::sync::Arc;

/// A Qwen-family tokenizer the machine has cached, when it does.
fn qwen_tokenizer() -> Option<(Arc<FastTokenizer>, PathBuf)> {
    let base = std::env::var("HOME").ok()?;
    let dir: PathBuf = base.parse().ok()?;
    let snapshots = dir.join(
        ".cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots",
    );
    let snapshot = std::fs::read_dir(&snapshots).ok()?.next()?.ok()?;
    let directory = snapshot.path();
    Some((
        Arc::new(FastTokenizer::from_file(&directory).expect("tokenizer loads")),
        directory,
    ))
}

/// The text the streamed decisions carry.
fn text_of(streamed: &[Streamed]) -> String {
    streamed
        .iter()
        .map(|decision| match decision {
            Streamed::Text(text) => text.clone(),
            Streamed::Stopped => String::new(),
        })
        .collect::<Vec<_>>()
        .concat()
}

#[test]
fn the_incremental_decode_agrees_with_the_whole_decode() {
    let Some((tokenizer, _)) = qwen_tokenizer() else {
        eprintln!("no local Qwen tokenizer; skipped");
        return;
    };
    let text = "The answer is 42, and then some more text follows here.";
    let tokens = tokenizer.encode(text, false).expect("encodes");
    let mut streamer = CallbackStreamer::new(&[]);
    let mut streamed = Vec::new();
    // One token at a time, as the engine sends them.
    for chunk in tokens.chunks(1) {
        streamed.extend(streamer.put_tokens(&tokenizer, chunk));
    }
    streamed.extend(streamer.end(&tokenizer));
    assert_eq!(text_of(&streamed), text);
    // The pieces' concatenation is the whole decode: no divergence.
    assert_eq!(streamer.token_ids(), &tokens[..]);
}

/// A multi-byte character split across tokens must not split the text at a
/// byte offset: the Python's streamer counts characters, and a byte offset
/// lands inside a character and panics.
#[test]
fn a_multibyte_reply_streams_without_splitting_a_character() {
    let Some((tokenizer, _)) = qwen_tokenizer() else {
        eprintln!("no local Qwen tokenizer; skipped");
        return;
    };
    let text = "思考过程：1 + 1 等于 2。然后给出答案。";
    let tokens = tokenizer.encode(text, false).expect("encodes");
    let mut streamer = CallbackStreamer::new(&[]);
    let mut streamed = Vec::new();
    for chunk in tokens.chunks(1) {
        streamed.extend(streamer.put_tokens(&tokenizer, chunk));
    }
    streamed.extend(streamer.end(&tokenizer));
    assert_eq!(text_of(&streamed), text);
}

#[test]
fn a_stop_sequence_holds_its_may_complete_end_and_ends_the_stream() {
    let Some((tokenizer, _)) = qwen_tokenizer() else {
        eprintln!("no local Qwen tokenizer; skipped");
        return;
    };
    let stop = "\u{003c}\u{002f}\u{0074}\u{0068}\u{0069}\u{006e}\u{006b}\u{003e}";
    let mut streamer = CallbackStreamer::new(&[stop]);
    let before = "thinking text ";
    let tokens = tokenizer.encode(&format!("{before}{stop} after"), false).expect("encodes");
    let streamed = streamer.put_tokens(&tokenizer, &tokens);
    // The stream ends at the stop: the text before it went out, the stop
    // and what follows did not.
    assert_eq!(text_of(&streamed), before);
    assert!(streamed.iter().any(|decision| *decision == Streamed::Stopped));
    assert_eq!(streamer.stop_sequence(), Some(stop));
    // Further tokens are refused.
    assert!(streamer.put_tokens(&tokenizer, &[42]).is_empty());
    assert!(streamer.end(&tokenizer).is_empty());
}

#[test]
fn a_stop_split_across_tokens_is_found() {
    let Some((tokenizer, _)) = qwen_tokenizer() else {
        eprintln!("no local Qwen tokenizer; skipped");
        return;
    };
    let stop = "\u{003c}\u{002f}\u{0074}\u{0068}\u{0069}\u{006e}\u{006b}\u{003e}";
    let mut streamer = CallbackStreamer::new(&[stop]);
    let before = "thinking ";
    let tokens = tokenizer.encode(&format!("{before}{stop}"), false).expect("encodes");
    // The stop's tokens arrive one at a time: the held text completes it.
    let streamed = streamer.put_tokens(&tokenizer, &tokens);
    assert_eq!(text_of(&streamed), before);
    assert!(streamed.iter().any(|decision| *decision == Streamed::Stopped));
}

#[test]
fn a_multibyte_character_split_across_tokens_is_held() {
    let Some((tokenizer, _)) = qwen_tokenizer() else {
        eprintln!("no local Qwen tokenizer; skipped");
        return;
    };
    // An emoji's bytes may split across tokens: the pieces decode to
    // U+FFFD until the character completes.
    let text = "ok \u{1f600} done";
    let tokens = tokenizer.encode(text, false).expect("encodes");
    let mut streamer = CallbackStreamer::new(&[]);
    let mut streamed = Vec::new();
    for chunk in tokens.chunks(1) {
        streamed.extend(streamer.put_tokens(&tokenizer, chunk));
    }
    streamed.extend(streamer.end(&tokenizer));
    // No U+FFFD leaks into the stream.
    assert_eq!(text_of(&streamed), text);
    assert!(!text_of(&streamed).contains('\u{fffd}'));
}
