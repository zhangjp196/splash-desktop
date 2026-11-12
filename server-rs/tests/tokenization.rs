//! The prompt tokenizer's tests: the boundary-support rules against
//! synthetic tokenizer.json pipelines, and the prefix reuse against the
//! real Qwen tokenizer when the machine has one.

use std::path::PathBuf;
use std::sync::Arc;

use serde_json::json;
use splash_server::tokenization::{FastTokenizer, PromptTokenizer};

/// A Qwen-family tokenizer.json the machine has cached, when it does.
fn qwen_tokenizer_path() -> Option<PathBuf> {
    let base = std::env::var("HOME").ok()?;
    let dir = base.parse::<PathBuf>().ok()?;
    let path = dir.join(
        ".cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots",
    );
    let snapshot = std::fs::read_dir(&path).ok()?.next()?.ok()?;
    let tokenizer = snapshot.path().join("tokenizer.json");
    tokenizer.is_file().then_some(tokenizer)
}

const MARKER: &str = "<|im_end|>";

fn qwen_prompt_tokenizer() -> Option<(PromptTokenizer, PathBuf)> {
    let path = qwen_tokenizer_path()?;
    let directory = path.parent().expect("snapshot dir").to_path_buf();
    let tokenizer = Arc::new(FastTokenizer::from_file(&directory).expect("tokenizer loads"));
    Some((
        PromptTokenizer::from_directory(tokenizer, &directory, MARKER),
        directory,
    ))
}

#[test]
fn the_qwen_pipeline_supports_boundaries() {
    let Some((tokenizer, _)) = qwen_prompt_tokenizer() else {
        eprintln!("no local Qwen tokenizer; skipped");
        return;
    };
    assert!(tokenizer.enabled);
    assert_eq!(tokenizer.marker_id, Some(248046));
}

#[test]
fn a_long_prompt_reuses_its_cached_prefix() {
    let Some((tokenizer, _)) = qwen_prompt_tokenizer() else {
        eprintln!("no local Qwen tokenizer; skipped");
        return;
    };
    // A prefix past the minimum, ending at the marker, then a tail.
    let marker_text = format!("You are helpful. {}", MARKER);
    let padding = "def solve(): return 42  # padding line\n".repeat(200);
    let prefix = format!("{padding}{marker_text}");
    assert!(prefix.len() > 4096);
    let tail = " Write a fib function.";
    let whole = format!("{prefix}{tail}");

    let first = tokenizer.encode(&whole);
    let stats = tokenizer.stats();
    assert!(!stats.enabled == false);
    // The first encode caches the prefix: one entry, no hits yet.
    assert_eq!(stats.entries, 1);
    assert_eq!(stats.hits, 0);

    // The second encode of a longer conversation reuses the prefix and
    // encodes only the tail.
    let longer = format!("{whole} and memoize it.");
    let second = tokenizer.encode(&longer);
    assert_eq!(&second[..first.len()], &first[..], "the prefix's tokens changed");
    let stats = tokenizer.stats();
    assert_eq!(stats.hits, 1);
    assert!(stats.reused_tokens > 0);
    assert_eq!(stats.entries, 1, "the extension replaced its earlier prefix");
}

#[test]
fn a_short_prompt_is_not_cached() {
    let Some((tokenizer, _)) = qwen_prompt_tokenizer() else {
        eprintln!("no local Qwen tokenizer; skipped");
        return;
    };
    let text = format!("hi {MARKER} there");
    let tokens = tokenizer.encode(&text);
    assert_eq!(tokenizer.stats().entries, 0, "below the minimum prefix");
    // The tokens are the plain encoding's.
    assert_eq!(tokens, tokenizer.encode_all_public(&text));
}

#[test]
fn split_cuts_past_the_last_marker() {
    let Some((tokenizer, _)) = qwen_prompt_tokenizer() else {
        eprintln!("no local Qwen tokenizer; skipped");
        return;
    };
    let head_text = format!("system prompt {MARKER}");
    let tail = "user turn";
    let (head, cut) = tokenizer.split(&format!("{head_text}{tail}"));
    assert!(!head.is_empty());
    assert_eq!(cut, tail);
    // The head's tokens are the plain encoding of the head text.
    assert_eq!(head, tokenizer.encode_all_public(&head_text));
    // No marker, no split.
    let (head, cut) = tokenizer.split("no marker here");
    assert!(head.is_empty());
    assert_eq!(cut, "no marker here");
}

#[test]
fn the_boundary_rules_reject_other_pipelines() {
    // The checks read the tokenizer.json's pipeline, so synthetic
    // documents decide them without a working tokenizer.
    let qwen_like = json!({
        "model": {"type": "BPE", "dropout": null},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": "..."}},
            {"type": "ByteLevel", "add_prefix_space": false}
        ]},
        "normalizer": {"type": "NFC"},
        "post_processor": {"type": "ByteLevel"}
    });
    assert!(splash_server::tokenization::prompt::supports_boundaries_for_test(
        &qwen_like, MARKER
    ));
    // A Metaspace pre-tokenizer depends on whether a segment starts the
    // input, so it does not split.
    let metaspace = json!({
        "model": {"type": "BPE"},
        "pre_tokenizer": {"type": "Metaspace", "prepend_scheme": "first"}
    });
    assert!(!splash_server::tokenization::prompt::supports_boundaries_for_test(
        &metaspace, MARKER
    ));
    // A BPE dropout makes the splits probabilistic.
    let dropout = json!({
        "model": {"type": "BPE", "dropout": 0.1},
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false}
    });
    assert!(!splash_server::tokenization::prompt::supports_boundaries_for_test(
        &dropout, MARKER
    ));
    // A prefix space makes the byte-level split depend on position.
    let prefix_space = json!({
        "model": {"type": "BPE"},
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": true}
    });
    assert!(!splash_server::tokenization::prompt::supports_boundaries_for_test(
        &prefix_space, MARKER
    ));
    // A normalizer other than NFC changes what the text reads.
    let nfd = json!({
        "model": {"type": "BPE"},
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
        "normalizer": {"type": "NFD"}
    });
    assert!(!splash_server::tokenization::prompt::supports_boundaries_for_test(
        &nfd, MARKER
    ));
    // No pre-tokenizer at all.
    let bare = json!({"model": {"type": "BPE"}});
    assert!(!splash_server::tokenization::prompt::supports_boundaries_for_test(
        &bare, MARKER
    ));
}
