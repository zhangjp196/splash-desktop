//! The constraints' tests, ported from the Python tests that drive
//! `ConstraintFactory` and `TokenConstraint`, each expectation checked
//! against the Python implementation's own output.

use splash_server::chat_format::qwen;
use splash_server::constraints::{ConstraintFactory, ConstraintTokenizer, Grammar};
use splash_server::tokenization::FastTokenizer;
use std::path::PathBuf;
use std::sync::Arc;

fn qwen_directory() -> Option<PathBuf> {
    let base = std::env::var("HOME").ok()?;
    let dir: PathBuf = base.parse().ok()?;
    let snapshots = dir.join(
        ".cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots",
    );
    let snapshot = std::fs::read_dir(&snapshots).ok()?.next()?.ok()?;
    Some(snapshot.path())
}

/// The grammar tokenizer and the encoding tokenizer of one model.
fn tokenizers() -> Option<(ConstraintFactory, Arc<FastTokenizer>)> {
    let directory = qwen_directory()?;
    let format = qwen();
    let env_tokenizer =
        ConstraintTokenizer::from_directory(&directory, format.vocabulary, &format.stop_ids())
            .expect("env builds");
    let factory = ConstraintFactory::new(&env_tokenizer, &format).expect("factory builds");
    let fast = Arc::new(FastTokenizer::from_file(&directory).expect("tokenizer loads"));
    Some((factory, fast))
}

/// Whether a mask's first row allows `token_id`.
fn allowed(mask: &[u8], token_id: u32) -> bool {
    let (word_index, bit) = ((token_id / 32) as usize, token_id % 32);
    let word = u32::from_le_bytes(
        mask[word_index * 4..word_index * 4 + 4]
            .try_into()
            .expect("mask word"),
    );
    (word >> bit) & 1 == 1
}

const HELLO_WORLD: &str =
    r#"{"grammars":[{"name":"regex","lark_grammar":"start: /hello [0-9]+ world/"}]}"#;

#[test]
fn a_regex_grammar_allows_its_first_tokens_only() {
    let Some((factory, _tokenizer)) = tokenizers() else {
        eprintln!("no local Qwen tokenizer; skipped");
        return;
    };
    let constraint = factory
        .create(&Grammar::Gbnf(HELLO_WORLD.to_string()))
        .expect("compiles");
    let mask = constraint.masks(&[]).expect("mask");
    // The grammar starts with the literal "hello": its first token is
    // allowed, a digit is not, and neither an end nor an unrelated token
    // is.
    assert!(allowed(&mask, 14556), "hello allowed");
    assert!(!allowed(&mask, 19), "digit refused");
    assert!(!allowed(&mask, 248046), "im_end refused");
    assert!(!allowed(&mask, 79420), "thinking refused");
}

#[test]
fn a_consumed_grammar_allows_its_end() {
    let Some((factory, tokenizer)) = tokenizers() else {
        eprintln!("no local Qwen tokenizer; skipped");
        return;
    };
    let constraint = factory
        .create(&Grammar::Gbnf(HELLO_WORLD.to_string()))
        .expect("compiles");
    let tokens = tokenizer
        .encode("hello 42 world", false)
        .expect("encodes");
    constraint.commit(&tokens).expect("commits");
    let mask = constraint.masks(&[]).expect("mask");
    // The grammar matched whole: it may end, and it may not start again.
    assert!(allowed(&mask, 248046), "im_end allowed");
    assert!(!allowed(&mask, 14556), "hello refused");
}

#[test]
fn a_generated_token_out_of_range_is_refused() {
    let Some((factory, _tokenizer)) = tokenizers() else {
        eprintln!("no local Qwen tokenizer; skipped");
        return;
    };
    let constraint = factory
        .create(&Grammar::Gbnf(HELLO_WORLD.to_string()))
        .expect("compiles");
    let error = constraint
        .commit(&[qwen().vocabulary])
        .expect_err("out of range");
    assert!(error.contains("out of range"), "{error}");
}

#[test]
fn the_cache_serves_the_same_grammar_again() {
    let Some((factory, _tokenizer)) = tokenizers() else {
        eprintln!("no local Qwen tokenizer; skipped");
        return;
    };
    let grammar = Grammar::Gbnf(HELLO_WORLD.to_string());
    let first = factory.create(&grammar).expect("compiles");
    let misses = *factory.misses.lock().unwrap();
    let hits = *factory.hits.lock().unwrap();
    let second = factory.create(&grammar).expect("compiles");
    assert_eq!(*factory.misses.lock().unwrap(), misses, "no new miss");
    assert_eq!(*factory.hits.lock().unwrap(), hits + 1, "one new hit");
    // The two constraints consume their matchers independently: the
    // first's consumption leaves the second's grammar at its start.
    let tokens = vec![14556u32];
    first.commit(&tokens).expect("commits");
    let mask = second.masks(&[]).expect("mask");
    assert!(allowed(&mask, 14556), "the second's grammar is untouched by the first's consumption");
    assert!(!allowed(&mask, 19), "the second's grammar has not advanced");
}
