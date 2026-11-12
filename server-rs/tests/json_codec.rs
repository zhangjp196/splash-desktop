//! The JSON codec's tests, ported from
//! `dev/tests/server/test_json_codec.py`. The lone-surrogate cases are the
//! documented deviation (see `support/json_codec`): serde_json refuses
//! them at parse, so the values that carry one are not representable here.

use serde_json::{json, Value};
use splash_server::support::json_codec as codec;

fn assert_roundtrip(value: &Value) -> String {
    let text = codec::dumps(value).expect("dumps");
    let data = codec::encode(value).expect("encode");
    assert_eq!(text.clone().into_bytes(), data);
    assert_eq!(codec::encoded_size(value).expect("size"), data.len());
    let reparsed = serde_json::from_slice::<Value>(&data).expect("reparse");
    assert_eq!(reparsed, *value);
    for separator in ["\u{85}", "\u{2028}", "\u{2029}"] {
        assert!(!text.contains(separator), "unescaped separator in {text}");
    }
    text
}

#[test]
fn unicode_controls_and_nested_json() {
    let value = json!({
        "中文🌍": [
            "café e\u{301} \"quoted\" C:\\Users\\file.txt\n\r\t\u{0}",
            "\u{85}\u{2028}\u{2029}\u{feff}\u{ffff}\u{10ffff}",
            "literal \\ud800 and \\u2028",
            {"integer": 1_152_921_504_606_846_976u64, "float": 1.25, "bool": true, "null": null},
        ]
    });
    let data = assert_roundtrip(&value);
    assert!(data.contains("中文🌍"));
    assert!(!data.contains("\\u4e2d"));
    let arguments = codec::dumps(&value).unwrap();
    let outer = assert_roundtrip(&json!({"function": {"arguments": arguments}}));
    let outer_value: Value = serde_json::from_str(&outer).unwrap();
    assert_eq!(
        outer_value["function"]["arguments"].as_str().unwrap(),
        arguments
    );
}

#[test]
fn fragment_encoding_matches_complete_strings() {
    // The Python test's alphabet without the lone surrogates, which the
    // Rust value type cannot hold (see the module note).
    let alphabet: Vec<char> = "a中🌍é\u{0}\n\r\t\"\\\u{85}\u{2028}\u{2029}".chars().collect();
    let mut state = 42u64;
    let mut random = move || {
        // A minimal LCG so the case stays deterministic without a
        // dependency.
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    for _ in 0..1000 {
        let length = random() % 80 + 1;
        let value: String = (0..length)
            .map(|_| alphabet[random() % alphabet.len()])
            .collect();
        let split = random() % (value.chars().count() + 1);
        let head: String = value.chars().take(split).collect();
        let tail: String = value.chars().skip(split).collect();
        fn inner(text: &str) -> &str {
            &text[1..text.len() - 1]
        }
        let partial = format!(
            "{}{}",
            inner(&codec::dumps(&Value::String(head)).unwrap()),
            inner(&codec::dumps(&Value::String(tail)).unwrap()),
        );
        let whole = codec::dumps(&Value::String(value.clone())).unwrap();
        let whole_inner = whole[1..whole.len() - 1].to_string();
        assert_eq!(partial, whole_inner, "split {split} of {value}");
        assert_roundtrip(&json!({"text": value}));
    }
}

#[test]
fn invalid_values_fail_without_affecting_later_encoding() {
    // A NaN the value carries directly: serde_json refuses it, as the
    // Python's allow_nan=False does.
    let nan = serde_json::Number::from_f64(f64::NAN);
    assert!(nan.is_none(), "serde_json builds no NaN number");
    // A later encoding still works.
    assert_roundtrip(&json!({"ok": true}));
}

#[test]
fn loading_refuses_json_nested_deeper_than_its_bound() {
    let mut deep = String::new();
    for _ in 0..=codec::MAX_DEPTH {
        deep.push('[');
    }
    deep.push('1');
    for _ in 0..=codec::MAX_DEPTH {
        deep.push(']');
    }
    let error = codec::loads(&deep).expect_err("deep JSON loaded");
    assert!(
        error.0.contains("deeper than") || error.0.contains("recursion"),
        "{error}"
    );
    // One level shallower loads.
    let mut shallow = String::new();
    for _ in 0..codec::MAX_DEPTH {
        shallow.push('[');
    }
    shallow.push('1');
    for _ in 0..codec::MAX_DEPTH {
        shallow.push(']');
    }
    codec::loads(&shallow).expect("shallow JSON refused");
}

#[test]
fn strict_loading_preserves_finite_numbers_and_numeric_strings() {
    let value = codec::loads(r#"{"a": 1.25, "b": "1.25", "c": 1e3, "d": -0.0}"#).unwrap();
    assert_eq!(value["a"].as_f64(), Some(1.25));
    assert_eq!(value["b"].as_str(), Some("1.25"));
    assert_eq!(value["c"].as_f64(), Some(1000.0));
    assert_eq!(value["d"].as_f64(), Some(-0.0));
    // The JSON constants the strict loader refuses.
    for text in ["NaN", "Infinity", "-Infinity"] {
        assert!(codec::loads(text).is_err(), "{text} loaded");
        assert!(codec::loads(&format!("[{text}]")).is_err(), "[{text}] loaded");
    }
}
