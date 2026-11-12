//! The error dialects' payloads, ported from the shapes
//! `dev/tests/server/`'s dialect tests assert against.

use splash_server::support::errors::{field_error, ApiError, ContextLength, RequestValidationError};

#[test]
fn openai_payloads() {
    let error = ApiError::new(400, "bad field", "invalid_request_error");
    assert_eq!(
        error.openai_payload(),
        serde_json::json!({
            "error": {"message": "bad field", "type": "invalid_request_error", "code": "invalid_request_error"}
        })
    );
    let server = ApiError::new(503, "overloaded", "engine_busy");
    assert_eq!(
        server.openai_payload()["error"]["type"],
        serde_json::json!("server_error")
    );
}

#[test]
fn anthropic_payloads() {
    let error = ApiError::new(429, "slow down", "rate_limit_error");
    assert_eq!(
        error.anthropic_payload(),
        serde_json::json!({
            "type": "error",
            "error": {"type": "rate_limit_error", "message": "slow down"}
        })
    );
    // A context-length error's message carries the counts.
    let context = ContextLength {
        input_tokens: 200_000,
        maximum_input_tokens: 131_072,
        image_tokens_only: false,
    }
    .into_api_error();
    assert_eq!(
        context.anthropic_payload()["error"]["message"],
        serde_json::json!("prompt is too long: 200000 tokens > 131072 maximum input tokens")
    );
    // An unnamed status reads as an api_error from 500.
    let unnamed = ApiError::new(529, "busy", "engine_busy");
    assert_eq!(
        unnamed.anthropic_payload()["error"]["type"],
        serde_json::json!("api_error")
    );
}

#[test]
fn systemone_answers_validation_with_422_and_overload_with_529() {
    let validation = RequestValidationError::new(vec![
        field_error(&["messages"], "must be a list", "type_error"),
        field_error(&[], "body is not an object", "value_error"),
    ]);
    let error = validation.into_api_error();
    let (status, code, payload) = error.systemone_answer();
    assert_eq!((status, code.as_str()), (422, "unprocessable_entity"));
    assert_eq!(payload["detail"].as_array().unwrap().len(), 2);
    assert_eq!(payload["detail"][0]["loc"][0], serde_json::json!("body"));
    assert_eq!(payload["detail"][0]["loc"][1], serde_json::json!("messages"));
    // An overload answers 529, as TypeSafe's does.
    let overload = ApiError::new(503, "overloaded", "engine_busy");
    let (status, _, _) = overload.systemone_answer();
    assert_eq!(status, 529);
}
