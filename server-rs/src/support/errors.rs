//! The statuses and shapes of the server's refusals, ported from
//! `server/errors.py`: the error each API answers with, and the payloads
//! its dialect carries them in.

use serde_json::Value;

/// The statuses of a refusal the client may retry shortly, which a response
/// invites with Retry-After: the server is overloaded, restarting its
/// engine or shutting down. Each API answers these with 503, /v1/systemone
/// with 529.
pub const RETRY_STATUSES: [u16; 2] = [503, 529];

/// A refusal: its HTTP status, message, code and the seconds a retry
/// status invites the client to wait (Retry-After). The optional fields
/// carry what two refusals add: a context-length error's counts, and a
/// validation error's per-field details.
#[derive(Debug, Clone, PartialEq)]
pub struct ApiError {
    pub status: u16,
    pub message: String,
    pub code: String,
    pub retry_after: u64,
    pub input_tokens: Option<u64>,
    pub maximum_input_tokens: Option<u64>,
    pub image_tokens_only: bool,
    pub validation_details: Option<Vec<FieldError>>,
}

impl ApiError {
    pub fn new(status: u16, message: impl Into<String>, code: impl Into<String>) -> ApiError {
        ApiError {
            status,
            message: message.into(),
            code: code.into(),
            retry_after: 1,
            input_tokens: None,
            maximum_input_tokens: None,
            image_tokens_only: false,
            validation_details: None,
        }
    }

    pub fn with_retry_after(mut self, seconds: u64) -> ApiError {
        self.retry_after = seconds;
        self
    }

    /// The error as an OpenAI-shaped refusal carries it.
    pub fn openai_payload(&self) -> Value {
        let error_type = if self.status >= 500 {
            "server_error"
        } else {
            "invalid_request_error"
        };
        serde_json::json!({
            "error": {
                "message": self.message,
                "type": error_type,
                "code": self.code,
            }
        })
    }

    /// The error as an Anthropic-shaped refusal carries it.
    pub fn anthropic_payload(&self) -> Value {
        let mut message = self.message.clone();
        if self.code == "context_length_exceeded" {
            if let (Some(input), Some(maximum)) = (self.input_tokens, self.maximum_input_tokens) {
                message = format!(
                    "prompt is too long: {input} tokens > {maximum} maximum input tokens"
                );
                if self.image_tokens_only {
                    message += " (image tokens alone; text not yet counted)";
                }
            }
        }
        let error_type = match self.status {
            401 => "authentication_error",
            403 => "permission_error",
            404 => "not_found_error",
            408 => "timeout_error",
            413 => "request_too_large",
            429 => "rate_limit_error",
            503 => "overloaded_error",
            504 => "timeout_error",
            other => {
                if other >= 500 {
                    "api_error"
                } else {
                    "invalid_request_error"
                }
            }
        };
        serde_json::json!({
            "type": "error",
            "error": {"type": error_type, "message": message}
        })
    }

    /// The status of the response that answers `error`, the code the
    /// console logs for it, and its payload: OpenAI's dialect, which also
    /// answers on the paths of no other API.
    pub fn answer(&self) -> (u16, &str, Value) {
        (self.status, &self.code, self.openai_payload())
    }

    /// /v1/systemone answers as the FastAPI service it stands in for:
    /// invalid fields with 422 and what is wrong with each, an overload
    /// with TypeSafe's 529, and other errors as OpenAI's APIs do.
    pub fn systemone_answer(&self) -> (u16, String, Value) {
        if let Some(details) = &self.validation_details {
            let payload = serde_json::json!({
                "detail": details.iter().map(FieldError::to_json).collect::<Vec<_>>()
            });
            return (422, "unprocessable_entity".to_string(), payload);
        }
        let (status, code, payload) = self.answer();
        let status = if status == 503 { 529 } else { status };
        (status, code.to_string(), payload)
    }
}

/// A prompt that does not fit the context window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextLength {
    pub input_tokens: u64,
    pub maximum_input_tokens: u64,
    pub image_tokens_only: bool,
}

impl ContextLength {
    pub fn into_api_error(self) -> ApiError {
        let mut error = ApiError::new(
            400,
            "prompt exceeds the context window",
            "context_length_exceeded",
        );
        error.input_tokens = Some(self.input_tokens);
        error.maximum_input_tokens = Some(self.maximum_input_tokens);
        error.image_tokens_only = self.image_tokens_only;
        error
    }
}

/// What is wrong with one body field, as FastAPI describes one.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldError {
    pub loc: Vec<String>,
    pub msg: String,
    pub error_type: String,
}

/// What is wrong with the body field at path `loc`, the body itself when
/// `loc` is empty.
pub fn field_error(loc: &[&str], msg: impl Into<String>, error_type: &str) -> FieldError {
    FieldError {
        loc: std::iter::once("body".to_string())
            .chain(loc.iter().map(|part| (*part).to_string()))
            .collect(),
        msg: msg.into(),
        error_type: error_type.to_string(),
    }
}

impl FieldError {
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "loc": self.loc,
            "msg": self.msg,
            "type": self.error_type,
        })
    }
}

/// Request fields that fail validation, each described as FastAPI
/// describes one. /v1/systemone answers with every one, as the FastAPI
/// service it stands in for does; the other APIs with the first one's
/// message, as with any invalid request.
#[derive(Debug, Clone, PartialEq)]
pub struct RequestValidationError {
    pub details: Vec<FieldError>,
}

impl RequestValidationError {
    pub fn new(details: Vec<FieldError>) -> RequestValidationError {
        RequestValidationError { details }
    }

    pub fn into_api_error(self) -> ApiError {
        let message = self
            .details
            .first()
            .map(|first| first.msg.clone())
            .unwrap_or_default();
        let mut error = ApiError::new(400, message, "invalid_request_error");
        error.validation_details = Some(self.details);
        error
    }
}
