//! A minimal HTTP/1.1 server over a thread per connection, the
//! architecture the Python's `http.server`-based frontend uses. It parses
//! a request's line, headers and body, and answers with JSON or an SSE
//! event stream.
//!
//! The security layer validates what a request names: its authority, its
//! browser origin and its credentials.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;

pub mod security;
pub use security::{authenticate, parse_allowed_origin, validate_api_key, validate_headers, AllowedOrigin, Headers, SecurityError};

/// One parsed request.
pub struct Request {
    pub method: String,
    pub path: String,
    pub body: Vec<u8>,
}

impl Request {
    /// The body as JSON, or the parse error.
    pub fn json(&self) -> Result<serde_json::Value, String> {
        serde_json::from_slice(&self.body)
            .map_err(|error| format!("invalid JSON body: {error}"))
    }
}

/// One response to write.
pub struct Response {
    pub status: u16,
    pub content_type: &'static str,
    pub body: Vec<u8>,
}

impl Response {
    pub fn json(status: u16, value: &serde_json::Value) -> Response {
        Response {
            status,
            content_type: "application/json",
            body: serde_json::to_vec(value).expect("json serializes"),
        }
    }

    pub fn error(status: u16, message: &str, code: &str) -> Response {
        Response::json(
            status,
            &serde_json::json!({
                "error": {"message": message, "type": "server_error", "code": code}
            }),
        )
    }

    /// An SSE stream's head: the caller writes the events and the
    /// terminator onto the stream afterwards.
    pub fn sse_head() -> Response {
        Response {
            status: 200,
            content_type: "text/event-stream",
            body: Vec::new(),
        }
    }
}

fn status_text(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "OK",
    }
}

/// Read one request from the connection, or `None` at EOF.
pub fn read_request(stream: &mut TcpStream) -> std::io::Result<Option<Request>> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();
    if method.is_empty() || path.is_empty() {
        return Ok(None);
    }
    let mut content_length = 0usize;
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header)? == 0 {
            return Ok(None);
        }
        let header = header.trim_end();
        if header.is_empty() {
            break;
        }
        if let Some((name, value)) = header.split_once(':') {
            if name.eq_ignore_ascii_case("content-length") {
                content_length = value.trim().parse().unwrap_or(0);
            }
        }
    }
    let mut body = vec![0u8; content_length];
    reader.read_exact(&mut body)?;
    Ok(Some(Request { method, path, body }))
}

/// Write one response.
pub fn write_response(stream: &mut TcpStream, response: &Response) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        response.status,
        status_text(response.status),
        response.content_type,
        response.body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(&response.body)?;
    stream.flush()
}

/// Write one SSE event: `data: ` + the payload, blank-line separated.
pub fn write_sse_event(stream: &mut TcpStream, payload: &str) -> std::io::Result<()> {
    write!(stream, "data: {payload}\n\n")?;
    stream.flush()
}
