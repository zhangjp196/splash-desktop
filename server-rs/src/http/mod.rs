//! A minimal HTTP/1.1 server over a thread per connection, the
//! architecture the Python's `http.server`-based frontend uses. It parses
//! a request's line, headers and body, and answers with JSON or an SSE
//! event stream.
//!
//! The security layer validates what a request names: its authority, its
//! browser origin and its credentials.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;

pub mod admission;
pub mod security;
pub use admission::{HttpAdmission, RequestBodyReservation, CONNECTION_OVERLOADED_RESPONSE, HEAD_WAIT_BYTES};
pub use security::{authenticate, parse_allowed_origin, validate_api_key, validate_headers, AllowedOrigin, Headers, SecurityError};

/// One parsed request.
pub struct Request {
    pub method: String,
    pub path: String,
    pub headers: Headers,
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
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "OK",
    }
}

/// Read one request from the connection, or `None` at EOF.
pub fn read_request(stream: &mut TcpStream) -> std::io::Result<Option<Request>> {
    Ok(read_timed_request(stream)?.map(|(request, _)| request))
}

pub(crate) fn read_timed_request(
    stream: &mut TcpStream,
) -> std::io::Result<Option<(Request, std::time::Instant)>> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    if (&mut reader).take(65537).read_line(&mut line)? == 0 {
        return Ok(None);
    }
    if line.len() > 65536 || !line.ends_with('\n') {
        return Err(std::io::ErrorKind::InvalidData.into());
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();
    if method.is_empty() || path.is_empty() {
        return Ok(None);
    }
    let mut headers = Headers::new();
    let mut remaining = 65536;
    loop {
        let mut header = String::new();
        if (&mut reader).take(remaining + 1).read_line(&mut header)? == 0 {
            return Ok(None);
        }
        if header.len() as u64 > remaining || !header.ends_with('\n') {
            return Err(std::io::ErrorKind::InvalidData.into());
        }
        remaining -= header.len() as u64;
        let header = header.trim_end();
        if header.is_empty() {
            break;
        }
        let (name, value) = header.split_once(':').ok_or(std::io::ErrorKind::InvalidData)?;
        headers.add(name, value.trim());
    }
    let content_length = match headers.get("content-length") {
        [] => 0,
        [length] => length.parse::<usize>().map_err(|_| std::io::ErrorKind::InvalidData)?,
        _ => return Err(std::io::ErrorKind::InvalidData.into()),
    };
    if content_length > 128 * 1024 * 1024 || !headers.get("transfer-encoding").is_empty() {
        return Err(std::io::ErrorKind::InvalidData.into());
    }
    let started_at = std::time::Instant::now();
    let mut body = vec![0u8; content_length];
    reader.read_exact(&mut body)?;
    Ok(Some((Request { method, path, headers, body }, started_at)))
}

/// Write one response.
pub fn write_response(stream: &mut TcpStream, response: &Response) -> std::io::Result<()> {
    if response.content_type == "text/event-stream" {
        stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")?;
        return stream.flush();
    }
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
