//! A minimal HTTP/1.1 server over a thread per connection, the
//! architecture the Python's `http.server`-based frontend uses. It parses
//! a request's line, headers and body, and answers with JSON or an SSE
//! event stream.
//!
//! The security layer validates what a request names: its authority, its
//! browser origin and its credentials.

use std::io::{BufReader, Read, Write};
use std::net::TcpStream;

pub mod admission;
pub mod security;
pub use admission::{HttpAdmission, RequestBodyReservation, CONNECTION_OVERLOADED_RESPONSE, HEAD_WAIT_BYTES};
pub use security::{authenticate, parse_allowed_origin, validate_api_key, validate_headers, AllowedOrigin, Headers, SecurityError};

pub const DEFAULT_MAX_REQUEST_SIZE: usize = 128 * 1024 * 1024;

/// One parsed request.
pub struct Request {
    pub method: String,
    pub path: String,
    pub headers: Headers,
    pub body: Vec<u8>,
    /// Time spent receiving the body after the request headers completed.
    pub upload_duration: Option<std::time::Duration>,
    /// The HTTP request's start point for request-level latency.
    pub headers_received_at: std::time::Instant,
    pub body_reservation: Option<RequestBodyReservation>,
}

impl Request {
    /// The body as JSON, or the parse error.
    pub fn json(&self) -> Result<serde_json::Value, String> {
        serde_json::from_slice(&self.body)
            .map_err(|error| format!("invalid JSON body: {error}"))
    }

    /// Release the shared body-byte budget once prompt preparation no
    /// longer needs the raw request body.
    pub fn release_body_reservation(&mut self) {
        if let Some(reservation) = &mut self.body_reservation {
            reservation.release();
        }
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
        let error_type = match status {
            401 => "authentication_error",
            403 => "permission_error",
            404 => "not_found_error",
            408 | 504 => "timeout_error",
            413 => "request_too_large",
            429 => "rate_limit_error",
            503 => "overloaded_error",
            _ if status >= 500 => "server_error",
            _ => "invalid_request_error",
        };
        Response::json(
            status,
            &serde_json::json!({
                "error": {"message": message, "type": error_type, "code": code}
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
        204 => "No Content",
        200 => "OK",
        408 => "Request Timeout",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        429 => "Too Many Requests",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "OK",
    }
}

/// Read one request from the connection, or `None` at EOF.
pub fn read_request(stream: &mut TcpStream) -> std::io::Result<Option<Request>> {
    Ok(read_timed_request_with_admission(stream, None, DEFAULT_MAX_REQUEST_SIZE, None)?
        .map(|(request, _)| request))
}

pub(crate) fn read_timed_request_with_admission(
    stream: &mut TcpStream,
    body_admission: Option<&std::sync::Arc<HttpAdmission>>,
    max_request_size: usize,
    request_timeout: Option<std::time::Duration>,
) -> std::io::Result<Option<(Request, std::time::Instant)>> {
    // The request deadline includes waiting for the request line and its
    // headers, not only the upload and work that follow them.
    let started_at = std::time::Instant::now();
    let mut reader = BufReader::new(stream.try_clone()?);
    let line = match read_bounded_line(&mut reader, stream, started_at, request_timeout, 65536)? {
        Some(line) => String::from_utf8(line).map_err(|_| std::io::ErrorKind::InvalidData)?,
        None => return Ok(None),
    };
    if !line.ends_with('\n') {
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
        let Some(header) = read_bounded_line(
            &mut reader,
            stream,
            started_at,
            request_timeout,
            remaining as usize,
        )? else {
            return Ok(None);
        };
        let header = String::from_utf8(header).map_err(|_| std::io::ErrorKind::InvalidData)?;
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
    if content_length > max_request_size {
        return Err(std::io::Error::new(
            std::io::ErrorKind::FileTooLarge,
            "request body exceeds its configured limit",
        ));
    }
    if !headers.get("transfer-encoding").is_empty() {
        return Err(std::io::ErrorKind::InvalidData.into());
    }
    let body_reservation = match (body_admission, content_length) {
        (Some(admission), size) if size > 0 => Some(
            RequestBodyReservation::new(admission, size)
                .map_err(|_| std::io::ErrorKind::WouldBlock)?,
        ),
        _ => None,
    };
    let headers_received_at = std::time::Instant::now();
    let upload_started = std::time::Instant::now();
    let mut body = vec![0u8; content_length];
    let mut written = 0;
    while written < body.len() {
        set_request_read_timeout(stream, started_at, request_timeout)?;
        let count = reader.read(&mut body[written..])?;
        if count == 0 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        written += count;
    }
    Ok(Some((
        Request {
            method,
            path,
            headers,
            body,
            upload_duration: (content_length > 0).then(|| upload_started.elapsed()),
            headers_received_at,
            body_reservation,
        },
        started_at,
    )))
}

fn set_request_read_timeout(
    stream: &TcpStream,
    started_at: std::time::Instant,
    request_timeout: Option<std::time::Duration>,
) -> std::io::Result<()> {
    let timeout = match request_timeout {
        Some(timeout) => timeout
            .checked_sub(started_at.elapsed())
            .filter(|remaining| !remaining.is_zero())
            .ok_or(std::io::ErrorKind::TimedOut)?,
        None => std::time::Duration::from_secs(15),
    };
    stream.set_read_timeout(Some(timeout))
}

fn read_bounded_line(
    reader: &mut BufReader<TcpStream>,
    stream: &TcpStream,
    started_at: std::time::Instant,
    request_timeout: Option<std::time::Duration>,
    limit: usize,
) -> std::io::Result<Option<Vec<u8>>> {
    use std::io::BufRead;

    let mut line = Vec::new();
    loop {
        set_request_read_timeout(stream, started_at, request_timeout)?;
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return if line.is_empty() {
                Ok(None)
            } else {
                Err(std::io::ErrorKind::InvalidData.into())
            };
        }
        let end = available.iter().position(|byte| *byte == b'\n');
        let count = end.map_or(available.len(), |index| index + 1);
        if line.len().saturating_add(count) > limit {
            return Err(std::io::ErrorKind::InvalidData.into());
        }
        line.extend_from_slice(&available[..count]);
        reader.consume(count);
        if end.is_some() {
            return Ok(Some(line));
        }
    }
}

/// Write one response.
pub fn write_response(stream: &mut TcpStream, response: &Response) -> std::io::Result<()> {
    write_response_with_cors(stream, response, None)
}

/// Write one response, adding the CORS headers for an admitted browser
/// origin where the security policy returned one.
pub fn write_response_with_cors(
    stream: &mut TcpStream,
    response: &Response,
    origin: Option<&str>,
) -> std::io::Result<()> {
    let cors = origin.map(|origin| {
        format!(
            "Access-Control-Allow-Origin: {origin}\r\nVary: Origin\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Authorization, Content-Type, X-API-Key\r\n"
        )
    }).unwrap_or_default();
    if response.content_type == "text/event-stream" {
        write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n{cors}\r\n")?;
        return stream.flush();
    }
    let head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n{cors}\r\n",
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
