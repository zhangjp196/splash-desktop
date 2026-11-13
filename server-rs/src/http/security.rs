//! Validate HTTP authority, browser origin and optional API credentials,
//! ported from `server/http_security.py` and `server/origins.py`.

use std::collections::HashMap;
use std::fmt;

/// Every origin: in --allowed-origin, and in Access-Control-Allow-Origin.
pub const ANY_ORIGIN: &str = "*";

/// The default port of each scheme, which an origin that names none uses.
pub fn default_port(scheme: &str) -> Option<u16> {
    match scheme {
        "http" => Some(80),
        "https" => Some(443),
        _ => None,
    }
}

/// A refusal the security layer answers with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SecurityError {
    pub status: u16,
    pub message: String,
    pub code: String,
}

impl SecurityError {
    fn forbidden(message: impl Into<String>) -> SecurityError {
        SecurityError {
            status: 403,
            message: message.into(),
            code: "forbidden".into(),
        }
    }

    fn unauthorized(message: impl Into<String>) -> SecurityError {
        SecurityError {
            status: 401,
            message: message.into(),
            code: "authentication_error".into(),
        }
    }
}

impl fmt::Display for SecurityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.message, self.code)
    }
}

impl std::error::Error for SecurityError {}

/// An origin as the server compares it: its scheme, host and port.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AllowedOrigin {
    Any,
    Origin(String, String, Option<u16>),
}

/// Parse an authority: a host and an optional port, nothing else.
pub fn parse_authority(value: &str) -> Result<(String, Option<u16>), String> {
    if value.is_empty() || value.chars().any(|c| c <= ' ' as char || c as u32 >= 127) {
        return Err("invalid authority".into());
    }
    // A host:port, where a bracketed IPv6 literal holds colons of its own.
    let (host, port) = if let Some(rest) = value.strip_prefix('[') {
        let Some(close) = rest.find(']') else {
            return Err("invalid authority".into());
        };
        let host = &rest[..close];
        let after = &rest[close + 1..];
        match after.strip_prefix(':') {
            Some(port) => (host, Some(port)),
            None if after.is_empty() => (host, None),
            _ => return Err("invalid authority".into()),
        }
    } else {
        match value.rsplit_once(':') {
            Some((host, port)) => (host, Some(port)),
            None => (value, None),
        }
    };
    if host.is_empty() || value.contains('@') || value.contains('/') {
        return Err("invalid authority".into());
    }
    let port = match port {
        None => None,
        Some(port) => Some(port.parse::<u16>().map_err(|_| "invalid authority")?),
    };
    Ok((host.to_lowercase().trim_end_matches('.').to_string(), port))
}

/// Parse an origin as a browser serializes it: a scheme and an authority
/// and nothing after them. A port it does not name is its scheme's
/// default. No browser sends a *, which only a pattern holds.
pub fn parse_origin(value: &str) -> Result<(String, String, Option<u16>), String> {
    if value.contains('*') || value.chars().any(|c| c <= ' ' as char || c as u32 >= 127) {
        return Err("invalid origin".into());
    }
    let Some((scheme, rest)) = value.split_once("://") else {
        return Err("invalid origin".into());
    };
    if scheme.is_empty() || rest.contains('/') || rest.contains('?') || rest.contains('#') {
        return Err("invalid origin".into());
    }
    let (host, port) = parse_authority(rest)?;
    let port = port.or_else(|| default_port(scheme));
    Ok((scheme.to_string(), host, port))
}

/// An --allowed-origin value as `validate_headers` compares it.
pub fn parse_allowed_origin(value: &str) -> Result<AllowedOrigin, String> {
    if value == ANY_ORIGIN {
        return Ok(AllowedOrigin::Any);
    }
    if value.contains('*') {
        return Err(format!(
            "{value} is not an origin: only a bare '*' admits every origin; \
             origins are matched exactly, so patterns such as tauri://* or \
             http://*.example.com are not supported"
        ));
    }
    parse_origin(value)
        .map(|(scheme, host, port)| AllowedOrigin::Origin(scheme, host, port))
        .map_err(|_| {
            format!(
                "{value} is not an origin: expected a scheme and a host, as in \
                 tauri://localhost or http://localhost:3000, or '*' for every origin"
            )
        })
}

/// An API key's rule: only visible ASCII characters.
pub fn validate_api_key(value: &str) -> Result<(), String> {
    if value.is_empty() || value.chars().any(|c| c as u32 <= 32 || c as u32 >= 127) {
        return Err("API key must contain only visible ASCII characters".into());
    }
    Ok(())
}

/// The headers of one request, as the security layer reads them: each
/// name's values in order.
pub struct Headers {
    values: HashMap<String, Vec<String>>,
}

impl Headers {
    pub fn new() -> Headers {
        Headers {
            values: HashMap::new(),
        }
    }

    pub fn add(&mut self, name: &str, value: &str) {
        self.values
            .entry(name.to_lowercase())
            .or_default()
            .push(value.to_string());
    }

    pub fn get(&self, name: &str) -> &[String] {
        self.values
            .get(&name.to_lowercase())
            .map(|values| values.as_slice())
            .unwrap_or(&[])
    }
}

impl Default for Headers {
    fn default() -> Headers {
        Headers::new()
    }
}

/// The credentials a request supplied: every one given must be the key,
/// each header only once.
pub fn authenticate(headers: &Headers, key: &str) -> Result<(), SecurityError> {
    let mut supplied: Vec<Option<String>> = headers.get("x-api-key").to_vec().iter().cloned().map(Some).collect();
    for header in headers.get("authorization") {
        let (scheme, value) = match header.split_once(' ') {
            Some((scheme, value)) => (scheme, Some(value.to_string())),
            None => (header.as_str(), None),
        };
        supplied.push(if scheme.eq_ignore_ascii_case("bearer") {
            value
        } else {
            None
        });
    }
    let authorization_count = headers.get("authorization").len();
    let api_key_count = headers.get("x-api-key").len();
    let all_match = !supplied.is_empty()
        && authorization_count <= 1
        && api_key_count <= 1
        && supplied.iter().all(|value| {
            value
                .as_ref()
                .map(|value| constant_time_eq(value.as_bytes(), key.as_bytes()))
                .unwrap_or(false)
        });
    if all_match {
        Ok(())
    } else {
        Err(SecurityError::unauthorized("invalid or missing API key"))
    }
}

/// A constant-time comparison, as credential checks read.
fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut difference = 0u8;
    for (a, b) in left.iter().zip(right.iter()) {
        difference |= a ^ b;
    }
    difference == 0
}

/// What a request's Host and Origin headers owe the response: the origin
/// of a page elsewhere that --allowed-origin admits, or `None`.
pub fn validate_headers(
    headers: &Headers,
    allowed_hosts: &[String],
    allowed_origins: &[AllowedOrigin],
) -> Result<Option<String>, SecurityError> {
    let hosts = headers.get("host");
    let origins = headers.get("origin");
    if hosts.len() != 1 || origins.len() > 1 {
        return Err(SecurityError::forbidden(
            "expected one Host header and at most one Origin header",
        ));
    }
    let (host, port) = parse_authority(&hosts[0])
        .map_err(|_| SecurityError::forbidden("invalid Host header"))?;
    if !allowed_hosts.iter().any(|allowed| *allowed == host) {
        // Only the bind address, loopback and --allowed-host names are
        // served, which keeps DNS-rebound pages out. Name the fix for the
        // operator.
        return Err(SecurityError::forbidden(format!(
            "Host {host} is not allowed; restart the server with --allowed-host {host} to accept it"
        )));
    }
    if origins.is_empty() {
        return Ok(None);
    }
    if allowed_origins.iter().any(|allowed| *allowed == AllowedOrigin::Any) {
        return Ok(Some(ANY_ORIGIN.to_string()));
    }
    let origin = parse_origin(&origins[0])
        .map_err(|_| SecurityError::forbidden("invalid Origin header"))?;
    let default_port = default_port(&origin.0);
    if default_port.is_some_and(|default| {
        origin.1 == host && origin.2 == Some(port.unwrap_or(default))
    }) {
        // The server's own pages.
        return Ok(None);
    }
    let admitted = allowed_origins.iter().any(|allowed| match allowed {
        AllowedOrigin::Any => true,
        AllowedOrigin::Origin(scheme, host, port) => {
            *scheme == origin.0 && *host == origin.1 && *port == origin.2
        }
    });
    if !admitted {
        return Err(SecurityError::forbidden(format!(
            "Origin {} is not allowed; restart the server with --allowed-origin {} to accept it",
            origins[0], origins[0]
        )));
    }
    Ok(Some(origins[0].clone()))
}
