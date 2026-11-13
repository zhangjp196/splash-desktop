//! The security layer's tests, each expectation checked against the
//! Python implementation's own behavior.

use splash_server::http::security::{
    authenticate, parse_allowed_origin, parse_authority, parse_origin, validate_api_key,
    validate_headers, AllowedOrigin, Headers,
};

fn headers(pairs: &[(&str, &str)]) -> Headers {
    let mut headers = Headers::new();
    for (name, value) in pairs {
        headers.add(name, value);
    }
    headers
}

#[test]
fn authorities_parse_to_a_host_and_an_optional_port() {
    assert_eq!(parse_authority("localhost:8000"), Ok(("localhost".into(), Some(8000))));
    assert_eq!(parse_authority("myhost.local"), Ok(("myhost.local".into(), None)));
    // A trailing dot is dropped, as a DNS name may carry.
    assert_eq!(parse_authority("myhost.local."), Ok(("myhost.local".into(), None)));
    // An IPv6 literal holds its own colons.
    assert_eq!(
        parse_authority("[::1]:8000"),
        Ok(("::1".into(), Some(8000)))
    );
    // Credentials, paths and queries are refused.
    assert!(parse_authority("user@host").is_err());
    assert!(parse_authority("host/path").is_err());
    assert!(parse_authority("").is_err());
}

#[test]
fn origins_parse_to_scheme_host_and_port() {
    assert_eq!(
        parse_origin("http://localhost:3000"),
        Ok(("http".into(), "localhost".into(), Some(3000)))
    );
    // A port the origin does not name is its scheme's default.
    assert_eq!(
        parse_origin("https://example.com"),
        Ok(("https".into(), "example.com".into(), Some(443)))
    );
    // A path or a * is refused, as no browser sends one.
    assert!(parse_origin("http://localhost/path").is_err());
    assert!(parse_origin("*").is_err());
}

#[test]
fn allowed_origins_admit_a_bare_star_alone() {
    assert_eq!(parse_allowed_origin("*"), Ok(AllowedOrigin::Any));
    assert!(parse_allowed_origin("tauri://*").is_err());
    assert!(parse_allowed_origin("http://*.example.com").is_err());
    assert_eq!(
        parse_allowed_origin("tauri://localhost"),
        Ok(AllowedOrigin::Origin(
            "tauri".into(),
            "localhost".into(),
            None
        ))
    );
}

#[test]
fn an_api_key_holds_only_visible_ascii() {
    assert!(validate_api_key("secret-key").is_ok());
    assert!(validate_api_key("has space").is_err());
    assert!(validate_api_key("").is_err());
}

#[test]
fn credentials_must_all_be_the_key() {
    // No credentials at all is refused.
    let error = authenticate(&headers(&[]), "key").expect_err("no credentials");
    assert_eq!(error.status, 401);
    // The bearer scheme's value is the credential.
    assert!(authenticate(&headers(&[("Authorization", "Bearer key")]), "key").is_ok());
    // x-api-key carries the credential directly.
    assert!(authenticate(&headers(&[("x-api-key", "key")]), "key").is_ok());
    // A wrong key is refused.
    let error = authenticate(&headers(&[("x-api-key", "wrong")]), "key").expect_err("wrong key");
    assert_eq!(error.status, 401);
    // A non-bearer authorization carries no credential.
    let error = authenticate(&headers(&[("Authorization", "Basic key")]), "key").expect_err("no credential");
    assert_eq!(error.status, 401);
}

#[test]
fn the_host_must_be_one_the_server_serves() {
    let allowed_hosts = vec!["localhost".to_string(), "127.0.0.1".to_string()];
    let allowed_origins = vec![];
    // The server's own host is served.
    let response = validate_headers(
        &headers(&[("Host", "localhost:8000")]),
        &allowed_hosts,
        &allowed_origins,
    );
    assert!(response.is_ok());
    // Another host is refused, with the fix named.
    let error = validate_headers(
        &headers(&[("Host", "evil.example.com")]),
        &allowed_hosts,
        &allowed_origins,
    )
    .expect_err("refused");
    assert_eq!(error.status, 403);
    assert!(error.message.contains("--allowed-host evil.example.com"), "{error}");
    // Two Host headers are refused.
    let error = validate_headers(
        &headers(&[("Host", "localhost"), ("Host", "localhost")]),
        &allowed_hosts,
        &allowed_origins,
    )
    .expect_err("two hosts");
    assert!(error.message.contains("one Host header"), "{error}");
}

#[test]
fn an_origin_must_be_one_the_operator_admitted() {
    let allowed_hosts = vec!["localhost".to_string()];
    let allowed_origins = vec![
        AllowedOrigin::Origin("http".into(), "localhost".into(), Some(3000)),
    ];
    // The admitted origin is echoed for Access-Control-Allow-Origin.
    let response = validate_headers(
        &headers(&[("Host", "localhost"), ("Origin", "http://localhost:3000")]),
        &allowed_hosts,
        &allowed_origins,
    )
    .expect("admitted");
    assert_eq!(response, Some("http://localhost:3000".to_string()));
    // The server's own pages owe no CORS header.
    let response = validate_headers(
        &headers(&[("Host", "localhost:8000"), ("Origin", "http://localhost:8000")]),
        &allowed_hosts,
        &allowed_origins,
    )
    .expect("own pages");
    assert_eq!(response, None);
    // Another origin is refused, with the fix named.
    let error = validate_headers(
        &headers(&[("Host", "localhost"), ("Origin", "http://evil.example.com")]),
        &allowed_hosts,
        &allowed_origins,
    )
    .expect_err("refused");
    assert_eq!(error.status, 403);
    assert!(
        error
            .message
            .contains("--allowed-origin http://evil.example.com"),
        "{error}"
    );
    // A bare * admits every origin.
    let any = vec![AllowedOrigin::Any];
    let response = validate_headers(
        &headers(&[("Host", "localhost"), ("Origin", "http://anything.example.com")]),
        &allowed_hosts,
        &any,
    )
    .expect("any origin");
    assert_eq!(response, Some("*".to_string()));
}
