//! The admission gate's tests, ported from the Python tests that drive
//! `HttpAdmission` and `RequestBodyReservation`.

use std::sync::Arc;

use splash_server::http::{HttpAdmission, RequestBodyReservation};

#[test]
fn the_gate_refuses_when_the_capacity_would_pass() {
    let admission = HttpAdmission::new(2).expect("builds");
    assert!(admission.acquire(1));
    assert!(admission.acquire(1));
    // The third request is refused at once, without waiting.
    assert!(!admission.acquire(1));
    assert_eq!(
        admission.stats(),
        splash_server::http::admission::AdmissionStats {
            active: 2,
            capacity: 2
        }
    );
    // A release makes room again.
    admission.release(1).expect("releases");
    assert!(admission.acquire(1));
}

#[test]
fn a_release_without_an_acquisition_is_a_bug() {
    let admission = HttpAdmission::new(1).expect("builds");
    assert!(admission.release(1).is_err());
}

#[test]
fn a_capacity_of_zero_is_a_construction_error() {
    assert!(HttpAdmission::new(0).is_err());
}

#[test]
fn a_body_reservation_accounts_its_bytes_until_released() {
    let admission = Arc::new(HttpAdmission::new(1024).expect("builds"));
    let mut reservation = RequestBodyReservation::new(&admission, 512).expect("reserves");
    assert_eq!(reservation.size(), 512);
    // The body grows as it reads.
    reservation.grow(256).expect("grows");
    assert_eq!(reservation.size(), 768);
    // A growth past the capacity is refused.
    assert!(reservation.grow(512).is_err());
    assert_eq!(reservation.size(), 768);
    // The release gives the bytes back.
    reservation.release();
    assert_eq!(reservation.size(), 0);
    assert_eq!(admission.stats().active, 0);
}

#[test]
fn an_oversized_body_is_refused_at_once() {
    let admission = Arc::new(HttpAdmission::new(512).expect("builds"));
    let error = RequestBodyReservation::new(&admission, 1024).expect_err("refused");
    assert!(error.contains("capacity is exhausted"), "{error}");
    // The refusal held nothing.
    assert_eq!(admission.stats().active, 0);
}

#[test]
fn the_reservation_releases_on_drop() {
    let admission = Arc::new(HttpAdmission::new(512).expect("builds"));
    {
        let _reservation = RequestBodyReservation::new(&admission, 512).expect("reserves");
        assert_eq!(admission.stats().active, 512);
    }
    // The drop released the bytes.
    assert_eq!(admission.stats().active, 0);
}
