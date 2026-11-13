//! The nonwaiting capacity gate, in request counts or input bytes, and
//! the input reservation a request's body holds until its preparation
//! releases it. Ported from `server/connections.py`'s `HttpAdmission` and
//! `RequestBodyReservation`.

use std::sync::{Arc, Mutex};

/// The response a refused connection reads: the capacity is exhausted,
/// with the seconds a retry may wait.
pub const CONNECTION_OVERLOADED_RESPONSE: &str =
    "{\"error\":{\"type\":\"server_error\",\"code\":\"frontend_overloaded\",\
      \"message\":\"HTTP connection capacity is exhausted\"}}";

/// The most of a request's head a connection waits for without a thread:
/// the kernel holds it unread, in a receive buffer of 128 KiB on macOS
/// unless tuned, so its client never waits for room. A thread reads a
/// longer head, under the Python's own limits.
pub const HEAD_WAIT_BYTES: usize = 64 * 1024;

/// The nonwaiting capacity gate: `acquire` refuses at once when the
/// active count or bytes would pass the capacity, and `release` gives
/// back.
#[derive(Debug)]
pub struct HttpAdmission {
    state: Mutex<AdmissionState>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmissionStats {
    pub active: usize,
    pub capacity: usize,
}

#[derive(Debug)]
struct AdmissionState {
    active: usize,
    capacity: usize,
}

impl HttpAdmission {
    /// The gate of `capacity` requests or bytes; a capacity of zero or
    /// fewer is a construction error.
    pub fn new(capacity: usize) -> Result<HttpAdmission, String> {
        if capacity == 0 {
            return Err("HTTP admission capacity must be a positive integer".into());
        }
        Ok(HttpAdmission {
            state: Mutex::new(AdmissionState {
                active: 0,
                capacity,
            }),
        })
    }

    /// Take `amount` of the capacity; `false` when it would pass it.
    pub fn acquire(&self, amount: usize) -> bool {
        let mut state = self.state.lock().expect("admission lock");
        if state.active + amount > state.capacity {
            return false;
        }
        state.active += amount;
        true
    }

    /// Give `amount` back; a release without an acquisition is a bug.
    pub fn release(&self, amount: usize) -> Result<(), String> {
        let mut state = self.state.lock().expect("admission lock");
        if amount > state.active {
            return Err("HTTP admission slot released without acquisition".into());
        }
        state.active -= amount;
        Ok(())
    }

    /// The gate's counts, as the status document reports them.
    pub fn stats(&self) -> AdmissionStats {
        let state = self.state.lock().expect("admission lock");
        AdmissionStats {
            active: state.active,
            capacity: state.capacity,
        }
    }
}

/// The input reservation one request's body holds: the bytes it accounts
/// until its preparation releases them, growing as the body reads.
#[derive(Debug)]
pub struct RequestBodyReservation {
    admission: Arc<HttpAdmission>,
    size: usize,
}

impl RequestBodyReservation {
    /// Account `size` bytes, or the refusal the exhausted capacity gives.
    pub fn new(admission: &Arc<HttpAdmission>, size: usize) -> Result<RequestBodyReservation, String> {
        if !admission.acquire(size) {
            return Err(
                "request body capacity is exhausted; retry shortly".into()
            );
        }
        Ok(RequestBodyReservation {
            admission: admission.clone(),
            size,
        })
    }

    /// The bytes the reservation holds.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Account more bytes, as a chunked body grows.
    pub fn grow(&mut self, size: usize) -> Result<(), String> {
        if !self.admission.acquire(size) {
            return Err("request body capacity is exhausted; retry shortly".into());
        }
        self.size += size;
        Ok(())
    }

    /// Give the bytes back.
    pub fn release(&mut self) {
        let _ = self.admission.release(self.size);
        self.size = 0;
    }
}

impl Drop for RequestBodyReservation {
    fn drop(&mut self) {
        self.release();
    }
}
