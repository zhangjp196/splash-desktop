//! Concurrency and deadline handling for frontend prompt preparation.

use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::support::errors::ApiError;

const PREPARATION_WAIT: Duration = Duration::from_secs(30);

#[derive(Clone, Copy)]
pub(super) struct Deadline(Option<Instant>);

impl Deadline {
    pub(super) fn from_body(
        body: &Value,
        started_at: Instant,
        default_timeout: Option<Duration>,
    ) -> Result<Self, ApiError> {
        let request_timeout = match body.get("timeout").filter(|value| !value.is_null()) {
            None => None,
            Some(timeout) => {
                let seconds = timeout.as_f64().filter(|seconds| seconds.is_finite() && *seconds > 0.0)
                    .ok_or_else(|| ApiError::new(400, "timeout must be positive", "invalid_request_error"))?;
                Some(Duration::try_from_secs_f64(seconds).map_err(|_| {
                    ApiError::new(400, "timeout must be positive", "invalid_request_error")
                })?)
            }
        };
        let timeout = match (request_timeout, default_timeout) {
            (Some(request), Some(default)) => Some(request.min(default)),
            (Some(request), None) => Some(request),
            (None, default) => default,
        };
        Ok(Self(timeout.and_then(|timeout| started_at.checked_add(timeout))))
    }

    pub(super) fn remaining(self) -> Result<Option<Duration>, ApiError> {
        self.0
            .map(|end| {
                end.checked_duration_since(Instant::now())
                    .filter(|remaining| !remaining.is_zero())
                    .ok_or_else(|| ApiError::new(504, "request timed out", "request_timeout"))
            })
            .transpose()
    }
}

pub(super) struct Preparation {
    capacity: usize,
    latency: std::sync::Arc<crate::metrics::LatencyMetrics>,
    counts: Mutex<Counts>,
    available: Condvar,
}

#[derive(Default)]
struct Counts {
    active: usize,
    waiting: usize,
}

impl Preparation {
    pub(super) fn new(
        capacity: usize,
        latency: std::sync::Arc<crate::metrics::LatencyMetrics>,
    ) -> Self {
        assert!(
            capacity > 0,
            "frontend preparation capacity must be positive"
        );
        Self {
            capacity,
            latency,
            counts: Mutex::new(Counts::default()),
            available: Condvar::new(),
        }
    }

    pub(super) fn status(&self) -> Value {
        let counts = self.counts.lock().expect("preparation lock");
        json!({
            "preparation_capacity": self.capacity,
            "active": counts.active,
            "waiting": counts.waiting,
        })
    }

    pub(super) fn acquire(&self, deadline: Deadline) -> Result<Slot<'_>, ApiError> {
        self.acquire_with_wait(deadline, PREPARATION_WAIT)
    }

    fn acquire_with_wait(&self, deadline: Deadline, max_wait: Duration) -> Result<Slot<'_>, ApiError> {
        let _latency = self.latency.timer("preparation_queue");
        let wait = deadline
            .remaining()?
            .map_or(max_wait, |remaining| remaining.min(max_wait));
        let mut counts = self.counts.lock().expect("preparation lock");
        counts.waiting += 1;
        let (mut counts, _) = self
            .available
            .wait_timeout_while(counts, wait, |counts| counts.active >= self.capacity)
            .expect("preparation lock");
        counts.waiting -= 1;
        deadline.remaining()?;
        if counts.active >= self.capacity {
            return Err(ApiError::new(
                503,
                "frontend preparation capacity is exhausted",
                "frontend_overloaded",
            ));
        }
        counts.active += 1;
        Ok(Slot(self))
    }
}

pub(super) struct Slot<'a>(&'a Preparation);

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        self.0.counts.lock().expect("preparation lock").active -= 1;
        self.0.available.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;

    use super::*;

    #[test]
    fn a_waiting_request_acquires_the_slot_after_release() {
        let latency = std::sync::Arc::new(crate::metrics::LatencyMetrics::default());
        let preparation = Preparation::new(1, latency.clone());
        let slot = preparation.acquire(Deadline(None)).unwrap();
        std::thread::scope(|scope| {
            let (acquired, received) = mpsc::channel();
            let gate = &preparation;
            scope.spawn(move || {
                let result = gate.acquire(Deadline(None));
                acquired.send(result.is_ok()).unwrap();
            });
            let end = Instant::now() + Duration::from_secs(2);
            while preparation.status()["waiting"] != 1 && Instant::now() < end {
                std::thread::yield_now();
            }
            let blocked = preparation.status();
            drop(slot);
            assert_eq!(
                blocked,
                json!({"preparation_capacity": 1, "active": 1, "waiting": 1})
            );
            assert!(received.recv_timeout(Duration::from_secs(2)).unwrap());
        });
        assert_eq!(preparation.status()["active"], 0);
        assert_eq!(preparation.status()["waiting"], 0);
        assert_eq!(latency.snapshot()["preparation_queue"]["count"], 2);
    }

    #[test]
    fn request_expiry_and_queue_overload_have_distinct_errors() {
        let preparation = Preparation::new(1, std::sync::Arc::new(crate::metrics::LatencyMetrics::default()));
        let slot = preparation.acquire(Deadline(None)).unwrap();
        let timeout = Deadline(Some(Instant::now() + Duration::from_millis(20)));
        let error = preparation.acquire(timeout).err().unwrap();
        assert_eq!((error.status, error.code.as_str()), (504, "request_timeout"));

        let error = preparation
            .acquire_with_wait(Deadline(None), Duration::ZERO)
            .err()
            .unwrap();
        assert_eq!(
            (error.status, error.code.as_str()),
            (503, "frontend_overloaded")
        );
        assert_eq!(preparation.status()["waiting"], 0);
        drop(slot);
        assert!(preparation.acquire(Deadline(None)).is_ok());
    }

    #[test]
    fn the_deadline_is_relative_to_the_start_of_the_upload() {
        let started_at = Instant::now() - Duration::from_secs(2);
        let expired = Deadline::from_body(&json!({"timeout": 1}), started_at, None).unwrap();
        assert_eq!(expired.remaining().unwrap_err().status, 504);

        for invalid in [json!(false), json!(0), json!("1"), json!(-1)] {
            let error = Deadline::from_body(&json!({"timeout": invalid}), started_at, None)
                .err()
                .unwrap();
            assert_eq!(error.status, 400);
        }
        let unlimited = Deadline::from_body(&json!({"timeout": null}), started_at, None).unwrap();
        assert_eq!(unlimited.remaining().unwrap(), None);
    }
}
