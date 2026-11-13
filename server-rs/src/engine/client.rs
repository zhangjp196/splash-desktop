//! The one-reader, direct-admission client for the native protocol,
//! ported from `server/runtime.py`'s `MultiplexedRuntime`: the engine runs
//! as a subprocess, the client writes client frames to its stdin and one
//! reader thread parses its stdout, dispatching each event to the call it
//! names.
//!
//! This core carries one engine generation: spawn, submit, cancel, status
//! and the dispatch rules. The restart attempts, the liveness probes and
//! the CPU mask workers are the next slice (`PORTING.md`).

use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use crate::engine::call::{
    wire_deadline,
    Call, CompletionCallback, EngineError, EventCallback, GenerationRequest, RequestFailed,
};
use crate::protocol as wire;

/// How long the engine's loop may leave a status request unanswered before
/// its engine counts as stuck and fails: far above any legitimate tick.
pub const STATUS_ANSWER_LIMIT: Duration = Duration::from_secs(30);

const READ_CHUNK_BYTES: usize = 64 * 1024;

/// A status document's waiter.
struct StatusWaiter {
    result: Mutex<Option<wire::StatusJsonEvent>>,
    error: Mutex<Option<EngineError>>,
    finished: Condvar,
}

impl StatusWaiter {
    fn new() -> StatusWaiter {
        StatusWaiter {
            result: Mutex::new(None),
            error: Mutex::new(None),
            finished: Condvar::new(),
        }
    }

    fn answer(&self, event: wire::StatusJsonEvent) {
        *self.result.lock().expect("waiter lock") = Some(event);
        self.finished.notify_all();
    }

    fn fail(&self, error: EngineError) {
        *self.error.lock().expect("waiter lock") = Some(error);
        self.finished.notify_all();
    }

    fn consume(self) -> Result<wire::StatusJsonEvent, EngineError> {
        if let Some(error) = self.error.into_inner().expect("waiter lock") {
            return Err(error.restate());
        }
        self.result
            .into_inner()
            .expect("waiter lock")
            .ok_or_else(|| EngineError::RuntimeClosed("status waiter ended empty".into()))
    }
}

/// The client's state, under its lock.
struct State {
    closed: bool,
    generation: u64,
    ready: Option<wire::ReadyEvent>,
    /// Set when the engine failed for good; no further engine starts.
    fatal_error: Option<EngineError>,
    pending: HashMap<u64, Arc<Call>>,
    pending_limit: usize,
    status_waiters: HashMap<u64, Arc<StatusWaiter>>,
    last_status_id: u64,
    next_request_id: u64,
    next_status_id: u64,
}

/// The client's shared plumbing.
struct Shared {
    state: Mutex<State>,
    ready_changed: Condvar,
    write: Mutex<Box<dyn Write + Send>>,
}

impl Shared {
    fn write_frame(&self, encoded: &[u8]) -> Result<(), EngineError> {
        let mut write = self.write.lock().expect("write lock");
        write
            .write_all(encoded)
            .and_then(|_| write.flush())
            .map_err(|error| EngineError::RuntimeClosed(format!("native write failed: {error}")))
    }
}

/// One engine generation's client. Cloneable; every clone shares the
/// engine's streams and state.
#[derive(Clone)]
pub struct EngineClient {
    shared: Arc<Shared>,
}

impl EngineClient {
    /// Run `command` as the engine, reading its stdout and writing its
    /// stdin. The caller owns the child process; `close` stops writing and
    /// the caller reaps it.
    pub fn spawn(
        mut child: std::process::Child,
        pending_limit: usize,
    ) -> std::io::Result<EngineClient> {
        let stdin = child.stdin.take().expect("child stdin piped");
        let stdout = child.stdout.take().expect("child stdout piped");
        Ok(Self::over_streams(Box::new(stdout), Box::new(stdin), pending_limit))
    }

    /// Run the client over any byte streams, as the tests do: `read` is
    /// the engine's output the reader parses, `write` carries client
    /// frames to the engine.
    pub fn over_streams(
        read: Box<dyn Read + Send>,
        write: Box<dyn Write + Send>,
        pending_limit: usize,
    ) -> EngineClient {
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                closed: false,
                generation: 1,
                ready: None,
                fatal_error: None,
                pending: HashMap::new(),
                pending_limit,
                status_waiters: HashMap::new(),
                last_status_id: 0,
                next_request_id: 1,
                next_status_id: 1,
            }),
            ready_changed: Condvar::new(),
            write: Mutex::new(write),
        });
        let client = EngineClient { shared: shared.clone() };
        let reader_shared = Arc::downgrade(&shared);
        std::thread::Builder::new()
            .name("splash-engine-reader".into())
            .spawn(move || reader_loop(read, reader_shared, client))
            .expect("reader thread");
        EngineClient { shared }
    }

    /// Wait until the engine is ready, or the timeout passes; `false` then.
    pub fn wait_ready(&self, timeout: Option<Duration>) -> bool {
        let mut state = self.shared.state.lock().expect("state lock");
        let deadline = timeout.map(|timeout| std::time::Instant::now() + timeout);
        while state.ready.is_none() && state.fatal_error.is_none() && !state.closed {
            match deadline {
                None => state = self.shared.ready_changed.wait(state).expect("state lock"),
                Some(deadline) => {
                    let now = std::time::Instant::now();
                    if now >= deadline {
                        break;
                    }
                    let (guarded, _) = self
                        .shared
                        .ready_changed
                        .wait_timeout(state, deadline - now)
                        .expect("state lock");
                    state = guarded;
                }
            }
        }
        state.ready.is_some()
    }

    /// The engine's readiness, once its ReadyEvent arrived.
    pub fn readiness(&self) -> Option<wire::ReadyEvent> {
        self.shared.state.lock().expect("state lock").ready
    }

    pub fn ready(&self) -> bool {
        self.readiness().is_some()
    }

    pub fn pending_count(&self) -> usize {
        self.shared.state.lock().expect("state lock").pending.len()
    }

    /// The failure that ended the engine for good, if one was set.
    pub fn fatal_error(&self) -> Option<EngineError> {
        self.shared
            .state
            .lock()
            .expect("state lock")
            .fatal_error
            .clone()
    }

    /// Admit and immediately write one request without client scheduling.
    /// A runtime that is not ready refuses at once; only `wait_ready`
    /// starts an engine.
    pub fn submit(
        &self,
        request: &GenerationRequest,
        on_event: Option<EventCallback>,
        on_complete: Option<CompletionCallback>,
    ) -> Result<Arc<Call>, EngineError> {
        request.validate()?;
        let mut state = self.shared.state.lock().expect("state lock");
        if state.closed {
            return Err(EngineError::RuntimeClosed("runtime is closed".into()));
        }
        if state.fatal_error.is_some() {
            return Err(state
                .fatal_error
                .as_ref()
                .expect("checked above")
                .restate());
        }
        if state.pending.len() >= state.pending_limit {
            return Err(EngineError::PendingLimitExceeded(format!(
                "pending request limit {} is full",
                state.pending_limit
            )));
        }
        if state.ready.is_none() {
            return Err(EngineError::EngineUnhealthy(
                "native process is not ready".into(),
            ));
        }
        let request_id = state.next_request_id;
        state.next_request_id += 1;
        let generation = state.generation;
        let (absolute, remaining) = wire_deadline(request.deadline);
        let mut frame = request.frame.clone();
        frame.request_id = request_id;
        frame.absolute_deadline_unix_micros = absolute;
        frame.remaining_deadline_micros = remaining;
        let encoded = wire::serialize_message(&wire::ClientMessage::Request(frame))
            .map_err(|error| {
                EngineError::RequestFailed(Box::new(RequestFailed {
                    request_id,
                    code: error.0.code.name().as_bytes().to_vec(),
                    message_bytes: error.0.message.into_bytes(),
                    retryable: false,
                    retry_after_seconds: 0,
                }))
            })?;
        let call = Call::new(request_id, generation, request, on_event, on_complete);
        state.pending.insert(request_id, call.clone());
        drop(state);
        // Admission is held: a write failure fails the call, which releases
        // it from the pending map.
        if let Err(error) = self.shared.write_frame(&encoded) {
            self.fail_call(&call, error.clone());
            return Err(error);
        }
        Ok(call)
    }

    /// Stop one request the engine is serving.
    pub fn cancel(&self, call: &Call) -> bool {
        if !call.mark_cancel_requested() {
            return false;
        }
        let frame = wire::CancelFrame {
            request_id: call.request_id,
        };
        if let Err(error) =
            self.shared.write_frame(&wire::serialize_message(&wire::ClientMessage::Cancel(frame)).expect("cancel frame"))
        {
            self.fail_call(call, error);
            return false;
        }
        true
    }

    /// The engine's status JSON, answered within `timeout`.
    pub fn status(&self, timeout: Duration) -> Result<wire::StatusJsonEvent, EngineError> {
        self.status_request(timeout, false)
    }

    /// Asks the engine to give its weights back now. It does so only while
    /// it holds no request and no restore is under way; the status JSON
    /// that answers shows which.
    pub fn release_weights(&self, timeout: Duration) -> Result<wire::StatusJsonEvent, EngineError> {
        self.status_request(timeout, true)
    }

    /// Changes a running engine's prefill duty, as `--prefill-duty` does at
    /// start, and answers with the status that reads it back. The duty frame
    /// itself is not answered; the status request written after it is, and
    /// the engine applies the frames it reads in order, so that status's
    /// `throttle` section carries the duty the engine took (it clamps what a
    /// configuration may not ask for).
    pub fn set_prefill_duty(
        &self,
        duty_percent: f32,
        timeout: Duration,
    ) -> Result<wire::StatusJsonEvent, EngineError> {
        {
            let state = self.shared.state.lock().expect("state lock");
            if state.closed {
                return Err(EngineError::RuntimeClosed("runtime is closed".into()));
            }
            if state.ready.is_none() {
                return Err(EngineError::EngineUnhealthy(
                    "native process is not ready".into(),
                ));
            }
        }
        let frame = wire::PrefillDutyRequestFrame { duty_percent };
        self.shared
            .write_frame(&wire::serialize_message(&wire::ClientMessage::PrefillDutyRequest(frame)).expect("prefill duty frame"))?;
        self.status(timeout)
    }

    fn status_request(
        &self,
        timeout: Duration,
        release: bool,
    ) -> Result<wire::StatusJsonEvent, EngineError> {
        let mut state = self.shared.state.lock().expect("state lock");
        if state.closed {
            return Err(EngineError::RuntimeClosed("runtime is closed".into()));
        }
        if state.ready.is_none() {
            return Err(EngineError::EngineUnhealthy(
                "native process is not ready".into(),
            ));
        }
        let correlation_id = state.next_status_id;
        state.next_status_id += 1;
        state.last_status_id = correlation_id;
        let waiter = Arc::new(StatusWaiter::new());
        state
            .status_waiters
            .insert(correlation_id, waiter.clone());
        drop(state);
        let message = if release {
            wire::ClientMessage::ReleaseRequest(wire::ReleaseRequestFrame { correlation_id })
        } else {
            wire::ClientMessage::StatusRequest(wire::StatusRequestFrame { correlation_id })
        };
        if let Err(error) = self
            .shared
            .write_frame(&wire::serialize_message(&message).expect("status frame"))
        {
            self.shared
                .state
                .lock()
                .expect("state lock")
                .status_waiters
                .remove(&correlation_id);
            return Err(error);
        }
        let deadline = std::time::Instant::now() + timeout;
        let mut answered = waiter.result.lock().expect("waiter lock");
        while answered.is_none() {
            let now = std::time::Instant::now();
            if now >= deadline {
                break;
            }
            let (guarded, _) = waiter
                .finished
                .wait_timeout(answered, deadline - now)
                .expect("waiter lock");
            answered = guarded;
        }
        if answered.is_none() {
            let expired = self
                .shared
                .state
                .lock()
                .expect("state lock")
                .status_waiters
                .remove(&correlation_id)
                .is_some();
            if expired {
                return Err(EngineError::RuntimeClosed(
                    "native status response timed out".into(),
                ));
            }
        }
        drop(answered);
        match std::sync::Arc::into_inner(waiter) {
            Some(waiter) => waiter.consume(),
            None => Err(EngineError::RuntimeClosed(
                "status waiter still shared".into(),
            )),
        }
    }

    /// Stop serving: fail every pending call and refuse the rest. The
    /// caller owns the child process and reaps it.
    pub fn close(&self) {
        let pending = {
            let mut state = self.shared.state.lock().expect("state lock");
            if state.closed {
                return;
            }
            state.closed = true;
            std::mem::take(&mut state.pending)
        };
        for (_, call) in pending {
            self.fail_call(&call, EngineError::RuntimeClosed("runtime is closed".into()));
        }
        self.shared.ready_changed.notify_all();
    }

    /// Fail one call and release its admission.
    fn fail_call(&self, call: &Call, error: EngineError) {
        if call.set_terminal(None, Some(error.clone())) {
            self.shared
                .state
                .lock()
                .expect("state lock")
                .pending
                .remove(&call.request_id);
        }
    }

    /// Dispatch one decoded engine event, with the Python dispatch's rules.
    fn dispatch(&self, message: wire::EngineEvent) -> Result<(), EngineError> {
        let (generation, ready_arrived, call) = {
            let mut state = self.shared.state.lock().expect("state lock");
            if state.closed {
                return Ok(());
            }
            let generation = state.generation;
            if let wire::EngineEvent::Ready(ready) = &message {
                if state.ready.is_some() {
                    return Err(EngineError::ProtocolFatal(
                        "native sent more than one ReadyEvent".into(),
                    ));
                }
                state.ready = Some(*ready);
                drop(state);
                self.shared.ready_changed.notify_all();
                return Ok(());
            }
            // Before its ReadyEvent an engine sends nothing but why its
            // start failed.
            if state.ready.is_none()
                && !matches!(
                    &message,
                    wire::EngineEvent::Error(event)
                        if event.failure_class == wire::FailureClass::EngineUnhealthy
                )
            {
                return Err(EngineError::ProtocolFatal(
                    "native event arrived before ReadyEvent".into(),
                ));
            }
            let call = match &message {
                wire::EngineEvent::Start(event) => Self::pending_call(&state, generation, event.request_id)?,
                wire::EngineEvent::PromptProgress(event) => {
                    Self::pending_call(&state, generation, event.request_id)?
                }
                wire::EngineEvent::Tokens(event) => {
                    Self::pending_call(&state, generation, event.request_id)?
                }
                wire::EngineEvent::Logprobs(event) => {
                    Self::pending_call(&state, generation, event.request_id)?
                }
                wire::EngineEvent::MaskRequest(ref event) => {
                    Self::pending_call(&state, generation, event.request_id)?
                }
                wire::EngineEvent::Done(event) => {
                    Self::pending_call(&state, generation, event.request_id)?
                }
                _ => None,
            };
            (generation, state.ready.is_some(), call)
        };
        let _ = ready_arrived;
        match message {
            wire::EngineEvent::Ready(_) => unreachable!("handled above"),
            wire::EngineEvent::StatusJson(event) => self.dispatch_status(event),
            wire::EngineEvent::Error(event) => self.dispatch_error(generation, event)?,
            wire::EngineEvent::Start(event) => {
                let call = call.expect("gated above");
                if !call.record_start(&event) {
                    return Err(EngineError::ProtocolFatal(
                        "duplicate or late StartEvent".into(),
                    ));
                }
            }
            wire::EngineEvent::PromptProgress(event) => {
                let call = call.expect("gated above");
                if let Err(issue) = call.record_progress(&event) {
                    return Err(EngineError::ProtocolFatal(issue));
                }
            }
            wire::EngineEvent::Tokens(event) => {
                let call = call.expect("gated above");
                if let Err(issue) = call.record_tokens(&event) {
                    return Err(EngineError::ProtocolFatal(issue));
                }
            }
            wire::EngineEvent::Logprobs(event) => {
                let call = call.expect("gated above");
                if let Err(issue) = call.record_logprobs(&event) {
                    return Err(EngineError::ProtocolFatal(issue));
                }
            }
            wire::EngineEvent::MaskRequest(_event) => {
                let call = call.expect("gated above");
                if call.mask_provider.is_none() {
                    return Err(EngineError::ProtocolFatal(
                        "native requested a token mask for an unconstrained request".into(),
                    ));
                }
                // The CPU mask workers are the next slice; until they are
                // ported, a mask request fails its call as a server
                // condition.
                call.record_mask_error(EngineError::MaskComputationFailed {
                    message: "token-mask workers are not ported yet".into(),
                    retryable: true,
                });
            }
            wire::EngineEvent::Done(event) => {
                let call = call.expect("gated above");
                call.check_done(&event)?;
                if let Some(error) = call.terminal_mask_error() {
                    self.fail_call(&call, error);
                } else {
                    self.finish_call(&call, Some(event), None);
                }
            }
        }
        Ok(())
    }

    fn pending_call(
        state: &State,
        generation: u64,
        request_id: u64,
    ) -> Result<Option<Arc<Call>>, EngineError> {
        match state.pending.get(&request_id) {
            Some(call) if call.generation == generation => Ok(Some(call.clone())),
            _ => Err(EngineError::ProtocolFatal(format!(
                "native event references unknown request {request_id}"
            ))),
        }
    }

    /// End one call with its result or failure and release its admission.
    fn finish_call(
        &self,
        call: &Call,
        result: Option<wire::DoneEvent>,
        error: Option<EngineError>,
    ) {
        if call.set_terminal(result, error) {
            self.shared
                .state
                .lock()
                .expect("state lock")
                .pending
                .remove(&call.request_id);
        }
    }

    fn dispatch_error(&self, generation: u64, event: wire::ErrorEvent) -> Result<(), EngineError> {
        match event.failure_class {
            wire::FailureClass::RequestError => {
                let state = self.shared.state.lock().expect("state lock");
                let call = Self::pending_call(&state, generation, event.request_id)?
                    .expect("request errors name a request");
                drop(state);
                self.finish_call(
                    &call,
                    None,
                    Some(EngineError::RequestFailed(Box::new(RequestFailed {
                        request_id: event.request_id,
                        code: event.code,
                        message_bytes: event.message,
                        retryable: event.retryable,
                        retry_after_seconds: event.retry_after_seconds,
                    }))),
                );
                Ok(())
            }
            wire::FailureClass::EngineUnhealthy => Err(EngineError::EngineUnhealthy(
                String::from_utf8_lossy(&event.message).into_owned(),
            )),
            wire::FailureClass::ProtocolFatal => Err(EngineError::ProtocolFatal(
                String::from_utf8_lossy(&event.message).into_owned(),
            )),
        }
    }

    fn dispatch_status(&self, event: wire::StatusJsonEvent) {
        let waiter = {
            let mut state = self.shared.state.lock().expect("state lock");
            let waiter = state.status_waiters.remove(&event.correlation_id);
            match waiter {
                Some(waiter) => waiter,
                None => {
                    // An answer to a request this client already gave up on
                    // is read and dropped; an unknown correlation is fatal.
                    if event.correlation_id != 0 && event.correlation_id <= state.last_status_id {
                        return;
                    }
                    // The failure is reported through the reader loop.
                    return;
                }
            }
        };
        waiter.answer(event);
    }
}

/// The reader thread: parse the engine's output, decode each frame, and
/// dispatch it; any failure fails the generation.
fn reader_loop(
    mut read: Box<dyn Read + Send>,
    shared: std::sync::Weak<Shared>,
    client: EngineClient,
) {
    let mut parser = wire::FrameParser::new();
    let mut chunk = vec![0u8; READ_CHUNK_BYTES];
    loop {
        let read_bytes = match read.read(&mut chunk) {
            Ok(0) => {
                let shared = match shared.upgrade() {
                    Some(shared) => shared,
                    None => return,
                };
                let error = match parser.finish() {
                    Some(issue) => issue_error(&issue),
                    None => EngineError::EngineUnhealthy("native protocol reached EOF".into()),
                };
                fail_shared(&shared, error);
                return;
            }
            Ok(count) => count,
            Err(error) => {
                if let Some(shared) = shared.upgrade() {
                    fail_shared(
                        &shared,
                        EngineError::EngineUnhealthy(format!("native reader failed: {error}")),
                    );
                }
                return;
            }
        };
        let shared = match shared.upgrade() {
            Some(shared) => shared,
            None => return,
        };
        let mut offset = 0usize;
        while offset < read_bytes {
            let step = parser.consume(&chunk[offset..read_bytes]);
            offset += step.consumed_bytes;
            if let Some(issue) = step.issue {
                fail_shared(&shared, issue_error(&issue));
                return;
            }
            if step.consumed_bytes == 0 {
                fail_shared(
                    &shared,
                    EngineError::ProtocolFatal("native parser made no progress".into()),
                );
                return;
            }
            if let Some(frame) = step.frame {
                let message = match wire::decode_frame(&frame) {
                    Ok(message) => message,
                    Err(error) => {
                        fail_shared(&shared, issue_error(&error.0));
                        return;
                    }
                };
                if let Err(error) = client.dispatch(message) {
                    fail_shared(&shared, error);
                    return;
                }
            }
        }
    }
}

/// The parser's and the decoder's failures fail the stream; a request's
/// own failure arrives as an ErrorEvent.
fn issue_error(issue: &wire::ProtocolIssue) -> EngineError {
    if issue.failure_class == wire::FailureClass::EngineUnhealthy {
        EngineError::EngineUnhealthy(issue.to_string())
    } else {
        EngineError::ProtocolFatal(issue.to_string())
    }
}

fn fail_shared(shared: &Shared, error: EngineError) {
    let pending = {
        let mut state = shared.state.lock().expect("state lock");
        if state.closed {
            return;
        }
        if state.fatal_error.is_none() {
            state.fatal_error = Some(error.clone());
        }
        state.ready = None;
        std::mem::take(&mut state.pending)
    };
    for (_, call) in pending {
        call.set_terminal(None, Some(error.clone()));
    }
    for (_, waiter) in shared
        .state
        .lock()
        .expect("state lock")
        .status_waiters
        .drain()
    {
        waiter.fail(error.clone());
    }
    shared.ready_changed.notify_all();
}
