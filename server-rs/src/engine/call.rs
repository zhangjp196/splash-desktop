//! The engine client's failures and one request's state machine, ported
//! from `server/runtime.py`'s error family and `RuntimeCall`. The
//! subprocess wrapper that drives it is `client.rs`.

use std::fmt;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use crate::protocol as wire;

/// The engine client's failures. Stored failures are shared across calls;
/// re-raising one retains its history, so [`EngineError::restate`] copies
/// the failure fresh, as the Python's `restate` does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    /// The engine's stream is closed, by shutdown or by the engine's exit.
    RuntimeClosed(String),
    /// The engine's pending queue is full.
    PendingLimitExceeded(String),
    /// The engine refused or failed one request.
    RequestFailed(Box<RequestFailed>),
    /// The engine is not able to serve.
    EngineUnhealthy(String),
    /// The engine's stream cannot be trusted.
    ProtocolFatal(String),
    /// A token mask was not delivered. `retryable` marks a server
    /// condition, such as a full mask queue or a stalled transport,
    /// rather than the request's constraint or provider.
    MaskComputationFailed { message: String, retryable: bool },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestFailed {
    pub request_id: u64,
    pub code: Vec<u8>,
    pub message_bytes: Vec<u8>,
    pub retryable: bool,
    /// When a retry may succeed, in seconds; 0 for at once.
    pub retry_after_seconds: u32,
}

impl RequestFailed {
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.message_bytes).into_owned()
    }

    pub fn machine_code(&self) -> String {
        String::from_utf8_lossy(&self.code).into_owned()
    }
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::RuntimeClosed(message)
            | EngineError::PendingLimitExceeded(message)
            | EngineError::EngineUnhealthy(message)
            | EngineError::ProtocolFatal(message) => f.write_str(message),
            EngineError::RequestFailed(failure) => write!(
                f,
                "request {} failed [{}]: {}",
                failure.request_id,
                failure.machine_code(),
                failure.text()
            ),
            EngineError::MaskComputationFailed { message, .. } => f.write_str(message),
        }
    }
}

impl std::error::Error for EngineError {}

impl EngineError {
    /// Copy this failure without retaining a history.
    pub fn restate(&self) -> EngineError {
        self.clone()
    }

    /// The seconds a retry status invites the client to wait, for the
    /// statuses a response answers with Retry-After.
    pub fn retry_after(&self) -> Option<u64> {
        match self {
            EngineError::RequestFailed(failure) if failure.retryable => {
                Some(failure.retry_after_seconds as u64)
            }
            EngineError::MaskComputationFailed { retryable: true, .. } => Some(0),
            _ => None,
        }
    }
}

/// `(absolute unix µs, remaining µs)` for a deadline `remaining` seconds
/// away; an infinite one saturates at the wire's u64 maximum.
pub fn wire_deadline(remaining: Option<Duration>) -> (u64, u64) {
    let wall = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since| since.as_micros() as u64)
        .unwrap_or(0);
    let micros = match remaining {
        None => u64::MAX - wall,
        Some(remaining) => {
            let remaining_micros = remaining.as_micros() as u64;
            let limit = u64::MAX - wall;
            if remaining_micros >= limit {
                limit
            } else {
                remaining_micros.max(1)
            }
        }
    };
    (wall + micros, micros)
}

/// What one native request carries past its wire frame: the deadline the
/// client enforces, and the mask provider a token-mask request needs. The
/// request id and both deadline fields of `frame` are zero; the client
/// fills them in.
#[derive(Clone)]
pub struct GenerationRequest {
    pub frame: wire::RequestFrame,
    /// `None` for no deadline.
    pub deadline: Option<Duration>,
    /// A token-mask request's mask provider, called with each
    /// MaskRequestEvent to produce the mask words.
    pub mask_provider: Option<MaskProvider>,
}

/// A token-mask request's mask provider.
pub type MaskProvider = Arc<dyn Fn(&wire::MaskRequestEvent) -> Vec<u8> + Send + Sync>;

impl GenerationRequest {
    pub fn validate(&self) -> Result<(), EngineError> {
        if (self.frame.constraint == wire::ConstraintMode::TokenMask) != self.mask_provider.is_some()
        {
            return Err(EngineError::RuntimeClosed(
                "a token-mask request needs exactly one mask provider".into(),
            ));
        }
        Ok(())
    }
}

/// The events a request's consumer reads, delivered as they arrive.
pub type EventCallback = Arc<dyn Fn(&Call, &wire::EngineEvent) + Send + Sync>;
/// The one callback a finished request receives.
pub type CompletionCallback = Arc<dyn Fn(&Call) + Send + Sync>;

/// The state one call tracks, under its lock.
struct State {
    start: Option<wire::StartEvent>,
    progress: Option<wire::PromptProgressEvent>,
    next_token_offset: u32,
    mask_error: Option<EngineError>,
    callback_error: Option<String>,
    cancel_requested: bool,
    terminal: Option<Terminal>,
    on_event: Option<EventCallback>,
    on_complete: Option<CompletionCallback>,
}

enum Terminal {
    Done(wire::DoneEvent),
    Error(EngineError),
}

/// Future-like handle for one directly admitted native request.
pub struct Call {
    pub request_id: u64,
    /// The engine generation the call belongs to; events of another are
    /// not its business.
    pub generation: u64,
    prompt_tokens: usize,
    logical_max: u32,
    score_tokens: usize,
    return_progress: bool,
    pub mask_provider: Option<MaskProvider>,
    state: Mutex<State>,
    finished: Condvar,
}

impl Call {
    pub fn new(
        request_id: u64,
        generation: u64,
        request: &GenerationRequest,
        on_event: Option<EventCallback>,
        on_complete: Option<CompletionCallback>,
    ) -> Arc<Call> {
        Arc::new(Call {
            request_id,
            generation,
            prompt_tokens: request.frame.prompt_tokens.len(),
            logical_max: request.frame.logical_max_output_tokens,
            score_tokens: request.frame.score_tokens.len(),
            return_progress: request.frame.flags & wire::flags::RETURN_PROGRESS != 0,
            mask_provider: request.mask_provider.clone(),
            state: Mutex::new(State {
                start: None,
                progress: None,
                next_token_offset: 0,
                mask_error: None,
                callback_error: None,
                cancel_requested: false,
                terminal: None,
                on_event,
                on_complete,
            }),
            finished: Condvar::new(),
        })
    }

    /// Whether the request reached its terminal state.
    pub fn done(&self) -> bool {
        self.state.lock().expect("call lock").terminal.is_some()
    }

    pub fn cancel_requested(&self) -> bool {
        self.state.lock().expect("call lock").cancel_requested
    }

    /// The first error an event or completion callback raised.
    pub fn callback_error(&self) -> Option<String> {
        self.state.lock().expect("call lock").callback_error.clone()
    }

    /// Wait for the request's end and return its DoneEvent, or the failure
    /// it ended with.
    pub fn result(&self, timeout: Option<Duration>) -> Result<wire::DoneEvent, EngineError> {
        let deadline = timeout.map(|timeout| std::time::Instant::now() + timeout);
        let mut state = self.state.lock().expect("call lock");
        while state.terminal.is_none() {
            match deadline {
                None => state = self.finished.wait(state).expect("call lock"),
                Some(deadline) => {
                    let now = std::time::Instant::now();
                    if now >= deadline {
                        break;
                    }
                    let (guarded, _) = self
                        .finished
                        .wait_timeout(state, deadline - now)
                        .expect("call lock");
                    state = guarded;
                }
            }
        }
        match &state.terminal {
            Some(Terminal::Done(done)) => Ok(done.clone()),
            Some(Terminal::Error(error)) => Err(error.restate()),
            None => Err(EngineError::RuntimeClosed(format!(
                "request {} did not finish in time",
                self.request_id
            ))),
        }
    }

    /// Mark the call cancelled and report whether the client should send a
    /// CancelFrame; the client sends it, so the call does not.
    pub fn mark_cancel_requested(&self) -> bool {
        let mut state = self.state.lock().expect("call lock");
        if state.terminal.is_some() || state.cancel_requested {
            return false;
        }
        state.cancel_requested = true;
        true
    }

    /// Deliver `message` to the event callback. A caller that already
    /// holds the state passes the callback it read there, so delivering a
    /// token costs one lock instead of two. The callback only ever goes
    /// from set to `None`, so a read of `None` is final.
    fn emit(&self, message: &wire::EngineEvent, callback: Option<EventCallback>) {
        let callback = match callback {
            Some(callback) => callback,
            None => match self.state.lock().expect("call lock").on_event.clone() {
                Some(callback) => callback,
                None => return,
            },
        };
        // The callback runs without the state held; a consumer that failed
        // cannot be delivered, so the request is stopped.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            callback(self, message)
        }));
        if outcome.is_err() {
            self.record_callback_error("an event callback panicked".into());
        }
    }

    /// Record a callback error and stop the request.
    pub fn record_callback_error(&self, error: String) {
        let stop = {
            let mut state = self.state.lock().expect("call lock");
            if state.callback_error.is_none() {
                state.callback_error = Some(error);
            }
            !state.cancel_requested && state.terminal.is_none()
        };
        if stop {
            self.mark_cancel_requested();
        }
    }

    /// Record a StartEvent; `false` when the call is terminal or started.
    pub fn record_start(&self, event: &wire::StartEvent) -> bool {
        let callback = {
            let mut state = self.state.lock().expect("call lock");
            if state.terminal.is_some() || state.start.is_some() {
                return false;
            }
            state.start = Some(*event);
            state.on_event.clone()
        };
        if let Some(callback) = callback {
            self.emit(&wire::EngineEvent::Start(*event), Some(callback));
        }
        true
    }

    /// Record a PromptProgressEvent, or the rule it broke.
    pub fn record_progress(&self, event: &wire::PromptProgressEvent) -> Result<(), String> {
        let callback = {
            let mut state = self.state.lock().expect("call lock");
            if !self.return_progress {
                return Err("unsolicited prompt progress".into());
            }
            if state.terminal.is_some() || state.start.is_none() || state.next_token_offset != 0 {
                return Err("prompt progress outside prefill".into());
            }
            let matched = state.start.as_ref().expect("checked above").matched_prompt_tokens;
            if matched > event.processed_tokens
                || event.processed_tokens as usize > self.prompt_tokens
            {
                return Err("prompt progress is outside the request's token range".into());
            }
            if let Some(previous) = &state.progress {
                if event.processed_tokens <= previous.processed_tokens
                    || event.elapsed_micros < previous.elapsed_micros
                {
                    return Err("prompt progress moved backwards or repeated".into());
                }
            }
            state.progress = Some(*event);
            state.on_event.clone()
        };
        if let Some(callback) = callback {
            self.emit(&wire::EngineEvent::PromptProgress(*event), Some(callback));
        }
        Ok(())
    }

    /// Record a TokensEvent, or the rule it broke.
    pub fn record_tokens(&self, event: &wire::TokensEvent) -> Result<(), String> {
        let callback = {
            let mut state = self.state.lock().expect("call lock");
            if state.terminal.is_some() {
                return Err("TokensEvent arrived after the request became terminal".into());
            }
            if state.start.is_none() {
                return Err("TokensEvent arrived before StartEvent".into());
            }
            if event.sequence_offset != state.next_token_offset {
                let relation = if event.sequence_offset < state.next_token_offset {
                    "overlaps or duplicates prior tokens"
                } else {
                    "leaves a gap in the token stream"
                };
                return Err(format!(
                    "TokensEvent offset {} {relation}; expected {}",
                    event.sequence_offset, state.next_token_offset
                ));
            }
            let next_offset = state.next_token_offset + event.tokens.len() as u32;
            if next_offset > self.logical_max {
                return Err(format!(
                    "TokensEvent stream length {next_offset} exceeds logical maximum {}",
                    self.logical_max
                ));
            }
            state.next_token_offset = next_offset;
            state.on_event.clone()
        };
        if let Some(callback) = callback {
            self.emit(&wire::EngineEvent::Tokens(event.clone()), Some(callback));
        }
        Ok(())
    }

    /// Record a LogprobsEvent, or the rule it broke.
    pub fn record_logprobs(&self, event: &wire::LogprobsEvent) -> Result<(), String> {
        let callback = {
            let state = self.state.lock().expect("call lock");
            if state.terminal.is_some() {
                return Err("LogprobsEvent arrived after the request became terminal".into());
            }
            if state.start.is_none() {
                return Err("LogprobsEvent arrived before StartEvent".into());
            }
            if event.sequence_offset != state.next_token_offset {
                return Err(format!(
                    "LogprobsEvent offset {} does not match the next token {}",
                    event.sequence_offset, state.next_token_offset
                ));
            }
            state.on_event.clone()
        };
        if let Some(callback) = callback {
            self.emit(&wire::EngineEvent::Logprobs(event.clone()), Some(callback));
        }
        Ok(())
    }

    /// Record a mask computation's failure, before the call ends.
    pub fn record_mask_error(&self, error: EngineError) {
        let mut state = self.state.lock().expect("call lock");
        if state.terminal.is_none() && state.mask_error.is_none() {
            state.mask_error = Some(error);
        }
    }

    /// The mask failure a terminal call ends with, if one was recorded.
    pub fn terminal_mask_error(&self) -> Option<EngineError> {
        self.state.lock().expect("call lock").mask_error.clone()
    }

    /// The rules a DoneEvent must satisfy to end this call.
    pub fn check_done(&self, done: &wire::DoneEvent) -> Result<(), EngineError> {
        let state = self.state.lock().expect("call lock");
        let completion_tokens = state.next_token_offset;
        if done.prompt_tokens as usize != self.prompt_tokens {
            return Err(EngineError::ProtocolFatal(format!(
                "DoneEvent prompt count {} does not match request count {}",
                done.prompt_tokens, self.prompt_tokens
            )));
        }
        if done.completion_tokens > self.logical_max {
            return Err(EngineError::ProtocolFatal(format!(
                "DoneEvent completion count {} exceeds logical maximum {}",
                done.completion_tokens, self.logical_max
            )));
        }
        if done.completion_tokens != completion_tokens {
            return Err(EngineError::ProtocolFatal(format!(
                "DoneEvent completion count {} does not match streamed count {completion_tokens}",
                done.completion_tokens
            )));
        }
        if state.start.is_none()
            && !(state.cancel_requested
                && done.reason == wire::FinishReason::Cancelled
                && completion_tokens == 0)
        {
            return Err(EngineError::ProtocolFatal(
                "DoneEvent arrived before StartEvent".into(),
            ));
        }
        // Only a bound of the engine's own ends a request with LENGTH short
        // of its logical maximum.
        if done.reason == wire::FinishReason::Length {
            let expected = if done.bounded {
                completion_tokens < self.logical_max
            } else {
                completion_tokens == self.logical_max
            };
            if !expected {
                return Err(EngineError::ProtocolFatal(format!(
                    "length-finished DoneEvent has {completion_tokens} tokens; expected {}logical maximum {}",
                    if done.bounded { "fewer than " } else { "" },
                    self.logical_max
                )));
            }
        }
        if self.score_tokens != 0 {
            if done.decode_micros != 0 {
                return Err(EngineError::ProtocolFatal(
                    "score DoneEvent reported autoregressive decoding".into(),
                ));
            }
            if !matches!(
                done.reason,
                wire::FinishReason::Stop | wire::FinishReason::Cancelled
            ) {
                return Err(EngineError::ProtocolFatal(
                    "score DoneEvent has an invalid finish reason".into(),
                ));
            }
            if done.reason == wire::FinishReason::Stop {
                if done.option_logits.len() != self.score_tokens {
                    return Err(EngineError::ProtocolFatal(format!(
                        "score DoneEvent returned {} option logits; expected {}",
                        done.option_logits.len(),
                        self.score_tokens
                    )));
                }
            } else if !done.option_logits.is_empty() {
                return Err(EngineError::ProtocolFatal(
                    "unfinished score DoneEvent returned option logits".into(),
                ));
            }
        } else if !done.option_logits.is_empty() {
            return Err(EngineError::ProtocolFatal(
                "DoneEvent returned option logits for a generation request".into(),
            ));
        }
        Ok(())
    }

    /// The mask failure a terminal call ends with, if one was recorded.
    /// See `terminal_mask_error`.
    pub fn take_mask_error_if_failed(&self, error: &EngineError) -> Option<EngineError> {
        let state = self.state.lock().expect("call lock");
        match (&state.terminal, &state.mask_error) {
            (Some(Terminal::Error(failure)), Some(mask_error)) if failure == error => {
                Some(mask_error.clone())
            }
            _ => None,
        }
    }

    /// End the call with its result or its failure; `false` when it was
    /// already terminal. The completion callback runs without the state
    /// held.
    pub fn set_terminal(
        &self,
        result: Option<wire::DoneEvent>,
        error: Option<EngineError>,
    ) -> bool {
        let callback = {
            let mut state = self.state.lock().expect("call lock");
            if state.terminal.is_some() {
                return false;
            }
            state.terminal = Some(match (result, error) {
                (Some(done), _) => Terminal::Done(done),
                (None, Some(error)) => Terminal::Error(error),
                (None, None) => unreachable!("a terminal call ends with one or the other"),
            });
            state.on_event = None;
            state.on_complete.take()
        };
        self.finished.notify_all();
        if let Some(callback) = callback {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| callback(self)));
            if outcome.is_err() {
                self.record_callback_error("a completion callback panicked".into());
            }
        }
        true
    }
}
