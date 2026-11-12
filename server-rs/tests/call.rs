//! The call state machine's rules, ported from the Python tests that drive
//! `RuntimeCall` directly (`test_prompt_progress.py`'s
//! `test_runtime_rejects_unsolicited_or_regressing_progress` and the
//! request-lifecycle cases the runtime tests cover).

use std::sync::Arc;

use splash_server::engine::{Call, EngineError, GenerationRequest};
use splash_server::protocol as wire;

fn request(prompt_tokens: usize, return_progress: bool) -> GenerationRequest {
    GenerationRequest {
        frame: wire::RequestFrame {
            logical_max_output_tokens: 8,
            prompt_tokens: vec![1; prompt_tokens],
            flags: if return_progress {
                wire::flags::RETURN_PROGRESS
            } else {
                0
            },
            ..wire::RequestFrame::default()
        },
        deadline: None,
        mask_provider: None,
    }
}

fn started_call(prompt_tokens: usize, return_progress: bool) -> Arc<Call> {
    let call = Call::new(1, 1, &request(prompt_tokens, return_progress), None, None);
    assert!(call.record_start(&wire::StartEvent {
        request_id: 1,
        lane: 0,
        matched_prompt_tokens: 1,
    }));
    call
}

fn progress(processed: u32, elapsed: u64) -> wire::PromptProgressEvent {
    wire::PromptProgressEvent {
        request_id: 1,
        processed_tokens: processed,
        elapsed_micros: elapsed,
    }
}

#[test]
fn unsolicited_progress_is_rejected() {
    let call = started_call(10, false);
    assert!(call.record_progress(&progress(1, 0)).is_err());
}

#[test]
fn progress_outside_the_request_range_or_backwards_is_rejected() {
    let call = started_call(10, true);
    // Before the matched prefix.
    assert!(call.record_progress(&progress(0, 100)).is_err());
    // Past the prompt.
    assert!(call.record_progress(&progress(11, 100)).is_err());
    // The first valid reading.
    assert!(call.record_progress(&progress(1, 100)).is_ok());
    // Repeated.
    assert!(call.record_progress(&progress(1, 101)).is_err());
    // Backwards in position.
    assert!(call.record_progress(&progress(2, 99)).is_err());
    // Backwards in time.
    assert!(call.record_progress(&progress(2, 99)).is_err());
    // Forward in both.
    assert!(call.record_progress(&progress(2, 101)).is_ok());
    // Tokens have started; prefill is over.
    assert!(
        call.record_tokens(&wire::TokensEvent {
            request_id: 1,
            sequence_offset: 0,
            tokens: vec![42],
        })
        .is_ok()
    );
    assert!(call.record_progress(&progress(2, 102)).is_err());
}

#[test]
fn the_token_stream_must_be_contiguous_within_the_maximum() {
    let call = started_call(10, true);
    assert!(call.record_progress(&progress(10, 1)).is_ok());
    // A gap.
    assert!(
        call.record_tokens(&wire::TokensEvent {
            request_id: 1,
            sequence_offset: 1,
            tokens: vec![7, 7],
        })
        .is_err()
    );
    // The contiguous stream, to the logical maximum.
    for offset in 0..8u32 {
        assert!(
            call.record_tokens(&wire::TokensEvent {
                request_id: 1,
                sequence_offset: offset,
                tokens: vec![42],
            })
            .is_ok()
        );
    }
    // Past it.
    assert!(
        call.record_tokens(&wire::TokensEvent {
            request_id: 1,
            sequence_offset: 8,
            tokens: vec![42],
        })
        .is_err()
    );
}

#[test]
fn tokens_before_start_and_after_the_end_are_rejected() {
    let call = Call::new(1, 1, &request(10, false), None, None);
    assert!(
        call.record_tokens(&wire::TokensEvent {
            request_id: 1,
            sequence_offset: 0,
            tokens: vec![42],
        })
        .is_err()
    );
    call.set_terminal(
        Some(wire::DoneEvent {
            request_id: 1,
            reason: wire::FinishReason::Cancelled,
            prompt_tokens: 10,
            completion_tokens: 0,
            prefill_micros: 0,
            decode_micros: 0,
            wall_micros: 0,
            option_logits: vec![],
            deduplicated_tokens: 0,
            drafted_tokens: 0,
            accepted_draft_tokens: 0,
            draft_cycles: 0,
            bounded: false,
        }),
        None,
    );
    assert!(
        call.record_tokens(&wire::TokensEvent {
            request_id: 1,
            sequence_offset: 0,
            tokens: vec![42],
        })
        .is_err()
    );
}

#[test]
fn a_done_event_must_agree_with_the_stream_it_ends() {
    let call = started_call(10, false);
    for offset in 0..4u32 {
        call.record_tokens(&wire::TokensEvent {
            request_id: 1,
            sequence_offset: offset,
            tokens: vec![42],
        })
        .expect("tokens");
    }
    let done = |completion_tokens: u32, reason: wire::FinishReason, bounded: bool| wire::DoneEvent {
        request_id: 1,
        reason,
        prompt_tokens: 10,
        completion_tokens,
        prefill_micros: 0,
        decode_micros: 0,
        wall_micros: 0,
        option_logits: vec![],
        deduplicated_tokens: 0,
        drafted_tokens: 0,
        accepted_draft_tokens: 0,
        draft_cycles: 0,
        bounded,
    };
    // A wrong prompt count.
    assert!(call
        .check_done(&done(4, wire::FinishReason::Stop, false).with_prompt(9))
        .is_err());
    // A count that disagrees with the stream.
    assert!(call.check_done(&done(3, wire::FinishReason::Stop, false)).is_err());
    // LENGTH short of the maximum without a bound.
    assert!(call.check_done(&done(4, wire::FinishReason::Length, false)).is_err());
    // LENGTH at the bound of the engine's own is fine.
    assert!(call.check_done(&done(4, wire::FinishReason::Length, true)).is_ok());
    // Stop at the streamed count.
    assert!(call.check_done(&done(4, wire::FinishReason::Stop, false)).is_ok());
}

#[test]
fn a_cancelled_call_may_end_without_a_start() {
    let call = Call::new(1, 1, &request(10, false), None, None);
    assert!(call.mark_cancel_requested());
    let done = wire::DoneEvent {
        request_id: 1,
        reason: wire::FinishReason::Cancelled,
        prompt_tokens: 10,
        completion_tokens: 0,
        prefill_micros: 0,
        decode_micros: 0,
        wall_micros: 0,
        option_logits: vec![],
        deduplicated_tokens: 0,
        drafted_tokens: 0,
        accepted_draft_tokens: 0,
        draft_cycles: 0,
        bounded: false,
    };
    assert!(call.check_done(&done).is_ok());
    assert!(call.set_terminal(Some(done), None));
    // A second terminal is refused.
    assert!(!call.set_terminal(None, Some(EngineError::RuntimeClosed("late".into()))));
}

#[test]
fn a_score_call_checks_its_logits() {
    let mut request = request(3, false);
    request.frame.score_tokens = vec![101, 202, 303];
    request.frame.logical_max_output_tokens = 0;
    let call = Call::new(1, 1, &request, None, None);
    call.record_start(&wire::StartEvent {
        request_id: 1,
        lane: 0,
        matched_prompt_tokens: 0,
    });
    let done = |logits: Vec<f32>, reason: wire::FinishReason| wire::DoneEvent {
        request_id: 1,
        reason,
        prompt_tokens: 3,
        completion_tokens: 0,
        prefill_micros: 0,
        decode_micros: 0,
        wall_micros: 0,
        option_logits: logits,
        deduplicated_tokens: 0,
        drafted_tokens: 0,
        accepted_draft_tokens: 0,
        draft_cycles: 0,
        bounded: false,
    };
    // The right count.
    assert!(call
        .check_done(&done(vec![1.0, -2.0, 3.5], wire::FinishReason::Stop))
        .is_ok());
    // The wrong count.
    assert!(call
        .check_done(&done(vec![1.0, -2.0], wire::FinishReason::Stop))
        .is_err());
    // Autoregressive decoding.
    let mut decoded = done(vec![1.0, -2.0, 3.5], wire::FinishReason::Stop);
    decoded.decode_micros = 5;
    assert!(call.check_done(&decoded).is_err());
    // A generation request with logits.
    let generation = started_call(10, false);
    assert!(generation
        .check_done(&done(vec![1.0], wire::FinishReason::Stop))
        .is_err());
}

#[test]
fn the_result_carries_the_done_event_or_the_failure() {
    let call = Call::new(1, 1, &request(10, false), None, None);
    // A short wait on an unfinished call times out.
    assert!(call
        .result(Some(std::time::Duration::from_millis(20)))
        .is_err());
    let done = wire::DoneEvent {
        request_id: 1,
        reason: wire::FinishReason::Stop,
        prompt_tokens: 10,
        completion_tokens: 0,
        prefill_micros: 0,
        decode_micros: 0,
        wall_micros: 0,
        option_logits: vec![],
        deduplicated_tokens: 0,
        drafted_tokens: 0,
        accepted_draft_tokens: 0,
        draft_cycles: 0,
        bounded: false,
    };
    assert!(call.set_terminal(Some(done.clone()), None));
    assert_eq!(call.result(None).unwrap(), done);
    assert!(call.done());
}

trait WithPrompt {
    fn with_prompt(self, prompt_tokens: u32) -> Self;
}

impl WithPrompt for wire::DoneEvent {
    fn with_prompt(mut self, prompt_tokens: u32) -> Self {
        self.prompt_tokens = prompt_tokens;
        self
    }
}
