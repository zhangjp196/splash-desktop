//! The stop search's and the reasoning splitter's tests, ported from the
//! Python tests that drive `StopSearch`, `hold_partial` and
//! `ReasoningSplitter` (`dev/tests/server/test_output.py`'s cases).

use splash_server::chat_format::{flint, qwen, THINK_END, THINK_START, TOOL_CALL_OPEN};
use splash_server::output::{hold_partial, ReasoningPart, ReasoningSplitter, StopSearch};

#[test]
fn hold_partial_splits_before_a_longest_end_that_begins_a_marker() {
    assert_eq!(hold_partial("abcde", &["xyz"]), ("abcde", ""));
    // "abcab" ends with "ab", a prefix of "abc": held back.
    assert_eq!(hold_partial("abcab", &["abc"]), ("abc", "ab"));
    assert_eq!(hold_partial("abcab", &["abca"]), ("abc", "ab"));
    // The longest end that begins a marker wins.
    assert_eq!(hold_partial("abcab", &["ab", "abca"]), ("abc", "ab"));
    assert_eq!(hold_partial("abcab", &["b"]), ("abcab", ""));
    // Nothing to hold.
    assert_eq!(hold_partial("", &["abc"]), ("", ""));
}

#[test]
fn the_first_stop_the_output_contains_wins() {
    let mut search = StopSearch::new(&["STOP", "END"]);
    assert_eq!(search.search("hello ST"), None);
    assert_eq!(search.held(), 2, "ST begins STOP");
    assert_eq!(search.search("o"), None);
    assert_eq!(search.held(), 0);
    // The output contains END first.
    let found = search.search(" the END of it").expect("found");
    assert_eq!(found.stop, "END");
    // It began before this text's end: back counts past the text's start.
    assert!(found.back > found.stop.len());
}

#[test]
fn a_stop_split_across_calls_is_the_streamer_s_business() {
    // The search's state tracks the text it was given: an end that does
    // not begin a stop holds nothing, and the streamer holds the text
    // that may complete a stop before the search sees it.
    let mut search = StopSearch::new(&[THINK_END]);
    // The stop's beginning arrives: the state holds how much of the text's
    // end begins it.
    assert_eq!(search.search("hello </thi"), None);
    assert_eq!(search.held(), 5);
    // The rest completes it, and how far back the match began is
    // reported: it began in the earlier text.
    let found = search.search("nk> there").expect("found");
    assert_eq!(found.stop, THINK_END);
    assert_eq!(found.back, 14);
}

#[test]
fn the_reasoning_opens_where_the_generation_prompt_opened_it() {
    let format = qwen();
    let mut splitter = ReasoningSplitter::new(true, Some(&format), false);
    let parts = splitter.put("thinking text");
    assert_eq!(parts, vec![ReasoningPart::Reasoning("thinking text".into())]);
    // The close ends the reasoning and is dropped. The newline before it
    // is the reasoning's; the newlines after it set the answer apart and
    // are dropped.
    let parts = splitter.put(&format!(
        "\n{THINK_END}\n\nThe answer is 42."
    ));
    assert_eq!(
        parts,
        vec![
            ReasoningPart::Reasoning("\n".into()),
            ReasoningPart::Content("The answer is 42.".into())
        ]
    );
    // Everything after is content.
    let parts = splitter.put(" More.");
    assert_eq!(parts, vec![ReasoningPart::Content(" More.".into())]);
    assert!(splitter.finish().is_empty());
}

#[test]
fn the_reasoning_opens_where_the_model_opens_it() {
    let format = flint();
    let mut splitter = ReasoningSplitter::new(false, Some(&format), false);
    // The output may yet open its reasoning: a prefix of THINK_START is
    // held.
    let prefix: String = THINK_START.chars().take(2).collect();
    assert!(splitter.put(&prefix).is_empty());
    // The opening is dropped and the reasoning starts.
    let parts = splitter.put(&format!("{}reasoning here", &THINK_START[2..]));
    assert_eq!(parts, vec![ReasoningPart::Reasoning("reasoning here".into())]);
    // A model that opens its reasoning itself closes it before it calls a
    // tool; without tool calls the close alone ends it.
    let parts = splitter.put(&format!("{THINK_END}answer"));
    assert_eq!(parts, vec![ReasoningPart::Content("answer".into())]);
    // The close alone, with no reasoning open, is content: it is longer
    // than THINK_START, so it is decided at once.
    let mut splitter = ReasoningSplitter::new(false, Some(&format), false);
    let parts = splitter.put(THINK_END);
    assert_eq!(parts, vec![ReasoningPart::Content(THINK_END.into())]);
}

#[test]
fn a_call_opening_ends_the_reasoning_where_a_call_may_follow() {
    let format = qwen();
    let mut splitter = ReasoningSplitter::new(true, Some(&format), true);
    let parts = splitter.put("thinking");
    assert_eq!(parts, vec![ReasoningPart::Reasoning("thinking".into())]);
    // The call's opening ends the reasoning where a call may follow: the
    // newlines before it are the reasoning's, and the opening begins the
    // answer.
    let parts = splitter.put(&format!(
        "\n\n{TOOL_CALL_OPEN}\n<function=run_tests>"
    ));
    assert_eq!(
        parts,
        vec![
            ReasoningPart::Reasoning("\n\n".into()),
            ReasoningPart::Content(format!(
                "{TOOL_CALL_OPEN}\n<function=run_tests>"
            ))
        ]
    );
}

#[test]
fn a_text_completion_reasons_only_where_thinking_says_so() {
    let mut splitter = ReasoningSplitter::new(false, None, false);
    // No chat format: no THINK_START opening, no call ends.
    let parts = splitter.put("plain answer");
    assert_eq!(parts, vec![ReasoningPart::Content("plain answer".into())]);
    // The close is content here, not a reasoning end.
    let parts = splitter.put("\n</think>more");
    assert_eq!(parts, vec![ReasoningPart::Content("\n</think>more".into())]);
}

#[test]
fn the_pending_text_ends_as_the_reasoning() {
    let format = qwen();
    let mut splitter = ReasoningSplitter::new(true, Some(&format), false);
    // A partial close is held, and the output ends inside the reasoning.
    let parts = splitter.put("thinking </thi");
    assert_eq!(parts, vec![ReasoningPart::Reasoning("thinking ".into())]);
    let parts = splitter.finish();
    assert_eq!(parts, vec![ReasoningPart::Reasoning("</thi".into())]);
}
