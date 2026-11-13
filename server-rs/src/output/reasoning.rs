//! The output's reasoning and its answer, as the output arrives. Ported
//! from `server/output.py`'s `ReasoningSplitter`.

use crate::chat_format::{ChatFormat, THINK_END, THINK_START};
use crate::output::stop::hold_partial;

/// What one `put` decided: the reasoning and the content it released.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReasoningPart {
    Reasoning(String),
    Content(String),
}

/// The output's reasoning and its answer, as the output arrives. The
/// reasoning starts the output where the generation prompt opened a think
/// block (`thinking`), and where the model opens its reasoning itself it
/// follows a THINK_START that starts the output, which is dropped. It ends
/// at THINK_END, which is dropped too, and where a call may follow (a
/// call's opening ends it) at a call's opening, which begins the answer.
/// A text completion, which has no chat format, reasons only where
/// `thinking` says so, and calls no tool.
pub struct ReasoningSplitter {
    reasoning: bool,
    /// Whether the new text may yet open its reasoning: until it shows
    /// whether it starts with THINK_START.
    opening: bool,
    pending: String,
    /// Whether the newlines after </think>, which set the answer apart in
    /// the chat template's layout of a turn, are still to be dropped.
    separator: bool,
    ends: Vec<String>,
}

impl ReasoningSplitter {
    pub fn new(thinking: bool, chat_format: Option<&ChatFormat>, tool_calls: bool) -> ReasoningSplitter {
        let mut ends = vec![THINK_END.to_string()];
        if let Some(format) = chat_format {
            if tool_calls && format.calls_end_reasoning() {
                ends.push(format.call_syntax.call_open());
            }
        }
        let opening = !thinking && chat_format.is_some();
        ReasoningSplitter {
            reasoning: thinking,
            opening,
            pending: String::new(),
            separator: false,
            ends,
        }
    }

    /// The text as it arrives, split into the reasoning and the content it
    /// released.
    pub fn put(&mut self, text: &str) -> Vec<ReasoningPart> {
        let mut text = text.to_string();
        if self.opening {
            self.pending.push_str(&text);
            if self.pending.len() < THINK_START.len() && THINK_START.starts_with(&self.pending) {
                return Vec::new();
            }
            self.opening = false;
            text = std::mem::take(&mut self.pending);
            if text.starts_with(THINK_START) {
                self.reasoning = true;
                text = text[THINK_START.len()..].to_string();
            }
        }
        if !self.reasoning {
            return self.content(text);
        }
        self.pending.push_str(&text);
        let mut ends: Vec<(usize, &String)> = Vec::new();
        for marker in &self.ends {
            if let Some(index) = self.pending.find(marker) {
                ends.push((index, marker));
            }
        }
        if let Some((end, marker)) = ends.iter().min_by_key(|(index, _)| *index).cloned() {
            let reasoning = self.pending[..end].to_string();
            // The close is dropped; a call's opening begins the answer,
            // so it stays in the content.
            let content_from = if *marker == THINK_END {
                end + marker.len()
            } else {
                end
            };
            let content = self.pending[content_from..].to_string();
            self.pending.clear();
            self.reasoning = false;
            self.separator = marker == THINK_END;
            let mut output = Vec::new();
            if !reasoning.is_empty() {
                output.push(ReasoningPart::Reasoning(reasoning));
            }
            output.extend(self.content(content));
            return output;
        }
        let markers: Vec<&str> = self.ends.iter().map(|marker| marker.as_str()).collect();
        let pending = std::mem::take(&mut self.pending);
        let (ready, held) = hold_partial(&pending, &markers);
        self.pending = held.to_string();
        if ready.is_empty() {
            Vec::new()
        } else {
            vec![ReasoningPart::Reasoning(ready.to_string())]
        }
    }

    fn content(&mut self, mut text: String) -> Vec<ReasoningPart> {
        if self.separator {
            text = text.trim_start_matches('\n').to_string();
            if text.is_empty() {
                return Vec::new();
            }
            self.separator = false;
        }
        if text.is_empty() {
            Vec::new()
        } else {
            vec![ReasoningPart::Content(text)]
        }
    }

    /// The pending text as the output ends: the reasoning's or the
    /// answer's, whichever was open.
    pub fn finish(&mut self) -> Vec<ReasoningPart> {
        if self.pending.is_empty() {
            return Vec::new();
        }
        let kind = if self.reasoning {
            ReasoningPart::Reasoning
        } else {
            ReasoningPart::Content
        };
        let text = std::mem::take(&mut self.pending);
        vec![kind(text)]
    }
}
