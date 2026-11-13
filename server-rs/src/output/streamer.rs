//! Incremental detokenization of a generation's tokens into callback
//! text, ported from `server/streamer.py`: each token decodes as the
//! engine sends it, the text a stop sequence may still change is held
//! back, and what is decided goes through one callback.

use crate::output::stop::StopSearch;

/// The streamer's decisions, delivered as they are made.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Streamed {
    Text(String),
    /// A stop sequence matched: the stream ends here.
    Stopped,
}

/// The streamer of one generation's output.
///
/// The decode is incremental by difference: the tokens decoded whole each
/// time, the text beyond what was already emitted is this step's, and a
/// trailing U+FFFD — a multi-byte character the output ends inside — is
/// held until more text decides it. The observable behavior is the
/// Python's `DecodeStream`, whose type in the tokenizers crate this port
/// uses borrows the tokenizer and cannot be stored beside it.
pub struct CallbackStreamer {
    stops: Option<StopSearch>,
    token_ids: Vec<u32>,
    emitted: Vec<String>,
    /// How much of the whole decode the stream has published.
    settled: usize,
    pending_text: String,
    stop_sequence: Option<String>,
}

impl CallbackStreamer {
    /// The streamer of an output whose stop sequences are `stops`.
    pub fn new(stops: &[&str]) -> CallbackStreamer {
        CallbackStreamer {
            stops: if stops.is_empty() {
                None
            } else {
                Some(StopSearch::new(stops))
            },
            token_ids: Vec::new(),
            emitted: Vec::new(),
            settled: 0,
            pending_text: String::new(),
            stop_sequence: None,
        }
    }

    /// The tokens the output has carried, as the reasoning's token count
    /// reads them.
    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    /// The tokens as they arrive, decoded incrementally and split by the
    /// stop sequences.
    pub fn put_tokens(
        &mut self,
        tokenizer: &crate::tokenization::FastTokenizer,
        token_ids: &[u32],
    ) -> Vec<Streamed> {
        if self.stop_sequence.is_some() {
            return Vec::new();
        }
        self.token_ids.extend_from_slice(token_ids);
        let whole = tokenizer.decode(&self.token_ids, true).unwrap_or_default();
        self.put_decoded(&whole)
    }

    /// The whole decode so far, as the streaming loop reads it: the text
    /// beyond what was published is this step's, and a trailing U+FFFD —
    /// a multi-byte character the output ends inside — is held until more
    /// text decides it.
    pub fn put_decoded(&mut self, whole: &str) -> Vec<Streamed> {
        let mut output = Vec::new();
        if self.stop_sequence.is_some() {
            return output;
        }
        let fresh = &whole[self.settled.min(whole.len())..];
        let complete = fresh.trim_end_matches('\u{fffd}');
        if complete.is_empty() {
            return output;
        }
        self.settled += complete.len();
        output.extend(self.emit(complete));
        output
    }

    /// How much of the whole decode the stream has published.
    pub fn settled(&self) -> usize {
        self.settled
    }

    /// The text as it arrives, held back where a stop may still change
    /// it. A stop's match ends the stream: the text before it goes out,
    /// then the stopped marker.
    fn emit(&mut self, text: &str) -> Vec<Streamed> {
        if text.is_empty() {
            return Vec::new();
        }
        let Some(stops) = &mut self.stops else {
            self.send(text);
            return vec![Streamed::Text(text.to_string())];
        };
        self.pending_text.push_str(text);
        let found = stops.search(text);
        if let Some(found) = found {
            self.stop_sequence = Some(found.stop.clone());
            let keep = self.pending_text.len().saturating_sub(found.back);
            let ready = self.pending_text[..keep].to_string();
            self.pending_text.clear();
            let mut output = Vec::new();
            if !ready.is_empty() {
                self.send(&ready);
                output.push(Streamed::Text(ready));
            }
            output.push(Streamed::Stopped);
            return output;
        }
        let held = stops.held();
        let ready_len = self.pending_text.len() - held;
        let ready = self.pending_text[..ready_len].to_string();
        self.pending_text = self.pending_text[ready_len..].to_string();
        if ready.is_empty() {
            Vec::new()
        } else {
            self.send(&ready);
            vec![Streamed::Text(ready)]
        }
    }

    fn send(&mut self, text: &str) {
        self.emitted.push(text.to_string());
    }

    /// The output's end: the whole sequence decoded at once must agree
    /// with what the pieces decoded to, and the pending text goes out.
    pub fn end(
        &mut self,
        tokenizer: &crate::tokenization::FastTokenizer,
    ) -> Vec<Streamed> {
        let mut output = Vec::new();
        if self.stop_sequence.is_some() {
            return output;
        }
        let decoded = tokenizer
            .decode(&self.token_ids, true)
            .unwrap_or_default();
        let handled: String = self.emitted.concat();
        if !decoded.starts_with(&handled) {
            panic!("incremental tokenizer output diverged");
        }
        // Bytes of a multi-byte character the output ends inside decode to
        // U+FFFD; the stream held them back for the rest. Drop the
        // trailing U+FFFD, as vLLM's detokenizer does; one that text
        // follows stays.
        let rest = decoded[handled.len()..]
            .trim_end_matches('\u{fffd}')
            .to_string();
        if !rest.is_empty() {
            self.send(&rest);
            output.push(Streamed::Text(rest));
        }
        if self.stop_sequence.is_none() && !self.pending_text.is_empty() {
            let pending = std::mem::take(&mut self.pending_text);
            self.send(&pending);
            output.push(Streamed::Text(pending));
        }
        output
    }

    /// The stop sequence the output ended at, if one did.
    pub fn stop_sequence(&self) -> Option<&str> {
        self.stop_sequence.as_deref()
    }
}
