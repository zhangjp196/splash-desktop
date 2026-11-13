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
/// The decode is incremental by difference, as the tokenizers crate's own
/// `DecodeStream` does it: the ids needed to produce a valid chunk are kept
/// with the string they last decoded to, and each new token's chunk is what
/// the longer decode adds past that string. A trailing U+FFFD — a
/// multi-byte character the output ends inside — is held until more text
/// decides it. The crate's `DecodeStream` borrows the tokenizer and cannot
/// be stored beside it, so its algorithm is carried here over
/// `FastTokenizer::decode`.
pub struct CallbackStreamer {
    stops: Option<StopSearch>,
    token_ids: Vec<u32>,
    emitted: Vec<String>,
    /// The ids needed to produce valid chunks: the read, the prefix and the
    /// rest, as the crate's `DecodeStream` keeps them.
    ids: Vec<u32>,
    /// The string those ids last decoded to, which the next chunk is what
    /// follows.
    prefix: String,
    /// The index within `ids` the prefix corresponds to.
    prefix_index: usize,
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
            ids: Vec::new(),
            prefix: String::new(),
            prefix_index: 0,
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
        let mut output = Vec::new();
        for token_id in token_ids {
            if self.stop_sequence.is_some() {
                break;
            }
            self.token_ids.push(*token_id);
            if let Some(text) = self.step(tokenizer, *token_id) {
                output.extend(self.emit(&text));
            }
        }
        output
    }

    /// One token's chunk, as the crate's `step_decode_stream` produces it:
    /// the text the longer decode adds past the string the kept ids last
    /// decoded to, or nothing while a multi-byte character is still
    /// incomplete.
    fn step(
        &mut self,
        tokenizer: &crate::tokenization::FastTokenizer,
        token_id: u32,
    ) -> Option<String> {
        if self.prefix.is_empty() && !self.ids.is_empty() {
            let new_prefix = tokenizer.decode(&self.ids, true).ok()?;
            if !new_prefix.ends_with('\u{fffd}') {
                self.prefix = new_prefix;
                self.prefix_index = self.ids.len();
            }
        }
        self.ids.push(token_id);
        let string = tokenizer.decode(&self.ids, true).ok()?;
        if string.len() <= self.prefix.len() || string.ends_with('\u{fffd}') {
            return None;
        }
        if !string.starts_with(&self.prefix) {
            // The decode changed behind the published text, which the
            // crate's stream reports as an invalid prefix: keep the text
            // already sent and start the prefix over from here.
            self.prefix.clear();
            self.prefix_index = 0;
            return None;
        }
        let new_text = string[self.prefix.len()..].to_string();
        let new_prefix_index = self.ids.len() - self.prefix_index;
        self.ids = self.ids[self.prefix_index..].to_vec();
        self.prefix = tokenizer.decode(&self.ids, true).ok()?;
        self.prefix_index = new_prefix_index;
        Some(new_text)
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
            // `back` counts characters, as the search does.
            let characters = self.pending_text.chars().count();
            let keep = characters.saturating_sub(found.back);
            let split = self
                .pending_text
                .char_indices()
                .nth(keep)
                .map(|(index, _)| index)
                .unwrap_or(self.pending_text.len());
            let ready = self.pending_text[..split].to_string();
            self.pending_text.clear();
            let mut output = Vec::new();
            if !ready.is_empty() {
                self.send(&ready);
                output.push(Streamed::Text(ready));
            }
            output.push(Streamed::Stopped);
            return output;
        }
        // `held` counts characters, so the split is at a character
        // boundary: a byte offset would cut a multi-byte character.
        let held = stops.held();
        let characters = self.pending_text.chars().count();
        let ready_len = characters.saturating_sub(held);
        let split = self
            .pending_text
            .char_indices()
            .nth(ready_len)
            .map(|(index, _)| index)
            .unwrap_or(self.pending_text.len());
        let ready = self.pending_text[..split].to_string();
        self.pending_text = self.pending_text[split..].to_string();
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
