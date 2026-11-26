"""Incremental detokenization of a generation's tokens into callback text.

The streamer decodes each token as the engine sends it, holds back the text
a stop sequence may still change, and reports what is decided through one
callback, so a stream or a collector receives output as it is generated.
"""

from tokenizers.decoders import DecodeStream

from .output import StopSearch


class CallbackStreamer:
    def __init__(self, tokenizer, callback, stop_sequences=(), on_stop=None, visible_tags=()):
        self.tokenizer = tokenizer
        self.callback = callback
        self.stops = StopSearch(stop_sequences) if stop_sequences else None
        self.on_stop = on_stop
        self.backend = tokenizer.backend_tokenizer
        # Special tokens whose text the output holds (ChatFormat.visible_tags).
        self.visible = {
            tokenizer.convert_tokens_to_ids(tag): tag for tag in visible_tags
        }
        # The tokens of the run since the last visible tag, and the text the
        # stream decoded of them.
        self.run = []
        self.run_text = ""
        self.decode_stream = DecodeStream(skip_special_tokens=True)
        self.token_ids = []
        self.emitted = []
        self.pending_text = ""
        self.stop_sequence = None

    def _send(self, text):
        if text:
            self.callback(text)
            self.emitted.append(text)

    def _emit(self, text):
        if not text or self.stop_sequence is not None:
            return
        if self.stops is None:
            self._send(text)
            return
        self.pending_text += text
        found = self.stops.search(text)
        if found is not None:
            self.stop_sequence, back = found
            self._send(self.pending_text[: len(self.pending_text) - back])
            self.pending_text = ""
            if self.on_stop is not None:
                self.on_stop()
            return

        retained = self.stops.held
        ready = self.pending_text[:-retained] if retained else self.pending_text
        self.pending_text = self.pending_text[-retained:] if retained else ""
        self._send(ready)

    def put_tokens(self, token_ids):
        if self.stop_sequence is not None:
            return
        for token_id in token_ids:
            token_id = int(token_id)
            self.token_ids.append(token_id)
            if token_id in self.visible:
                self._end_run()
                self._emit(self.visible[token_id])
                if self.stop_sequence is not None:
                    break
                continue
            if self.visible:
                self.run.append(token_id)
            if self.decode_stream is None:
                continue
            try:
                text = self.decode_stream.step(self.backend, token_id)
            except Exception as error:
                # tokenizers raises a plain Exception with this message when an
                # incremental decode cannot continue; fall back to decoding the
                # whole sequence at the end.
                if not str(error).startswith("Invalid prefix encountered"):
                    raise
                self.decode_stream = None
                continue
            if text:
                if self.visible:
                    self.run_text += text
                self._emit(text)
                if self.stop_sequence is not None:
                    break

    def _end_run(self):
        """Finish the run before a visible tag as _decode decodes it, and
        start a new stream for the run after it."""
        decoded = self._decode_run(self.run)
        if not decoded.startswith(self.run_text):
            raise RuntimeError("incremental tokenizer output diverged")
        self._emit(decoded[len(self.run_text) :])
        self.run, self.run_text = [], ""
        self.decode_stream = DecodeStream(skip_special_tokens=True)

    def end(self):
        if self.stop_sequence is not None:
            return
        decoded = self._decode(self.token_ids)
        handled = "".join(self.emitted) + self.pending_text
        if not decoded.startswith(handled):
            raise RuntimeError("incremental tokenizer output diverged")
        # Bytes of a multi-byte character the output ends inside decode to
        # U+FFFD; DecodeStream held them back for the rest. Drop the trailing
        # U+FFFD, as vLLM's detokenizer does; one that text follows stays.
        self._emit(decoded[len(handled) :].rstrip("\ufffd"))
        if self.stop_sequence is None:
            self._send(self.pending_text)
            self.pending_text = ""

    def _decode(self, token_ids):
        """The text of token_ids, each visible tag as its text between the
        runs of other tokens, decoded without the other special tokens."""
        if not self.visible:
            return self._decode_run(token_ids)
        text, run = [], []
        for token_id in token_ids:
            if token_id in self.visible:
                text += [self._decode_run(run), self.visible[token_id]]
                run = []
            else:
                run.append(token_id)
        return "".join([*text, self._decode_run(run)])

    def _decode_run(self, token_ids):
        return self.tokenizer.decode(
            token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def count_reasoning_tokens(self, chat_format, thinking, tool_calls=False):
        """The reasoning's tokens, as ReasoningSplitter reads the text: from
        the start of the output where the generation prompt opened the
        reasoning (thinking), or after the model's own opening of it
        (chat_format.think_start_ids), to its close or, where a call may follow
        and a call's opening ends the reasoning, a call's opening; where the
        format's reasoning returns after text, those inside each of its
        reasoning blocks; none for a request without a chat format."""
        if chat_format is None:
            return 0
        opening = chat_format.think_start_ids
        if chat_format.reasoning_returns:
            count, index, inside = 0, 0, thinking
            while index < len(self.token_ids):
                if inside:
                    inside = self.token_ids[index] != chat_format.think_end_id
                    count += inside
                    index += 1
                elif (
                    opening
                    and tuple(self.token_ids[index : index + len(opening)]) == opening
                ):
                    inside = True
                    index += len(opening)
                else:
                    index += 1
            return count
        if thinking:
            start = 0
        elif opening and tuple(self.token_ids[: len(opening)]) == opening:
            start = len(opening)
        else:
            return 0
        calls_end = tool_calls and chat_format.calls_end_reasoning
        for index in range(start, len(self.token_ids)):
            token = self.token_ids[index]
            if token == chat_format.think_end_id or (
                calls_end
                and token == chat_format.call_open_id
                and self._opens_call(index, chat_format.call_syntax.call_open)
            ):
                return index - start
        return len(self.token_ids) - start

    def _opens_call(self, index, call_open):
        """Whether the text from the call-open token at `index` on begins
        call_open, decoded from as few tokens as decide it."""
        text = ""
        for end in range(index + 1, len(self.token_ids) + 1):
            text = self.tokenizer.decode(self.token_ids[index:end])
            if len(text) >= len(call_open) or not call_open.startswith(text):
                break
        return text.startswith(call_open)
