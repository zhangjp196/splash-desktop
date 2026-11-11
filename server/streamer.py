"""Incremental detokenization of a generation's tokens into callback text.

The streamer decodes each token as the engine sends it, holds back the text
a stop sequence may still change, and reports what is decided through one
callback, so a stream or a collector receives output as it is generated.
"""

from tokenizers.decoders import DecodeStream

from .output import StopSearch


class CallbackStreamer:
    def __init__(self, tokenizer, callback, stop_sequences=(), on_stop=None):
        self.tokenizer = tokenizer
        self.callback = callback
        self.stops = StopSearch(stop_sequences) if stop_sequences else None
        self.on_stop = on_stop
        self.backend = tokenizer.backend_tokenizer
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
                self._emit(text)
                if self.stop_sequence is not None:
                    break

    def end(self):
        if self.stop_sequence is not None:
            return
        decoded = self.tokenizer.decode(
            self.token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
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

    def count_reasoning_tokens(self, chat_format, thinking, tool_calls=False):
        """The reasoning's tokens, as ReasoningSplitter reads the text: from
        the start of the output where the generation prompt opened the
        reasoning (thinking), or after the model's own opening of it
        (chat_format.think_start_id), to its close or, where a call may follow
        and a call's opening ends the reasoning, a call's opening; none for
        a request without a chat format."""
        if chat_format is None:
            return 0
        if thinking:
            start = 0
        elif chat_format.think_start_id is not None and self.token_ids[:1] == [
            chat_format.think_start_id
        ]:
            start = 1
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
