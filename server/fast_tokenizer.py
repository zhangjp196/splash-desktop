"""The tokenizer the server serves by, read straight from its files.

A model's ``tokenizer`` directory holds ``tokenizer.json``, the chat
template files and ``tokenizer_config.json``. `FastTokenizer` loads the
first with the `tokenizers` library and wraps it with the few calls the
server makes: encode and decode as requests and the engine read them, the
chat template's string rendering, and the vocabulary and special-token
lookups validation and constraints read. No `transformers`, whose import
and its own tokenizer wrapper the server does not need.
"""

import datetime
import json
from copy import copy
from functools import lru_cache
from pathlib import Path

from jinja2.exceptions import TemplateError
from jinja2.ext import Extension
from jinja2.nodes import Scope
from jinja2.sandbox import ImmutableSandboxedEnvironment
from llguidance import LLTokenizer
from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream

__all__ = ["FastTokenizer", "DecodeStream", "TemplateError", "guidance_tokenizer"]


class FastTokenizer:
    """A `tokenizers.Tokenizer` with the surface the server reads tokenizers
    by: the calls transformers' fast tokenizer answered, without
    transformers."""

    def __init__(self, tokenizer, chat_template, eos_token_id):
        self._tokenizer = tokenizer
        self._chat_template = chat_template
        self._eos_token_id = eos_token_id

    @classmethod
    def from_pretrained(cls, directory):
        """Load a model's ``tokenizer`` directory: `tokenizer.json` for the
        vocabulary and merges, the chat template files for its templates,
        and `tokenizer_config.json` for the EOS token."""
        directory = Path(directory)
        tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        chat_template = _load_chat_templates(directory)
        config = _read_json(directory / "tokenizer_config.json")
        eos = config.get("eos_token") if config else None
        if isinstance(eos, dict):
            eos = eos.get("content")
        eos_token_id = tokenizer.token_to_id(eos) if isinstance(eos, str) else None
        return cls(tokenizer, chat_template, eos_token_id)

    # The tokenizers library itself, which the streamer and the prompt
    # tokenizer read directly.

    @property
    def backend_tokenizer(self):
        return self._tokenizer

    # Encoding and decoding, as transformers spelled them.

    def __call__(self, text, add_special_tokens=False, **_):
        return {"input_ids": self.encode(text, add_special_tokens)}

    def encode(self, text, add_special_tokens=False, **_):
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(
        self,
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
        **_,
    ):
        # clean_up_tokenization_spaces=False is transformers' identity; the
        # tokenizers library never applies the cleanup.
        return self._tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )

    # Vocabulary and special tokens, as validation and constraints read them.

    def get_vocab(self):
        return self._tokenizer.get_vocab()

    def convert_tokens_to_ids(self, token):
        return self._tokenizer.token_to_id(token)

    @property
    def eos_token_id(self):
        return self._eos_token_id

    @property
    def added_tokens_decoder(self):
        return self._tokenizer.get_added_tokens_decoder()

    # The chat template, rendered as transformers renders it: the same
    # environment, filters and globals, the string path alone.

    @property
    def chat_template(self):
        return self._chat_template

    def apply_chat_template(
        self,
        messages,
        *,
        chat_template=None,
        tools=None,
        documents=None,
        add_generation_prompt=False,
        tokenize=False,
        **kwargs,
    ):
        if tokenize:
            raise ValueError(
                "the server renders chat templates as text and tokenizes the"
                " result itself: apply_chat_template takes tokenize=False"
            )
        source = _select_chat_template(self._chat_template, chat_template, tools)
        template = _compile_chat_template(source)
        return template.render(
            messages=messages,
            tools=tools,
            documents=documents,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )


def guidance_tokenizer(tokenizer, *, n_vocab=None, eos_token=None, slices=None):
    """The llguidance tokenizer for a FastTokenizer, serialized as
    llguidance's own transformers bridge serializes a fast tokenizer: a
    private copy, without padding or truncation, as JSON. An expensive
    call; the constraint factory builds it once."""
    backend = copy(tokenizer.backend_tokenizer)
    backend.no_padding()
    backend.no_truncation()
    return LLTokenizer(
        backend.to_str(), n_vocab=n_vocab, eos_token=eos_token, slices=slices
    )


def _select_chat_template(defined, requested, tools):
    """The template a request renders by: its own source, else the named
    template its name or the tools pick, else the default — as
    transformers' get_chat_template chooses."""
    if requested is not None:
        if isinstance(defined, dict) and requested in defined:
            return defined[requested]
        return requested
    if isinstance(defined, dict):
        if tools is not None and "tool_use" in defined:
            return defined["tool_use"]
        if "default" in defined:
            return defined["default"]
        raise TemplateError(
            "the model has multiple chat templates with no default:"
            f" {sorted(defined.keys())}"
        )
    if defined is None:
        raise TemplateError("the model's tokenizer defines no chat template")
    return defined


class _Generation(Extension):
    """transformers' ``{% generation %}`` block, which bounds the tokens the
    model itself generates. The string path renders the block's body as a
    plain scope."""

    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return Scope(body, lineno=lineno)


@lru_cache
def _compile_chat_template(source):
    def raise_exception(message):
        raise TemplateError(message)

    def tojson(
        value, ensure_ascii=False, indent=None, separators=None, sort_keys=False
    ):
        # Jinja's own filter escapes HTML characters; templates read JSON.
        return json.dumps(
            value,
            ensure_ascii=ensure_ascii,
            indent=indent,
            separators=separators,
            sort_keys=sort_keys,
        )

    def strftime_now(format):
        return datetime.datetime.now().strftime(format)

    environment = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=["jinja2.ext.loopcontrols", _Generation],
    )
    environment.filters["tojson"] = tojson
    environment.globals["raise_exception"] = raise_exception
    environment.globals["strftime_now"] = strftime_now
    return environment.from_string(source)


def _load_chat_templates(directory):
    """The tokenizer's chat templates: `chat_template.jinja` as the default,
    `additional_chat_templates/*.jinja` as named templates beside it, else
    the legacy `chat_template` entry of `tokenizer_config.json`. Files take
    priority over the config, as transformers reads them."""
    templates = {}
    default = directory / "chat_template.jinja"
    if default.is_file():
        templates["default"] = default.read_text(encoding="utf-8")
    extra = directory / "additional_chat_templates"
    if extra.is_dir():
        for file in sorted(extra.glob("*.jinja")):
            templates[file.name.removesuffix(".jinja")] = file.read_text(
                encoding="utf-8"
            )
    if len(templates) == 1 and "default" in templates:
        return templates["default"]
    if templates:
        return templates
    config = _read_json(directory / "tokenizer_config.json")
    chat_template = config.get("chat_template") if config else None
    if isinstance(chat_template, (list, tuple)):
        chat_template = {entry["name"]: entry["template"] for entry in chat_template}
    return chat_template


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return None
