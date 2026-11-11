"""Prepare API requests for generation and manage Responses history."""

import hashlib
import json
import secrets
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from itertools import count
from pathlib import Path

from jinja2 import TemplateError

from . import compaction, generation_options, image_prompts, judgments
from . import images as image_input
from . import protocol as wire
from .backend import Job, remaining_request_time
from .chat_format import THINK_END, THINK_START, TOOL_CALL_OPEN
from .chat_messages import (
    IMAGE_PAD_TOKEN,
    normalize_messages,
    template_messages,
)
from .chat_templates import (
    LATER_SYSTEM_UNSUPPORTED,
    RESERVED_TEMPLATE_KWARGS,
    render_chat_template,
    template_options,
)
from .constraints import grammar_for_source
from .diagnostics import print_status
from .errors import (
    APIError,
    ContextLengthError,
    RequestValidationError,
    field_error,
)
from .job import ReasoningRequest
from .latency import LatencyMetrics
from .metrics import is_finite_number
from .response_store import ResponseStore
from .serve_options import REASONING_EFFORTS, parse_served_model_name
from .tokenization import PromptTokenizer, tokenizable
from .tool_schema import (
    ToolPolicy,
    json_grammar,
    normalize_response_format,
    normalize_tools,
    tool_grammar,
)

PREPARATION_WAIT_SECONDS = 30.0


# Text completions' output budget when max_tokens is omitted: OpenAI's
# default for the endpoint, which vLLM and SGLang also use.
COMPLETION_DEFAULT_MAX_TOKENS = 16


def _generation_prompt(probed, rendered, tokens):
    """Whether the generation prompt opens a think block, and how many of the
    prompt's last tokens it is (zero where they differ from its tokens).
    `probed` is its text and tokens, as the startup probe found them for the
    request's template options, and the rendered prompt must end with that
    text."""
    text, ids = probed
    if not text or not rendered.endswith(text):
        raise APIError(
            400, "chat template must end with an assistant generation prefix"
        )
    count = len(ids)
    return (
        text.rfind(THINK_START) > text.rfind(THINK_END),
        count if count < len(tokens) and tuple(tokens[-count:]) == ids else 0,
    )


@dataclass
class Prompt:
    messages: list
    tools: list | None
    tool_policy: ToolPolicy | None
    reasoning_effort: str | None
    response_schema: dict | bool | None = None
    response_validator: object = None
    preserve_thinking: bool | None = None
    # The request's own chat_template_kwargs enable_thinking, which outranks
    # its reasoning effort (template_options); None where it sets none.
    enable_thinking: bool | None = None
    # Its other template variables, which outrank Splash's own.
    template_kwargs: dict = field(default_factory=dict)
    # The reasoning effort the request sent; None where it sent none.
    sent_effort: str | None = None


@dataclass
class RenderedPrompt:
    text: str
    tokens: list[int]
    images: list
    image_positions: list[int]
    thinking: bool
    # Images precede the generation prompt; expanding them keeps this count.
    generation_prompt_tokens: int


class Frontend:
    def __init__(
        self,
        tokenizer,
        backend,
        model,
        max_context,
        request_timeout,
        preparation_capacity,
        *,
        constraint_factory,
        chat_templates,
        chat_format,
        thinking_codec,
        vision,
        max_image_pixels=image_input.MAX_PIXELS,
        served_model_names=(),
        announce_served_name=False,
        default_reasoning_effort=None,
        default_thinking_token_budget=None,
        compact_at=None,
        compact_keep=None,
        compact_memory=None,
    ):
        if not isinstance(preparation_capacity, int) or preparation_capacity <= 0:
            raise ValueError("frontend preparation capacity must be positive")
        # Announced by the engine in its Ready event. Without it, message
        # normalization rejects image and PDF input before any decoding.
        self.vision = vision
        self.latencies = LatencyMetrics()
        self.tokenizer = tokenizer
        # Probed at startup; requests choose among these, never the
        # tokenizer's own.
        self.chat_templates = chat_templates
        # The served family's: how its output reasons and calls tools.
        self.chat_format = chat_format
        self.prompt_tokenizer = PromptTokenizer(tokenizer, chat_format.message_boundary)
        self.backend = backend
        # The model id the engine loaded. /status reports it, so an alias
        # can never hide what served a request (#81).
        self.model = model
        served = tuple(parse_served_model_name(name) for name in served_model_names)
        if announce_served_name and not served:
            raise ValueError("announce_served_name needs a served model name")
        # The name generation and scoring responses report.
        self.response_model = served[0] if announce_served_name else model
        # /v1/models order: the name responses report first, so a client
        # configured from data[0] (splash <client>) requests the name it gets
        # back. Every name stays accepted.
        self.model_names = tuple(dict.fromkeys((self.response_model, model, *served)))
        if (
            default_reasoning_effort is not None
            and default_reasoning_effort not in REASONING_EFFORTS
        ):
            raise ValueError("invalid default_reasoning_effort")
        self.default_reasoning_effort = default_reasoning_effort
        if default_thinking_token_budget is not None and (
            not isinstance(default_thinking_token_budget, int)
            or isinstance(default_thinking_token_budget, bool)
            or default_thinking_token_budget < 0
        ):
            raise ValueError("invalid default_thinking_token_budget")
        self.default_thinking_token_budget = default_thinking_token_budget
        self.max_context = max_context
        # --compact-at: "auto", or the share of the window a prompt may reach
        # before a request compacts; None turns compaction off.
        if (
            compact_at is not None
            and compact_at != "auto"
            and not (
                isinstance(compact_at, (int, float))
                and not isinstance(compact_at, bool)
                and 0.0 < compact_at < 1.0
            )
        ):
            raise ValueError("invalid compact_at")
        self.compact_at = compact_at
        if compact_keep is not None and (
            not isinstance(compact_keep, int)
            or isinstance(compact_keep, bool)
            or compact_keep <= 0
        ):
            raise ValueError("invalid compact_keep")
        if compact_keep is not None and compact_at is None:
            raise ValueError("compact_keep needs compact_at")
        self.compact_keep = compact_keep
        if compact_memory is not None and compact_at is None:
            raise ValueError("compact_memory needs compact_at")
        self.compact_memory = (
            Path(compact_memory) if compact_memory is not None else None
        )
        if self.compact_memory is not None:
            self.compact_memory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.request_timeout = request_timeout
        self.constraint_factory = constraint_factory
        self.max_image_pixels = max_image_pixels
        self.images = image_input.ImageCache()
        self.ids = count(1)
        self.preparation_capacity = preparation_capacity
        self.preparation_slots = threading.BoundedSemaphore(preparation_capacity)
        self.preparation_lock = threading.Lock()
        self.preparation_active = 0
        self.preparation_waiting = 0
        self.response_store = ResponseStore()
        self.thinking_codec = thinking_codec

    def accepts_model(self, model):
        """Whether a request may name `model`: a name the server answers to,
        or none, absent or empty, which means the served model."""
        return model in (None, "") or (
            isinstance(model, str) and model in self.model_names
        )

    @property
    def input_modalities(self):
        """The one list /status, /v1/models and client setup report."""
        return ["text", "image", "pdf"] if self.vision else ["text"]

    @property
    def reasoning(self):
        """What /v1/models reports of reasoning: the default template's
        (ChatTemplate.reasoning), with the server's default effort in place of
        the template's own, named by the listed effort it renders like; no
        efforts for a model that decides itself whether to reason."""
        if not self.chat_format.reasoning_switch:
            return {"thinks": None, "efforts": [], "default": None}
        template = self.chat_templates.select(None)
        reasoning = dict(template.reasoning)
        if (effort := self.default_reasoning_effort) is not None:
            reasoning["thinks"] = effort != "none"
            reasoning["default"] = template.effort_names.get(effort)
        return reasoning

    def status(self, cached=False):
        # The cache is the backend's decision; a probe without the flag is
        # the plain live status it always was.
        status = self.backend.status(cached=True) if cached else self.backend.status()
        status["vision"] = self.vision
        status["input_modalities"] = self.input_modalities
        status["reasoning_switch"] = self.chat_format.reasoning_switch
        status["chat_template"] = self.chat_templates.status()
        with self.preparation_lock:
            status["frontend"] = {
                "preparation_capacity": self.preparation_capacity,
                "active": self.preparation_active,
                "waiting": self.preparation_waiting,
            }
        status["grammar_cache"] = self.constraint_factory.stats()
        status["response_store"] = self.response_store.stats()
        status["image_cache"] = self.images.stats()
        status["tokenizer_cache"] = self.prompt_tokenizer.stats()
        status["latency"] = self.latencies.snapshot()
        return status

    def _prepare_images(self, messages, deadline, *, check_context=True):
        return image_prompts.prepare_images(
            self, messages, deadline, check_context=check_context
        )

    def _check_image_request_size(
        self,
        tokens,
        image_count,
        pixel_bytes,
        *,
        check_context=True,
        image_tokens_only=False,
    ):
        return image_prompts.check_image_request_size(
            self,
            tokens,
            image_count,
            pixel_bytes,
            check_context=check_context,
            image_tokens_only=image_tokens_only,
        )

    def _render_image_tokens(self, messages, template):
        return image_prompts.render_image_tokens(self, messages, template)

    def _image_token_count(self, prompt_tokens, prepared, positions):
        return image_prompts.image_token_count(self, prompt_tokens, prepared, positions)

    def _expand_image_pads(self, prompt_tokens, prepared, positions):
        return image_prompts.expand_image_pads(self, prompt_tokens, prepared, positions)

    def request_deadline(self, body, started_at):
        timeout = body.get("timeout")
        if timeout is None:
            timeout = self.request_timeout
        elif not is_finite_number(timeout) or timeout <= 0:
            raise RequestValidationError(
                [field_error(["timeout"], "timeout must be positive")]
            )
        return started_at + min(timeout, self.request_timeout)

    def prepare(
        self,
        body,
        *,
        deadline,
        output_field=None,
        thinking_display="summarized",
    ):
        """A Chat request, or a Messages request converted to Chat.
        Output limit errors name output_field, that API's own field.
        thinking_display "omitted" hides Messages reasoning behind a
        signature."""
        with self._preparation(deadline) as without_slot:
            return self._prepare(
                body,
                None,
                deadline,
                without_slot,
                output_field=output_field,
                thinking_display=thinking_display,
            )

    def preparation(self, deadline):
        """The preparation slot a dialect prepares under (prepare)."""
        return self._preparation(deadline)

    def chat_job(
        self,
        body,
        tool_namespaces,
        deadline,
        without_slot,
        *,
        output_field,
        thinking_display,
    ):
        """The generation job of a Chat-shaped body a dialect has read."""
        return self._prepare(
            body,
            tool_namespaces,
            deadline,
            without_slot,
            output_field=output_field,
            thinking_display=thinking_display,
        )

    def prepare_completion(self, body, *, deadline):
        """A text completion: the prompt generates as given, with no chat
        template, reasoning split, tools or images."""
        with self._preparation(deadline):
            if not self.accepts_model(body.get("model")):
                raise APIError(
                    404, f"model {body['model']} not found", "model_not_found"
                )
            # A request generates one text and returns only that text.
            for field in ("suffix", "logprobs"):
                if field in body:
                    raise APIError(400, f"{field} is not supported")
            if body.get("echo", False) is not False:
                raise APIError(400, "echo is not supported")
            for field in ("best_of",):
                value = body.get(field, 1)
                if not isinstance(value, int) or isinstance(value, bool) or value != 1:
                    raise APIError(400, f"{field} must be 1")
            n = body.get("n", 1)
            if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 4:
                raise APIError(400, "n must be an integer from 1 to 4")
            options = self._generation_options(body)
            # A text completion has no reasoning split, so no thinking is
            # tracked to close.
            if body.get("thinking_token_budget") not in (None, -1):
                raise APIError(
                    400, "thinking_token_budget is not supported for text completions"
                )
            grammar_source = body.get("grammar")
            regex_source = body.get("regex")
            if grammar_source is not None and regex_source is not None:
                raise APIError(400, "grammar and regex cannot be combined")
            for field_name, value in (
                ("grammar", grammar_source),
                ("regex", regex_source),
            ):
                if value is not None and (
                    not isinstance(value, str) or not value.strip()
                ):
                    raise APIError(400, f"{field_name} must be a nonempty string")
            if options.ignore_eos and (
                grammar_source is not None or regex_source is not None
            ):
                raise APIError(
                    400, "ignore_eos cannot be combined with grammar or regex"
                )
            prompt_tokens = self._completion_prompt(body.get("prompt"))
            remaining_request_time(deadline)
            constraint = None
            if grammar_source is not None or regex_source is not None:
                with self.latencies.measure("grammar"):
                    if grammar_source is not None:
                        grammar = grammar_for_source("gbnf", grammar_source)
                    else:
                        grammar = grammar_for_source("regex", regex_source)
                    constraint = self.constraint_factory.create(
                        grammar, timeout=remaining_request_time(deadline)
                    )
            requested = body.get("max_tokens")
            max_new = self._output_budget(
                COMPLETION_DEFAULT_MAX_TOKENS if requested is None else requested,
                prompt_tokens,
                "max_tokens",
            )
            job = self._generation_job(
                options, prompt_tokens, max_new, deadline, constraint=constraint
            )
            job.n = n
            return job

    def _completion_prompt(self, prompt):
        """A string encodes as a raw prompt, with the tokenizer's own special
        tokens such as a BOS; token ids must be in its vocabulary."""
        if isinstance(prompt, str):
            try:
                tokens = self._tokenize(prompt, add_special_tokens=True)["input_ids"]
            except Exception as error:
                raise APIError(400, "prompt could not be tokenized") from error
        elif isinstance(prompt, list) and all(type(token) is int for token in prompt):
            vocabulary = len(self.tokenizer)
            if any(not 0 <= token < vocabulary for token in prompt):
                raise APIError(400, "prompt token ids must be in the vocabulary")
            tokens = prompt
        else:
            raise APIError(400, "prompt must be one string or one array of token ids")
        if not tokens:
            raise APIError(400, "prompt must not be empty")
        return list(tokens)

    def count_tokens(self, body, *, deadline):
        with self._preparation(deadline) as without_slot:
            prompt = self._prepare_prompt(
                body, None, deadline=deadline, without_slot=without_slot
            )
            rendered = self._render_prompt(prompt, deadline, check_context=False)
            return self._image_token_count(
                rendered.tokens, rendered.images, rendered.image_positions
            )

    def tokenize(self, body, *, deadline):
        content = body.get("content")
        if not isinstance(content, str):
            raise APIError(400, "content must be a string")
        add_special = body.get("add_special", False)
        if not isinstance(add_special, bool):
            raise APIError(400, "add_special must be a boolean")
        for option, supported in (("parse_special", True), ("with_pieces", False)):
            if body.get(option, supported) is not supported:
                raise APIError(
                    400, f"only {option}={str(supported).lower()} is supported"
                )
        with self._preparation(deadline):
            try:
                tokens = self._tokenize(content, add_special_tokens=add_special)[
                    "input_ids"
                ]
            except Exception as error:
                raise APIError(400, "content could not be tokenized") from error
            remaining_request_time(deadline)
            return tokens

    def detokenize(self, body, *, deadline):
        """Token ids back to text, as llama-server's /detokenize does."""
        tokens = body.get("tokens")
        if not isinstance(tokens, list) or any(
            not isinstance(token, int) or isinstance(token, bool) for token in tokens
        ):
            raise APIError(400, "tokens must be an array of token ids")
        if any(not 0 <= token < self.chat_format.vocabulary for token in tokens):
            raise APIError(400, "a token id is outside the vocabulary")
        special = body.get("special", False)
        if not isinstance(special, bool):
            raise APIError(400, "special must be a boolean")
        with self._preparation(deadline):
            try:
                text = self.tokenizer.decode(tokens, skip_special_tokens=not special)
            except Exception as error:
                raise APIError(400, "tokens could not be detokenized") from error
            remaining_request_time(deadline)
            return text

    def prepare_judgment(self, body, *, deadline, disconnected):
        return judgments.prepare_judgment(
            self, body, deadline=deadline, disconnected=disconnected
        )

    def prepare_systemone(self, body, *, deadline, disconnected):
        return judgments.prepare_systemone(
            self, body, deadline=deadline, disconnected=disconnected
        )

    def apply_template(self, body, *, deadline):
        add_generation_prompt = body.get("add_generation_prompt", True)
        if not isinstance(add_generation_prompt, bool):
            raise APIError(400, "add_generation_prompt must be a boolean")
        with self._preparation(deadline) as without_slot:
            prompt = self._prepare_prompt(
                body, None, deadline=deadline, without_slot=without_slot
            )
            return self._render_prompt(
                prompt,
                deadline,
                check_context=False,
                add_generation_prompt=add_generation_prompt,
            ).text

    @contextmanager
    def _preparation(self, deadline):
        """Hold a preparation slot while the request prepares. Yields a
        context manager under which the request holds none: it gives its
        slot back to wait for and use the PDF worker, which slots do not
        bound, and takes one again after, unless it failed meanwhile."""
        self._take_preparation_slot(deadline)
        held = True

        @contextmanager
        def without_slot():
            nonlocal held
            self._give_preparation_slot()
            held = False
            yield
            self._take_preparation_slot(deadline)
            held = True

        try:
            remaining_request_time(deadline)
            with self.latencies.measure("preparation"):
                yield without_slot
        finally:
            if held:
                self._give_preparation_slot()

    def _take_preparation_slot(self, deadline):
        remaining = remaining_request_time(deadline)
        with self.preparation_lock:
            self.preparation_waiting += 1
        with self.latencies.measure("preparation_queue"):
            acquired = self.preparation_slots.acquire(
                timeout=min(remaining, PREPARATION_WAIT_SECONDS)
            )
        with self.preparation_lock:
            self.preparation_waiting -= 1
            if acquired:
                self.preparation_active += 1
        if not acquired:
            remaining_request_time(deadline)
            raise APIError(
                503,
                "frontend preparation capacity is exhausted",
                "frontend_overloaded",
            )

    def _give_preparation_slot(self):
        with self.preparation_lock:
            self.preparation_active -= 1
        self.preparation_slots.release()

    def _prepare_prompt(self, body, tool_namespaces, *, deadline, without_slot):
        if not self.accepts_model(body.get("model")):
            raise APIError(404, f"model {body['model']} not found", "model_not_found")
        reasoning_effort = body.get("reasoning_effort")
        if reasoning_effort is None:
            reasoning_effort = self.default_reasoning_effort
        if reasoning_effort is not None and (
            not isinstance(reasoning_effort, str)
            or reasoning_effort not in REASONING_EFFORTS
        ):
            raise APIError(400, "invalid reasoning_effort")
        preserve_thinking = body.get("preserve_thinking")
        if preserve_thinking is not None and not isinstance(preserve_thinking, bool):
            raise APIError(400, "preserve_thinking must be a boolean")
        template_kwargs = body.get("chat_template_kwargs")
        if template_kwargs is None:
            template_kwargs = {}
        elif not isinstance(template_kwargs, dict):
            raise APIError(400, "chat_template_kwargs must be an object")
        elif reserved := sorted(RESERVED_TEMPLATE_KWARGS & template_kwargs.keys()):
            raise APIError(400, f"chat_template_kwargs cannot set {reserved[0]}")
        template_kwargs = dict(template_kwargs)
        enable_thinking = template_kwargs.pop("enable_thinking", None)
        if enable_thinking is not None and not isinstance(enable_thinking, bool):
            raise APIError(
                400, "chat_template_kwargs enable_thinking must be a boolean"
            )
        messages = template_messages(
            normalize_messages(
                body.get("messages"),
                vision=self.vision,
                deadline=deadline,
                without_slot=without_slot,
            )
        )
        tools, tool_policy = normalize_tools(
            body.get("tools"),
            body.get("tool_choice"),
            body.get("parallel_tool_calls", True),
            tool_namespaces,
        )
        response_schema, response_validator = normalize_response_format(
            body.get("response_format")
        )
        return Prompt(
            messages,
            tools,
            tool_policy,
            reasoning_effort,
            response_schema,
            response_validator,
            preserve_thinking,
            enable_thinking,
            template_kwargs,
            body.get("reasoning_effort"),
        )

    def _tokenize(self, text, **options):
        with self.latencies.measure("tokenization"):
            return self.tokenizer(tokenizable(text), **options)

    def _apply_chat_template(self, messages, template):
        with self.latencies.measure("template"):
            return render_chat_template(self.tokenizer, messages, template)

    def _render_prompt(
        self, prompt, deadline, *, check_context=True, add_generation_prompt=True
    ):
        chat_template = self.chat_templates.select(prompt.tools)
        if not chat_template.accepts(prompt.messages):
            raise APIError(400, LATER_SYSTEM_UNSUPPORTED)
        template = {
            "tokenize": False,
            "return_dict": False,
            "chat_template": chat_template.source,
            **template_options(
                reasoning_effort=prompt.reasoning_effort,
                preserve_thinking=prompt.preserve_thinking,
                tools=prompt.tools,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=prompt.enable_thinking,
            ),
            **prompt.template_kwargs,
        }
        with self.latencies.measure("images"):
            images = self._prepare_images(
                prompt.messages, deadline, check_context=check_context
            )
        remaining_request_time(deadline)
        if images and self.tokenizer.convert_tokens_to_ids(IMAGE_PAD_TOKEN) is None:
            raise APIError(400, "the tokenizer does not define the image pad token")
        positions = []
        try:
            if images:
                tokens, positions, rendered = self._render_image_tokens(
                    prompt.messages, template
                )
            else:
                rendered = self._apply_chat_template(prompt.messages, template)
                with self.latencies.measure("tokenization"):
                    tokens = self.prompt_tokenizer.encode(rendered)
        except APIError:
            raise
        except Exception as error:
            frame = error.__traceback__
            template_frame = None
            while True:
                if frame.tb_frame.f_code.co_filename == "<template>":
                    template_frame = frame
                if frame.tb_next is None:
                    break
                frame = frame.tb_next
            frame = template_frame or frame
            location = Path(frame.tb_frame.f_code.co_filename).name
            print_status(
                f"Template error · {type(error).__name__} · {location}:{frame.tb_lineno}",
                error=True,
            )
            # The client reads a template's own refusal, such as a system
            # message it does not take; other errors may carry internals, and
            # the log keeps no request content.
            reason = f": {error}" if isinstance(error, TemplateError) else ""
            raise APIError(400, f"messages could not be rendered{reason}") from error
        remaining_request_time(deadline)
        thinking, generation_prompt_tokens = False, 0
        if add_generation_prompt:
            thinking, generation_prompt_tokens = _generation_prompt(
                chat_template.generation_prompt(template), rendered, tokens
            )
        return RenderedPrompt(
            rendered, tokens, images, positions, thinking, generation_prompt_tokens
        )

    def _prepare(
        self,
        body,
        tool_namespaces,
        deadline,
        without_slot,
        *,
        output_field,
        thinking_display,
    ):
        # Fields checked without the prompt fail before it is prepared, which
        # can render documents and build validators.
        options = self._generation_options(body)
        grammar_source = body.get("grammar")
        regex_source = body.get("regex")
        if grammar_source is not None and regex_source is not None:
            raise APIError(400, "grammar and regex cannot be combined")
        for field_name, value in (
            ("grammar", grammar_source),
            ("regex", regex_source),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise APIError(400, f"{field_name} must be a nonempty string")
        n = body.get("n", 1)
        logprobs = body.get("logprobs", False)
        top_logprobs = body.get("top_logprobs", 0)
        if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 4:
            raise APIError(400, "n must be an integer from 1 to 4")
        if not isinstance(logprobs, bool):
            raise APIError(400, "logprobs must be a boolean")
        if (
            not isinstance(top_logprobs, int)
            or isinstance(top_logprobs, bool)
            or not 0 <= top_logprobs <= 20
        ):
            raise APIError(400, "top_logprobs must be an integer from 0 to 20")
        if top_logprobs and not logprobs:
            raise APIError(400, "top_logprobs requires logprobs")
        if logprobs and not options.cache_salt:
            # A cold prompt keeps the first anchor's logits row, which a
            # restored prefix does not carry: a salt of its own gives a
            # logprobs request that exact row.
            options = replace(options, cache_salt=secrets.randbits(128))
        prompt = self._prepare_prompt(
            body, tool_namespaces, deadline=deadline, without_slot=without_slot
        )
        tools, tool_policy = prompt.tools, prompt.tool_policy
        response_schema, response_validator = (
            prompt.response_schema,
            prompt.response_validator,
        )
        # Constrained tool calls and structured output generate under a
        # grammar, which decides where the output ends; tools beside a
        # response schema share one. A request's own GBNF or regex constrains
        # the whole output instead, and composes with neither.
        tool_constrained = tool_policy is not None and (
            tool_policy.constrained or response_schema is not None
        )
        grammar_constrained = grammar_source is not None or regex_source is not None
        if grammar_constrained and (
            tools or tool_policy is not None or response_schema is not None
        ):
            raise APIError(
                400,
                "grammar and regex cannot be combined with tools or response_format",
            )
        constrained = (
            tool_constrained or response_schema is not None or grammar_constrained
        )
        if options.ignore_eos and constrained:
            raise APIError(
                400,
                "ignore_eos cannot be combined with grammar or regex"
                if grammar_constrained
                else "ignore_eos cannot be combined with constrained tool calls, "
                "tool_choice none or structured output",
            )
        rendered = self._render_prompt(prompt, deadline)
        requested = body.get("max_completion_tokens", body.get("max_tokens"))
        if output_field is None:
            # Chat takes either field; errors name the one the client sent.
            output_field = (
                "max_completion_tokens"
                if "max_completion_tokens" in body
                else "max_tokens"
            )
        # --compact-at: a prompt that leaves no room for the reply becomes the
        # newest turns plus a note (compaction.py).
        prompt, rendered, compaction_report = self._compact(
            prompt, rendered, deadline, requested
        )
        prompt_tokens, prepared_images = rendered.tokens, rendered.images
        image_positions, thinking = rendered.image_positions, rendered.thinking
        if grammar_constrained and thinking:
            raise APIError(
                400,
                "grammar and regex require reasoning off; set reasoning_effort to none",
            )
        if logprobs and thinking:
            raise APIError(
                400,
                "logprobs require reasoning off; set reasoning_effort to none",
            )
        if logprobs and (constrained or tools):
            raise APIError(
                400,
                "logprobs cannot be combined with tools, structured output, "
                "grammar or regex",
            )
        constraint = None
        remaining_request_time(deadline)
        if constrained:
            with self.latencies.measure("grammar"):
                if grammar_source is not None:
                    constraint = self.constraint_factory.create(
                        grammar_for_source("gbnf", grammar_source),
                        timeout=remaining_request_time(deadline),
                    )
                elif regex_source is not None:
                    constraint = self.constraint_factory.create(
                        grammar_for_source("regex", regex_source),
                        timeout=remaining_request_time(deadline),
                    )
                elif tool_constrained:
                    constraint = self.constraint_factory.create(
                        tool_grammar(
                            tool_policy,
                            thinking,
                            response_schema,
                            chat_format=self.chat_format,
                        ),
                        timeout=remaining_request_time(deadline),
                        prefixes=lambda: self._call_openings(tool_policy, thinking),
                    )
                else:
                    constraint = self.constraint_factory.create(
                        json_grammar(
                            response_schema, thinking, chat_format=self.chat_format
                        ),
                        timeout=remaining_request_time(deadline),
                    )
        remaining_request_time(deadline)
        tools_signature = None
        if tools:
            # The signature needs key-sorted canonical JSON, which the
            # strict codec's encoder does not offer; the tools are request
            # JSON that json_codec.loads already validated, so stdlib
            # encoding here is only a digest, never a payload.
            digest = hashlib.sha1(
                json.dumps(tools, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()[:8]
            tools_signature = (len(tools), digest)
        image_spans, image_pixels = (), b""
        if prepared_images:
            prompt_tokens, image_spans, image_pixels = self._expand_image_pads(
                prompt_tokens, prepared_images, image_positions
            )
        remaining_request_time(deadline)
        max_new = self._output_budget(requested, prompt_tokens, output_field)
        # A budget the output limit reaches first closes nothing. Where the
        # context leaves less room than the output limit, a budget larger
        # than that room keeps its share of it, and the answer keeps the
        # rest: 24,576 of 32,768 with 20,000 tokens left ends thinking at
        # 15,000 rather than taking all of it.
        budget = options.thinking_token_budget
        if not thinking or budget is None:
            budget = None
        elif requested is not None and max_new < requested and budget > max_new:
            budget = max(1, budget * max_new // requested)
        if budget is not None and budget >= max_new:
            budget = None
        job = self._generation_job(
            options,
            prompt_tokens,
            max_new,
            deadline,
            thinking=thinking,
            reasoning=ReasoningRequest(
                prompt.sent_effort, prompt.reasoning_effort, prompt.enable_thinking
            ),
            chat_format=self.chat_format,
            thinking_token_budget=budget,
            thinking_display=thinking_display,
            tool_policy=tool_policy,
            response_validator=response_validator,
            response_format=body.get("response_format"),
            constraint=constraint,
            image_spans=image_spans,
            image_pixels=image_pixels,
            image_owner=prepared_images if prepared_images else None,
            tools_signature=tools_signature,
            generation_prompt_tokens=rendered.generation_prompt_tokens,
            output_clamped_to_context=requested is not None and max_new < requested,
        )
        job.compaction = compaction_report
        job.n = n
        job.logprobs = logprobs
        job.top_logprobs = top_logprobs
        return job

    def _call_openings(self, policy, thinking):
        """The tokens that begin each strict tool's call, after the close of
        the reasoning the generation prompt opened. Its parameter names may
        come next, so a tool with more of them than the parser admits fails
        there."""
        chat_format = self.chat_format
        reasoning = [chat_format.think_end_id] if thinking else []
        return [
            (
                reasoning
                + self._tokenize(
                    TOOL_CALL_OPEN + chat_format.call_syntax.function_opening(name),
                    add_special_tokens=False,
                )["input_ids"],
                f"tool {name} has too many parameters to constrain",
            )
            for name in policy.schemas
            if name in policy.strict
        ]

    def _generation_options(self, body):
        return generation_options.from_request(self, body)

    def _output_budget(self, requested, prompt_tokens, field):
        """The output token budget requested under the API's field name. It is
        a ceiling: a request generates up to the end of the context window
        its prompt leaves, all of it when the request names no budget."""
        if len(prompt_tokens) >= self.max_context:
            raise ContextLengthError(len(prompt_tokens), self.max_context - 1)
        remaining = self.max_context - len(prompt_tokens)
        max_new = remaining if requested is None else requested
        if not isinstance(max_new, int) or isinstance(max_new, bool) or max_new <= 0:
            raise APIError(400, f"{field} must be a positive integer")
        return min(max_new, remaining)

    def _compact(self, prompt, rendered, deadline, requested):
        return compaction.compact(self, prompt, rendered, deadline, requested)

    def _generation_job(self, options, prompt_tokens, max_new, deadline, **fields):
        """The job for a prepared prompt, with the endpoint's own fields."""
        return Job(
            request_id=next(self.ids),
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new,
            seed=options.seed,
            sampling=options.sampling,
            deadline=deadline,
            priority=options.priority,
            cache_salt=options.cache_salt,
            stop_sequences=options.stop_sequences,
            logit_bias=options.logit_bias,
            flags=(
                wire.RequestFlag.IGNORE_END_OF_SEQUENCE
                if options.ignore_eos
                else wire.RequestFlag(0)
            ),
            public_id=secrets.token_hex(16),
            **fields,
        )

    def persist_response(self, job, response, output):
        if not job.response_store:
            return
        # The store encodes the history at once, so later changes to the
        # request's items or the output cannot reach it.
        history = [*job.response_history_items, *output]
        if not self.response_store.put(response, history):
            response["store"] = False
            job.response_store = False
