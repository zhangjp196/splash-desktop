"""The job a prepared request becomes, and the result the backend reports.

A Job is one request's full serving state, from preparation through the
engine's answer; NativeResult is what a completed job reports to the APIs
that render responses from it.
"""

import functools
import queue
import threading
import time
from dataclasses import dataclass, field

from . import protocol as wire
from .chat_format import ChatFormat
from .constraints import TokenConstraint
from .latency import RequestLatency
from .metrics import metrics_dict
from .tool_schema import ToolPolicy


@dataclass(frozen=True)
class CacheInfo:
    status: str = "unknown"
    matched_tokens: int = 0
    lane: int = -1
    # Tokens whose KV the request read from cached blocks another request
    # computed, past its cached prefix; known once it is done.
    deduplicated_tokens: int = 0


@dataclass(frozen=True)
class ReasoningRequest:
    """How a chat request asked to reason: the reasoning effort it sent, None
    when absent; the effort its template rendered with, the server's default
    for an absent one and None for the template's own; and its
    chat_template_kwargs enable_thinking, None when unset."""

    sent: str | None
    effort: str | None
    enable_thinking: bool | None


@dataclass
class NativeResult:
    reason: str
    prompt_tokens: int
    completion_tokens: int
    start_to_first_token_ms: float
    first_token_to_done_ms: float
    request_wall_ms: float
    prefill_tokens: int = 0
    cache: CacheInfo = field(default_factory=CacheInfo)
    stop_sequence: str | None = None
    first_token_batch_tokens: int = 0
    # Raw option logits for score-only jobs, in requested token order.
    option_logits: tuple = ()
    drafted_tokens: int = 0
    accepted_draft_tokens: int = 0
    draft_cycles: int = 0
    # The engine ended the response at a bound of its own, short of
    # max_new_tokens: what the SSD cache or memory holds of its context.
    bounded: bool = False
    # Per generated token: its id, log probability and top (token, logprob)
    # pairs, aligned with the streamer's token ids.
    logprobs: tuple = ()

    @functools.cached_property
    def metrics(self):
        """The per-request metrics every response and log record reports."""
        return metrics_dict(self)


@dataclass
class Job:
    request_id: int
    prompt_tokens: list
    max_new_tokens: int
    seed: int
    sampling: wire.SamplingParameters
    deadline: float
    priority: wire.RequestPriority = wire.RequestPriority.NORMAL
    stop_sequences: tuple[str, ...] = ()
    thinking: bool = False
    # A chat request's reasoning options, which its Started line shows; None
    # for a request without a chat template.
    reasoning: ReasoningRequest | None = None
    # The format a chat request's output is read in; None for a request
    # without a chat template.
    chat_format: ChatFormat | None = None
    # The thinking tokens the job may select before its thinking closes, set
    # only where that comes before the output limit; None for no cap.
    thinking_token_budget: int | None = None
    thinking_display: str = "summarized"
    reasoning_tokens: int = 0
    events: queue.Queue = field(default_factory=queue.Queue)
    cancelled: threading.Event = field(default_factory=threading.Event)
    tool_policy: ToolPolicy | None = None
    response_validator: object | None = None
    response_format: dict | None = None
    constraint: TokenConstraint | None = None
    cache: CacheInfo = field(default_factory=CacheInfo)
    # Image placeholder spans with their grids and digests, plus the
    # concatenated resized pixels the engine encodes during prefill.
    image_spans: tuple = ()
    image_pixels: bytes = b""
    image_owner: object | None = None
    public_id: str = ""
    created_at: int = field(default_factory=lambda: int(time.time()))
    response_store: bool = False
    # (count, short hash) of the normalized tool list, so the console shows
    # when a client's tool block changes between turns and breaks its prefix.
    tools_signature: tuple | None = None
    response_previous_id: str | None = None
    response_history_items: list | None = None
    # --compact-at: the note a compaction wrote and what it replaced, or None
    # (compaction.Compaction). The response may report it.
    compaction: object | None = None
    # Option token ids for score-only jobs; empty means ordinary generation.
    score_tokens: tuple = ()
    # How many independent choices the request asks for (Chat and text
    # completions' n); one for every other endpoint.
    n: int = 1
    # Tokens above each generated token in its logprobs; 0 disables them.
    top_logprobs: int = 0
    # Return each generated token's log probability (chosen only when
    # top_logprobs is 0).
    logprobs: bool = False
    # Raw-logit adjustments, token id to value.
    logit_bias: tuple = ()
    # Trailing prompt tokens of the chat template's generation prompt; zero
    # when unknown.
    generation_prompt_tokens: int = 0
    flags: wire.RequestFlag = wire.RequestFlag(0)
    # The request asked for more output than the context leaves, and
    # max_new_tokens was lowered to what it leaves.
    output_clamped_to_context: bool = False
    # The digest of a judgment's rendered prompt, which its response reports.
    prompt_sha256: str | None = None
    # The request's cache_salt hashed to 128 bits; zero without one.
    cache_salt: int = 0
    latency: RequestLatency | None = None

    @property
    def may_call_tools(self):
        """Whether the output may call a tool: tools are offered under a
        choice other than none."""
        return self.tool_policy is not None and bool(self.tool_policy.schemas)
