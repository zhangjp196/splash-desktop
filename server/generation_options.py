"""The sampling, stop and scheduling options a generation request names.

Every generation API validates its request's numbers alike, through
`from_request`; the table and its bounds are the one source of that.
"""

import hashlib
import secrets
from dataclasses import dataclass

from . import protocol as wire
from .errors import APIError
from .metrics import is_finite_number
from .output import MAX_STOP_CHARACTERS

MIN_FLOAT32_SUBNORMAL = float.fromhex("0x1p-149")
FLOAT32_MAX = float.fromhex("0x1.fffffep127")
# vLLM raises a nonzero temperature below this to it (_MAX_TEMP in
# vllm/sampling_params.py); Splash does the same.
MIN_SAMPLING_TEMPERATURE = 0.01

# The sampling numbers a request may set: each with its default (Qwen's
# generation config for temperature and top_p, and otherwise the value that
# leaves the logits unchanged, as in vLLM), the values the engine accepts and
# how a 400 names them. A nonzero top_p or repetition_penalty must not round
# to zero in the engine's float32, which would make the value invalid.
SAMPLING_NUMBERS = {
    "temperature": (
        1.0,
        lambda value: 0 <= value <= 2,
        "a number in [0, 2]",
    ),
    "top_p": (
        0.95,
        lambda value: MIN_FLOAT32_SUBNORMAL <= value <= 1,
        "a number in (0, 1]",
    ),
    "presence_penalty": (0.0, lambda value: -2 <= value <= 2, "a number in [-2, 2]"),
    "frequency_penalty": (0.0, lambda value: -2 <= value <= 2, "a number in [-2, 2]"),
    "repetition_penalty": (
        1.0,
        lambda value: MIN_FLOAT32_SUBNORMAL <= value <= FLOAT32_MAX,
        "a positive number",
    ),
    "min_p": (0.0, lambda value: 0 <= value <= 1, "a number in [0, 1]"),
}
# top_k's default, from Qwen's generation config as temperature's and top_p's
# are. With it, the table covers every sampling option the frame carries.
TOP_K_DEFAULT = 20
assert set(SAMPLING_NUMBERS) | {"top_k"} == set(wire.SAMPLING_FIELDS)

# The priority a request names, by its lowercase wire name.
_PRIORITIES = {priority.name.lower(): priority for priority in wire.RequestPriority}


@dataclass(frozen=True)
class GenerationOptions:
    """Sampling, stop and scheduling options, validated alike by every
    generation API."""

    sampling: wire.SamplingParameters
    stop_sequences: tuple[str, ...]
    ignore_eos: bool
    seed: int
    priority: wire.RequestPriority
    cache_salt: int
    # The thinking tokens of a request that thinks, after which its thinking
    # closes; None for no cap.
    thinking_token_budget: int | None
    # Raw-logit adjustments, token id to value.
    logit_bias: tuple[tuple[int, float], ...] = ()


def priority(body):
    """The request's scheduling priority, by its lowercase wire name."""
    priority_name = body.get("priority", "normal")
    if not isinstance(priority_name, str) or priority_name not in _PRIORITIES:
        raise APIError(400, "priority must be foreground, normal, or background")
    return _PRIORITIES[priority_name]


def cache_salt(body):
    """The request's cache_salt as the engine keys its cache: 128 bits of
    its hash, zero without one. A request shares cached prompts only with
    requests of the same salt."""
    salt = body.get("cache_salt")
    if salt is None:
        return 0
    if not isinstance(salt, str) or not salt:
        raise APIError(400, "cache_salt must be a non-empty string")
    digest = hashlib.blake2b(
        salt.encode("utf-8", "surrogatepass"), digest_size=16, person=b"splash-salt"
    ).digest()
    return int.from_bytes(digest, "little")


def from_request(frontend, body):
    """The request's GenerationOptions, every field validated."""
    numbers = {}
    for name, (default, accepts, requirement) in SAMPLING_NUMBERS.items():
        value = body.get(name, default)
        if not is_finite_number(value) or not accepts(value):
            raise APIError(400, f"{name} must be {requirement}")
        numbers[name] = float(value)
    if 0 < numbers["temperature"] < MIN_SAMPLING_TEMPERATURE:
        numbers["temperature"] = MIN_SAMPLING_TEMPERATURE
    # A top_k of 0 or -1 keeps every token, which the frame says with 0. A
    # top_k past the vocabulary keeps every token too, so one past the
    # frame's u32 is sent as its largest value.
    top_k = body.get("top_k", TOP_K_DEFAULT)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < -1:
        raise APIError(400, "top_k must be 0 or -1 (disabled) or a positive integer")
    top_k = min(max(top_k, 0), 0xFFFFFFFF)
    stop = body.get("stop")
    if stop in (None, []):
        stop_sequences = ()
    elif isinstance(stop, str) and stop:
        stop_sequences = (stop,)
    elif (
        isinstance(stop, list)
        and 1 <= len(stop) <= 4
        and all(isinstance(value, str) and value for value in stop)
    ):
        stop_sequences = tuple(stop)
    else:
        raise APIError(400, "stop must be a string or up to four strings")
    if len(stop_sequences) > 4:
        raise APIError(400, "stop must be a string or up to four strings")
    if any(len(value) > MAX_STOP_CHARACTERS for value in stop_sequences):
        raise APIError(
            400,
            f"each of stop must be at most {MAX_STOP_CHARACTERS} characters",
        )
    raw_bias = body.get("logit_bias")
    logit_bias = []
    if raw_bias is not None:
        if not isinstance(raw_bias, dict):
            raise APIError(400, "logit_bias must be an object")
        if len(raw_bias) > 256:
            raise APIError(400, "logit_bias takes at most 256 entries")
        for key, value in raw_bias.items():
            try:
                token = int(key)
            except (TypeError, ValueError):
                raise APIError(400, "logit_bias keys must be token ids") from None
            if str(token) != key or not 0 <= token < frontend.chat_format.vocabulary:
                raise APIError(400, f"logit_bias token {key} is outside the vocabulary")
            if (
                isinstance(value, bool)
                or not is_finite_number(value)
                or not -100.0 <= value <= 100.0
            ):
                raise APIError(400, "logit_bias values must be numbers in [-100, 100]")
            logit_bias.append((token, float(value)))
    ignore_eos = body.get("ignore_eos", False)
    if not isinstance(ignore_eos, bool):
        raise APIError(400, "ignore_eos must be a boolean")
    seed = body.get("seed", -1)
    if (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or not -(2**63) <= seed < 2**64
    ):
        raise APIError(400, "seed must be a 64-bit integer")
    # -1 asks for a random seed, and another negative one is taken as
    # its 64 bits.
    seed = secrets.randbits(64) if seed == -1 else seed % 2**64
    # -1 sets no cap; an absent or null budget takes the server's.
    budget = body.get("thinking_token_budget")
    if budget is None:
        budget = frontend.default_thinking_token_budget
    elif not isinstance(budget, int) or isinstance(budget, bool) or budget < -1:
        raise APIError(
            400,
            "thinking_token_budget must be a non-negative integer or -1 for no cap",
        )
    elif budget == -1:
        budget = None
    return GenerationOptions(
        wire.SamplingParameters(top_k=top_k, **numbers),
        stop_sequences,
        ignore_eos,
        seed,
        priority(body),
        cache_salt(body),
        budget,
        tuple(logit_bias),
    )
