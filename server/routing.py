"""A POST endpoint's route, as the router and each API dialect describe it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .errors import ErrorDialect


@dataclass(frozen=True, slots=True)
class PostRoute:
    """A POST endpoint, as FrontendHandler serves it."""

    # What serves a request: a function of the exchange, body and deadline,
    # or the name of the FrontendHandler method that does. It answers the
    # request, or prepares the generation the request asks for and returns its
    # job and the function that answers with the job's output.
    handler: Callable | str
    # How the endpoint's API answers errors.
    errors: ErrorDialect
    # Whether it answers from the prompt alone: it needs no engine, so the
    # engine's state refuses none of its requests, which hold slots of the
    # token-count gate instead of the request gate.
    prompt_only: bool = False
    # Whether it is an LLM API the queue admits: it submits the engine's
    # work and holds a `--queue-size` slot from its arrival until its
    # response ends, so a request that arrives with every slot taken is
    # refused 503 frontend_overloaded (529 on /v1/systemone). Chat
    # Completions, Responses, Messages, text completions, judgments and
    # systemone are these. The prompt tools keep their own preparation gate
    # and a control endpoint answers at once, so neither holds a slot.
    queued: bool = False
    # Whether it takes no fields: a request carries no body or an empty JSON
    # object.
    fieldless: bool = False
