"""The request fields every generation API reads alike."""

from __future__ import annotations

from . import protocol as wire
from .errors import APIError


def progress(body):
    """The request flag of a generation request's return_progress, which only
    a stream takes."""
    value = body.get("return_progress", False)
    if not isinstance(value, bool) or (value and body.get("stream") is not True):
        raise APIError(
            400, "return_progress requires stream: true and must be a boolean"
        )
    return wire.RequestFlag.RETURN_PROGRESS if value else wire.RequestFlag(0)


def stream_options(body):
    """A Chat or text completion request's stream and its stream_options."""
    stream = body.get("stream", False)
    options = body.get("stream_options", {})
    if not isinstance(stream, bool) or not isinstance(options, dict):
        raise APIError(400, "invalid streaming options")
    # A null include_usage is one left out, as a null field is.
    include_usage = options.get("include_usage")
    if include_usage is None:
        include_usage = False
    if not isinstance(include_usage, bool):
        raise APIError(400, "invalid streaming options")
    return stream, {"include_usage": include_usage}
