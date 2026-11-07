"""OpenAI's Chat and text completions: their answers."""

from __future__ import annotations

from functools import partial

from .. import request_fields
from ..chat_messages import compaction_dict
from ..errors import OPENAI_ERRORS, APIError
from ..metrics import timings_dict, usage_dict
from ..routing import PostRoute


def finish_reason(result, called):
    """OpenAI finish_reason: the output's whole tool calls count only when
    the token limit did not end it."""
    if called and result.reason != "length":
        return "tool_calls"
    return result.reason


def completion_response(model, job, result, message, called):
    response = {
        "id": f"chatcmpl-{job.public_id}",
        "object": "chat.completion",
        "created": job.created_at,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason(result, called),
            }
        ],
        "usage": usage_dict(result, job),
        "metrics": result.metrics,
        "timings": timings_dict(result),
    }
    if (compaction := compaction_dict(job)) is not None:
        response["compaction"] = compaction
    return response


def text_completion_response(model, job, result, text):
    response = {
        "id": f"cmpl-{job.public_id}",
        "object": "text_completion",
        "created": job.created_at,
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": text,
                "logprobs": None,
                "finish_reason": result.reason,
            }
        ],
        "usage": usage_dict(result, job),
        "metrics": result.metrics,
        "timings": timings_dict(result),
    }
    if (compaction := compaction_dict(job)) is not None:
        response["compaction"] = compaction
    return response


def _chunk(
    object_type,
    chunk_id,
    created,
    model,
    choice,
    usage,
    metrics,
    timings,
    compaction=None,
):
    chunk = {
        "id": chunk_id,
        "object": object_type,
        "created": created,
        "model": model,
        "choices": [choice],
    }
    if usage is not None:
        chunk["choices"] = []
        chunk["usage"] = usage
    if metrics is not None:
        chunk["metrics"] = metrics
    if timings is not None:
        chunk["timings"] = timings
    if compaction is not None:
        chunk["compaction"] = compaction
    return chunk


def stream_chunk(
    model,
    request_id,
    created,
    delta,
    finish_reason=None,
    usage=None,
    metrics=None,
    timings=None,
    compaction=None,
):
    return _chunk(
        "chat.completion.chunk",
        f"chatcmpl-{request_id}",
        created,
        model,
        {"index": 0, "delta": delta, "finish_reason": finish_reason},
        usage,
        metrics,
        timings,
        compaction,
    )


def text_completion_chunk(
    model,
    request_id,
    created,
    text,
    finish_reason=None,
    usage=None,
    metrics=None,
    timings=None,
    compaction=None,
):
    return _chunk(
        "text_completion",
        f"cmpl-{request_id}",
        created,
        model,
        {"index": 0, "text": text, "logprobs": None, "finish_reason": finish_reason},
        usage,
        metrics,
        timings,
        compaction,
    )


def complete(exchange, job):
    collected = exchange.collect(job)
    message = {"role": "assistant", "content": collected.content or None}
    if collected.reasoning:
        message["reasoning_content"] = collected.reasoning
    if collected.tool_calls:
        message["tool_calls"] = collected.tool_calls
    response = completion_response(
        exchange.app.response_model,
        job,
        collected.result,
        message,
        collected.called,
    )
    if job.logprobs:
        response["choices"][0]["logprobs"] = exchange.chat_logprobs(
            collected.result, collected.content
        )
    exchange.send_json(200, response)


def complete_choices(exchange, jobs):
    choices = []
    first = None
    prompt_tokens = 0
    completion_tokens = 0
    for index, job in enumerate(jobs):
        collected = exchange.collect(job)
        message = {"role": "assistant", "content": collected.content or None}
        if collected.reasoning:
            message["reasoning_content"] = collected.reasoning
        if collected.tool_calls:
            message["tool_calls"] = collected.tool_calls
        response = completion_response(
            exchange.app.response_model,
            job,
            collected.result,
            message,
            collected.called,
        )
        if job.logprobs:
            response["choices"][0]["logprobs"] = exchange.chat_logprobs(
                collected.result, collected.content
            )
        if first is None:
            first = response
        choice = response["choices"][0]
        choice["index"] = index
        choices.append(choice)
        prompt_tokens = max(prompt_tokens, response["usage"]["prompt_tokens"])
        completion_tokens += response["usage"]["completion_tokens"]
    first["choices"] = choices
    first["usage"]["prompt_tokens"] = prompt_tokens
    first["usage"]["completion_tokens"] = completion_tokens
    first["usage"]["total_tokens"] = prompt_tokens + completion_tokens
    exchange.send_json(200, first)


def text_completion(exchange, job):
    collected = exchange.collect(job)
    exchange.send_json(
        200,
        text_completion_response(
            exchange.app.response_model, job, collected.result, collected.content
        ),
    )


def text_completion_choices(exchange, jobs):
    choices = []
    first = None
    prompt_tokens = 0
    completion_tokens = 0
    for index, job in enumerate(jobs):
        collected = exchange.collect(job)
        response = text_completion_response(
            exchange.app.response_model, job, collected.result, collected.content
        )
        if first is None:
            first = response
        choice = response["choices"][0]
        choice["index"] = index
        choices.append(choice)
        prompt_tokens = max(prompt_tokens, response["usage"]["prompt_tokens"])
        completion_tokens += response["usage"]["completion_tokens"]
    first["choices"] = choices
    first["usage"]["prompt_tokens"] = prompt_tokens
    first["usage"]["completion_tokens"] = completion_tokens
    first["usage"]["total_tokens"] = prompt_tokens + completion_tokens
    exchange.send_json(200, first)


def stream(exchange, job, stream_options, *, chat):
    """A Chat or text completion stream; text completions have no tools."""
    chunk = partial(
        stream_chunk if chat else text_completion_chunk,
        exchange.app.response_model,
        job.public_id,
        job.created_at,
    )
    empty = {} if chat else ""
    started = False

    def payload(field, text):
        return {field: text} if chat else text

    def start():
        nonlocal started
        started = True
        exchange.start_event_stream()
        if chat:
            exchange.sse(chunk({"role": "assistant", "content": ""}))

    def keepalive():
        # After the start, a chunk that adds nothing: tool arguments of
        # arrays and objects are buffered until complete, which can take
        # minutes, and clients that time out on missing data events
        # ignore SSE comments.
        if started:
            exchange.sse(chunk(empty))
        else:
            exchange.sse_keepalive()

    def run():
        collected = exchange.collect(
            job,
            on_start=start,
            on_text=lambda field, text: exchange.sse(chunk(payload(field, text))),
            on_tool_delta=lambda delta: exchange.sse(chunk({"tool_calls": [delta]})),
            on_idle=keepalive,
            on_progress=lambda progress: exchange.sse(
                chunk(empty) | {"prompt_progress": progress}
            ),
        )
        result = collected.result
        exchange.sse(
            chunk(
                empty,
                finish_reason(result, collected.called),
                timings=timings_dict(result),
                compaction=compaction_dict(job),
            )
        )
        if stream_options.get("include_usage"):
            exchange.sse(
                chunk(
                    empty,
                    usage=usage_dict(result, job),
                    metrics=result.metrics,
                    compaction=compaction_dict(job),
                )
            )
        exchange.sse("[DONE]")

    exchange.guarded_stream(job, run, exchange.sse_error)


def post_chat(exchange, body, deadline):
    """Prepare a Chat request, and the answer that writes its output."""
    progress = request_fields.progress(body)
    streamed, stream_options = request_fields.stream_options(body)
    job = exchange.app.prepare(body, deadline=deadline)
    job.flags |= progress
    if job.logprobs and streamed:
        raise APIError(400, "streaming logprobs are not supported")
    jobs = exchange.choice_jobs(job)
    if isinstance(jobs, list):
        if streamed:
            raise APIError(400, "n > 1 is not supported with stream")
        return jobs, partial(complete_choices, exchange)
    if streamed:
        return job, partial(stream, exchange, stream_options=stream_options, chat=True)
    return job, partial(complete, exchange)


def post_completion(exchange, body, deadline):
    """Prepare a text completion, and the answer that writes its output."""
    progress = request_fields.progress(body)
    streamed, stream_options = request_fields.stream_options(body)
    job = exchange.app.prepare_completion(body, deadline=deadline)
    job.flags |= progress
    jobs = exchange.choice_jobs(job)
    if isinstance(jobs, list):
        if streamed:
            raise APIError(400, "n > 1 is not supported with stream")
        return jobs, partial(text_completion_choices, exchange)
    if streamed:
        return job, partial(stream, exchange, stream_options=stream_options, chat=False)
    return job, partial(text_completion, exchange)


def post_chat_count_tokens(exchange, body, deadline):
    tokens = exchange.app.count_tokens(body, deadline=deadline)
    exchange.send_json(200, {"input_tokens": tokens})


# Chat and text completions, and the prompt-only token count.
ROUTES = {
    "/v1/chat/completions": PostRoute(post_chat, OPENAI_ERRORS, queued=True),
    "/v1/chat/completions/count_tokens": PostRoute(
        post_chat_count_tokens, OPENAI_ERRORS, prompt_only=True
    ),
    "/v1/completions": PostRoute(post_completion, OPENAI_ERRORS, queued=True),
}
