"""The scoring APIs: typed judgments and systemone's finite-choice answers."""

from __future__ import annotations

from .. import judgments
from ..errors import OPENAI_ERRORS, SYSTEMONE_ERRORS
from ..routing import PostRoute


def post_judgments(exchange, body, deadline):
    job, row = exchange.app.prepare_judgment(
        body, deadline=deadline, disconnected=exchange.client_disconnected
    )
    exchange.submit(job)
    result = exchange.await_done(job)
    exchange.send_json(
        200,
        judgments.judgment_response(exchange.app.response_model, row, job, result),
    )


def post_systemone(exchange, body, deadline):
    answers = {}
    input_tokens = 0
    entries = exchange.app.prepare_systemone(
        body, deadline=deadline, disconnected=exchange.client_disconnected
    )
    for qid, spec, job in entries:
        if job is None:
            answers[qid] = judgments.deterministic_answer(spec)
            continue
        # One admitted job per HTTP request preserves the existing queue bound
        # and lets later questions reuse the state prefix.
        exchange.submit(job)
        result = exchange.await_done(job)
        input_tokens += result.prompt_tokens
        answers[qid] = judgments.systemone_answer(
            spec, judgments.softmax(list(result.option_logits))
        )
    exchange.send_json(
        200,
        {
            "model": exchange.app.response_model,
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        },
    )


ROUTES = {
    "/v1/judgments": PostRoute(post_judgments, OPENAI_ERRORS, queued=True),
    "/v1/systemone": PostRoute(post_systemone, SYSTEMONE_ERRORS, queued=True),
}
