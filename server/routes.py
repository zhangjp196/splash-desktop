"""The server's POST route table, every API's routes beside the control ones.

Each API's routes live in its dialect; this module gathers them into the one
table the handler routes by, and says how every path answers its errors.
"""

from .dialects.anthropic import ROUTES as ANTHROPIC_ROUTES
from .dialects.openai import ROUTES as OPENAI_ROUTES
from .dialects.responses import ROUTES as RESPONSES_ROUTES
from .dialects.scoring import ROUTES as SCORING_ROUTES
from .errors import ANTHROPIC_ERRORS, OPENAI_ERRORS
from .routing import PostRoute

POST_ROUTES = {
    **OPENAI_ROUTES,
    **RESPONSES_ROUTES,
    **ANTHROPIC_ROUTES,
    **SCORING_ROUTES,
    "/tokenize": PostRoute("_post_tokenize", OPENAI_ERRORS, prompt_only=True),
    "/detokenize": PostRoute("_post_detokenize", OPENAI_ERRORS, prompt_only=True),
    "/apply-template": PostRoute(
        "_post_apply_template", OPENAI_ERRORS, prompt_only=True
    ),
    "/v1/memory/release": PostRoute(
        "_post_memory_release", OPENAI_ERRORS, fieldless=True
    ),
    "/v1/thermal": PostRoute("_post_thermal", OPENAI_ERRORS),
}


def path_errors(path):
    """How every error on `path` is answered, whatever its method and
    whether it arises before a handler runs or in one: as the API of its
    POST route answers it, on Anthropic's other paths, those under
    /v1/messages, as that API does, and elsewhere as OpenAI's."""
    route = POST_ROUTES.get(path)
    if route is not None:
        return route.errors
    return ANTHROPIC_ERRORS if path.startswith("/v1/messages/") else OPENAI_ERRORS
