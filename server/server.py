"""HTTP routing and serving-process startup. One table names each endpoint's
handler; each API's handlers, request conversion and answers live in its module
in dialects/."""

import argparse
import json
import math
import os
import queue
import re
import resource
import secrets
import select
import signal
import socket
import sys
import time
from dataclasses import dataclass, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import unquote

from transformers import AutoTokenizer

from install import assembly
from install.models import ModelError

from . import images as image_input
from . import json_codec, serve_options, serve_record
from . import runtime as engine_runtime
from .backend import (
    CacheInfo,
    NativeBackend,
    NativeResult,
    ThinkingEndError,
    remaining_request_time,
    thinking_end_tokens,
)
from .chat_templates import ChatTemplateError, ChatTemplates
from .connections import ConnectionSlots, HttpAdmission, RequestBodyReservation
from .constraints import ConstraintFactory, validate_tokenizer
from .diagnostics import log_unexpected, print_request, print_status
from .dialects.anthropic import ROUTES as ANTHROPIC_ROUTES
from .dialects.openai import ROUTES as OPENAI_ROUTES
from .dialects.responses import ROUTES as RESPONSES_ROUTES
from .dialects.scoring import ROUTES as SCORING_ROUTES
from .errors import (
    ANTHROPIC_ERRORS,
    OPENAI_ERRORS,
    RETRY_STATUSES,
    APIError,
    RequestValidationError,
    field_error,
)
from .frontend import Frontend
from .http_security import (
    OriginRefused,
    RefusedOriginLog,
    authenticate,
    validate_headers,
)
from .latency import RequestLatency
from .metrics import prometheus_metrics
from .origins import ANY_ORIGIN
from .output import (
    ReasoningSplitter,
    StreamingToolCallProjector,
    validate_response_content,
)
from .routing import PostRoute
from .thinking import ThinkingCodec, ThinkingKeyError, load_thinking_key

# Request bodies held at once, from upload through preparation and, for what
# a generation retains of them, until it ends, take at most max(this, twice
# --max-request-size) bytes, so two of the largest requests always fit.
DEFAULT_REQUEST_BODY_BUDGET = 512 * 1024 * 1024
HTTP_IO_TIMEOUT = 30.0
HTTP_UPLOAD_BYTES_PER_SECOND = 512 * 1024
# Native events wake a waiting request at once; this only bounds how late a
# client disconnect is noticed.
CLIENT_DISCONNECT_POLL = 0.1
# How long a connection refused unread may take its client to close.
REFUSED_LINGER_SECONDS = 2.0
# The soft descriptor limit the server raises its own to: macOS starts a
# process with 256, fewer than its connection slots, the refused connections
# lingering after a burst and the engine's pipes and files can need. 10240 is
# macOS's OPEN_MAX.
DESCRIPTOR_LIMIT = 10240
SSE_KEEPALIVE_SECONDS = 2.0
NATIVE_START_TIMEOUT = 600.0
ROOT = Path(__file__).parents[1]
CHAT_HTML = Path(__file__).with_name("chat.html").read_bytes()
CHAT_ASSETS = {
    f"/chat-assets/{name}": (Path(__file__).parent / "chat-assets" / name).read_bytes()
    for name in (
        "markdown-it-15.0.2.min.js",
        "highlight-11.12.0.min.js",
        "github-11.12.0.min.css",
        "github-dark-11.12.0.min.css",
    )
}
# The chat page's brand mark, in its text colors, which the page shows too.
# Browsers, and other clients, ask for a site's icon at /favicon.ico.
FAVICON_SVG = Path(__file__).with_name("favicon.svg").read_bytes()


def _content_length(value):
    """A Content-Length value as a byte count; None when it is not a
    decimal count, or has more digits than int() converts."""
    if not value.isascii() or not value.isdigit():
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _normalize_path(raw_path):
    """Canonicalize a request target for route and header decisions.

    Strips any query string or fragment, percent-decodes, and resolves
    ``.`` and ``..`` segments so encoded or dotted spellings of a route
    are treated exactly like the route itself.
    """
    decoded = unquote(raw_path.partition("?")[0].partition("#")[0])
    if not decoded.startswith("/"):
        return decoded
    trailing = decoded.endswith("/") and len(decoded) > 1
    segments = []
    for segment in decoded.split("/"):
        if segment in ("", "."):
            continue
        if segment == "..":
            if segments:
                segments.pop()
            continue
        segments.append(segment)
    normalized = "/" + "/".join(segments)
    if trailing and normalized != "/":
        normalized += "/"
    return normalized


# Each API's routes live in its dialect; the control routes stay here.
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


@dataclass(slots=True)
class Collected:
    """A generation as FrontendHandler._collect gathered it."""

    reasoning: str
    content: str
    tool_calls: list
    result: NativeResult
    # The output ended inside its reasoning.
    reasoning_open: bool
    # The token limit or a stop sequence ended the output inside a call.
    cut_in_call: bool
    # The id of that call, whose arguments are unfinished; None when it cut
    # none, or none whose name it had read.
    cut_call: str | None

    @property
    def cut(self):
        """Whether the output's end cut its last block short: the token limit
        cuts any block, a stop sequence only a call, as text may end where
        the client asks. A call cut at its name has no block, and a stop
        sequence there leaves the block before it whole."""
        return self.result.reason == "length" or self.cut_call is not None

    @property
    def called(self):
        """Whether the output's tool calls count as made: it cut none."""
        return bool(self.tool_calls) and not self.cut_in_call


class FrontendHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    methods = "GET, HEAD, POST, DELETE, OPTIONS"

    def setup(self):
        self._response_started = False
        # What the response owes its request's origin, once the request has
        # passed validate_headers.
        self._allow_origin = None
        self._unread_body = 0
        self._last_sse_write = time.monotonic()
        super().setup()
        self.connection.settimeout(HTTP_IO_TIMEOUT)

    def finish(self):
        if self._unread_body:
            self._discard_unread_body()
        super().finish()

    def _discard_unread_body(self):
        # Closing with request bytes unread resets the connection, and the
        # reset can destroy the response before a client still uploading
        # reads it. Half-close, then receive the rest of the upload on the
        # terms a body is read, waiting on the client as a connection with
        # no request yet does.
        self.server.connections.draining(self.connection)
        try:
            self.connection.shutdown(socket.SHUT_WR)
            while self._unread_body > 0:
                remaining = self._upload_deadline - time.monotonic()
                if remaining <= 0:
                    return
                self.connection.settimeout(min(remaining, HTTP_IO_TIMEOUT))
                chunk = self.rfile.read1(min(65536, self._unread_body))
                if not chunk:
                    return
                self._unread_body -= len(chunk)
        except OSError:
            pass

    def log_message(self, format, *args):
        pass

    def parse_request(self):
        if not super().parse_request():
            return False
        if self.request_version not in {"HTTP/1.0", "HTTP/1.1"}:
            self.close_connection = True
            self.send_error(505, "HTTP version not supported")
            return False
        # The body still to come, which finish() receives if no handler
        # reads it before responding: its stated length or, without a valid
        # one, as much as the server accepts, in the time an upload gets,
        # counted from here.
        lengths = self.headers.get_all("Content-Length", [])
        length = _content_length(lengths[0]) if len(lengths) == 1 else None
        if length is not None:
            self._unread_body = length
        elif lengths or self.headers.get_all("Transfer-Encoding"):
            self._unread_body = self.server.max_request_bytes
        self._upload_deadline = time.monotonic() + self._upload_seconds(
            min(self._unread_body, self.server.max_request_bytes)
        )
        try:
            allowed_hosts = self.server.allowed_hosts | {
                self.connection.getsockname()[0].lower()
            }
            self._allow_origin = validate_headers(
                self.headers, allowed_hosts, self.server.allowed_origins
            )
            public = self.command == "OPTIONS" or (
                self.command in ("GET", "HEAD")
                and (
                    self.route
                    in ("/", "/index.html", "/favicon.ico", "/health", "/ready")
                    or self.route in CHAT_ASSETS
                )
            )
            if not public:
                authenticate(self.headers, self.server.api_key)
        except APIError as error:
            if isinstance(error, OriginRefused):
                self.server.refused_origins.report(error.origin)
            self.close_connection = True
            self._safe_error(error, log=False)
            return False
        return True

    def send_error(self, code, message=None, explain=None):
        # The stdlib's send_error, which answers requests it cannot parse or
        # route and parse_request's 505, writes an HTML page. After a request
        # line it cannot parse, or HTTP/0.9's, request_version is HTTP/0.9,
        # and it writes that page with no status line or headers. Answer as
        # any other error, over HTTP/1.1, a method or version the server does
        # not implement as a server error.
        self.request_version = self.protocol_version
        status = HTTPStatus(code)
        error_code = status.name.lower() if code >= 500 else "invalid_request_error"
        self._safe_error(
            APIError(code, message or status.phrase, error_code), log=False
        )

    @property
    def app(self):
        return self.server.app

    @property
    def route(self):
        """The request's path as routing, authentication and error dialects
        all read it; empty before a request line parses."""
        return _normalize_path(getattr(self, "path", ""))

    @property
    def errors(self):
        """How errors on the request's path are answered (path_errors)."""
        return path_errors(self.route)

    def end_headers(self):
        # A browser hands a page the response from another origin only when
        # the response names that origin, so every response to an admitted
        # origin does, errors and event streams too, and exposes the retry
        # and authentication hints, which CORS hides by default.
        if self._allow_origin is not None:
            self.send_header("Access-Control-Allow-Origin", self._allow_origin)
            if self._allow_origin != ANY_ORIGIN:
                self.send_header("Vary", "Origin")
            self.send_header(
                "Access-Control-Expose-Headers", "Retry-After, WWW-Authenticate"
            )
        super().end_headers()

    def _send(self, status, data, content_type, retry_after=1):
        self.send_response(status)
        if status in RETRY_STATUSES:
            self.send_header("Retry-After", str(retry_after))
        if status == 401:
            self.send_header("WWW-Authenticate", "Bearer")
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        route = self.route
        if (
            route == "/v1/systemone"
            or route == "/v1/models"
            or route.startswith("/v1/models/")
        ):
            self.send_header("x-typesafe-request-id", f"req_{secrets.token_hex(12)}")
        self._response_started = True
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _json(self, status, payload, retry_after=1):
        try:
            data = json_codec.encode(payload)
        except json_codec.JSONEncodingError as error:
            log_unexpected(error)
            self._safe_error(
                APIError(500, "internal server error", "internal_server_error"),
                log=False,
            )
            return
        self._send(status, data, "application/json", retry_after)

    def _log_api_error(self, code):
        path = "".join(char if char.isprintable() else "?" for char in self.route)
        print_status(f"Error · {code} · {self.command} {path[:256]}", error=True)

    def _safe_error(self, error, *, log=True):
        """Answer with `error` as its path's API answers it, unless the
        response has begun."""
        if self._response_started:
            return
        status, code, payload = self.errors.answer(error)
        if log:
            self._log_api_error(code)
        try:
            self._json(status, payload, error.retry_after)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass

    def _upload_seconds(self, length):
        # Inactivity allowed at any point, plus the body at the upload rate.
        return HTTP_IO_TIMEOUT + length / HTTP_UPLOAD_BYTES_PER_SECOND

    def _read_json_body(self, deadline):
        """The request's body, a JSON object, read by `deadline`."""
        if self.headers.get_all("Transfer-Encoding"):
            raise APIError(400, "transfer encoding is not supported")
        encodings = self.headers.get_all("Content-Encoding", [])
        if len(encodings) > 1 or (
            encodings and encodings[0].strip().lower() != "identity"
        ):
            raise APIError(415, "content encoding is not supported")
        content_types = self.headers.get_all("Content-Type", [])
        content_type = self.headers.get_content_type().lower()
        if len(content_types) != 1 or not (
            content_type == "application/json"
            or (
                content_type.startswith("application/")
                and content_type.endswith("+json")
            )
        ):
            raise APIError(415, "Content-Type must be application/json")
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1:
            raise APIError(400, "exactly one Content-Length header is required")
        length = _content_length(lengths[0])
        if length is None:
            raise APIError(400, "invalid Content-Length header")
        if length <= 0:
            raise APIError(400, "request body must not be empty")
        if length > self.server.max_request_bytes:
            raise APIError(
                413,
                f"request body is {length} bytes; limit is "
                f"{self.server.max_request_bytes} bytes (--max-request-size)",
                "request_too_large",
            )
        # Bound total upload time even when a client keeps the socket active.
        deadline = min(deadline, time.monotonic() + self._upload_seconds(length))
        self._body_reservation = RequestBodyReservation(
            self.server.request_bodies, length
        )
        payload = bytearray()
        try:
            while len(payload) < length:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                self.connection.settimeout(min(remaining, HTTP_IO_TIMEOUT))
                chunk = self.rfile.read1(min(65536, length - len(payload)))
                if not chunk:
                    raise APIError(400, "request body ended before Content-Length")
                payload.extend(chunk)
        finally:
            self._unread_body = length - len(payload)
            self.connection.settimeout(HTTP_IO_TIMEOUT)
        try:
            text = payload.decode(json.detect_encoding(payload), "surrogatepass")
            payload.clear()
            body = json_codec.loads(text)
        except ValueError:
            # Text that is not JSON, or JSON nested past json_codec.MAX_DEPTH.
            raise RequestValidationError(
                [field_error([], "invalid JSON request body")]
            ) from None
        if not isinstance(body, dict):
            raise RequestValidationError(
                [field_error([], "request body must be an object")]
            )
        return body

    def do_HEAD(self):
        self.do_GET()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Allow", self.methods)
        if (
            self._allow_origin is not None
            and "Access-Control-Request-Method" in self.headers
        ):
            # A browser's preflight, which asks what the request it holds back
            # may use: every method the server has and, as in vLLM, any header.
            self.send_header("Access-Control-Allow-Methods", self.methods)
            requested = self.headers.get("Access-Control-Request-Headers")
            if requested is not None and requested.isprintable():
                self.send_header("Access-Control-Allow-Headers", requested)
            self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()

    def version_string(self):
        return "Splash"

    def do_GET(self):
        path = self.route
        if path in ("/", "/index.html", "/favicon.ico") or path in CHAT_ASSETS:
            if not self.server.webui:
                self._safe_error(APIError(404, "not found", "not_found"))
            elif path == "/favicon.ico":
                self._send(200, FAVICON_SVG, "image/svg+xml")
            elif path in CHAT_ASSETS:
                mime = "text/css" if path.endswith(".css") else "text/javascript"
                self._send(200, CHAT_ASSETS[path], f"{mime}; charset=utf-8")
            else:
                self._send(200, CHAT_HTML, "text/html; charset=utf-8")
            return
        if path == "/health":
            self._json(200, {"status": "ok"})
            return
        if path == "/ready":
            ready = self.app.backend.is_ready()
            self._json(
                200 if ready else 503, {"status": "ready" if ready else "unavailable"}
            )
            return
        if path == "/status":
            self._json(200, self.server.status())
            return
        if path == "/metrics":
            self._send(
                200,
                prometheus_metrics(self.server.status()).encode(),
                "text/plain; version=0.0.4; charset=utf-8",
            )
            return
        response_match = re.fullmatch(r"/v1/responses/(resp_[A-Za-z0-9_]+)", path)
        if response_match:
            stored = self.app.response_store.get(response_match.group(1))
            if stored is None:
                self._safe_error(APIError(404, "response not found", "not_found_error"))
            else:
                self._json(200, stored.response)
            return
        if path == "/v1/models" or path.startswith("/v1/models/"):
            models = [
                {
                    "id": name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "splash",
                    "max_model_len": self.app.max_context,
                    "context_length": self.app.max_context,
                    "vision": self.app.vision,
                    "input_modalities": self.app.input_modalities,
                    "reasoning": self.app.reasoning,
                    **(
                        {"root": self.app.response_model}
                        if name != self.app.response_model
                        else {}
                    ),
                }
                for name in self.app.model_names
            ]
            if path == "/v1/models":
                # TypeSafe SDK compatibility: models.list() reads "models" entries.
                typed = [
                    {
                        "name": item["id"],
                        "description": "Splash resident model",
                        "release_date": "",
                    }
                    for item in models
                ]
                self._json(200, {"object": "list", "data": models, "models": typed})
            else:
                name = path.removeprefix("/v1/models/")
                model = next((item for item in models if item["id"] == name), None)
                if model is None:
                    self._safe_error(
                        APIError(404, "model not found", "model_not_found")
                    )
                else:
                    self._json(200, model)
            return
        self._safe_error(APIError(404, "not found", "not_found"))

    def do_DELETE(self):
        path = self.route
        response_match = re.fullmatch(r"/v1/responses/(resp_[A-Za-z0-9_]+)", path)
        if response_match is None:
            self._safe_error(APIError(404, "not found", "not_found"))
            return
        response_id = response_match.group(1)
        if not self.app.response_store.delete(response_id):
            self._safe_error(APIError(404, "response not found", "not_found_error"))
            return
        self._json(
            200,
            {"id": response_id, "object": "response", "deleted": True},
        )

    def do_POST(self):
        started_at = time.monotonic()
        route = POST_ROUTES.get(self.route)
        if route is None:
            self._safe_error(APIError(404, "not found", "not_found"))
            return
        refusal = None if route.prompt_only else self.app.backend.refusal()
        if refusal is not None:
            self._safe_error(refusal, log=False)
            return
        # Hold one ingress slot through body parsing, preparation, and the
        # complete response. Slow uploads/readers cannot accumulate outside
        # the native pending limit. Only the LLM APIs the queue admits hold
        # one of the request gate's slots; the prompt tools hold a
        # preparation slot of the token-count gate, and a control endpoint,
        # which answers at once, holds none.
        if route.prompt_only:
            admission = self.server.token_counts
        elif route.queued:
            admission = self.server.requests
        else:
            admission = None
        if admission is not None and not admission.acquire():
            self._safe_error(
                APIError(
                    503,
                    "frontend request capacity is exhausted",
                    "frontend_overloaded",
                )
            )
            return
        self._body_reservation = None
        # The native job submitted last, which a failure cancels, and every
        # job of this request (n choices), which a failure cancels too.
        self._submitted = None
        self._submitted_jobs = []
        try:
            generation = self._serve(route, started_at)
            if generation is not None:
                self._generate(*generation, started_at)
        except (BrokenPipeError, ConnectionResetError):
            self._cancel_submitted()
        except TimeoutError:
            self._cancel_submitted()
            error = APIError(408, "HTTP I/O timed out", "request_timeout")
            self._safe_error(error, log=self._submitted is None)
        except APIError as error:
            self._cancel_submitted()
            # The native outcome was already logged; a server-side failure
            # after submission must still reach the console.
            self._safe_error(error, log=self._submitted is None or error.status >= 500)
        except Exception as error:
            self._cancel_submitted()
            log_unexpected(error)
            error = APIError(500, "internal server error", "internal_server_error")
            self._safe_error(error, log=False)
        finally:
            if self._body_reservation is not None:
                self._body_reservation.release()
                self._body_reservation = None
            # The handler outlives the request, through what finish()
            # still reads of an upload refused unread; the job must not, as
            # freeing it returns the input it retains.
            self._submitted = None
            self._submitted_jobs = []
            if admission is not None:
                admission.release()
            self.app.latencies.observe("http_request", time.monotonic() - started_at)

    def _serve(self, route, started_at):
        """Read the request and serve it: answer it, or prepare the
        generation it asks for and return its job and the method that answers
        with the job's output."""
        if route.fieldless and not self._unread_body:
            body = {}
        else:
            with self.app.latencies.measure("upload"):
                body = self._read_json_body(started_at + self.app.request_timeout)
            # A field set to null is one left out, as clients that write every
            # field they do not set as null mean it.
            body = {key: value for key, value in body.items() if value is not None}
            if route.fieldless and body:
                raise APIError(400, "this endpoint takes no fields")
        deadline = self.app.request_deadline(body, started_at)
        if isinstance(route.handler, str):
            return getattr(self, route.handler)(body, deadline)
        return route.handler(self, body, deadline)

    def _generate(self, job, respond, started_at):
        """Submit a prepared generation and answer with its output. The
        request's input bytes stay reserved only for what the job retains.
        `job` is one Job, or the list of a request's n choices."""
        jobs = job if isinstance(job, (list, tuple)) else [job]
        self._body_reservation.retain_for(jobs[0])
        self._body_reservation = None
        for choice in jobs:
            choice.latency = RequestLatency(self.app.latencies, started_at)
            self._submit(choice)
        respond(jobs if len(jobs) > 1 else jobs[0])

    def _submit(self, job):
        """Submit `job` unless its deadline has passed or its client has
        left; from then on a failure cancels it."""
        remaining_request_time(job.deadline)
        if self._client_disconnected():
            raise ConnectionResetError("client disconnected before submission")
        self.app.backend.submit(job)
        self._submitted = job
        self._submitted_jobs.append(job)

    def _cancel_submitted(self):
        for job in self._submitted_jobs:
            self.app.backend.cancel(job)
        self._submitted_jobs = []

    def _choice_jobs(self, job):
        """One job per choice of an n-request, or the job itself."""
        if job.n <= 1:
            return job
        jobs = [job]
        for index in range(1, job.n):
            jobs.append(
                replace(
                    job,
                    request_id=next(self.app.ids),
                    seed=(job.seed + index) % (2**64),
                    events=queue.Queue(),
                    cache=CacheInfo(),
                    latency=None,
                    cache_salt=(
                        secrets.randbits(128) if job.logprobs else job.cache_salt
                    ),
                    constraint=(
                        job.constraint.clone() if job.constraint is not None else None
                    ),
                )
            )
        return jobs

    def _post_tokenize(self, body, deadline):
        self._json(200, {"tokens": self.app.tokenize(body, deadline=deadline)})

    def _post_detokenize(self, body, deadline):
        self._json(200, {"content": self.app.detokenize(body, deadline=deadline)})

    def _post_apply_template(self, body, deadline):
        self._json(200, {"prompt": self.app.apply_template(body, deadline=deadline)})

    def _post_memory_release(self, body, deadline):
        weights = self.app.backend.release_weights().get("weights") or {}
        # A restore leaves the weights released until its last image is back.
        if weights.get("released") is True and weights.get("restoring") is False:
            self._json(200, {"released": True, "weights": weights})
            return
        # An answer about the engine's state, not a failure: the console does
        # not log it.
        self._safe_error(
            APIError(
                409,
                "the engine is serving requests or restoring its weights; it "
                "releases them only while idle",
                "engine_busy",
            ),
            log=False,
        )

    def _post_thermal(self, body, deadline):
        """Set the heat-and-noise rung of the running engine: the share of the
        wall time prefill may keep the GPU busy, applied without restarting it.
        The answer names the duty the engine took, which is what its next
        status reports too."""
        requested = body.get("prefill_duty_percent") if isinstance(body, dict) else None
        if requested is None:
            self._safe_error(
                APIError(
                    400,
                    "prefill_duty_percent is required",
                    "invalid_request_error",
                )
            )
            return
        unknown = sorted(set(body) - {"prefill_duty_percent"})
        if unknown:
            self._safe_error(
                APIError(
                    400,
                    "this endpoint takes only prefill_duty_percent; got "
                    + ", ".join(unknown),
                    "invalid_request_error",
                )
            )
            return
        try:
            duty = serve_options.prefill_duty(requested)
        except ValueError as error:
            self._safe_error(
                APIError(400, f"prefill_duty_percent {error}", "invalid_request_error")
            )
            return
        snapshot = self.app.backend.set_prefill_duty(duty)
        throttle = snapshot.get("throttle") or {}
        self._json(200, {"prefill_duty_percent": throttle.get("duty_percent", duty)})

    def _await_done(self, job):
        """The result of a score job, which emits only start and done."""
        while (event := self._next_event(job))[0] != "done":
            pass
        return event[1]

    def _next_event(self, job, on_idle=None):
        while True:
            if self._client_disconnected():
                self.app.backend.cancel(job)
                raise ConnectionResetError
            remaining = job.deadline - time.monotonic()
            if remaining <= 0:
                self.app.backend.cancel(job)
                raise APIError(504, "request timed out", "request_timeout")
            try:
                event = job.events.get(timeout=min(remaining, CLIENT_DISCONNECT_POLL))
            except queue.Empty:
                event = None
            # A queued failure takes precedence over a keepalive: until headers
            # are sent, the caller can still return its HTTP error status.
            if event is not None and event[0] == "error":
                raise event[1]
            if time.monotonic() >= job.deadline:
                continue
            # Native token activity can be buffered by the tool projector.
            # Measure silence on the HTTP stream across calls, not time spent
            # waiting for an empty native-event queue.
            if (
                on_idle is not None
                and time.monotonic() - self._last_sse_write >= SSE_KEEPALIVE_SECONDS
            ):
                on_idle()
            if event is not None:
                return event

    def _client_disconnected(self):
        # A poll object holds no descriptor: running out of descriptors
        # must not read as a disconnect.
        poller = select.poll()
        poller.register(self.connection, select.POLLIN)
        try:
            # The body is already consumed and every response closes the
            # connection. Drain unexpected trailing bytes so they cannot hide EOF.
            return bool(poller.poll(0)) and not self.connection.recv(
                65536, socket.MSG_DONTWAIT
            )
        except BlockingIOError:
            return False
        except ConnectionError:
            return True

    def _finalize_content(self, content, job, result, projector, cut_in_call):
        """The content and calls of the output, and the events the stream
        still owes. The token limit cuts the output wherever it ends, and a
        stop sequence the call it ends inside; text before a stop sequence is
        the text the client asked for. An answer without calls is validated
        against the response format only when the model ended it. `content`
        is read only without a projector, which otherwise holds it."""
        ended = result.reason != "length" and result.stop_sequence is None
        if projector is None:
            if ended:
                validate_response_content(content, job.response_validator)
            return content, [], []
        content, tool_calls, events = projector.finish(
            result.reason == "length" or cut_in_call
        )
        if ended and not tool_calls:
            validate_response_content(content, job.response_validator)
        return content, tool_calls, events

    def _collect(
        self,
        job,
        *,
        on_start=None,
        on_text=None,
        on_tool_delta=None,
        on_idle=None,
        on_progress=None,
    ):
        """Gather a generation until it is done. The callbacks receive the
        start, each piece of output, the idle waits and prompt progress:
        streams send them, and complete Messages and Responses gather the
        output into blocks."""
        splitter = ReasoningSplitter(job.thinking, job.may_call_tools, job.chat_format)
        # A chat answer is parsed as it arrives whether it streams or not:
        # its calls where tools may be called, and in any answer, a </think>
        # the model writes after its reasoning ended is dropped. A text
        # completion, without a chat template, is its text as written.
        projector = (
            StreamingToolCallProjector(
                job.tool_policy,
                job.public_id,
                job.response_validator is not None,
                job.chat_format.call_syntax,
            )
            if job.tool_policy is not None or job.reasoning is not None
            else None
        )
        reasoning, content, result = [], [], None

        def publish(events):
            if on_text is None:
                return
            for kind, value in events:
                if kind == "tool":
                    on_tool_delta(value)
                else:
                    on_text(kind, value)

        def append(field, text):
            if field == "content" and projector is not None:
                publish(projector.put(text))
            else:
                (reasoning if field == "reasoning_content" else content).append(text)
                publish([(field, text)])

        while result is None:
            kind, value = self._next_event(job, on_idle)
            if kind == "start" and on_start is not None:
                on_start()
            elif kind == "text":
                for field, text in splitter.put(value):
                    append(field, text)
            elif kind == "progress" and on_progress is not None:
                on_progress(value)
            elif kind == "done":
                result = value
        for field, text in splitter.finish():
            append(field, text)
        cut_in_call = (
            projector is not None
            and projector.in_call
            and (result.reason == "length" or result.stop_sequence is not None)
        )
        content_text, tool_calls, events = self._finalize_content(
            "".join(content), job, result, projector, cut_in_call
        )
        publish(events)
        return Collected(
            "".join(reasoning),
            content_text,
            tool_calls,
            result,
            splitter.reasoning,
            cut_in_call,
            None if projector is None else projector.call_id,
        )

    def _chat_logprobs(self, result, content):
        """A generated token list as Chat's logprobs content, trimmed to the
        text a stop sequence left."""
        entries = []
        for token, logprob, top in result.logprobs:
            text = self.app.tokenizer.decode([token], skip_special_tokens=True)
            if not text:
                # A stop or other special token ends the output rather than
                # being content; it has no Chat logprobs entry.
                continue
            entries.append(
                {
                    "token": text,
                    "logprob": logprob,
                    "bytes": list(text.encode("utf-8")),
                    "top_logprobs": [
                        {
                            "token": self.app.tokenizer.decode(
                                [top_token], skip_special_tokens=False
                            ),
                            "logprob": top_logprob,
                            "bytes": list(
                                self.app.tokenizer.decode(
                                    [top_token], skip_special_tokens=False
                                ).encode("utf-8")
                            ),
                        }
                        for top_token, top_logprob in top
                    ],
                }
            )
        while entries:
            joined = "".join(entry["token"] for entry in entries)
            if joined == content or content.startswith(joined):
                break
            entries.pop()
        return {"content": entries, "refusal": None}

    def _start_event_stream(self):
        if self._response_started:
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self._response_started = True

    def _write_sse(self, frame):
        self.wfile.write(frame)
        self.wfile.flush()
        self._last_sse_write = time.monotonic()

    def _sse(self, payload):
        data = (
            payload.encode("utf-8")
            if isinstance(payload, str)
            else json_codec.encode(payload)
        )
        self._write_sse(b"data: " + data + b"\n\n")

    def _event_sse(self, event, payload):
        data = json_codec.encode(payload)
        self._write_sse(f"event: {event}\ndata: ".encode() + data + b"\n\n")

    def _sse_keepalive(self):
        # An SSE comment is traffic, so socket read timeouts and proxies do
        # not take a long prefill or resource wait for a dead connection, but
        # event decoders skip it and clients that time out on missing data
        # events ignore it. Streams send it only where no data event fits: the
        # chat and text completion streams until the request starts, the
        # Responses stream once output has begun.
        self._start_event_stream()
        self._write_sse(b": splash-keepalive\n\n")

    def _sse_error(self, error):
        self._sse(self.errors.payload(error))
        self._sse("[DONE]")

    def _guarded_stream(self, job, run, send_error):
        try:
            run()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.app.backend.cancel(job)
        except APIError as error:
            self.app.backend.cancel(job)
            if not self._response_started:
                raise
            if error.status >= 500:
                self._log_api_error(error.code)
            try:
                send_error(error)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass
        except Exception as error:
            self.app.backend.cancel(job)
            if not self._response_started:
                raise
            log_unexpected(error)
            try:
                send_error(
                    APIError(500, "internal server error", "internal_server_error")
                )
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass

    # The public surface a dialect's handlers use. Each wrapper calls its
    # private method by name, so a test's patch of that method still holds.
    def send_json(self, status, payload, retry_after=1):
        return self._json(status, payload, retry_after)

    def choice_jobs(self, job):
        return self._choice_jobs(job)

    def collect(self, job, **kwargs):
        return self._collect(job, **kwargs)

    def chat_logprobs(self, result, content):
        return self._chat_logprobs(result, content)

    def event_sse(self, event, payload):
        return self._event_sse(event, payload)

    def guarded_stream(self, job, run, send_error):
        return self._guarded_stream(job, run, send_error)

    def start_event_stream(self):
        return self._start_event_stream()

    def sse(self, payload):
        return self._sse(payload)

    def sse_error(self, error):
        return self._sse_error(error)

    def sse_keepalive(self):
        return self._sse_keepalive()

    def client_disconnected(self):
        return self._client_disconnected()

    def submit(self, job):
        return self._submit(job)

    def await_done(self, job):
        return self._await_done(job)

    def safe_error(self, error, log=True):
        return self._safe_error(error, log=log)

    @property
    def response_started(self):
        return self._response_started

    @property
    def reserve_input(self):
        """Reserve request body budget for a resource a job retains."""
        return self._body_reservation.grow


class FrontendServer(HTTPServer):
    allow_reuse_address = True
    # Connections the kernel holds until the accept loop takes them. A burst
    # beyond this queue is reset by the kernel, unseen by the server, so ask
    # for as many as uvicorn does; the kernel caps it (128 on macOS).
    # Queued connections take no thread or descriptor.
    request_queue_size = 2048
    # Keep control/catalog capacity separate from generation capacity.
    # Neither gate allocates workers in advance.
    control_connection_capacity = 64
    # Connections answered without their requests read that wait at once,
    # on one thread, for their clients to close. A burst refuses all that
    # arrive while every slot is held, and a client that sends its request
    # after its connection closed is reset before it reads the 503, so
    # enough linger for a burst well beyond the kernel's queue, within the
    # descriptors the server raises its limit to.
    refused_connection_capacity = 1024

    def __init__(
        self,
        address,
        app,
        bind_and_activate=True,
        request_capacity=serve_options.DEFAULT_QUEUE_SIZE,
        allowed_hosts=(),
        api_key=None,
        webui=True,
        max_request_bytes=serve_options.DEFAULT_MAX_REQUEST_BYTES,
        allowed_origins=(),
    ):
        if (
            isinstance(max_request_bytes, bool)
            or not isinstance(max_request_bytes, int)
            or max_request_bytes <= 0
        ):
            raise ValueError("max_request_bytes must be a positive integer")
        self.max_request_bytes = max_request_bytes
        self.request_bodies = HttpAdmission(
            max(DEFAULT_REQUEST_BODY_BUDGET, 2 * max_request_bytes)
        )
        self.api_key = api_key
        self.webui = webui
        self.allowed_hosts = {
            host.lower().rstrip(".")
            for host in (*allowed_hosts, address[0], "localhost", "127.0.0.1", "::1")
            if host not in ("0.0.0.0", "::")
        }
        # As serve_options.parse_allowed_origin returns them.
        self.allowed_origins = frozenset(allowed_origins)
        self.refused_origins = RefusedOriginLog()
        self.instance_id = secrets.token_hex(12)
        self.started_at = time.time()
        self.requests = HttpAdmission(request_capacity)
        self.token_counts = HttpAdmission(request_capacity)
        self.connections = ConnectionSlots(
            request_capacity + self.control_connection_capacity,
            self._serve_connection,
            self.refused_connection_capacity,
            REFUSED_LINGER_SECONDS,
        )
        super().__init__(address, FrontendHandler, bind_and_activate)
        self.app = app

    def status(self):
        status = self.app.status()
        status["instance"] = {
            "id": self.instance_id,
            "pid": os.getpid(),
            "model": self.app.model,
            "host": self.server_address[0],
            "port": self.server_address[1],
            "started_at": self.started_at,
        }
        status["http"] = {
            "requests": self.requests.stats(),
            "request_body_bytes": self.request_bodies.stats(),
            "max_request_bytes": self.max_request_bytes,
            "token_counts": self.token_counts.stats(),
            "connections": self.connections.stats(),
        }
        return status

    def server_activate(self):
        super().server_activate()
        # accept() then returns at once when the kernel holds no connection.
        self.socket.setblocking(False)

    def _handle_request_noblock(self):
        """Take every connection the kernel holds at each wakeup of the accept
        loop, as asyncio's accept does, not one: a burst arrives faster than a
        select per connection takes it, and macOS resets what overflows the
        kernel's queue."""
        while True:
            try:
                request, client_address = self.get_request()
            except OSError:
                return
            # macOS gives a connection accepted from a nonblocking socket that
            # socket's mode.
            request.setblocking(True)
            if not self.verify_request(request, client_address):
                self.shutdown_request(request)
                continue
            try:
                self.process_request(request, client_address)
            except Exception:
                self.handle_error(request, client_address)
                self.shutdown_request(request)
            except BaseException:
                self.shutdown_request(request)
                raise

    def process_request(self, request, client_address):
        if not self.connections.admit(
            request, client_address, time.monotonic() + HTTP_IO_TIMEOUT
        ):
            self.connections.refuse(request)

    def _serve_connection(self, request, client_address):
        """Serve the connection `request`, ready for its thread, on a
        connection worker, as ThreadingMixIn's thread for a request does."""
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)

    def shutdown_request(self, request):
        # An answered connection is half-closed and lingers: a request still
        # unread on it, as a pipelined one, must not reset the answer.
        self.connections.close(request)

    def server_close(self):
        super().server_close()
        self.connections.stop()
        self.connections.idle.wait(min(2.0, HTTP_IO_TIMEOUT))

    def handle_error(self, request, client_address):
        error = sys.exc_info()[1]
        if isinstance(error, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def _parse_model_id(value):
    """The server's --model: a repository ID, or the bare id of a local model
    directory (--model-dir), which the launcher derives from the folder name."""
    if "/" not in value:
        if (
            not value
            or any(character in value for character in " \t\n\\%?#")
            or ":" in value
        ):
            raise argparse.ArgumentTypeError(
                "model must be a repository ID (owner/repo[:variant]) or a bare name"
            )
        return value
    return serve_options.parse_model_id(value)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_root",
        metavar="MODEL_DIRECTORY",
        help="the directory of a model Splash prepared, or a selection link to "
        "one: the server holds and serves the directory it names at startup",
    )
    parser.add_argument(
        "--model",
        type=_parse_model_id,
        required=True,
        metavar="OWNER/REPO",
    )
    parser.add_argument("--port", type=int, default=serve_options.DEFAULT_PORT)
    parser.add_argument("--binary", default=str(ROOT / "build" / "splash"))
    parser.add_argument(
        "--serve-record",
        type=Path,
        default=None,
        metavar="PATH",
        help="the JSON record of this start the launcher and the app follow "
        "(serve_record)",
    )
    serve_options.add_serve_arguments(parser)
    args = parser.parse_args(argv)
    serve_options.check_serve_arguments(parser, args)
    if not 0 <= args.port <= 65535:
        parser.error("--port must be in [0, 65535]")
    return args


def _native_command(args, model_root, relaunch_context=None):
    """The engine's command line serving model_root, a relaunch's with
    `relaunch_context`, the context the first engine served: without
    --max-context the relaunch keeps it, where its own would follow a Neural
    Engine split calibrated anew."""
    command = [
        args.binary,
        "serve-native",
        str(model_root),
        "auto" if args.max_context is None else str(args.max_context),
        "auto" if args.max_memory is None else str(args.max_memory),
    ]
    if args.max_cache_disk is not None:
        command.append(str(args.max_cache_disk))
    if args.max_cache_disk_writes:
        command.extend(("--max-cache-disk-writes", str(args.max_cache_disk_writes)))
    if args.persistent_cache:
        command.extend(("--persistent-cache", "on"))
    if args.cache_dir is not None:
        command.extend(("--cache-dir", str(args.cache_dir)))
    if args.cache_eviction != "lru":
        command.extend(("--cache-eviction", args.cache_eviction))
    if args.cache_eviction_protect:
        command.extend(("--cache-eviction-protect", str(args.cache_eviction_protect)))
    if args.kv_format != "int8":
        command.extend(("--kv-format", args.kv_format))
    if args.state_format != "fp32":
        command.extend(("--state-format", args.state_format))
    if args.idle_release is not None:
        command.extend(
            ("--idle-release", serve_options.idle_release_text(args.idle_release))
        )
    if args.cache_idle_spill is not None:
        command.extend(
            (
                "--cache-idle-spill",
                serve_options.idle_release_text(args.cache_idle_spill),
            )
        )
    if args.disable_ane:
        command.extend(("--ane", "off"))
    if args.disable_ane_mixer:
        command.extend(("--ane-mixer", "off"))
    if share := os.environ.get("SPLASH_ANE_SHARE"):
        try:
            value = float(share)
        except ValueError:
            value = -1.0
        if not 0.0 <= value < 1.0:
            raise ValueError("SPLASH_ANE_SHARE must be a share from 0 to just below 1")
        if args.disable_ane:
            raise ValueError("SPLASH_ANE_SHARE cannot be combined with --disable-ane")
        command.extend(("--ane-share", share))
    policy = serve_options.thermal_policy(args)
    if args.verify_rows != "auto":
        command.extend(("--verify-rows", args.verify_rows))
    if args.decode_share is not None:
        command.extend(("--decode-share", str(args.decode_share)))
    if args.prefill_slice_ms is not None:
        command.extend(("--prefill-slice-ms", str(args.prefill_slice_ms)))
    if policy.prefill_duty != serve_options.DEFAULT_PREFILL_DUTY:
        command.extend(("--prefill-duty", str(policy.prefill_duty)))
    if args.max_image_pixels != image_input.MAX_PIXELS:
        command.extend(
            ("--max-image-patches", str(image_input.max_patches(args.max_image_pixels)))
        )
    if args.allow_idle_sleep:
        command.extend(("--idle-sleep", "allow"))
    if args.loop_guard:
        command.extend(("--loop-guard", "on"))
    if relaunch_context is not None and args.max_context is None:
        command.extend(("--relaunch-context", str(relaunch_context)))
    for flag, value in (
        ("--ple-memory", args.ple_memory),
        ("--expert-memory", args.expert_memory),
        ("--sparse-kv-memory", args.sparse_kv_memory),
    ):
        if value is not None:
            command.extend((flag, str(value)))
    return command


def _raise_descriptor_limit():
    """Raise the soft descriptor limit to DESCRIPTOR_LIMIT, or to the hard
    limit when that is lower; a higher soft limit stays."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    wanted = DESCRIPTOR_LIMIT
    if hard != resource.RLIM_INFINITY:
        wanted = min(wanted, hard)
    if soft < wanted:
        resource.setrlimit(resource.RLIMIT_NOFILE, (wanted, hard))


def _interrupt(_signum, _frame):
    raise KeyboardInterrupt


class ListenError(Exception):
    """The HTTP server cannot listen at its address."""


def _start_failed(args, error, failure):
    """End a start that failed, saying why in the log and, for the app, in
    the launcher's serve record (serve_record)."""
    print_status(f"Error · {error}", error=True)
    serve_record.failed(args.serve_record, failure)
    raise SystemExit(1) from None


def main():
    args = parse_args()
    server = None
    runtime = None
    backend = None
    # A server started in the background from a non-interactive shell inherits
    # SIGINT as ignored and Python then leaves it alone; install both stop
    # signals explicitly so scripts and supervisors can interrupt it.
    signal.signal(signal.SIGTERM, _interrupt)
    signal.signal(signal.SIGINT, _interrupt)
    try:
        # The launcher blocks both across its exec: one sent while this module
        # imported arrives here and ends the startup cleanly.
        signal.pthread_sigmask(signal.SIG_UNBLOCK, (signal.SIGINT, signal.SIGTERM))
        _raise_descriptor_limit()
        # Bind before loading the tokenizer or model so duplicates fail early.
        # Activate only after the runtime is ready, keeping a partially started
        # service from receiving requests.
        server = FrontendServer(
            (args.host, args.port),
            None,
            bind_and_activate=False,
            request_capacity=args.queue_size,
            allowed_hosts=args.allowed_host,
            api_key=args.api_key,
            webui=not args.no_webui,
            max_request_bytes=args.max_request_size,
            allowed_origins=args.allowed_origin,
        )
        try:
            server.server_bind()
        except OSError as error:
            raise ListenError(f"unable to start HTTP server: {error}") from None
        if ANY_ORIGIN in args.allowed_origin and args.api_key is None:
            print_status(
                "Warning · --allowed-origin '*' without --api-key lets every web "
                "page open in a browser that reaches this server use it",
                error=True,
            )
        thinking_codec = ThinkingCodec(load_thinking_key())
        print_status(f"Loading · {args.model}")
        serve_record.entered(args.serve_record, "loading")
        # Nothing reads `held`: it keeps the assembly held until the server
        # exits, so that no installation collects the assembly its tokenizer
        # and engines, a relaunched one too, serve.
        model_root, held = assembly.hold(Path(args.model_root))
        # The family the installation found the model to be, whose chat
        # format its output is read in.
        family = assembly.family(model_root)
        if family is None:
            raise ModelError(
                f"{args.model} is not prepared as a model of a family this "
                "release serves: prepare it again"
            )
        tokenizer = AutoTokenizer.from_pretrained(
            model_root / "tokenizer", local_files_only=True, trust_remote_code=False
        )
        validate_tokenizer(tokenizer, family.chat_format)
        thinking_end = thinking_end_tokens(
            tokenizer, args.thinking_end_text, family.chat_format
        )
        chat_templates = ChatTemplates(tokenizer)
        print_status(f"Chat template · {chat_templates.describe()}")
        runtime = engine_runtime.MultiplexedRuntime(
            _native_command(args, model_root),
            relaunch_command=lambda first: _native_command(
                args, model_root, first.max_context_tokens
            ),
            startup_timeout=NATIVE_START_TIMEOUT,
            pending_limit=args.queue_size,
        )
        backend = NativeBackend(
            runtime,
            tokenizer,
            request_logger=print_request,
            thinking_end=thinking_end,
            loop_guard=args.loop_guard,
        )
        if not runtime.wait_ready():
            raise engine_runtime.EngineUnhealthy("native runtime did not become ready")
        readiness = runtime.readiness
        # The engine serves --max-context as given, and without it a context
        # the native window holds.
        if readiness is None or not (
            readiness.max_context_tokens == args.max_context
            if args.max_context is not None
            else 1 <= readiness.max_context_tokens <= serve_options.MAX_CONTEXT_TOKENS
        ):
            raise engine_runtime.EngineUnhealthy(
                "native runtime reported an invalid context window"
            )
        effective_context = readiness.max_context_tokens
        constraint_factory = ConstraintFactory(tokenizer, family.chat_format)
        app = Frontend(
            tokenizer,
            backend,
            args.model,
            effective_context,
            # No deadline unless given, as in vLLM and SGLang; a request still
            # ends when its client disconnects.
            math.inf if args.request_timeout is None else args.request_timeout,
            readiness.max_concurrent_requests,
            constraint_factory=constraint_factory,
            chat_templates=chat_templates,
            chat_format=family.chat_format,
            max_image_pixels=args.max_image_pixels,
            thinking_codec=thinking_codec,
            served_model_names=args.served_model_name,
            announce_served_name=args.announce_served_name,
            default_reasoning_effort=args.default_reasoning_effort,
            default_thinking_token_budget=args.thinking_token_budget,
            compact_at=args.compact_at,
            compact_keep=args.compact_keep,
            compact_memory=args.compact_memory,
            vision=readiness.vision,
        )
        server.app = app
        server.server_activate()
        # The app takes the server for ready from its record alone: one that
        # cannot say so has not started (serve_record).
        serve_record.ready(args.serve_record)
        address = f"http://{args.host}:{server.server_port}"
        context = (
            f"{effective_context // 1024}K"
            if effective_context % 1024 == 0
            else f"{effective_context:,}"
        )
        mode = "" if readiness.vision else " · text only"
        print_status(f"Ready · {args.model} · context {context}{mode} · {address}")
        server.serve_forever()
    except ListenError as error:
        _start_failed(args, error, serve_record.failure(serve_record.PORT, str(error)))
    except (
        ModelError,
        ChatTemplateError,
        ThinkingEndError,
    ) as error:
        _start_failed(args, error, serve_record.failure(serve_record.MODEL, str(error)))
    except (engine_runtime.EngineRuntimeError, ThinkingKeyError, OSError) as error:
        _start_failed(
            args, error, serve_record.failure(serve_record.INTERNAL, str(error))
        )
    except serve_record.NotRecorded as error:
        # The record that cannot say ready cannot say why either: the app
        # takes the exit for the failure, which the log words.
        print_status(f"Error · {error}", error=True)
        raise SystemExit(1) from None
    except Exception as error:
        # An error no clause above names is a bug: the record still names the
        # failure, so that the app reports it instead of a crash, and the
        # traceback reaches the log.
        serve_record.failed(
            args.serve_record,
            serve_record.failure(
                serve_record.INTERNAL, f"{type(error).__name__}: {error}"
            ),
        )
        raise
    except KeyboardInterrupt:
        pass
    finally:
        # main owns this process. Keep stop signals idempotent through child
        # cleanup and interpreter teardown, including after this function
        # returns, except that a second Ctrl+C during cleanup stops the engine
        # without waiting for its graceful exit.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(
            signal.SIGINT,
            signal.SIG_IGN if runtime is None else lambda *_: runtime.kill(),
        )
        try:
            if backend is not None:
                print_status("Stopping · releasing engine resources")
                backend.close()
        finally:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            if server is not None:
                server.server_close()


if __name__ == "__main__":
    main()
