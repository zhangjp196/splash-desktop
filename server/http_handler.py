"""The HTTP request handler every API's answers flow through.

One class routes each request by the path table (routes), reads and bounds
its body, and hands the prepared generations to the backend; the answer
writers it holds are the public surface a dialect's handlers call.
"""

import json
import queue
import re
import secrets
import select
import socket
import time
from dataclasses import dataclass, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from urllib.parse import unquote

from . import json_codec, serve_options
from .backend import CacheInfo, NativeResult, remaining_request_time
from .chat_assets import CHAT_ASSETS, CHAT_HTML, FAVICON_SVG
from .connections import RequestBodyReservation
from .diagnostics import log_unexpected, print_status
from .errors import (
    RETRY_STATUSES,
    APIError,
    RequestValidationError,
    field_error,
)
from .http_security import OriginRefused, authenticate, validate_headers
from .latency import RequestLatency
from .metrics import prometheus_metrics
from .origins import ANY_ORIGIN
from .output import (
    ReasoningSplitter,
    StreamingToolCallProjector,
    validate_response_content,
)
from .routes import POST_ROUTES, path_errors

HTTP_IO_TIMEOUT = 30.0
HTTP_UPLOAD_BYTES_PER_SECOND = 512 * 1024
# Native events wake a waiting request at once; this only bounds how late a
# client disconnect is noticed.
CLIENT_DISCONNECT_POLL = 0.1
SSE_KEEPALIVE_SECONDS = 2.0


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
        # One poller for the connection's whole life: _client_disconnected
        # runs at every token event, and a poll object holds no descriptor,
        # so registering once asks the same question of the same socket
        # without rebuilding the poller each time.
        self._disconnect_poller = select.poll()
        self._disconnect_poller.register(self.connection, select.POLLIN)

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
            # A scrape reads the engine through the probes' short-TTL cache:
            # scrapers and health checkers poll far faster than an engine's
            # status changes, and each scrape must not pay a native round
            # trip. /status stays live for the console's diagnostics.
            self._send(
                200,
                prometheus_metrics(self.server.status(cached=True)).encode(),
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
            speculation = self.app.backend.status(cached=True).get("speculation")
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
                    "speculation": speculation,
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
        # The route table calls every POST handler with (body, deadline);
        # this endpoint reads neither, taking no fields and answering at
        # once, and keeps the uniform signature.
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
        # A handler that never ran setup, as a test's bare instance, has no
        # poller yet; make one then, as setup would have.
        poller = getattr(self, "_disconnect_poller", None)
        if poller is None:
            poller = select.poll()
            poller.register(self.connection, select.POLLIN)
            self._disconnect_poller = poller
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
            tops = []
            for top_token, top_logprob in top:
                # One decode serves both the text and its byte lengths.
                top_text = self.app.tokenizer.decode(
                    [top_token], skip_special_tokens=False
                )
                tops.append(
                    {
                        "token": top_text,
                        "logprob": top_logprob,
                        "bytes": list(top_text.encode("utf-8")),
                    }
                )
            entries.append(
                {
                    "token": text,
                    "logprob": logprob,
                    "bytes": list(text.encode("utf-8")),
                    "top_logprobs": tops,
                }
            )
        # A longer join of the entries only extends a shorter one, so the
        # joins that are a prefix of `content` are exactly the first ones:
        # one pass from the front finds where they stop being a prefix, and
        # the entries up to there are what the repeated pop-and-rejoin of a
        # trim from the end would have left.
        joined = ""
        for cut, entry in enumerate(entries):
            longer = joined + entry["token"]
            if not content.startswith(longer):
                entries = entries[:cut]
                break
            joined = longer
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
