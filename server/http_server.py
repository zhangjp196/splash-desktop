"""The listening HTTP server: connection admission and the status document.

FrontendServer accepts on a nonblocking socket, hands every admitted
connection to a worker, and refuses the rest on terms their clients can
read, while the handler answers the requests themselves.
"""

import os
import secrets
import sys
import time
from http.server import HTTPServer

from . import diagnostics, http_handler, serve_options
from .connections import ConnectionSlots, HttpAdmission
from .http_handler import FrontendHandler
from .http_security import RefusedOriginLog

# Request bodies held at once, from upload through preparation and, for what
# a generation retains of them, until it ends, take at most max(this, twice
# --max-request-size) bytes, so two of the largest requests always fit.
DEFAULT_REQUEST_BODY_BUDGET = 512 * 1024 * 1024
# How long a connection refused unread may take its client to close.
REFUSED_LINGER_SECONDS = 2.0


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

    def status(self, cached=False):
        status = self.app.status(cached=cached)
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
        # The console is its own backpressure domain: a line it could not take
        # is counted here rather than waited on by a request.
        status["console"] = diagnostics.console_stats()
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
            request, client_address, time.monotonic() + http_handler.HTTP_IO_TIMEOUT
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
        self.connections.idle.wait(min(2.0, http_handler.HTTP_IO_TIMEOUT))

    def handle_error(self, request, client_address):
        error = sys.exc_info()[1]
        if isinstance(error, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)
