"""The serving process's entry module.

The launcher and the packaged app start the server as ``python -m
server.server``; this module runs that start and gathers the serving
surface — the HTTP handler, the listening server and the startup — so
tests and tools reach them from one import. The handler lives in
http_handler, the server in http_server, the start in serve.
"""

import queue
import resource
import secrets
import signal
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from install import assembly

from . import json_codec, serve_options
from . import runtime as engine_runtime
from .backend import NativeBackend
from .chat_assets import CHAT_ASSETS
from .connections import ConnectionSlots, HttpAdmission, RequestBodyReservation
from .errors import APIError
from .fast_tokenizer import FastTokenizer
from .http_handler import (
    HTTP_IO_TIMEOUT,
    HTTP_UPLOAD_BYTES_PER_SECOND,
    SSE_KEEPALIVE_SECONDS,
    FrontendHandler,
)
from .http_server import DEFAULT_REQUEST_BODY_BUDGET, FrontendServer
from .routes import POST_ROUTES, path_errors
from .serve import (
    DESCRIPTOR_LIMIT,
    NATIVE_START_TIMEOUT,
    ROOT,
    ListenError,
    _interrupt,
    _native_command,
    _raise_descriptor_limit,
    main,
    parse_args,
)

__all__ = (
    "CHAT_ASSETS",
    "POST_ROUTES",
    "path_errors",
    "FrontendHandler",
    "HTTP_IO_TIMEOUT",
    "HTTP_UPLOAD_BYTES_PER_SECOND",
    "SSE_KEEPALIVE_SECONDS",
    "DEFAULT_REQUEST_BODY_BUDGET",
    "FrontendServer",
    "DESCRIPTOR_LIMIT",
    "NATIVE_START_TIMEOUT",
    "ROOT",
    "ListenError",
    "_interrupt",
    "_native_command",
    "_raise_descriptor_limit",
    "main",
    "parse_args",
    "assembly",
    "engine_runtime",
    "json_codec",
    "serve_options",
    "ConnectionSlots",
    "HttpAdmission",
    "RequestBodyReservation",
    "APIError",
    "FastTokenizer",
    "BaseHTTPRequestHandler",
    "HTTPServer",
    "NativeBackend",
    "queue",
    "resource",
    "secrets",
    "signal",
    "sys",
    "time",
)

if __name__ == "__main__":
    main()
