"""The serving process's startup: the command line, the engine's command,
and `main`, which assembles the backend and the server and serves.

The launcher starts this as ``python -m server.server`` (server.py, the
entry module, forwards here); the app follows the start through the serve
record this writes (serve_record).
"""

import argparse
import math
import os
import resource
import signal
import threading
from pathlib import Path

from transformers import AutoTokenizer

from install import assembly
from install.models import ModelError

from . import diagnostics, serve_options, serve_record
from . import images as image_input
from . import runtime as engine_runtime
from .backend import NativeBackend, ThinkingEndError, thinking_end_tokens
from .chat_templates import ChatTemplateError, ChatTemplates
from .constraints import ConstraintFactory, validate_tokenizer
from .diagnostics import print_request, print_status
from .frontend import Frontend
from .http_server import FrontendServer
from .origins import ANY_ORIGIN
from .thinking import ThinkingCodec, ThinkingKeyError, load_thinking_key

# The soft descriptor limit the server raises its own to: macOS starts a
# process with 256, fewer than its connection slots, the refused connections
# lingering after a burst and the engine's pipes and files can need. 10240 is
# macOS's OPEN_MAX.
DESCRIPTOR_LIMIT = 10240
NATIVE_START_TIMEOUT = 600.0
ROOT = Path(__file__).parents[1]


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
    if args.cache_idle_evict is not None:
        command.extend(("--cache-idle-evict", args.cache_idle_evict))
    if args.residency_keepalive is not None:
        command.extend(
            (
                "--residency-keepalive",
                serve_options.idle_release_text(args.residency_keepalive),
            )
        )
    if args.expert_memory_grow is not None:
        command.extend(("--expert-memory-grow", args.expert_memory_grow))
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


class EngineStopped(Exception):
    """The engine stopped for good: no relaunch can serve, so the process
    ends and leaves the restart to its supervisor (the launcher or the app)
    instead of answering 500s to every request."""


# The exit status of a server whose engine stopped for good: sysexits'
# EX_UNAVAILABLE, "a service is unavailable", which a supervisor reads as
# "restart me" rather than as a crash of the server itself.
EX_UNAVAILABLE = 69


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
        # From here the process serves: its console becomes a bounded writer
        # that a request thread or the engine's reader never waits on.
        diagnostics.start_console()
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
        # The engine stopping for good ends the process: a server that can
        # only answer 500s is worse than one its supervisor restarts.
        stopped = threading.Event()
        stop_reason = []

        def engine_stopped(reason):
            stop_reason.append(reason)
            stopped.set()
            # shutdown() blocks until serve_forever returns, so it runs on a
            # thread of its own.
            threading.Thread(
                target=server.shutdown,
                name="splash-engine-stopped",
                daemon=True,
            ).start()

        backend.on_engine_stopped = engine_stopped
        server.serve_forever()
        if stopped.is_set():
            raise EngineStopped(stop_reason[0] if stop_reason else "engine stopped")
    except EngineStopped as error:
        # The record says why, so the app words it instead of a bare status;
        # the exit status tells a supervisor to start a new server.
        serve_record.failed(
            args.serve_record, serve_record.failure(serve_record.ENGINE, str(error))
        )
        print_status(f"Error · {error}", error=True)
        raise SystemExit(EX_UNAVAILABLE) from None
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
            # The last lines the console holds reach it before the process
            # ends; after this a print writes in place again.
            diagnostics.close_console()
