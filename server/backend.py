"""Native request submission, cancellation, recovery and completion ownership."""

import functools
import hashlib
import math
import queue
import signal
import threading
import time
from dataclasses import dataclass, field, replace

from . import json_codec
from . import protocol as wire
from . import runtime as engine_runtime
from .chat_format import THINK_END
from .diagnostics import print_status
from .errors import APIError, ConstraintError
from .job import CacheInfo, Job, NativeResult
from .streamer import CallbackStreamer

# A failure counts toward a crash loop unless its engine served this long.
CRASH_LOOP_WINDOW_SECONDS = 60.0
# Consecutive such failures, failed relaunches included, that stop relaunching.
# Six: the first waits 0, then 5 s doubling to the cap at the sixth.
CRASH_LOOP_LIMIT = 6
# The n-th consecutive relaunch waits RESTART_BACKOFF_SECONDS * 2**(n-2), up
# to RESTART_BACKOFF_CAP_SECONDS; the first waits 0.
RESTART_BACKOFF_SECONDS = 5.0
RESTART_BACKOFF_CAP_SECONDS = 60.0
# A start that failed for want of memory is retried for as long as it fails,
# at most the cap apart, and does not count toward the crash loop: the memory
# a request or another application frees may let it start.
MEMORY_START_MARKERS = (
    "memory",
    "allocation",
    "allocate",
    "budget",
    "does not fit",
    "out of memory",
    "insufficient",
)
# The signals an engine that a request crashed may die by.
FAULT_SIGNALS = frozenset(
    {
        signal.SIGABRT,
        signal.SIGBUS,
        signal.SIGFPE,
        signal.SIGILL,
        signal.SIGSEGV,
        signal.SIGTRAP,
    }
)


class Quarantine:
    """Refuses a request, until the server restarts, whose fingerprint was
    among the requests the engine had started at each of the last two
    failures a request may have caused. A failure no request could have
    caused neither adds to the suspects nor clears them."""

    def __init__(self):
        self._lock = threading.Lock()
        self._suspects = frozenset()
        self._refused = frozenset()

    def refuses(self, fingerprint):
        return fingerprint in self._refused

    def after(self, started):
        with self._lock:
            refused = (self._suspects & started) - self._refused
            self._refused |= refused
            self._suspects = started
            return refused


def request_fingerprint(job):
    """A request's identity across retries: its prompt, sampling and seed,
    so a client that resends the same request can be recognised."""
    digest = hashlib.sha256()
    digest.update(repr(job.prompt_tokens).encode())
    digest.update(repr(job.max_new_tokens).encode())
    digest.update(repr(job.seed).encode())
    digest.update(repr(tuple(job.stop_sequences)).encode())
    digest.update(
        repr(
            tuple(getattr(job.sampling, name) for name in wire.SAMPLING_FIELDS)
        ).encode()
    )
    return digest.digest()


# Control requests use a short live probe and explicitly label stale snapshots.
STATUS_REFRESH_TIMEOUT_SECONDS = 0.05
# How long a status answer from a serving engine stands in for the frequent
# probes (/ready, /metrics) before one of them asks the engine again. Half a
# second is well under the one-second Retry-After every SDK honors and under
# any scraper's interval, so a readiness flip shows at the next probe after
# it, while a burst of probes costs one native round trip instead of one
# each. The cache answers only while the engine document says ready and the
# transport is live and owes no relaunch or refresh: an engine that has
# failed, is recovering, or has gone quiet is probed live, so no flip out of
# readiness hides behind the TTL. What it does mask, for at most the TTL, is
# an engine turning not-ready under memory pressure or a Metal fault while
# it keeps serving.
STATUS_CACHE_TTL_SECONDS = 0.5


class ThinkingEndError(ValueError):
    """A --thinking-end-text the model's tokenizer cannot end thinking with."""


def thinking_end_tokens(tokenizer, text, chat_format):
    """The tokens the engine selects where a thinking budget runs out:
    --thinking-end-text's, which must encode as text then chat_format's
    close of thinking as its own token, at most wire.MAX_THINKING_END_TOKENS
    tokens. The text holds none of the tokenizer's added tokens, the
    template's special tokens. None without a text, where the close alone
    ends thinking (NativeBackend)."""
    if text is None:
        return None
    tokens = tuple(tokenizer.encode(text, add_special_tokens=False))
    if (
        not tokens
        or tokens[-1] != chat_format.think_end_id
        or not tokenizer.added_tokens_decoder.keys().isdisjoint(tokens[:-1])
    ):
        raise ThinkingEndError(
            f"--thinking-end-text must end with {THINK_END} and hold no other "
            "special token"
        )
    if len(tokens) > wire.MAX_THINKING_END_TOKENS:
        raise ThinkingEndError(
            f"--thinking-end-text takes {len(tokens)} tokens; the most is "
            f"{wire.MAX_THINKING_END_TOKENS}"
        )
    return tokens


def remaining_request_time(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise APIError(504, "request timed out", "request_timeout")
    # Callers wait this long, and waits reject a timeout above TIMEOUT_MAX;
    # a request without a deadline has infinite time left.
    return min(remaining, threading.TIMEOUT_MAX)


@dataclass
class _JobState:
    job: Job
    streamer: CallbackStreamer
    call: object | None = None
    terminal_enqueued: bool = False
    # Whether the engine has started it, and its identity across retries.
    started: bool = False
    fingerprint: bytes = b""
    detached: bool = False
    shutdown_requested: bool = False
    first_token_batch_tokens: int = 0
    logprobs: list = field(default_factory=list)

    def detach(self):
        self.detached = True
        self.call = None
        self.streamer.on_stop = None


class NativeBackend:
    """Write admitted requests directly to native inference.

    A CPU finalizer runs tokenizer flushes and request logging off the reader
    thread.
    """

    _FINISH_NAMES = {
        wire.FinishReason.STOP: "stop",
        wire.FinishReason.LENGTH: "length",
        wire.FinishReason.CANCELLED: "cancelled",
    }

    def __init__(
        self,
        runtime,
        tokenizer,
        request_logger,
        thinking_end=None,
        loop_guard=False,
    ):
        self.runtime = runtime
        self.tokenizer = tokenizer
        self.request_logger = request_logger
        # What the engine selects where a thinking budget runs out
        # (thinking_end_tokens); None for the close of thinking alone, in
        # each job's chat format.
        self.thinking_end = thinking_end
        # `--loop-guard`: every thinking request carries a closable thinking
        # budget, so the engine can end a repeated cycle early.
        self.loop_guard = loop_guard
        self.active = {}
        self.closing = False
        self.lock = threading.RLock()
        self.status_snapshot = None
        self.status_snapshot_at = None
        # Since when the engine loop has left status probes unanswered.
        self.status_unanswered_since = None
        # Recovery state, under self.lock: the one background worker, when a
        # relaunch is owed (monotonic), whether a status refresh is owed,
        # consecutive failures without a full window of service, and why
        # relaunching stopped for good.
        self.recovery = None
        self.restart_due = None
        self.refresh_due = False
        self.failures = 0
        # Starts that failed for want of memory, in a row, which back off but
        # never stop relaunching.
        self.memory_start_failures = 0
        self.fatal_error = None
        # Requests refused until the server restarts: those the engine had
        # started at two failures in a row a request may have caused.
        self.quarantine = Quarantine()
        # Set by close() to cut a relaunch's backoff short.
        self.wakeup = threading.Event()
        # Why the engine cannot serve: its failure, the last failed restart or
        # why restarts stopped.
        self.engine_error = None
        # What the finalizer runs, in order: requests' Started lines and
        # terminals, which must not hold up the runtime's reader thread.
        self.finalizer_queue = queue.Queue()
        self.finalizer = threading.Thread(
            target=self._finalize_loop,
            name="splash-http-finalizer",
            daemon=True,
        )
        self.finalizer.start()
        runtime.on_engine_failure = self._engine_failed

    def _engine_failed(self, error, served_seconds):
        """The runtime's failure listener: an engine that reached Ready failed
        after serving for `served_seconds`."""
        self._failed("Engine failed", error, served_seconds)

    def _failed(self, event, error, served_seconds):
        # Clients get the reason with every refusal until a status succeeds.
        returncode = getattr(error, "returncode", None)
        with self.lock:
            if self.closing or self.fatal_error is not None:
                return
            started = frozenset(
                state.fingerprint
                for state in self.active.values()
                if state.started and state.fingerprint
            )
            self.engine_error = str(error)
            if self.status_snapshot is not None:
                # A failed engine's readiness is no longer evidence.
                self.status_snapshot["ready"] = False
            if (unchangeable := self.runtime.fatal_error) is not None:
                # Changed limits would recur at every relaunch; the error says
                # how to recover.
                self._stop_restarting_locked(str(unchangeable))
            elif (
                delay := self._restart_delay_locked(
                    served_seconds, self._wanted_memory(error, served_seconds)
                )
            ) is None:
                # The runtime writes an engine's crash trace before it reports
                # that engine's failure, to the listener or from wait_ready.
                trace = self.runtime.last_crash_trace
                self._stop_restarting_locked(
                    f"the inference engine failed {self.failures} times in a row, "
                    f"each within {CRASH_LOOP_WINDOW_SECONDS:.0f} s of starting "
                    f"({error}); Splash stopped restarting it. "
                    + (
                        f"Crash trace: {trace}. "
                        if trace
                        else "Set SPLASH_CRASH_TRACE=1 to record a crash trace. "
                    )
                    + "Restart the Splash server after fixing the cause."
                )
            else:
                self.restart_due = time.monotonic() + delay
            stopped = self.fatal_error
            self._ensure_recovery_locked()
        # A signal a request may have crashed the engine with counts: a
        # request started at each of two such failures in a row is refused
        # until the server restarts, rather than crashing it again.
        if returncode is not None and returncode < 0 and -returncode in FAULT_SIGNALS:
            refused = self.quarantine.after(started)
            if refused:
                print_status(
                    f"Quarantine · {len(refused)} request"
                    f"{'' if len(refused) == 1 else 's'} in flight at two engine "
                    "failures in a row refused until the server restarts",
                    error=True,
                )
        print_status(f"{event} · {error}", error=True)
        if stopped is not None:
            print_status(f"Engine stopped · {stopped}", error=True)

    @staticmethod
    def _wanted_memory(error, served_seconds):
        """Whether a start failed for want of memory, which a later one may
        meet once memory is freed, rather than for a setting or a model file
        every relaunch would meet again."""
        if served_seconds > 0.0:
            return False
        message = str(error).lower()
        return any(marker in message for marker in MEMORY_START_MARKERS)

    def _restart_delay_locked(self, served_seconds, wanted_memory=False):
        """Seconds until the relaunch a failure owes; None once a crash loop
        stops relaunching. A start that wanted memory backs off and is
        retried without limit, and neither counts toward the crash loop nor
        clears it."""
        if wanted_memory:
            self.memory_start_failures += 1
            if self.memory_start_failures == 1:
                return 0.0
            return min(
                RESTART_BACKOFF_CAP_SECONDS,
                RESTART_BACKOFF_SECONDS * 2 ** (self.memory_start_failures - 2),
            )
        self.memory_start_failures = 0
        if served_seconds >= CRASH_LOOP_WINDOW_SECONDS:
            self.failures = 0
        self.failures += 1
        if self.failures >= CRASH_LOOP_LIMIT:
            return None
        if self.failures == 1:
            return 0.0
        return min(
            RESTART_BACKOFF_CAP_SECONDS,
            RESTART_BACKOFF_SECONDS * 2 ** (self.failures - 2),
        )

    def _stop_restarting_locked(self, message):
        """Stop relaunching for good; there is no way back but a server
        restart."""
        self.restart_due = None
        self.fatal_error = message
        self.engine_error = message

    def _ensure_recovery_locked(self):
        """Start the recovery worker when work is owed and none runs.

        Whoever finds work owed calls this, so a worker that failed to start
        is started by the next probe or request."""
        if (
            self.closing
            or self.fatal_error is not None
            or (self.restart_due is None and not self.refresh_due)
            or (self.recovery is not None and self.recovery.is_alive())
        ):
            return
        self.recovery = threading.Thread(
            target=self._recover,
            name="splash-engine-recovery",
            daemon=True,
        )
        try:
            self.recovery.start()
        except BaseException:
            self.recovery = None
            raise

    def _recover(self):
        """Run owed relaunches, then owed status refreshes, until none is
        owed."""
        while True:
            with self.lock:
                if self.closing or self.fatal_error is not None:
                    self.recovery = None
                    return
                due = self.restart_due
                refresh = due is None and self.refresh_due
                if refresh:
                    self.refresh_due = False
                elif due is None:
                    self.recovery = None
                    return
            if due is not None:
                if not self.wakeup.wait(max(0.0, due - time.monotonic())):
                    self._relaunch(due)
                continue
            try:
                restarts = self.runtime.restart_count
                event = self.runtime.status(
                    timeout=engine_runtime.STATUS_ANSWER_LIMIT_SECONDS,
                    fail_unanswered=True,
                )
                self._cache_status(self._decode_status_event(event), restarts)
            except Exception as error:
                # The snapshot stays. A refresh left unanswered has failed its
                # engine, whose relaunch is now owed; after any other failure
                # no status request is left waiting.
                if not isinstance(error, TimeoutError):
                    with self.lock:
                        self.status_unanswered_since = None

    def _relaunch(self, due):
        try:
            # It returns once the engine is Ready. The failure listener counts
            # a failure after Ready, even one before wait_ready returns.
            self.runtime.wait_ready()
        except Exception as error:
            self._failed("Engine restart failed", error, 0.0)
            return
        with self.lock:
            # A failure after Ready may already owe the next relaunch.
            if self.restart_due == due:
                self.restart_due = None
            self.refresh_due = True

    def refusal(self):
        """The error a request not yet admitted gets now; None while the
        engine serves."""
        with self.lock:
            if self.closing:
                return APIError(503, "server is shutting down", "server_shutdown")
            if self.fatal_error is not None:
                return APIError(500, self.fatal_error, "engine_failed")
            self._ensure_recovery_locked()
            if self.runtime.ready:
                return None
            failure = self.engine_error
            due = self.restart_due
        # The longest Retry-After every OpenAI and Anthropic SDK follows: the
        # JavaScript ones follow less than a minute, the Python ones up to a
        # minute.
        retry_after = 1
        if due is not None:
            retry_after = max(1, min(59, math.ceil(due - time.monotonic())))
        return APIError(
            503,
            "engine is recovering; retry shortly"
            + (f" (last failure: {failure})" if failure else ""),
            "engine_recovering",
            retry_after,
        )

    def is_ready(self):
        # The engine's own ready folds in memory pressure and Metal health.
        return self.status(cached=True)["ready"] is True

    @staticmethod
    def _decode_status_event(event):
        snapshot = json_codec.loads(event.json)
        if (
            not isinstance(snapshot, dict)
            or snapshot.get("schema_version") != wire.STATUS_SCHEMA_VERSION
        ):
            raise ValueError("native status does not match the current schema")
        return snapshot

    def _cache_status(self, snapshot, restarts):
        """Cache `snapshot` unless the engine that answered it has failed or
        been replaced since `restarts` was read; return whether it was cached."""
        with self.lock:
            # An answer from an engine that has failed since is no evidence,
            # even once a relaunched one serves. Ready is read first, so an
            # unchanged count means the engine found Ready is the one that
            # answered.
            if (
                self.closing
                or not self.runtime.ready
                or self.runtime.restart_count != restarts
            ):
                return False
            # A shallow copy serves: every later change to a snapshot is a
            # top-level key assignment (`ready`, `transport`), never a write
            # into a nested document, so the cache shares the engine's
            # sections with the copies handed out.
            self.status_snapshot = dict(snapshot)
            self.status_snapshot_at = time.monotonic()
            self.status_unanswered_since = None
            restarted = self.engine_error is not None
            self.engine_error = None
        if restarted:
            print_status("Engine restarted")
        return True

    def status(self, timeout=STATUS_REFRESH_TIMEOUT_SECONDS, cached=False):
        if cached:
            # A fresh answer from a serving engine is evidence enough for a
            # probe; the transport accounting below is still read live.
            with self.lock:
                snapshot = self.status_snapshot
                captured_at = self.status_snapshot_at
                if (
                    snapshot is not None
                    and captured_at is not None
                    and time.monotonic() - captured_at < STATUS_CACHE_TTL_SECONDS
                    and snapshot.get("ready") is True
                    and not self.closing
                    and self.runtime.ready
                    and self.restart_due is None
                    and not self.refresh_due
                ):
                    age_ms = (time.monotonic() - captured_at) * 1000.0
                    return self._add_transport(dict(snapshot), None, age_ms)
        stale_error = None
        stale_age_ms = None
        with self.lock:
            # The worker refreshes status unless a relaunch is owed. A probe
            # would wait behind its request in a busy loop.
            refresh_pending = (
                self.recovery is not None
                and self.restart_due is None
                and self.status_snapshot is not None
            )
        try:
            if refresh_pending:
                raise TimeoutError("native status refresh is pending")
            restarts = self.runtime.restart_count
            event = self.runtime.status(timeout=timeout)
            snapshot = self._decode_status_event(event)
        except Exception as error:
            stale_error = error
            busy = isinstance(error, TimeoutError)
            now = time.monotonic()
            with self.lock:
                # Shallow like the cache itself: the caller's changes land on
                # top-level keys, and the engine's sections are read-only.
                # No cache yet copies as no cache at all.
                snapshot = (
                    dict(self.status_snapshot)
                    if self.status_snapshot is not None
                    else None
                )
                captured_at = self.status_snapshot_at
                transport_ready = not self.closing and self.runtime.ready
                # A probe skipped behind the refresh sent nothing to answer.
                if busy and not refresh_pending:
                    if self.status_unanswered_since is None:
                        self.status_unanswered_since = now
                    # The refresh waits as long as a busy loop may take, and
                    # fails the engine of a loop that never answers.
                    if transport_ready:
                        self.refresh_due = True
                unanswered = self.status_unanswered_since
                self._ensure_recovery_locked()
            if snapshot is None or captured_at is None:
                snapshot = {
                    "schema_version": wire.STATUS_SCHEMA_VERSION,
                    "ready": False,
                }
            else:
                stale_age_ms = max(0.0, (now - captured_at) * 1000.0)
                # A busy loop keeps its last readiness until it has left a
                # status request unanswered as long as a refresh waits for one;
                # any other failure is not evidence of readiness.
                snapshot["ready"] = (
                    busy
                    and snapshot.get("ready") is True
                    and (
                        unanswered is None
                        or now - unanswered < engine_runtime.STATUS_ANSWER_LIMIT_SECONDS
                    )
                )
        else:
            if not self._cache_status(snapshot, restarts):
                snapshot["ready"] = False
        return self._add_transport(snapshot, stale_error, stale_age_ms)

    def _add_transport(self, snapshot, stale_error, age_ms):
        """The transport accounting every status answer carries, read live
        however the engine document behind it was obtained. `age_ms` is how
        old the engine document is: zero for one answered now, the cache's
        age for one a probe reused."""
        with self.lock:
            transport_ready = not self.closing and self.runtime.ready
            engine_error = self.engine_error
            stopped = self.fatal_error is not None
        snapshot["transport"] = {
            "ready": transport_ready,
            "recovering": not self.closing and not stopped and not transport_ready,
            "stopped": stopped,
            "pending": self.runtime.pending_count,
            "pending_limit": self.runtime.pending_limit,
            "restarts": self.runtime.restart_count,
            "last_crash_trace": self.runtime.last_crash_trace,
            "status_stale": stale_error is not None,
            "status_age_ms": 0.0 if age_ms is None else age_ms,
        }
        if stale_error is not None:
            snapshot["transport"]["error"] = engine_error or str(stale_error)
        if not transport_ready:
            snapshot["ready"] = False
        return snapshot

    @staticmethod
    def _mask_provider(job):
        if job.constraint is None:
            return None

        def provide(event):
            # runtime sends the simulated context sequence, not merely draft
            # proposals: [] for the initial mask, then [pending anchor,
            # draft...] for verification. TokenConstraint returns the mask
            # before the first simulated token and after each token.
            return job.constraint.masks(event.simulation_tokens)

        return provide

    def _generation_request(self, job):
        constraint = (
            wire.ConstraintMode.TOKEN_MASK
            if job.constraint is not None
            else wire.ConstraintMode.NONE
        )
        frame = wire.RequestFrame(
            request_id=0,
            priority=job.priority,
            absolute_deadline_unix_micros=0,
            remaining_deadline_micros=0,
            logical_max_output_tokens=job.max_new_tokens,
            prompt_tokens=tuple(job.prompt_tokens),
            sampling=job.sampling,
            seed=job.seed,
            constraint=constraint,
            image_spans=job.image_spans,
            image_pixels=job.image_pixels,
            score_tokens=job.score_tokens,
            generation_prompt_tokens=job.generation_prompt_tokens,
            flags=(
                job.flags | wire.RequestFlag.LOGPROBS if job.logprobs else job.flags
            ),
            top_logprobs=job.top_logprobs,
            bias_tokens=tuple(token for token, _ in job.logit_bias),
            bias_values=tuple(value for _, value in job.logit_bias),
            cache_salt=job.cache_salt,
            thinking_budget=self._thinking_budget(job),
        )
        return engine_runtime.GenerationRequest(
            frame, job.deadline, self._mask_provider(job), job.image_owner
        )

    def _thinking_budget(self, job):
        """The engine's cutoff of the thinking a job generates in, which it
        ends with the server's thinking end. The engine's thinking ends at its
        close or, where a call may follow and a call's opening ends the
        reasoning, at any call-open token, while count_reasoning_tokens ends
        the reasoning at a call-open token only where it begins the call's
        opening: a stray call-open token in the thinking ends the engine's
        count, and the reasoning can then run past the budget. With
        --loop-guard a thinking job without a budget carries an uncapped one,
        so the engine can close a repeated cycle (it is otherwise inert)."""
        if not job.thinking:
            return None
        if job.thinking_token_budget is not None:
            tokens = job.thinking_token_budget
        elif self.loop_guard:
            tokens = wire.NO_THINKING_BUDGET
        else:
            return None
        chat_format = job.chat_format
        return wire.ThinkingBudget(
            tokens,
            self.thinking_end or (chat_format.think_end_id,),
            chat_format.call_open_id
            if job.may_call_tools and chat_format.calls_end_reasoning
            else chat_format.think_end_id,
        )

    def release_weights(self, timeout=5.0):
        """Asks the engine to give its weights back now; returns its status.

        It releases them only while no request is in service and no restore is
        under way, with `--idle-release off` too; the returned snapshot's
        `weights` says whether they are released: `released` and not
        `restoring`. The next request restores the weights as after the idle
        release.
        """
        with self.lock:
            if self.closing:
                raise APIError(503, "server is shutting down", "server_shutdown")
        restarts = self.runtime.restart_count
        try:
            event = self.runtime.release_weights(timeout=timeout)
        except TimeoutError:
            # The engine answers between commands: it is running a long one.
            raise APIError(
                503,
                f"the engine did not answer within {timeout:g} s; retry shortly",
                "engine_busy",
            ) from None
        except (
            engine_runtime.EngineUnhealthy,
            engine_runtime.ProtocolFatal,
            engine_runtime.RuntimeClosed,
        ) as error:
            raise self.refusal() or self._api_error(error) from None
        snapshot = self._decode_status_event(event)
        self._cache_status(snapshot, restarts)
        return snapshot

    def set_prefill_duty(self, duty_percent, timeout=5.0):
        """Change a running engine's prefill duty; returns its status.

        The `throttle.duty_percent` of the returned snapshot is the duty the
        engine took: it clamps to what a configuration may ask for, and the
        engine holds the new duty from its next prefill command on.
        """
        with self.lock:
            if self.closing:
                raise APIError(503, "server is shutting down", "server_shutdown")
        restarts = self.runtime.restart_count
        try:
            event = self.runtime.set_prefill_duty(duty_percent, timeout=timeout)
        except TimeoutError:
            raise APIError(
                503,
                f"the engine did not answer within {timeout:g} s; retry shortly",
                "engine_busy",
            ) from None
        except (
            engine_runtime.EngineUnhealthy,
            engine_runtime.ProtocolFatal,
            engine_runtime.RuntimeClosed,
        ) as error:
            raise self.refusal() or self._api_error(error) from None
        snapshot = self._decode_status_event(event)
        self._cache_status(snapshot, restarts)
        return snapshot

    def submit(self, job):
        state = None

        def stop_matched():
            call = None
            with self.lock:
                call = state.call
            if call is not None:
                call.cancel()

        def emit(text):
            job.events.put(("text", text))

        streamer = CallbackStreamer(
            self.tokenizer, emit, job.stop_sequences, stop_matched
        )
        state = _JobState(job, streamer)
        state.fingerprint = request_fingerprint(job)
        request = self._generation_request(job)
        # The engine copies the pixels at admission; only the frame written
        # below needs them, and the job lives until the response ends.
        job.image_pixels = b""

        def on_event(call, event):
            with self.lock:
                if state.detached:
                    return
                if state.call is None:
                    state.call = call
                cancel = job.cancelled.is_set() or self.closing
            # A raise here is the call's callback error, which cancels it.
            self._on_event(state, event)
            if cancel:
                call.cancel()

        def on_complete(call):
            with self.lock:
                if state.detached or state.terminal_enqueued:
                    return
                if state.call is None:
                    state.call = call
                state.terminal_enqueued = True
                self.finalizer_queue.put(functools.partial(self._finalize, state, call))

        with self.lock:
            if self.closing:
                state.detach()
                job.events.put(
                    (
                        "error",
                        APIError(
                            503,
                            "server is shutting down",
                            "server_shutdown",
                        ),
                    )
                )
                return
            if self.quarantine.refuses(state.fingerprint):
                state.detach()
                trace = self.runtime.last_crash_trace
                job.events.put(
                    (
                        "error",
                        APIError(
                            400,
                            "this request was in flight at two engine failures in "
                            "a row; Splash refuses it until the server restarts"
                            + (f" (crash trace: {trace})" if trace else ""),
                            "request_quarantined",
                        ),
                    )
                )
                return
            self.active[job.request_id] = state
        try:
            call = self.runtime.submit(
                request, on_event=on_event, on_complete=on_complete
            )
            with self.lock:
                if not state.detached:
                    state.call = call
                cancel = state.detached or self.closing or job.cancelled.is_set()
            if cancel:
                call.cancel()
        except Exception as error:
            with self.lock:
                deliver = not state.detached and not state.terminal_enqueued
                if deliver:
                    self._detach_locked(state)
            if deliver:
                refusal = None
                if isinstance(
                    error,
                    (
                        engine_runtime.EngineUnhealthy,
                        engine_runtime.ProtocolFatal,
                        engine_runtime.RuntimeClosed,
                    ),
                ):
                    # Not admitted: refused as a request arriving now would be.
                    refusal = self.refusal()
                job.events.put(("error", refusal or self._api_error(error)))

    def _detach_locked(self, state):
        state.detach()
        if self.active.get(state.job.request_id) is state:
            del self.active[state.job.request_id]

    def _on_event(self, state, event):
        job = state.job
        if isinstance(event, wire.StartEvent):
            cache = CacheInfo(
                "hit" if event.matched_prompt_tokens else "miss",
                event.matched_prompt_tokens,
                event.lane,
            )
            with self.lock:
                state.started = True
                job.cache = cache
            record = {
                "outcome": "started",
                "prompt_tokens": len(job.prompt_tokens),
                "cached_tokens": cache.matched_tokens,
            }
            if job.reasoning is not None:
                record["reasoning"] = {
                    "sent_effort": job.reasoning.sent,
                    "effort": job.reasoning.effort,
                    "enable_thinking": job.reasoning.enable_thinking,
                    "thinking": job.thinking,
                }
            # The finalizer prints it, as it prints the Done line: a console
            # that stops draining must not hold up the reader thread, which
            # delivers every request's tokens and masks.
            self.finalizer_queue.put(functools.partial(self._log, record))
            job.events.put(("start", None))
        elif isinstance(event, wire.PromptProgressEvent):
            job.events.put(
                (
                    "progress",
                    {
                        "total": len(job.prompt_tokens),
                        "cache": job.cache.matched_tokens,
                        "processed": event.processed_tokens,
                        "time_ms": event.elapsed_micros / 1000.0,
                    },
                )
            )
        elif isinstance(event, wire.TokensEvent):
            if job.latency is not None and event.tokens:
                job.latency.tokens()
            if event.sequence_offset == 0:
                state.first_token_batch_tokens = len(event.tokens)
            if job.constraint is not None:
                job.constraint.commit(event.tokens)
            state.streamer.put_tokens(event.tokens)
        elif isinstance(event, wire.LogprobsEvent):
            state.logprobs.extend(event.values)

    def cancel(self, job):
        call = None
        with self.lock:
            job.cancelled.set()
            state = self.active.get(job.request_id)
            if state is not None:
                call = state.call
        if call is not None:
            call.cancel()

    def _finalize_loop(self):
        while (work := self.finalizer_queue.get()) is not None:
            work()
            # Do not keep a finished request alive while waiting: its image
            # batch returns the request budget only once nothing references
            # it.
            del work

    def _finalize(self, state, call):
        job = state.job
        error = None
        result = None
        try:
            done = call.result(0)
            if call.callback_error is not None:
                raise self._api_error(call.callback_error)
            if (
                job.constraint is not None
                and done.reason != wire.FinishReason.CANCELLED
            ):
                # The grammar checks what was generated after the last mask. A
                # cancelled request needs no check, and its last mask may
                # still be computing.
                job.constraint.finish()
            state.streamer.end()
            job.reasoning_tokens = state.streamer.count_reasoning_tokens(
                job.chat_format, job.thinking, job.may_call_tools
            )
            stop_sequence = state.streamer.stop_sequence
            result = NativeResult(
                reason=(
                    "stop"
                    if stop_sequence is not None
                    else self._FINISH_NAMES[done.reason]
                ),
                prompt_tokens=done.prompt_tokens,
                completion_tokens=(
                    len(state.streamer.token_ids)
                    if stop_sequence is not None
                    else done.completion_tokens
                ),
                option_logits=done.option_logits,
                start_to_first_token_ms=done.prefill_micros / 1000.0,
                first_token_to_done_ms=done.decode_micros / 1000.0,
                request_wall_ms=done.wall_micros / 1000.0,
                prefill_tokens=max(0, done.prompt_tokens - job.cache.matched_tokens),
                cache=replace(job.cache, deduplicated_tokens=done.deduplicated_tokens),
                stop_sequence=stop_sequence,
                first_token_batch_tokens=state.first_token_batch_tokens,
                drafted_tokens=done.drafted_tokens,
                accepted_draft_tokens=done.accepted_draft_tokens,
                draft_cycles=done.draft_cycles,
                bounded=done.bounded,
                logprobs=tuple(state.logprobs),
            )
            if job.latency is not None:
                latency = result.metrics["request_latency"]
                queued = latency.get("queue_to_start_ms")
                if queued is not None:
                    job.latency.metrics.observe("native_queue", queued / 1000.0)
        except (engine_runtime.EngineUnhealthy, engine_runtime.ProtocolFatal):
            # An admitted request ends with EngineUnhealthy or ProtocolFatal
            # only when the engine running it fails: it stopped or broke the
            # protocol. The listener has already decided whether that engine
            # restarts; the console names the failure.
            with self.lock:
                fatal_error = self.fatal_error
            if fatal_error is not None:
                error = APIError(500, fatal_error, "engine_failed")
            else:
                error = APIError(
                    503,
                    "the inference engine stopped unexpectedly and is restarting; "
                    "retry the request",
                    "runtime_unavailable",
                )
        except Exception as unexpected:
            error = self._api_error(unexpected)
        finally:
            with self.lock:
                shutdown_requested = state.shutdown_requested
                self._detach_locked(state)
        if shutdown_requested:
            # Once close() has claimed an active HTTP request, its public
            # terminal is deterministic even if the native cancellation and
            # RuntimeClosed delivery race each other. A completion delivered
            # before close() acquires the state lock remains a normal result.
            result = None
            error = APIError(503, "server is shutting down", "server_shutdown")
        self._record(job, result=result, error=error)
        job.events.put(("error", error) if error else ("done", result))

    @staticmethod
    def _api_error(error):
        if isinstance(error, APIError):
            return APIError(error.status, error.message, error.code)
        if isinstance(error, ConstraintError):
            return APIError(400, str(error), "constraint_error")
        if isinstance(error, engine_runtime.RequestFailed):
            code = error.code.decode("ascii", "replace")
            message = error.message_bytes.decode("utf-8", "replace")
            if code == "deadline_exceeded":
                return APIError(504, message, "request_timeout")
            if code == "capacity_exhausted":
                return APIError(
                    400,
                    "the request does not fit in the memory this server may use, even "
                    "after every cached prefix was evicted; restart the server with a "
                    f"larger --max-memory or a smaller --max-context ({message})",
                    code,
                )
            # The SSD cache cannot hold even the prompt's KV of a request that
            # does not fit in memory; the message names its quota.
            if code == "kv_disk_quota":
                return APIError(400, message, code)
            request_codes = {
                "invalid_constraint",
                "invalid_count",
                "invalid_deadline",
                "invalid_enum_value",
                "invalid_request",
                "invalid_sampling",
                "limit_exceeded",
            }
            status = 503 if error.retryable else 400 if code in request_codes else 500
            # The engine says when a retry may succeed where that is not at
            # once, as when the SSD cache's write budget refills.
            return APIError(status, message, code, error.retry_after_seconds or 1)
        if isinstance(error, engine_runtime.MaskComputationFailed):
            if error.retryable:
                return APIError(503, str(error), "runtime_busy")
            return APIError(400, str(error), "constraint_error")
        if isinstance(error, engine_runtime.PendingLimitExceeded):
            # The native pending limit is --queue-size, the HTTP request
            # gate's capacity, so only a race with the gate reaches it, as
            # when a cancelled request still holds its native slot: an
            # overload like the gate's own, retried the same way.
            return APIError(503, "request queue is full", "frontend_overloaded")
        if isinstance(
            error,
            (
                engine_runtime.EngineUnhealthy,
                engine_runtime.ProtocolFatal,
                engine_runtime.RuntimeClosed,
            ),
        ):
            return APIError(503, str(error), "runtime_unavailable")
        if isinstance(error, TimeoutError):
            return APIError(504, "request timed out", "request_timeout")
        return APIError(500, str(error), "runtime_error")

    def _record(self, job, result=None, error=None):
        record = {
            "outcome": result.reason if result else "error",
            "prompt_tokens": len(job.prompt_tokens),
        }
        if result:
            record["completion_tokens"] = result.completion_tokens
            record["metrics"] = result.metrics
        if job.tools_signature:
            record["tools"] = {
                "count": job.tools_signature[0],
                "signature": job.tools_signature[1],
            }
        if error:
            record["error_code"] = error.code
        self._log(record)

    def _log(self, record):
        # A request's start, when the engine admits it, and its end.
        try:
            self.request_logger(record)
        except Exception:
            pass

    def close(self):
        with self.lock:
            if self.closing:
                return
            self.closing = True
            calls = []
            for state in self.active.values():
                state.shutdown_requested = True
                calls.append(state.call)
        self.wakeup.set()
        for call in calls:
            if call is not None:
                call.cancel()
        try:
            # Also fails a relaunch in progress.
            self.runtime.close()
        finally:
            with self.lock:
                recovery = self.recovery
            if recovery is not None and recovery is not threading.current_thread():
                recovery.join(timeout=1.0)
            with self.lock:
                stranded = []
                for state in list(self.active.values()):
                    if state.terminal_enqueued:
                        continue
                    self._detach_locked(state)
                    stranded.append(state)
            for state in stranded:
                error = APIError(503, "server is shutting down", "server_shutdown")
                self._record(state.job, error=error)
                state.job.events.put(("error", error))
            self.finalizer_queue.put(None)
            self.finalizer.join()
