"""Concise console diagnostics without request-body logging."""

import collections
import sys
import threading
import time
import traceback
from pathlib import Path

# The server package, whose innermost frame locates an unexpected error.
_PACKAGE = Path(__file__).resolve().parent

# Lines a console may hold before it starts dropping them. A terminal, a log
# file on a slow disk or a reader that stopped reading must never hold up a
# request or the engine loop: the console is its own backpressure domain, and
# a burst past this many lines is dropped and counted rather than waited on.
CONSOLE_QUEUE_LINES = 1024


class Console:
    """A console writer that never blocks its caller.

    Every line is queued and one thread writes them in order, so a request
    thread, a finalizer or the engine's reader never waits on a terminal or a
    log file. A queue past CONSOLE_QUEUE_LINES drops the line and counts it
    (dropped_lines), which /status reports: a console that cannot keep up is
    visible as a number instead of as a stalled server. The writer is a daemon
    and is stopped by close(), which flushes what it holds.
    """

    def __init__(self, queue_lines=CONSOLE_QUEUE_LINES):
        if isinstance(queue_lines, bool) or not isinstance(queue_lines, int):
            raise ValueError("console queue size must be an integer")
        if queue_lines <= 0:
            raise ValueError("console queue size must be positive")
        self._queue = collections.deque()
        self._queue_lines = queue_lines
        self._lock = threading.Lock()
        self._wakeup = threading.Condition(self._lock)
        self._closed = False
        self._dropped = 0
        self._written = 0
        self._thread = threading.Thread(
            target=self._write_loop,
            name="splash-console",
            daemon=True,
        )
        self._thread.start()

    def write(self, text, *, error=False):
        """Queue one line, newline included, for the writer thread. A full
        queue drops it and counts it; this never blocks."""
        line = (sys.stderr if error else sys.stdout, text)
        with self._lock:
            if self._closed:
                return False
            if len(self._queue) >= self._queue_lines:
                self._dropped += 1
                return False
            self._queue.append(line)
            self._wakeup.notify()
            return True

    def stats(self):
        """What the console holds and did: the lines waiting, the lines
        written, and the lines dropped because the queue was full."""
        with self._lock:
            return {
                "queued": len(self._queue),
                "queue_lines": self._queue_lines,
                "written": self._written,
                "dropped_lines": self._dropped,
            }

    def close(self, timeout=2.0):
        """Stop taking lines and let the writer finish what it holds, for at
        most `timeout`; a writer still busy then is left to the interpreter,
        which does not wait for a daemon thread."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._wakeup.notify_all()
        self._thread.join(timeout)

    def flush(self, timeout=2.0):
        """Wait until the writer has written every line queued so far, for at
        most `timeout`. A caller that must see its line on the console, such
        as a test or a start that is about to fail, uses this; serving code
        never does."""
        deadline = time.monotonic() + timeout
        with self._lock:
            while self._queue:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._wakeup.wait(remaining)
            return True

    def _write_loop(self):
        while True:
            with self._lock:
                while not self._queue and not self._closed:
                    self._wakeup.wait()
                if not self._queue:
                    return
                stream, text = self._queue.popleft()
                self._written += 1
            try:
                stream.write(text)
                stream.flush()
            except Exception:
                # A console that cannot be written to must not stop the
                # server; the line is lost, as a dropped one is.
                pass
            finally:
                with self._lock:
                    self._wakeup.notify_all()


# The process's console. None until start_console() installs the writer, so
# that a module imported by a test, or a start that has not begun serving,
# writes its line in place: nothing concurrent runs yet, and a test that
# captures sys.stdout sees its line at once. The serving process installs the
# writer before it answers a request (serve.main).
_console = None
_console_lock = threading.Lock()


def start_console(queue_lines=CONSOLE_QUEUE_LINES):
    """Install the bounded console writer. A second call does nothing."""
    global _console
    with _console_lock:
        if _console is not None:
            return _console
        _console = Console(queue_lines)
        return _console


def console_stats():
    """The console's counters, or an empty console's before it was used."""
    with _console_lock:
        writer = _console
    if writer is None:
        return {
            "queued": 0,
            "queue_lines": CONSOLE_QUEUE_LINES,
            "written": 0,
            "dropped_lines": 0,
        }
    return writer.stats()


def close_console(timeout=2.0):
    """Flush and stop the process's console, if it was started."""
    global _console
    with _console_lock:
        writer = _console
        _console = None
    if writer is not None:
        writer.close(timeout)


def flush_console(timeout=2.0):
    """Wait until the process's console wrote every line queued so far."""
    with _console_lock:
        writer = _console
    if writer is not None:
        writer.flush(timeout)


def log_unexpected(error):
    """Print the type of an unexpected error and the innermost line of the
    server it passed through, never its message, which may carry request
    data."""
    try:
        location = ""
        for frame, line in traceback.walk_tb(error.__traceback__):
            path = Path(frame.f_code.co_filename).resolve()
            if path.parent == _PACKAGE:
                location = f" · {_PACKAGE.name}/{path.name}:{line}"
        print_status(
            f"Error · internal_server_error · {type(error).__name__}{location}",
            error=True,
        )
    except Exception:
        pass


def print_status(message, *, error=False):
    # One write per line, newline included, so that lines written at once by
    # request threads, or by the native runtime on the shared stderr, stay
    # whole in a terminal or in one log file. With the bounded writer
    # installed (start_console) the write is queued: a console that cannot
    # keep up drops lines and counts them instead of holding up the caller.
    with _console_lock:
        writer = _console
    if writer is not None:
        writer.write(f"{time.strftime('%H:%M:%S')} {message}\n", error=error)
        return
    stream = sys.stderr if error else sys.stdout
    stream.write(f"{time.strftime('%H:%M:%S')} {message}\n")
    stream.flush()


def _reasoning_parts(reasoning):
    """A chat request's reasoning: the effort it sent, or absent and the
    server's default where that applied, its enable_thinking where it set
    one, and whether it thinks."""
    if reasoning["sent_effort"] is not None:
        parts = [f"effort {reasoning['sent_effort']}"]
    elif reasoning["effort"] is not None:
        parts = [f"effort absent → {reasoning['effort']}"]
    else:
        parts = ["effort absent"]
    if reasoning["enable_thinking"] is not None:
        parts.append(f"enable_thinking {str(reasoning['enable_thinking']).lower()}")
    parts.append("thinking on" if reasoning["thinking"] else "thinking off")
    return parts


def print_request(record):
    outcome = record["outcome"]
    if outcome == "error":
        print_status(f"Error · {record.get('error_code', 'runtime_error')}", error=True)
        return
    if outcome == "started":
        # The engine admitted the request: a long prompt's prefill runs
        # from here until its Done line.
        parts = [
            "Started",
            f"input {record['prompt_tokens']:,}",
            f"cached {record['cached_tokens']:,}",
        ]
        if "reasoning" in record:
            parts.extend(_reasoning_parts(record["reasoning"]))
        print_status(" · ".join(parts))
        return
    metrics = record.get("metrics", {})
    latency = metrics.get("request_latency", {})
    cache = metrics.get("cache", {})
    parts = [
        "Cancelled" if outcome == "cancelled" else "Done",
        f"input {record['prompt_tokens']:,}",
        f"cached {cache.get('matched_tokens', 0):,}",
    ]
    # Tokens whose KV the request read from cached blocks another request
    # computed, past its cached prefix.
    if cache.get("deduplicated_tokens"):
        parts.append(f"deduplicated {cache['deduplicated_tokens']:,}")
    parts.append(f"output {record.get('completion_tokens', 0):,}")
    tools = record.get("tools")
    if isinstance(tools, dict) and tools.get("count"):
        parts.append(f"tools {tools['count']}·{tools.get('signature', '')}")
    ttft = latency.get("ttft_ms")
    speed = latency.get("stream_tokens_per_second")
    if ttft is not None:
        parts.append(f"TTFT {ttft / 1000:.1f}s")
    if speed is not None:
        parts.append(f"{speed:.1f} tok/s")
    speculation = metrics.get("speculation", {})
    if speculation.get("cycles"):
        # Each verified draft yields the tokens it accepted and one more.
        per_cycle = 1 + speculation["accepted_draft_tokens"] / speculation["cycles"]
        parts.append(f"{per_cycle:.1f} tok/cycle")
    print_status(" · ".join(parts))
