"""Context compaction (`--compact-at`): older turns become a memory note.

The frontend renders a conversation before it knows how much of the window the
reply needs. When a rendered request leaves too little room, the older turns
are replaced by a note and the newest ones stay word for word. A note never
separates a tool call from its results, and it joins the leading system message
so a template that rejects a later system message still renders.

The functions that take only messages and numbers are pure policy. The ones
that take the frontend render trial prompts, generate the note with its own
small request, and keep notes across runs in --compact-memory.
"""

import hashlib
import os
import queue
from dataclasses import dataclass, replace

from . import json_codec
from . import protocol as wire
from .backend import remaining_request_time
from .errors import APIError
from .generation_options import GenerationOptions

# The note's first line, so a client that sends the compacted conversation
# back carries its marker with it.
NOTE_HEADING = "Earlier turns, compacted into a note:"
# What the newest turns keep by default: a quarter of the window, at most this.
DEFAULT_KEEP_TOKENS_CAP = 20_000

# --compact-at: the room a reply keeps before a request compacts, and the
# reply a request that names no output limit is assumed to want.
COMPACT_RESERVE_TOKENS = 1024
COMPACT_DEFAULT_REPLY_TOKENS = 4096
# The most the note's own generation may produce, and what it is asked for.
COMPACT_MAX_NOTE_TOKENS = 1024
COMPACT_INSTRUCTION = (
    "Summarize the conversation below into a note for yourself, so you can "
    "continue the work: the goal, the constraints, what has been done, the "
    "decisions and why, the immediate next steps, and the exact names, "
    "numbers and file paths it mentions. Reply with the note alone, with no "
    "preamble and no advice to the user."
)


@dataclass(frozen=True)
class Compaction:
    """One compaction's report: the note, the turns it replaced and the
    rendered tokens before and after."""

    note: str
    removed_messages: int
    prompt_tokens_before: int
    prompt_tokens_after: int


def keep_budget(context, configured):
    """The tokens the newest turns keep: `--compact-keep`, or a quarter of the
    window, at most DEFAULT_KEEP_TOKENS_CAP."""
    if configured is not None:
        return configured
    return min(DEFAULT_KEEP_TOKENS_CAP, max(1, context // 4))


def should_compact(prompt_tokens, max_new, context, reserve, setting):
    """Whether a rendered request compacts under `--compact-at`: "auto"
    compacts when the prompt and the reply would pass the window minus
    `reserve`; a fraction compacts once the prompt reaches that share of the
    window. None (the flag absent) never compacts."""
    if setting is None:
        return False
    if setting == "auto":
        return prompt_tokens + max_new > context - reserve
    return prompt_tokens >= setting * context


def leading_system_end(messages):
    """How many leading system messages the conversation starts with; the
    normalizer merges every client system message into the first."""
    end = 0
    while end < len(messages) and messages[end]["role"] == "system":
        end += 1
    return end


def adjust_cut(messages, cut):
    """A cut that does not separate a tool result from its call: a cut on a
    tool message moves back to the assistant that called the tool, whose own
    call the kept suffix then holds."""
    start = leading_system_end(messages)
    while cut > start and messages[cut]["role"] == "tool":
        cut -= 1
    return cut


def choose_cut(messages, count, budget):
    """The earliest cut whose kept suffix fits `budget` rendered tokens,
    always keeping the most recent user turn however long it is. The leading
    system messages are always kept; `count(messages)` renders the given list
    and returns its token count."""
    start = leading_system_end(messages)
    system = messages[:start]
    if start >= len(messages):
        return start
    best = len(messages)
    cut = len(messages)
    while cut > start:
        candidate = adjust_cut(messages, cut - 1)
        if candidate <= start or candidate >= cut:
            break
        if count([*system, *messages[candidate:]]) > budget:
            break
        best = candidate
        cut = candidate
    last_user = next(
        (
            index
            for index in range(len(messages) - 1, start - 1, -1)
            if messages[index]["role"] == "user"
        ),
        None,
    )
    if last_user is not None and last_user < best:
        best = adjust_cut(messages, last_user)
    return best


def note_body(note):
    """The note as it travels inside the conversation."""
    text = (note or "").strip()
    return f"{NOTE_HEADING}\n{text}" if text else NOTE_HEADING


def compacted_messages(messages, cut, note):
    """`messages` with everything before `cut` replaced by `note`, which joins
    the leading system message (or becomes one when the conversation has
    none)."""
    start = leading_system_end(messages)
    body = note_body(note)
    if start:
        system = dict(messages[0])
        content = system.get("content") or ""
        system["content"] = f"{content}\n\n{body}" if content else body
        return [system, *messages[cut:]]
    return [{"role": "system", "content": body}, *messages[cut:]]


def transcript(messages):
    """The removed turns as plain text, for the note's own generation."""
    lines = []
    for message in messages:
        role = message.get("role", "")
        parts = [part for part in (message.get("content"),) if part]
        for call in message.get("tool_calls") or ():
            function = call.get("function") or {}
            parts.append(f"{function.get('name', '')}({function.get('arguments', '')})")
        lines.append(f"{role}: {' '.join(parts)}")
    return "\n".join(lines)


def compact(frontend, prompt, rendered, deadline, requested):
    """The request's prompt after `--compact-at`: the newest turns and a
    note in place of the older ones, or the prompt unchanged. Returns the
    prompt, its render, and the note's report."""
    if frontend.compact_at is None:
        return prompt, rendered, None
    reply = (
        requested
        if isinstance(requested, int) and not isinstance(requested, bool)
        else COMPACT_DEFAULT_REPLY_TOKENS
    )
    if not should_compact(
        len(rendered.tokens),
        reply,
        frontend.max_context,
        COMPACT_RESERVE_TOKENS,
        frontend.compact_at,
    ):
        return prompt, rendered, None
    messages = prompt.messages
    budget = keep_budget(frontend.max_context, frontend.compact_keep)

    def count(candidate):
        trial = replace(prompt, messages=candidate)
        return len(frontend._render_prompt(trial, deadline, check_context=False).tokens)

    cut = choose_cut(messages, count, budget)
    start = leading_system_end(messages)
    if cut <= start or cut >= len(messages):
        return prompt, rendered, None
    removed = messages[start:cut]
    text = transcript(removed)
    digest = hashlib.sha256(text.encode()).hexdigest()
    key = compaction_key(messages)
    note = read_note(frontend, key, digest)
    if note is None:
        note = summarize(frontend, text, deadline)
        write_note(frontend, key, digest, note)
    compacted = replace(prompt, messages=compacted_messages(messages, cut, note))
    final = frontend._render_prompt(compacted, deadline)
    report = Compaction(note, cut - start, len(rendered.tokens), len(final.tokens))
    return compacted, final, report


def summarize(frontend, text, deadline):
    """The note's own generation: the removed turns and one instruction,
    greedy and at background priority."""
    from .frontend import Prompt  # deferred: frontend imports this module

    text = fit_transcript(frontend, text)
    prompt = Prompt(
        messages=[{"role": "user", "content": f"{COMPACT_INSTRUCTION}\n\n{text}"}],
        tools=None,
        tool_policy=None,
        reasoning_effort="none",
    )
    rendered = frontend._render_prompt(prompt, deadline, check_context=False)
    max_new = min(
        COMPACT_MAX_NOTE_TOKENS,
        max(1, frontend.max_context - len(rendered.tokens) - COMPACT_RESERVE_TOKENS),
    )
    options = GenerationOptions(
        sampling=wire.SamplingParameters(temperature=0.0),
        stop_sequences=(),
        ignore_eos=False,
        seed=0,
        priority=wire.RequestPriority.BACKGROUND,
        cache_salt=0,
        thinking_token_budget=None,
    )
    job = frontend._generation_job(options, rendered.tokens, max_new, deadline)
    frontend.backend.submit(job)
    parts = []
    while True:
        try:
            kind, value = job.events.get(timeout=remaining_request_time(deadline))
        except queue.Empty:
            frontend.backend.cancel(job)
            raise APIError(504, "request timed out", "request_timeout") from None
        if kind == "text":
            parts.append(value)
        elif kind == "done":
            break
        elif kind == "error":
            raise value
    return "".join(parts).strip()


def fit_transcript(frontend, text):
    """The transcript shortened to fit the note's own prompt, keeping its
    start (the goal) and its end (the newest progress)."""
    limit = max(
        256, frontend.max_context - COMPACT_RESERVE_TOKENS - COMPACT_MAX_NOTE_TOKENS
    )
    for _ in range(8):
        tokens = frontend._tokenize(
            f"{COMPACT_INSTRUCTION}\n\n{text}", add_special_tokens=False
        )["input_ids"]
        if len(tokens) <= limit:
            return text
        keep = max(1, int(len(text) * limit / max(1, len(tokens)) * 0.9))
        head = int(keep * 0.6)
        tail = keep - head
        text = f"{text[:head]}\n\n[...]\n\n{text[-tail:]}" if tail else text[:head]
    return text


def compaction_key(messages):
    """A conversation's key for --compact-memory: its leading system text
    and its first user message, which later turns keep."""
    system = (
        messages[0].get("content", "")
        if messages and messages[0].get("role") == "system"
        else ""
    )
    first_user = next(
        (
            message.get("content", "")
            for message in messages
            if message.get("role") == "user"
        ),
        "",
    )
    seed = f"{system}\0{first_user}"
    return hashlib.sha256(seed.encode()).hexdigest()


def read_note(frontend, key, digest):
    """The note --compact-memory kept for this conversation and this run
    of removed turns, or None."""
    if frontend.compact_memory is None:
        return None
    try:
        data = json_codec.loads(
            (frontend.compact_memory / f"{key}.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
    note = data.get("note") if isinstance(data, dict) else None
    return note if isinstance(note, str) and data.get("removed") == digest else None


def write_note(frontend, key, digest, note):
    """Keep a compaction's note for a later identical compaction; a
    failure leaves the reply alone."""
    if frontend.compact_memory is None:
        return
    path = frontend.compact_memory / f"{key}.json"
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as target:
            # The strict codec escapes lone surrogates, which a text
            # file otherwise cannot hold, and read_note reads the note
            # back with the same codec.
            target.write(json_codec.dumps({"removed": digest, "note": note}))
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    except OSError:
        try:
            temporary.unlink()
        except OSError:
            pass
