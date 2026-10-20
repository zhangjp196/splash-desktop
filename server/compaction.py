"""Context compaction (`--compact-at`): older turns become a memory note.

The frontend renders a conversation before it knows how much of the window the
reply needs. When a rendered request leaves too little room, the older turns
are replaced by a note and the newest ones stay word for word. A note never
separates a tool call from its results, and it joins the leading system message
so a template that rejects a later system message still renders.

This module is pure policy: the summarizer that writes the note and the token
counts are the caller's.
"""

from dataclasses import dataclass

# The note's first line, so a client that sends the compacted conversation
# back carries its marker with it.
NOTE_HEADING = "Earlier turns, compacted into a note:"
# What the newest turns keep by default: a quarter of the window, at most this.
DEFAULT_KEEP_TOKENS_CAP = 20_000


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
