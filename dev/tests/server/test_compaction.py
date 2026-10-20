import tempfile
import time
import unittest
from itertools import count as iter_count
from pathlib import Path

from dev.tests.server_fixtures import make_job
from server import api_shapes, compaction
from server.frontend import Frontend, Prompt, RenderedPrompt


class SummarizingBackend:
    """The backend as compaction uses it: one internal job, answered with a
    fixed note."""

    def __init__(self, note="NOTE"):
        self.note = note
        self.calls = 0

    def submit(self, job):
        self.calls += 1
        job.events.put(("text", self.note))
        job.events.put(("done", None))

    def cancel(self, job):
        pass


class StubFrontend(Frontend):
    """The compaction path over a stand-in renderer: one token per four
    characters, so cut selection is exact without a tokenizer."""

    def __init__(
        self,
        backend,
        context=4096,
        compact_at="auto",
        compact_keep=512,
        compact_memory=None,
    ):
        self.backend = backend
        self.max_context = context
        self.compact_at = compact_at
        self.compact_keep = compact_keep
        self.compact_memory = (
            Path(compact_memory) if compact_memory is not None else None
        )
        if self.compact_memory is not None:
            self.compact_memory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.ids = iter_count(1)

    def _render_prompt(
        self, prompt, deadline, *, check_context=True, add_generation_prompt=True
    ):
        text = "".join(str(message.get("content") or "") for message in prompt.messages)
        tokens = list(range(max(1, len(text) // 4)))
        return RenderedPrompt(text, tokens, [], [], False, 0)

    def _tokenize(self, text, **options):
        return {"input_ids": list(range(max(1, len(text) // 4)))}


def long_messages():
    document = "The archive catalogues every ship that left the harbour. " * 200
    return [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Here is a document:\n\n" + document},
        {"role": "assistant", "content": "Understood."},
        {"role": "user", "content": "What does the archive catalogue?"},
    ]


def compact(app, messages=None):
    messages = long_messages() if messages is None else messages
    prompt = Prompt(
        messages=messages, tools=None, tool_policy=None, reasoning_effort=None
    )
    rendered = app._render_prompt(prompt, time.monotonic() + 10)
    return app._compact(prompt, rendered, time.monotonic() + 10, 512)


class FrontendCompactionTests(unittest.TestCase):
    def test_a_large_request_compacts_and_keeps_the_newest_turn(self):
        backend = SummarizingBackend()
        app = StubFrontend(backend)
        prompt, _rendered, report = compact(app)
        self.assertIsNotNone(report)
        self.assertEqual(report.note, "NOTE")
        self.assertGreater(report.removed_messages, 0)
        self.assertLess(report.prompt_tokens_after, report.prompt_tokens_before)
        self.assertEqual(prompt.messages[-1], long_messages()[-1])
        self.assertEqual(backend.calls, 1)

    def test_a_request_that_fits_is_untouched(self):
        backend = SummarizingBackend()
        app = StubFrontend(backend)
        small = [dict(message) for message in long_messages()]
        small[1]["content"] = "A short note."
        _prompt, _rendered, report = compact(app, small)
        self.assertIsNone(report)
        self.assertEqual(backend.calls, 0)

    def test_compaction_off_does_not_summarize(self):
        backend = SummarizingBackend()
        app = StubFrontend(backend, compact_at=None)
        _prompt, _rendered, report = compact(app)
        self.assertIsNone(report)
        self.assertEqual(backend.calls, 0)

    def test_compact_memory_reuses_a_note_for_the_same_conversation(self):
        backend = SummarizingBackend()
        with tempfile.TemporaryDirectory() as directory:
            app = StubFrontend(backend, compact_memory=directory)
            _prompt, _rendered, first = compact(app)
            _prompt, _rendered, second = compact(app)
            self.assertEqual(backend.calls, 1)
            self.assertEqual(first.note, second.note)
            (note_file,) = Path(directory).glob("*.json")
            self.assertEqual(note_file.stat().st_mode & 0o777, 0o600)


def count(messages):
    """A deterministic stand-in for a rendered token count: four characters a
    token, one token for each message's own scaffolding."""
    return sum(len(message.get("content") or "") // 4 + 1 for message in messages)


def text(tokens):
    """Content whose stand-in count is about `tokens`."""
    return "x" * (4 * tokens)


class CompactionPolicyTests(unittest.TestCase):
    def test_leading_system_messages_are_one_run(self):
        messages = [
            {"role": "system", "content": "a"},
            {"role": "system", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        self.assertEqual(compaction.leading_system_end(messages), 2)

    def test_a_cut_never_separates_a_tool_call_from_its_result(self):
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": text(100)},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "read", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "content": text(50), "tool_call_id": "1"},
            {"role": "user", "content": text(20)},
        ]
        # A cut on the result moves back to the assistant that called it.
        self.assertEqual(compaction.adjust_cut(messages, 3), 2)
        self.assertEqual(compaction.adjust_cut(messages, 4), 4)

    def test_the_newest_turns_keep_their_budget(self):
        messages = [
            {"role": "system", "content": text(10)},
            {"role": "user", "content": text(400)},
            {"role": "assistant", "content": text(400)},
            {"role": "user", "content": text(100)},
            {"role": "assistant", "content": text(100)},
            {"role": "user", "content": text(30)},
        ]
        cut = compaction.choose_cut(messages, count, budget=200)
        # The last three messages fit (about 143 tokens); one older turn
        # does not.
        self.assertEqual(cut, 4)
        self.assertLessEqual(count(messages[:1] + messages[cut:]), 200)
        self.assertGreater(count(messages[:1] + messages[cut - 1 :]), 200)

    def test_the_most_recent_user_turn_always_stays(self):
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": text(10)},
            {"role": "assistant", "content": text(900)},
            {"role": "user", "content": text(900)},
        ]
        # Neither turn fits the budget, so the answer keeps the newest user
        # turn rather than everything.
        cut = compaction.choose_cut(messages, count, budget=10)
        self.assertEqual(cut, 3)

    def test_the_note_joins_the_leading_system_message(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": text(400)},
            {"role": "assistant", "content": text(400)},
            {"role": "user", "content": text(20)},
        ]
        compacted = compaction.compacted_messages(messages, 3, "The user asked X.")
        self.assertEqual(len(compacted), 2)
        self.assertEqual(compacted[0]["role"], "system")
        self.assertIn("You are helpful.", compacted[0]["content"])
        self.assertIn("The user asked X.", compacted[0]["content"])
        self.assertTrue(compacted[0]["content"].startswith("You are helpful."))
        self.assertEqual(compacted[1], messages[3])

    def test_a_conversation_without_a_system_message_gains_one(self):
        messages = [
            {"role": "user", "content": text(400)},
            {"role": "user", "content": text(20)},
        ]
        compacted = compaction.compacted_messages(messages, 1, "note")
        self.assertEqual(compacted[0]["role"], "system")
        self.assertEqual(compacted[1], messages[1])

    def test_should_compact_follows_the_setting(self):
        self.assertFalse(compaction.should_compact(1000, 100, 4096, 256, None))
        self.assertTrue(compaction.should_compact(4000, 100, 4096, 256, "auto"))
        self.assertFalse(compaction.should_compact(1000, 100, 4096, 256, "auto"))
        self.assertTrue(compaction.should_compact(2050, 1, 4096, 256, 0.5))
        self.assertFalse(compaction.should_compact(2000, 1, 4096, 256, 0.5))

    def test_keep_budget_defaults_to_a_quarter_of_the_window_at_most_20000(self):
        self.assertEqual(compaction.keep_budget(262144, None), 20000)
        self.assertEqual(compaction.keep_budget(40000, None), 10000)
        self.assertEqual(compaction.keep_budget(4096, None), 1024)
        self.assertEqual(compaction.keep_budget(4096, 777), 777)

    def test_the_transcript_keeps_tool_calls_and_results(self):
        messages = [
            {
                "role": "assistant",
                "content": "checking",
                "tool_calls": [{"function": {"name": "read", "arguments": '{"p": 1}'}}],
            },
            {"role": "tool", "content": "42"},
        ]
        rendered = compaction.transcript(messages)
        self.assertIn('assistant: checking read({"p": 1})', rendered)
        self.assertIn("tool: 42", rendered)

    def test_the_report_is_a_response_extension(self):
        job = make_job()
        self.assertIsNone(api_shapes.compaction_dict(job))
        job.compaction = compaction.Compaction("note", 2, 100, 20)
        self.assertEqual(
            api_shapes.compaction_dict(job),
            {
                "note": "note",
                "removed_messages": 2,
                "prompt_tokens_before": 100,
                "prompt_tokens_after": 20,
            },
        )

    def test_a_stream_chunk_carries_the_report_only_when_present(self):
        report = {
            "note": "n",
            "removed_messages": 1,
            "prompt_tokens_before": 9,
            "prompt_tokens_after": 3,
        }
        self.assertEqual(
            api_shapes.stream_chunk("m", "id", 0, {}, compaction=report)["compaction"],
            report,
        )
        self.assertNotIn("compaction", api_shapes.stream_chunk("m", "id", 0, {}))


if __name__ == "__main__":
    unittest.main()
