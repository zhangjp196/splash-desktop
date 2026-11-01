import json

from dev.tests.server_fixtures import (
    FakeConstraintFactory,
    HarnessTestCase,
    anthropic_body,
    chat_body,
    responses_body,
)
from server import constraints
from server.errors import APIError


class GrammarSourceTests(HarnessTestCase):
    def test_gbnf_becomes_a_compiled_grammar(self):
        compiled = constraints.grammar_for_source("gbnf", 'root ::= "yes" | "no"')
        self.assertIn("start:", compiled)
        self.assertIn('"yes"', compiled)
        self.assertIn('"no"', compiled)

    def test_regex_becomes_a_compiled_grammar(self):
        compiled = constraints.grammar_for_source("regex", r"[0-9]{3}")
        self.assertIn("regex", compiled)
        self.assertIn("[0-9]{3}", compiled)

    def test_a_source_the_converter_refuses_is_the_requests_error(self):
        with self.assertRaises(APIError) as caught:
            constraints.grammar_for_source("gbnf", "root ::= [")
        self.assertEqual(caught.exception.status, 400)
        self.assertIn("unsupported gbnf grammar", caught.exception.message)


class GrammarRequestTests(HarnessTestCase):
    def post(self, body, path="/v1/chat/completions", factory=None):
        factory = factory or FakeConstraintFactory()
        harness = self.harness(constraint_factory=factory)
        status, _, payload = harness.request("POST", path, body)
        return status, payload, factory

    def test_grammar_reaches_the_constraint_factory(self):
        status, payload, factory = self.post(
            chat_body(
                grammar='root ::= "yes" | "no"',
                reasoning_effort="none",
                max_tokens=16,
            )
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(len(factory.grammars), 1)
        self.assertIn("start:", factory.grammars[0])

    def test_regex_reaches_the_constraint_factory(self):
        status, payload, factory = self.post(
            chat_body(
                regex=r"[0-9]{3}",
                reasoning_effort="none",
                max_tokens=16,
            )
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(len(factory.grammars), 1)
        self.assertIn("regex", factory.grammars[0])

    def test_text_completions_accept_a_grammar(self):
        status, payload, factory = self.post(
            {
                "model": "test-model",
                "prompt": "answer",
                "max_tokens": 16,
                "temperature": 0,
                "grammar": 'root ::= "yes" | "no"',
            },
            path="/v1/completions",
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(len(factory.grammars), 1)

    def test_grammar_cannot_be_combined_with_regex(self):
        status, payload, _ = self.post(
            chat_body(
                grammar='root ::= "yes"',
                regex=r"[0-9]{3}",
                reasoning_effort="none",
            )
        )
        self.assertEqual(status, 400, payload)
        self.assertIn("cannot be combined", json.loads(payload)["error"]["message"])

    def test_grammar_cannot_be_combined_with_tools(self):
        status, payload, _ = self.post(
            chat_body(
                grammar='root ::= "yes"',
                reasoning_effort="none",
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "f",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            )
        )
        self.assertEqual(status, 400, payload)
        self.assertIn("grammar and regex", json.loads(payload)["error"]["message"])

    def test_grammar_cannot_be_combined_with_response_format(self):
        status, payload, _ = self.post(
            chat_body(
                grammar='root ::= "yes"',
                reasoning_effort="none",
                response_format={"type": "json_object"},
            )
        )
        self.assertEqual(status, 400, payload)

    def test_grammar_requires_thinking_off(self):
        status, payload, _ = self.post(
            chat_body(grammar='root ::= "yes"', max_tokens=16)
        )
        self.assertEqual(status, 400, payload)
        self.assertIn("reasoning off", json.loads(payload)["error"]["message"])

    def test_a_grammar_the_converter_refuses_is_refused(self):
        status, payload, _ = self.post(
            chat_body(grammar="root ::= [", reasoning_effort="none")
        )
        self.assertEqual(status, 400, payload)
        self.assertIn(
            "unsupported gbnf grammar", json.loads(payload)["error"]["message"]
        )

    def test_an_empty_grammar_is_refused(self):
        status, payload, _ = self.post(chat_body(grammar="  ", reasoning_effort="none"))
        self.assertEqual(status, 400, payload)
        self.assertIn("nonempty string", json.loads(payload)["error"]["message"])

    def test_responses_do_not_take_a_grammar(self):
        status, payload, _ = self.post(
            responses_body(grammar='root ::= "yes"'),
            path="/v1/responses",
        )
        self.assertEqual(status, 400, payload)
        self.assertIn(
            "Chat Completions and text completions",
            json.loads(payload)["error"]["message"],
        )

    def test_messages_do_not_take_a_grammar(self):
        status, payload, _ = self.post(
            anthropic_body(grammar='root ::= "yes"'),
            path="/v1/messages",
        )
        self.assertEqual(status, 400, payload)
        self.assertIn(
            "Chat Completions and text completions",
            json.loads(payload)["error"]["message"],
        )
