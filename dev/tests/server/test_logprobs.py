import json
from types import SimpleNamespace

from dev.tests.server_fixtures import FakeTokenizer, HarnessTestCase, chat_body
from server import server as api


class SpecialTokenizer(FakeTokenizer):
    def decode(self, token_ids, **kwargs):
        if any(token in (26, 27) for token in token_ids):
            return "" if kwargs.get("skip_special_tokens") else "</think>"
        return super().decode(token_ids, **kwargs)


class ChatLogprobsTests(HarnessTestCase):
    def handler(self):
        return SimpleNamespace(app=SimpleNamespace(tokenizer=SpecialTokenizer()))

    def test_content_entries_skip_special_tokens(self):
        result = SimpleNamespace(
            logprobs=(
                (4, -0.5, ((4, -0.5), (27, -2.0))),
                (27, -0.1, ((27, -0.1),)),
            )
        )
        logprobs = api.FrontendHandler._chat_logprobs(
            self.handler(), result, "plain answer\n"
        )
        self.assertEqual(len(logprobs["content"]), 1)
        entry = logprobs["content"][0]
        self.assertEqual(entry["token"], "plain answer\n")
        self.assertEqual(entry["logprob"], -0.5)
        self.assertEqual(entry["bytes"], list(b"plain answer\n"))
        self.assertEqual(
            [item["token"] for item in entry["top_logprobs"]],
            ["plain answer\n", "</think>"],
        )
        self.assertEqual(entry["top_logprobs"][1]["logprob"], -2.0)
        self.assertIsNone(logprobs["refusal"])

    def test_an_empty_output_has_no_content(self):
        logprobs = api.FrontendHandler._chat_logprobs(
            self.handler(), SimpleNamespace(logprobs=()), ""
        )
        self.assertEqual(logprobs, {"content": [], "refusal": None})

    def test_a_stop_sequence_trims_its_own_tokens(self):
        result = SimpleNamespace(logprobs=((14, -0.5, ()), (15, -0.1, ())))
        logprobs = api.FrontendHandler._chat_logprobs(self.handler(), result, "first ")
        self.assertEqual([entry["token"] for entry in logprobs["content"]], ["first "])

    def test_top_logprobs_requires_logprobs(self):
        harness = self.harness()
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", chat_body(top_logprobs=2)
        )
        self.assertEqual(status, 400, payload)
        self.assertIn(
            "top_logprobs requires logprobs",
            json.loads(payload)["error"]["message"],
        )
