import json

from dev.tests.server_fixtures import HarnessTestCase, chat_body


class ChoiceTests(HarnessTestCase):
    def post(self, path, body):
        harness = self.harness()
        status, _, payload = harness.request("POST", path, body)
        return status, json.loads(payload)

    def test_chat_returns_one_choice_per_n(self):
        status, payload = self.post(
            "/v1/chat/completions", chat_body(n=2, max_tokens=8)
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual([choice["index"] for choice in payload["choices"]], [0, 1])
        self.assertEqual(len(payload["choices"]), 2)
        self.assertEqual(payload["usage"]["prompt_tokens"], 2)
        self.assertEqual(payload["usage"]["completion_tokens"], 2)
        self.assertEqual(payload["usage"]["total_tokens"], 4)

    def test_text_completions_return_one_choice_per_n(self):
        status, payload = self.post(
            "/v1/completions",
            {
                "model": "test-model",
                "prompt": "hello",
                "max_tokens": 8,
                "temperature": 0,
                "n": 2,
            },
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual([choice["index"] for choice in payload["choices"]], [0, 1])
        self.assertEqual(payload["usage"]["prompt_tokens"], 2)
        self.assertEqual(payload["usage"]["completion_tokens"], 2)

    def test_streaming_n_is_refused(self):
        status, payload = self.post(
            "/v1/chat/completions", chat_body(n=2, max_tokens=8, stream=True)
        )
        self.assertEqual(status, 400, payload)
        self.assertIn("n > 1", payload["error"]["message"])

    def test_n_above_four_is_refused(self):
        status, payload = self.post(
            "/v1/chat/completions", chat_body(n=5, max_tokens=8)
        )
        self.assertEqual(status, 400, payload)
        self.assertIn("n must be", payload["error"]["message"])
