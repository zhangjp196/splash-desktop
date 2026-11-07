import json
from unittest import mock

from dev.tests.server_fixtures import (
    FakeRuntime,
    HarnessTestCase,
    chat_body,
)
from server import frontend


class ChatCountTokensTest(HarnessTestCase):
    def test_counts_the_chat_prompt_without_inference(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime, max_context=32768)
        body = chat_body(
            messages=[
                {"role": "user", "content": "你好 👋"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "And then?"},
            ]
        )
        with mock.patch.object(
            frontend, "Job", side_effect=AssertionError("inference")
        ):
            status, _, payload = harness.request(
                "POST", "/v1/chat/completions/count_tokens?beta=true", body
            )
        self.assertEqual(status, 200, payload)
        count = json.loads(payload)
        self.assertGreater(count["input_tokens"], 0)
        self.assertEqual(runtime.requests, [])

    def test_counts_a_prompt_larger_than_the_context(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime, max_context=1)
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions/count_tokens", chat_body()
        )
        self.assertEqual(status, 200, payload)
        self.assertGreater(json.loads(payload)["input_tokens"], 1)
        self.assertEqual(runtime.requests, [])

    def test_authentication_and_validation(self):
        harness = self.harness(api_key="secret")
        path = "/v1/chat/completions/count_tokens"
        self.assertEqual(harness.request("POST", path, chat_body())[0], 401)
        headers = {"Authorization": "Bearer secret", "Content-Type": "application/json"}
        status, _, payload = harness.request(
            "POST", path, {"messages": "bad"}, headers=headers
        )
        self.assertEqual(status, 400, payload)
        self.assertEqual(json.loads(payload)["error"]["type"], "invalid_request_error")
        status, _, payload = harness.request("POST", path, chat_body(), headers=headers)
        self.assertEqual(status, 200, payload)
        self.assertIn("input_tokens", json.loads(payload))
