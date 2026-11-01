import json

from dev.tests.server_fixtures import (
    FakeRuntime,
    HarnessTestCase,
    chat_body,
)


class LogitBiasTests(HarnessTestCase):
    def test_a_valid_bias_reaches_the_native_request(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime)
        status, _, payload = harness.request(
            "POST",
            "/v1/chat/completions",
            chat_body(logit_bias={"4": -1.5, "10": 2}),
        )
        self.assertEqual(status, 200, payload)
        frame = runtime.requests[0].frame
        self.assertEqual(frame.bias_tokens, (4, 10))
        self.assertEqual(frame.bias_values, (-1.5, 2.0))

    def test_an_empty_bias_sends_none(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime)
        status, _, payload = harness.request(
            "POST", "/v1/chat/completions", chat_body(logit_bias={})
        )
        self.assertEqual(status, 200, payload)
        frame = runtime.requests[0].frame
        self.assertEqual(frame.bias_tokens, ())
        self.assertEqual(frame.bias_values, ())

    def test_invalid_biases_are_refused(self):
        harness = self.harness()
        cases = (
            ({"4": -101}, "[-100, 100]"),
            ({"4": 101}, "[-100, 100]"),
            ({"4": True}, "[-100, 100]"),
            ({"4": "high"}, "[-100, 100]"),
            ({"not-a-token": 1}, "token ids"),
            ({"248320": 1}, "outside the vocabulary"),
            ({f"{index}": 1 for index in range(257)}, "at most 256"),
            ([], "must be an object"),
        )
        for value, message in cases:
            with self.subTest(value=value):
                status, _, payload = harness.request(
                    "POST", "/v1/chat/completions", chat_body(logit_bias=value)
                )
                self.assertEqual(status, 400, payload)
                self.assertIn(message, json.loads(payload)["error"]["message"])
