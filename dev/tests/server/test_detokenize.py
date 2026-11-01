import json

from dev.tests.server_fixtures import HarnessTestCase
from server.chat_format import QWEN


class DetokenizeTests(HarnessTestCase):
    def post(self, body):
        harness = self.harness()
        status, _, payload = harness.request("POST", "/detokenize", body)
        return status, payload

    def test_tokens_decode_with_the_loaded_tokenizer(self):
        status, payload = self.post({"tokens": [4]})
        self.assertEqual(status, 200, payload)
        self.assertEqual(json.loads(payload), {"content": "plain answer\n"})

    def test_special_tokens_may_be_kept(self):
        status, payload = self.post({"tokens": [4], "special": True})
        self.assertEqual(status, 200, payload)
        self.assertEqual(json.loads(payload), {"content": "plain answer\n"})

    def test_empty_tokens_decode_to_empty_text(self):
        status, payload = self.post({"tokens": []})
        self.assertEqual(status, 200, payload)
        self.assertEqual(json.loads(payload), {"content": ""})

    def test_a_token_outside_the_vocabulary_is_refused(self):
        status, payload = self.post({"tokens": [QWEN.vocabulary]})
        self.assertEqual(status, 400, payload)
        self.assertIn("outside the vocabulary", json.loads(payload)["error"]["message"])

    def test_tokens_must_be_integers(self):
        status, payload = self.post({"tokens": ["4"]})
        self.assertEqual(status, 400, payload)
        self.assertIn("token ids", json.loads(payload)["error"]["message"])

    def test_special_must_be_a_boolean(self):
        status, payload = self.post({"tokens": [4], "special": "yes"})
        self.assertEqual(status, 400, payload)
        self.assertIn("special", json.loads(payload)["error"]["message"])
