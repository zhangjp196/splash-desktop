"""POST /v1/thermal over HTTP: the heat-and-noise rung of a running engine
changes without a restart, and the answer names the duty the engine took."""

import http.client
import json
import unittest
from unittest import mock

from dev.tests.server_fixtures import FakeRuntime, HarnessTestCase
from server import runtime as engine_runtime
from server import serve_options


class ThermalTests(HarnessTestCase):
    def post(self, harness, body, headers=None):
        """A POST of this JSON body to /v1/thermal, and its status and answer."""
        data = json.dumps(body).encode() if not isinstance(body, bytes) else body
        headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(data)),
            **(headers or {}),
        }
        connection = http.client.HTTPConnection(
            *harness.server.server_address, timeout=3
        )
        self.addCleanup(connection.close)
        connection.putrequest("POST", "/v1/thermal")
        for name, value in headers.items():
            connection.putheader(name, value)
        connection.endheaders(data)
        response = connection.getresponse()
        return response.status, json.loads(response.read())

    def duty(self, rung):
        return serve_options.THERMAL_POLICIES[rung].prefill_duty

    def test_every_rung_of_the_ladder_reaches_the_engine(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime)
        for rung in serve_options.THERMAL_POLICIES:
            duty = self.duty(rung)
            with self.subTest(rung=rung):
                self.assertEqual(
                    self.post(harness, {"prefill_duty_percent": duty}),
                    (200, {"prefill_duty_percent": duty}),
                )
        self.assertEqual(
            runtime.duty_calls,
            [self.duty(rung) for rung in serve_options.THERMAL_POLICIES],
        )

    def test_a_duty_between_the_rungs_is_taken(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime)
        self.assertEqual(
            self.post(harness, {"prefill_duty_percent": 55}),
            (200, {"prefill_duty_percent": 55}),
        )
        self.assertEqual(runtime.duty_calls, [55.0])

    def test_the_answer_names_the_duty_the_engine_took(self):
        # The engine clamps what a configuration may not ask for, so a value
        # the server accepted can still be answered differently: the status
        # the engine sends back is what the answer reports.
        runtime = FakeRuntime()
        runtime.duty_taken = 10.0
        harness = self.harness(runtime)
        self.assertEqual(
            self.post(harness, {"prefill_duty_percent": 12}),
            (200, {"prefill_duty_percent": 10.0}),
        )
        self.assertEqual(runtime.duty_calls, [12.0])

    def test_a_duty_outside_the_range_is_refused_before_the_engine(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime)
        for body in (
            {"prefill_duty_percent": 9},
            {"prefill_duty_percent": 101},
            {"prefill_duty_percent": "half"},
            {"prefill_duty_percent": True},
            # A field set to null is one left out.
            {"prefill_duty_percent": None},
            {"duty": 60},
            {},
        ):
            with self.subTest(body=body):
                status, payload = self.post(harness, body)
                self.assertEqual(status, 400)
                self.assertEqual(payload["error"]["code"], "invalid_request_error")
        self.assertEqual(runtime.duty_calls, [])

    def test_a_field_the_endpoint_does_not_take_is_refused(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime)
        status, payload = self.post(
            harness, {"prefill_duty_percent": 60, "unexpected": 1}
        )
        self.assertEqual(status, 400)
        self.assertIn("unexpected", payload["error"]["message"])
        self.assertEqual(runtime.duty_calls, [])

    def test_a_body_that_is_not_an_object_is_refused(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime)
        for body in (b"[]", b'"quiet"'):
            with self.subTest(body=body):
                self.assertEqual(self.post(harness, body)[0], 400)
        self.assertEqual(runtime.duty_calls, [])

    def test_an_engine_that_does_not_answer_or_fails_is_unavailable(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime)
        for error, code in (
            (TimeoutError("native status response timed out"), "engine_busy"),
            (
                engine_runtime.EngineUnhealthy("native loop failed"),
                "runtime_unavailable",
            ),
        ):
            with self.subTest(code=code):
                with mock.patch.object(runtime, "set_prefill_duty", side_effect=error):
                    status, payload = self.post(harness, {"prefill_duty_percent": 60})
                self.assertEqual((status, payload["error"]["code"]), (503, code))

    def test_the_change_needs_the_api_key(self):
        runtime = FakeRuntime()
        harness = self.harness(runtime, api_key="test-server-key")
        status, _payload = self.post(harness, {"prefill_duty_percent": 60})
        self.assertEqual(status, 401)
        self.assertEqual(
            self.post(
                harness,
                {"prefill_duty_percent": 60},
                {"Authorization": "Bearer test-server-key"},
            )[0],
            200,
        )
        self.assertEqual(runtime.duty_calls, [60.0])


if __name__ == "__main__":
    unittest.main()
