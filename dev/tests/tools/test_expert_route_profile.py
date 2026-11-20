import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from dev.benchmarks.expert_route_profile import load_priority_hints, profile


class ExpertRouteProfileTest(unittest.TestCase):
    def test_profile_counts_and_same_block_order(self):
        records = [
            {"v": 1, "block": 0, "phase": "decode", "rows": 2, "experts": [1, 3]},
            {"v": 1, "block": 1, "phase": "decode", "rows": 1, "experts": [1]},
            {"v": 1, "block": 0, "phase": "decode", "rows": 1, "experts": [3, 4]},
            {"v": 1, "block": 0, "phase": "prefill", "rows": 8, "experts": [3]},
            {"v": 1, "block": 0, "phase": "decode", "rows": 1, "experts": [3]},
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "routes.jsonl"
            path.write_text("".join(json.dumps(record) + "\n" for record in records))
            result = profile(path)
        self.assertEqual(result["model_identity"], "unknown")
        self.assertEqual(result["records"], 5)
        group = result["groups"][0]
        self.assertEqual(
            (group["block"], group["phase"], group["loads"]), (0, "decode", 3)
        )
        self.assertEqual(group["rows_sum"], 4)
        self.assertEqual(group["expert_selections"], 5)
        self.assertEqual(
            group["experts"],
            [
                {"expert": 3, "loads": 3},
                {"expert": 1, "loads": 1},
                {"expert": 4, "loads": 1},
            ],
        )
        self.assertEqual(
            (
                group["consecutive_pairs"],
                group["reused_selections"],
                group["current_selections_in_pairs"],
            ),
            (1, 1, 2),
        )
        self.assertEqual(group["reuse_fraction"], 0.5)
        self.assertIsNone(result["groups"][1]["reuse_fraction"])

    def test_empty_and_invalid_traces(self):
        valid = {"v": 1, "block": 0, "phase": "decode", "rows": 1, "experts": []}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trace"
            path.write_bytes(b"")
            self.assertEqual(profile(path)["groups"], [])
            for raw in [
                b"{}",
                b"{\n",
                b"\xff\n",
                b"[]\n",
                *(
                    json.dumps({**valid, **change}).encode() + b"\n"
                    for change in [
                        {"v": 2},
                        {"v": True},
                        {"block": -1},
                        {"rows": True},
                        {"phase": "other"},
                        {"experts": [2, 1]},
                        {"experts": [1, 1]},
                    ]
                ),
            ]:
                with self.subTest(raw=raw):
                    path.write_bytes(json.dumps(valid).encode() + b"\n" + raw)
                    with self.assertRaisesRegex(ValueError, ":2:"):
                        profile(path)

    def test_identity_is_preserved_and_mixed_files_are_rejected(self):
        base = {"v": 1, "block": 0, "phase": "decode", "rows": 1, "experts": []}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trace"
            path.write_text(json.dumps({**base, "model_identity": "qwen-test"}) + "\n")
            self.assertEqual(profile(path)["model_identity"], "qwen-test")
            for records in (
                [{**base, "model_identity": "one"}, {**base, "model_identity": "two"}],
                [{**base, "model_identity": "one"}, base],
                [{**base, "model_identity": ""}],
            ):
                path.write_text(
                    "".join(json.dumps(record) + "\n" for record in records)
                )
                with self.assertRaises(ValueError):
                    profile(path)

    def test_fingerprinted_trace_and_mixed_fingerprints(self):
        base = {
            "v": 2,
            "model_identity": "model",
            "weight_manifest_hash": "a" * 64,
            "identity_status": "known",
            "priority_eligible": True,
            "block": 0,
            "phase": "decode",
            "rows": 1,
            "experts": [1],
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trace"
            path.write_text(json.dumps(base) + "\n")
            result = profile(path)
            self.assertEqual(result["schema_version"], 3)
            self.assertEqual(result["weight_manifest_hash"], "a" * 64)
            self.assertTrue(result["priority_eligible"])
            path.write_text(
                "".join(
                    json.dumps({**base, **record}) + "\n"
                    for record in (
                        {},
                        {"weight_manifest_hash": "b" * 64},
                    )
                )
            )
            with self.assertRaisesRegex(
                ValueError, "mixed or missing weight fingerprints"
            ):
                profile(path)
            unknown = {
                **base,
                "weight_manifest_hash": "unknown",
                "identity_status": "unknown",
                "priority_eligible": False,
            }
            path.write_text(json.dumps(unknown) + "\n")
            self.assertFalse(profile(path)["priority_eligible"])

    def test_priority_hints_require_opt_in_and_exact_identity(self):
        document = {
            "schema_version": 3,
            "model_identity": "descriptor-name",
            "weight_manifest_hash": "a" * 64,
            "priority_eligible": True,
            "records": 3,
            "groups": [
                {
                    "block": 2,
                    "experts": [{"expert": 4, "loads": 3}, {"expert": 1, "loads": 2}],
                },
                {
                    "block": 2,
                    "experts": [{"expert": 1, "loads": 5}, {"expert": 7, "loads": 1}],
                },
            ],
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(document))
            self.assertEqual(load_priority_hints(path, "descriptor-name", "a" * 64), {})
            self.assertEqual(
                load_priority_hints(path, "other", "a" * 64, opt_in=True), {}
            )
            self.assertEqual(
                load_priority_hints(
                    path, "descriptor-name", "a" * 64, opt_in=True, top_k=2
                ),
                {2: [1, 4]},
            )
            for change in (
                {"schema_version": 1},
                {"model_identity": "unknown"},
                {"weight_manifest_hash": "b" * 64},
                {"groups": [{"block": 2, "experts": [{"expert": -1, "loads": 1}]}]},
                {"groups": [{"block": 2, "experts": [{"expert": 1, "loads": True}]}]},
            ):
                path.write_text(json.dumps({**document, **change}))
                self.assertEqual(
                    load_priority_hints(path, "descriptor-name", "a" * 64, opt_in=True),
                    {},
                )
            path.write_text("{broken")
            self.assertEqual(
                load_priority_hints(path, "descriptor-name", "a" * 64, opt_in=True), {}
            )

    def test_cli_rejects_partial_input_without_output(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trace"
            path.write_bytes(b'{"v":1}')
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "dev.benchmarks.expert_route_profile",
                    str(path),
                ],
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("unterminated line", result.stderr)


if __name__ == "__main__":
    unittest.main()
