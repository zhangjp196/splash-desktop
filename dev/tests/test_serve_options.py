"""Serve-option plumbing from the launcher's CLI down to the native engine.

These tests pin the argument plumbing only: the engine owns the behavior of
each switch, and its own suite covers that.
"""

import unittest
from types import SimpleNamespace

from server import server as api


def _args(**overrides):
    defaults = {
        "binary": "/engine/splash",
        "target": "/models/target",
        "draft": "/models/draft",
        "max_context": None,
        "max_memory": None,
        "max_cache_disk": None,
        "kv_format": "int8",
        "idle_offload_seconds": 0,
        "residency_seconds": api.DEFAULT_RESIDENCY_SECONDS,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class NativeCommandTest(unittest.TestCase):
    def test_defaults_stop_after_the_optional_quota(self):
        # The engine rejects unknown trailing arguments, so an option left at
        # its default must not appear on the line at all.
        self.assertEqual(
            api._native_command(_args(max_cache_disk="4G")),
            [
                "/engine/splash",
                "serve-native",
                "/models/target",
                "/models/draft",
                "auto",
                "auto",
                "4G",
            ],
        )

    def test_idle_offload_and_residency_follow_the_quota(self):
        command = api._native_command(
            _args(max_cache_disk="4G", idle_offload_seconds=10, residency_seconds=2)
        )
        self.assertEqual(
            command[6:],
            ["4G", "--idle-offload-seconds", "10", "--residency-seconds", "2"],
        )

    def test_switches_survive_without_a_disk_quota(self):
        # The engine reads the quota as the only positional trailing argument,
        # so the switches must not stand in for it.
        command = api._native_command(
            _args(idle_offload_seconds=10, residency_seconds=30)
        )
        self.assertEqual(
            command[6:],
            ["--idle-offload-seconds", "10", "--residency-seconds", "30"],
        )

    def test_kv_format_precedes_the_idle_switches(self):
        command = api._native_command(
            _args(kv_format="bf16", idle_offload_seconds=10, residency_seconds=30)
        )
        self.assertEqual(
            command[6:],
            [
                "--kv-format",
                "bf16",
                "--idle-offload-seconds",
                "10",
                "--residency-seconds",
                "30",
            ],
        )


if __name__ == "__main__":
    unittest.main()
