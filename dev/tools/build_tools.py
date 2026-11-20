#!/usr/bin/env python3
"""Record native build configurations and verify build identities."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dev.tools import build_config, build_identity


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["build-identity"]:
        return build_identity.main(arguments[1:])

    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    config = modes.add_parser("build-config")
    config_modes = config.add_subparsers(dest="action", required=True)
    stale = config_modes.add_parser("stale")
    stale.add_argument("--config", required=True)
    stale.add_argument("outputs", nargs="*")
    record = config_modes.add_parser("record")
    record.add_argument("--config", required=True)
    record.add_argument("--output", required=True)
    record.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(arguments)
    if args.action == "stale":
        print(" ".join(path for path in args.outputs if not build_config.matches(path, args.config)))
        return 0
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("record requires a compiler or linker command after --")
    return build_config.run_configured(args.output, args.config, command)


if __name__ == "__main__":
    raise SystemExit(main())
