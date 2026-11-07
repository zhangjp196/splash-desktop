#!/usr/bin/env python3
"""Refuse merge commits after the 1.7.1 release.

This fork ports external work by reading it and applying the change by hand,
with its tests and documentation. A merge commit would import another tree's
history and make that rule impossible to audit, so `make check-source` fails
on any merge after the `1.7.1` tag.
"""

import subprocess
import sys

BASE = "1.7.1"


def git(*arguments):
    result = subprocess.run(
        ["git", *arguments], capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise SystemExit(result.stderr.strip() or "git failed")
    return result.stdout


def main():
    try:
        git("rev-parse", "--verify", f"refs/tags/{BASE}")
    except SystemExit:
        raise SystemExit(f"the {BASE} release tag is missing") from None
    merges = git("rev-list", "--merges", f"{BASE}..HEAD").split()
    if not merges:
        print(f"history: no merges after {BASE}")
        return 0
    print(
        f"history: merge commits after {BASE} are not allowed; "
        "port the change by hand instead:",
        file=sys.stderr,
    )
    for commit in merges:
        subject = git("show", "-s", "--format=%h %s", commit).strip()
        print(f"  {subject}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
