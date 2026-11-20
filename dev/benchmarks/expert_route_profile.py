#!/usr/bin/env python3
"""Summarize streamed expert route sets; this is not a cache simulator."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def profile(path):
    groups = {}
    previous = {}
    records = 0
    identity = None
    identity_seen = False
    weight_hash = None
    with open(path, "rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            try:
                if not raw.endswith(b"\n"):
                    raise ValueError("unterminated line")
                record = json.loads(raw)
                if not isinstance(record, dict):
                    raise ValueError("record must be an object")
                if type(record.get("v")) is not int or record["v"] not in (1, 2):
                    raise ValueError("unsupported trace version")
                current_identity = record.get("model_identity")
                if current_identity is not None and (
                    not isinstance(current_identity, str)
                    or not current_identity.strip()
                ):
                    raise ValueError("invalid model_identity")
                if identity_seen and current_identity != identity:
                    raise ValueError("trace contains mixed or missing model identities")
                current_hash = record.get("weight_manifest_hash")
                if record["v"] == 2:
                    if record.get("identity_status") not in ("known", "unknown"):
                        raise ValueError("invalid identity_status")
                    if record.get("priority_eligible") is not (
                        record["identity_status"] == "known"
                    ):
                        raise ValueError(
                            "priority_eligible disagrees with identity_status"
                        )
                    if record["identity_status"] == "known" and (
                        not isinstance(current_hash, str) or len(current_hash) != 64
                    ):
                        raise ValueError("invalid weight_manifest_hash")
                    if (
                        record["identity_status"] == "unknown"
                        and current_hash != "unknown"
                    ):
                        raise ValueError(
                            "unknown identity must not carry a fingerprint"
                        )
                else:
                    current_hash = None
                if identity_seen and current_hash != weight_hash:
                    raise ValueError(
                        "trace contains mixed or missing weight fingerprints"
                    )
                identity_seen = True
                identity = current_identity
                weight_hash = current_hash
                block, phase, rows, experts = (
                    record.get(k) for k in ("block", "phase", "rows", "experts")
                )
                if type(block) is not int or not 0 <= block <= 0xFFFFFFFF:
                    raise ValueError("invalid block")
                if phase not in ("prefill", "decode"):
                    raise ValueError("invalid phase")
                if type(rows) is not int or not 0 <= rows <= 0xFFFFFFFF:
                    raise ValueError("invalid rows")
                if (
                    not isinstance(experts, list)
                    or any(
                        type(e) is not int or not 0 <= e <= 0xFFFFFFFF for e in experts
                    )
                    or any(a >= b for a, b in zip(experts, experts[1:]))
                ):
                    raise ValueError("experts must be sorted unique unsigned integers")
            except (ValueError, UnicodeDecodeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error

            records += 1
            key = (block, phase)
            if key not in groups:
                groups[key] = {
                    "loads": 0,
                    "rows_sum": 0,
                    "expert_selections": 0,
                    "expert_loads": Counter(),
                    "consecutive_pairs": 0,
                    "reused_selections": 0,
                    "current_selections_in_pairs": 0,
                }
            group = groups[key]
            group["loads"] += 1
            group["rows_sum"] += rows
            group["expert_selections"] += len(experts)
            group["expert_loads"].update(experts)
            current = set(experts)
            # The preceding load of this block can have a different phase.
            if block in previous and previous[block][0] == phase:
                group["consecutive_pairs"] += 1
                group["reused_selections"] += len(current & previous[block][1])
                group["current_selections_in_pairs"] += len(current)
            previous[block] = (phase, current)

    result = []
    for (block, phase), group in sorted(groups.items()):
        counts = group.pop("expert_loads")
        denominator = group["current_selections_in_pairs"]
        result.append(
            {
                "block": block,
                "phase": phase,
                **group,
                "reuse_fraction": group["reused_selections"] / denominator
                if denominator
                else None,
                "experts": [
                    {"expert": expert, "loads": count}
                    for expert, count in sorted(
                        counts.items(), key=lambda item: (-item[1], item[0])
                    )
                ],
            }
        )
    return {
        "schema_version": 3,
        "model_identity": identity if identity is not None else "unknown",
        "weight_manifest_hash": weight_hash if weight_hash is not None else "unknown",
        "priority_eligible": isinstance(weight_hash, str) and len(weight_hash) == 64,
        "records": records,
        "groups": result,
    }


def load_priority_hints(
    path, current_identity, current_weight_manifest_hash, *, opt_in=False, top_k=8
):
    """Return block -> ranked experts only for an explicitly trusted profile.

    Invalid, unknown-identity, and model or weight-mismatched profiles are
    ignored as a whole.
    """
    if (
        not opt_in
        or not isinstance(current_identity, str)
        or not current_identity.strip()
        or not isinstance(current_weight_manifest_hash, str)
        or len(current_weight_manifest_hash) != 64
    ):
        return {}
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            document = json.load(stream)
        if (
            not isinstance(document, dict)
            or type(document.get("schema_version")) is not int
            or document["schema_version"] != 3
            or document.get("model_identity") != current_identity
            or document.get("weight_manifest_hash") != current_weight_manifest_hash
            or document.get("priority_eligible") is not True
            or not isinstance(document.get("groups"), list)
            or type(top_k) is not int
            or not 0 < top_k <= 64
        ):
            return {}
        hints = {}
        for group in document["groups"]:
            if not isinstance(group, dict):
                return {}
            block, entries = group.get("block"), group.get("experts")
            if (
                type(block) is not int
                or not 0 <= block <= 0xFFFFFFFF
                or not isinstance(entries, list)
            ):
                return {}
            ranked = []
            seen = set()
            for entry in entries:
                if not isinstance(entry, dict):
                    return {}
                expert, loads = entry.get("expert"), entry.get("loads")
                if (
                    type(expert) is not int
                    or not 0 <= expert <= 0xFFFFFFFF
                    or type(loads) is not int
                    or loads <= 0
                    or expert in seen
                ):
                    return {}
                seen.add(expert)
                ranked.append((expert, loads))
            ranked.sort(key=lambda item: (-item[1], item[0]))
            # Multiple phases can name the same block; combine counts without
            # allowing file order to affect the resulting order.
            aggregate = hints.setdefault(block, Counter())
            aggregate.update(dict(ranked))
        return {
            block: [
                expert
                for expert, _ in sorted(
                    counts.items(), key=lambda item: (-item[1], item[0])
                )[:top_k]
            ]
            for block, counts in hints.items()
        }
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", help="v1 JSONL route trace")
    args = parser.parse_args()
    try:
        result = profile(args.trace)
    except (OSError, ValueError) as error:
        print(f"expert route profile: {error}", file=sys.stderr)
        return 1
    json.dump(result, sys.stdout, indent=2, allow_nan=False)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
