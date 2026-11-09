"""External MTP head metadata: the configuration an installation validates
before it links the head as plan data.

This release does not run an MTP head (docs/speculation.zh.md P2). It still
checks the head's declaration against the target: the family, hidden size,
vocabulary and target layer count must agree, and the proposal width must fit
a verify block. A head whose declaration is missing or different is refused by
name, rather than installed as data nothing can ever use.
"""

from __future__ import annotations

import json
from pathlib import Path

from .models import ModelError

# The most rows a cycle could verify; an MTP depth past it is not one the
# engine's block can carry when its runtime lands.
MAX_PROPOSAL_ROWS = 8


def _read(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ModelError(f"cannot read MTP head config {path}: {error}") from None
    if not isinstance(value, dict):
        raise ModelError(f"MTP head config {path} is not an object")
    return value


def _target_text(config: dict) -> dict:
    text = config.get("text_config")
    return text if isinstance(text, dict) else config


def validate(target_config: Path, head_config: Path, family: str) -> dict:
    """The head's alignment, checked against the target's configuration.

    The head states `alignment` with family, hidden_size, vocabulary_size and
    target_layers; an optional `proposal_rows` (1 to 8) is the width it was
    trained for. Returns the alignment as it is recorded in model.json.
    """
    target = _target_text(_read(target_config))
    head = _read(head_config)
    architectures = head.get("architectures")
    if not isinstance(architectures, list) or "MtpHead" not in architectures:
        raise ModelError("MTP head config architectures must name MtpHead")
    rows = head.get("proposal_rows", 1)
    if type(rows) is not int or not 1 <= rows <= MAX_PROPOSAL_ROWS:
        raise ModelError("MTP head proposal_rows must be a whole number from 1 to 8")
    alignment = head.get("alignment")
    if not isinstance(alignment, dict):
        raise ModelError("MTP head config must state alignment")
    expected = {
        "family": family,
        "hidden_size": target.get("hidden_size"),
        "vocabulary_size": target.get("vocab_size"),
        "target_layers": target.get("num_hidden_layers"),
    }
    for key, value in expected.items():
        found = alignment.get(key)
        if found != value:
            raise ModelError(
                f"MTP head alignment {key} is {found!r}, target is {value!r}"
            )
    return {
        "family": family,
        "hidden_size": expected["hidden_size"],
        "vocabulary_size": expected["vocabulary_size"],
        "target_layers": expected["target_layers"],
        "proposal_rows": rows,
    }
