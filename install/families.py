"""The model families Splash serves, the DFlash2 draft released for each and
the chat format the server reads and writes each one's output in.

A target's family is the one the engine finds its configuration to describe,
by the rules it holds every start to (upstream.check_model), never its
repository's name.
"""

from __future__ import annotations

from dataclasses import dataclass

from server.chat_format import FLINT, QWEN, ChatFormat


@dataclass(frozen=True)
class ModelFamily:
    # The family's name, as the engine names it (each target layout's family,
    # runtime/model/Qwen3_8.hpp, Qwen3_6Moe.hpp, Qwen4Exp.hpp and Flint.hpp).
    name: str
    # The repository of the DFlash2 draft released for the family, as its
    # release publishes it (config.json and BF16 safetensors). Installations
    # follow its default branch as they follow the target's. None while no
    # draft is released for the family: an installation names one with
    # --draft-model.
    draft_repo: str | None
    # How the family's model reasons and calls tools, and the tokens it
    # shares with the engine (server/chat_format.py).
    chat_format: ChatFormat
    # Whether the family's models have a vision tower, which an installation
    # loads from the target repository unless --language-only. The models of
    # a family without one serve text only, whatever their repository holds.
    vision: bool = True
    # Other repositories of the same checkpoint, tried after draft_repo when
    # it cannot be resolved (a z-lab mirror, for example). Their names are
    # data: compatibility is the checkpoint's own config, never its owner.
    draft_mirrors: tuple[str, ...] = ()


FAMILIES = (
    # Qwen3.8-4B and its fine-tunes, such as the Heretic ablation. No DFlash2
    # draft is released for it.
    ModelFamily("Qwen3.8-4B", None, QWEN, vision=False),
    # Qwen3.5-9B and its fine-tunes, such as Ornith-1.5-9B. No DFlash2 draft is
    # released for it; auto serves it without one.
    ModelFamily("Qwen3.5-9B", None, QWEN, vision=False),
    ModelFamily(
        "Qwen3.8-27B",
        "incoai/Qwen3.8-27B-DFlash2",
        QWEN,
        draft_mirrors=("z-lab/Qwen3.8-27B-DFlash2",),
    ),
    ModelFamily(
        "Qwen3.6-35B-A3B",
        "incoai/Qwen3.6-35B-A3B-DFlash2",
        QWEN,
        draft_mirrors=("z-lab/Qwen3.6-35B-A3B-DFlash2",),
    ),
    # Its fine-tune Ornith-1.5-122B is the family too, and the draft
    # released for it the family's.
    ModelFamily("Qwen3.5-122B-A10B", "incoai/Ornith-1.5-122B-DFlash2", QWEN),
    ModelFamily("Qwen3.8-Flash-Next", "incoai/Qwen3.8-Flash-Next-DFlash2", QWEN),
    # It has no vision tower.
    ModelFamily("Flint", "incoai/Flint-DFlash2", FLINT, vision=False),
)


def named(name):
    """The family called name, or None."""
    return next((family for family in FAMILIES if family.name == name), None)


def draft_candidates(family):
    """The family's draft repositories, the released one then its mirrors,
    without duplicates."""
    candidates = []
    for repo in (family.draft_repo, *family.draft_mirrors):
        if repo and repo not in candidates:
            candidates.append(repo)
    return tuple(candidates)
