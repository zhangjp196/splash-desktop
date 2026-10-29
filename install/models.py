#!/usr/bin/env python3

"""Install and verify the model a Splash selection names.

Terms the installer modules share:
- selection: what `--model OWNER/REPO[:VARIANT]` or `--model-dir DIRECTORY`
  and its source options name (Selection), or the local files a model is
  prepared from (local.py); its selection link, under the models root, points
  to what serves it.
- assembly: the directory an upstream model is served from, of links to
  source snapshot files and its record, model.json (assembly.py), built and
  published by upstream.py. A local target directory is assembled the same
  way, of links to the directory's own files.
- source snapshot: a repository at one commit in the Hub cache, or a local
  target or draft directory (hub.py), or what local.py prepares.

This module holds what the installers share, and the command line.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

if __name__ == "__main__" and not __package__:
    # Run as a script by make, the launcher and `python install/models.py`:
    # import siblings as the install package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "install"

# The model ID syntax, which --model shares with the server, in modules of
# the standard library alone: the launcher imports this before .venv exists.
from server import serve_options

from . import paths

MODELS = paths.MODELS
# The bound on one JSON metadata file the installer reads, a larger one being
# refused unread: configs are kilobytes, and a shard index names every tensor,
# 14 MB for a checkpoint that stores each routed expert's tensors of its own
# (Model Optimizer's). The engine bounds the files it reads by its own rule.
MAX_JSON_BYTES = 64 * 1024 * 1024
# What installation_kind finds at a selection link: an assembly, or a Splash
# package an earlier release installed.
ASSEMBLY, PACKAGE = "assembly", "package"
# Splash packages, the prebuilt format that predated upstream loading, are no
# longer loaded: each package format names the MLX model of its family.
PACKAGE_REPLACEMENTS = {
    "splash-packed-q4": "mlx-community/Qwen3.8-27B-4bit",
    "splash-packed-q4-moe": "mlx-community/Qwen3.6-35B-A3B-4bit",
}
# Staging an interrupted installation leaves, which garbage collection
# removes: a selection link being replaced (beside the link), and an assembly
# or derived metadata entry being written (beside the published entries).
LINK_STAGING = ".prepare-"
ENTRY_STAGING = ".loading-"


class ModelError(RuntimeError):
    pass


class VisionError(ModelError):
    """A refusal of the model's vision alone, which serving text only avoids:
    each caller names its own way to ask for that."""


def warn(message):
    """Report something the user may need to act on, on stderr."""
    print(f"Warning: {message}", file=sys.stderr, flush=True)


def run_engine(arguments, check):
    """The output of one of the engine's checks, `ENGINE ARGUMENTS`, which
    check names. A ModelError names an engine that cannot run, or that
    refuses the command line as one built before the check does, and what
    replaces it; else it gives the engine's refusal, its error: line without
    that prefix, or the whole report of an engine that ended before main()
    (dyld on an older macOS)."""
    engine = paths.BINARY
    replace = "reinstall Splash" if paths.PACKAGED else "rebuild it with make"
    try:
        result = subprocess.run(
            [str(engine), *arguments], capture_output=True, text=True
        )
    except OSError as error:
        raise ModelError(
            f"cannot run the engine {engine} ({error.strerror}); {replace}"
        ) from None
    if result.returncode == os.EX_USAGE:
        raise ModelError(f"the engine {engine} does not know this {check}; {replace}")
    if result.returncode:
        report = result.stderr.strip()
        refusals = [
            line.removeprefix("error: ")
            for line in report.splitlines()
            if line.startswith("error: ")
        ]
        raise ModelError(
            refusals[-1]
            if result.returncode > 0 and refusals
            else f"the engine's {check} failed: {report or f'status {result.returncode}'}"
        )
    return result.stdout


def is_hex_digest(value, length: int) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(rf"[0-9a-fA-F]{{{length}}}", value) is not None
    )


def is_safe_path(name) -> bool:
    """Whether name is a plain relative POSIX path: no absolute, empty, "."
    or ".." part, no backslash or control character, and none of the
    characters Hub download patterns read as globs (* ? [ ])."""
    if not isinstance(name, str):
        return False
    path = PurePosixPath(name)
    return (
        bool(path.parts)  # "" and "." have none
        and not path.is_absolute()
        and path.as_posix() == name
        and ".." not in path.parts
        and not any(character in name for character in "\\*?[]")
        and all(ord(character) >= 32 for character in name)
    )


def validate_repo_id(value: str) -> str:
    try:
        return serve_options.check_repo_id(value)
    except ValueError as error:
        raise ModelError(str(error)) from None


def split_model_id(value: str) -> tuple[str, str | None]:
    """owner/repo[:variant] -> (repository ID, variant or None)."""
    try:
        return serve_options.split_model_id(value)
    except ValueError as error:
        raise ModelError(str(error)) from None


def parse_draft_model(value: str) -> str:
    # A local draft directory is recorded as an absolute path, so the
    # installation it selects does not depend on the working directory.
    if value and (local := Path(value).expanduser()).is_dir():
        return str(local.resolve())
    try:
        return validate_repo_id(value)
    except ModelError:
        raise argparse.ArgumentTypeError(
            "must be a local DFlash2 draft directory or a Hugging Face "
            "repository ID (owner/repo)"
        ) from None


def local_target_path(value) -> Path:
    """value as the absolute path of a local target. A directory is resolved
    whole; a file keeps its own name, since a GGUF may be a link to a Hub cache
    blob and resolving the file would replace its .gguf name with the blob's
    digest."""
    path = Path(value).expanduser()
    if path.is_dir():
        return path.resolve()
    path = Path(os.path.abspath(path))
    return path.parent.resolve() / path.name


def parse_model_dir(value: str) -> str:
    # Like a local draft, a local target is recorded as an absolute path, so the
    # installation it selects does not depend on the working directory. It is a
    # directory holding an MLX target or one GGUF, or a GGUF file itself, which
    # local.py serves with the parts beside it.
    path = local_target_path(value) if value else None
    if path is None or not (path.is_dir() or path.is_file()):
        raise argparse.ArgumentTypeError(f"local model directory not found: {value}")
    if path.is_file() and path.suffix.lower() != ".gguf":
        raise argparse.ArgumentTypeError(f"a local model file is a GGUF: {value}")
    return str(path)


def parse_mmproj(value: str) -> str:
    # A vision projector is a GGUF file wherever it lies, recorded as an
    # absolute path as a local draft is: an installation it selects does not
    # depend on the working directory.
    path = Path(value).expanduser() if value else None
    if path is None or not path.is_file():
        raise argparse.ArgumentTypeError(f"vision projector not found: {value}")
    if path.suffix.lower() != ".gguf":
        raise argparse.ArgumentTypeError(f"a vision projector is a GGUF file: {value}")
    return str(path.resolve())


def derived_model_id(directory) -> str:
    """The API model ID a local target serves as: its folder name, or a GGUF
    file's name without the extension, reduced to the characters a repository
    ID may hold. A name that reduces to nothing becomes `model`."""
    path = Path(directory)
    name = re.sub(r"[^A-Za-z0-9._-]", "-", path.stem if path.is_file() else path.name)
    name = re.sub(r"\.{2,}", ".", re.sub(r"-{2,}", "-", name))
    # 96 is the repository-ID bound the server's validation enforces.
    name = name[:96].strip("-.") or "model"
    return name


def hash_file(path: Path, digest) -> str:
    """The hex digest of path's content fed to digest, a hashlib object."""
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256(path: Path) -> str:
    return hash_file(path, hashlib.sha256())


def read_json(path: Path):
    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            raise ModelError(f"JSON metadata is too large: {path}")
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelError(f"could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ModelError(f"expected a JSON object in {path}")
    return value


def json_bytes(value) -> bytes:
    """Canonical JSON: the same value always encodes to the same bytes."""
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    ).encode()


def selection_link(
    models: Path, model_id: str, *, revision=None, language_only=False, draft_model=None
) -> Path:
    """The selection link of model_id with these source options:
    OWNER/REPO[:VARIANT] under models for the model alone, else
    .selections/<hash of the selection>."""
    repo_id, variant = split_model_id(model_id)
    if revision or language_only or draft_model:
        return _hashed_link(models, [model_id, revision, language_only, draft_model])
    if variant is None:
        return models / repo_id
    return models / f"{repo_id}{serve_options.VARIANT_SEPARATOR}{variant}"


def local_selection_link(
    models: Path, directory: Path, *, language_only=False, draft_model=None, mmproj=None
) -> Path:
    """The selection link of a local target directory with these source
    options, in the same .selections place as a selected Hub model: the
    directory has no repository ID to name a link. A named vision projector is
    part of the selection: the same target with the installer's projector and
    with this one links apart. Without one, the link is what it always was."""
    selection = [str(directory), language_only, draft_model]
    if mmproj is not None:
        selection.append(str(mmproj))
    selection = json.dumps(selection, separators=(",", ":"))
    return models / ".selections" / hashlib.sha256(selection.encode()).hexdigest()


def prepared_selection_link(models: Path, target, draft, mmproj, language_only) -> Path:
    """The selection link of a model prepared from local files (local.py):
    .selections/<hash of its paths and options>, apart from every Hub
    selection's, whose hashed form is a list."""
    return _hashed_link(
        models,
        {
            "target": str(target),
            "draft": draft and str(draft),
            "mmproj": mmproj and str(mmproj),
            "language_only": language_only,
        },
    )


def _hashed_link(models: Path, selection) -> Path:
    """.selections/<SHA-256 of selection's compact JSON> under models."""
    encoded = json.dumps(selection, separators=(",", ":"))
    return models / ".selections" / hashlib.sha256(encoded.encode()).hexdigest()


def selection_links(models: Path):
    """Every selection link under models, in the places selection_link names."""
    return [path for path in models.glob("*/*") if path.is_symlink()]


@dataclass(frozen=True)
class Selection:
    """What --model or --model-dir and its source options select, and its
    selection link."""

    model: str
    repo_id: str | None
    variant: str | None
    revision: str | None
    language_only: bool
    draft_model: str | None
    models_root: Path
    link: Path
    directory: Path | None = None
    # The vision projector a local GGUF target names, instead of the one the
    # installer selects from the directory and the Hub.
    mmproj: Path | None = None

    @classmethod
    def of(
        cls,
        models_root,
        model=None,
        *,
        directory=None,
        revision=None,
        language_only=False,
        draft_model=None,
        mmproj=None,
    ):
        models_root = Path(models_root).resolve()
        if mmproj is not None:
            mmproj = Path(mmproj).expanduser().resolve()
            if directory is None:
                raise ModelError(
                    f"vision projector {mmproj} requires a local model directory "
                    "(--model-dir): a Hub model's projector is chosen from its "
                    "repository"
                )
            if language_only:
                raise ModelError(
                    f"{mmproj} is a vision projector, which a model that serves "
                    "text only (--language-only) does not load"
                )
        if directory is not None:
            if model is not None:
                raise ModelError(
                    "a local model directory and a repository ID are exclusive"
                )
            directory = local_target_path(directory)
            model = derived_model_id(directory)
            repo_id = variant = None
            link = local_selection_link(
                models_root,
                directory,
                language_only=language_only,
                draft_model=draft_model,
                mmproj=mmproj,
            )
        else:
            repo_id, variant = split_model_id(model)
            link = selection_link(
                models_root,
                model,
                revision=revision,
                language_only=language_only,
                draft_model=draft_model,
            )
        return cls(
            model,
            repo_id,
            variant,
            revision,
            language_only,
            draft_model,
            models_root,
            link,
            directory,
            mmproj,
        )


def installation_kind(link: Path):
    """What serves a selection link: an assembly (its model.json), a Splash
    package an earlier release installed (its manifest.json), or nothing
    (None)."""
    if (link / "model.json").exists():
        return ASSEMBLY
    if (link / "manifest.json").exists():
        return PACKAGE
    return None


def refuse_package(model, manifest, link=None):
    """Raise for a Splash package, whose manifest.json names a package format,
    with the MLX model to serve instead; another tool's manifest.json passes.
    For a package an earlier release installed at link, a link to its Hub
    snapshot, the message names the Hub cache folder holding its files, which
    nothing reads any more."""
    format_ = manifest.get("format") if isinstance(manifest, dict) else None
    name = format_.get("name") if isinstance(format_, dict) else None
    if name in PACKAGE_REPLACEMENTS:
        snapshot = link.resolve() if link is not None and link.is_symlink() else None
        files = ""
        if snapshot and snapshot.parent.name == "snapshots":
            folder = snapshot.parent.parent
            if folder.name.startswith("models--"):
                files = f" (its files in {folder} can be deleted)"
        raise ModelError(
            f"{model} is a Splash package, which Splash no longer loads{files}; "
            f"serve the MLX model of its family instead: splash serve --model "
            f"{PACKAGE_REPLACEMENTS[name]}"
        )


@contextmanager
def installation_lock(models: Path):
    """Exclusive access to everything written under models."""
    lock_path = models / ".install.lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(
                "Another Splash model installation is running; waiting...",
                flush=True,
            )
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def link_selection(link: Path, target: Path):
    """Point the selection link at target, atomically. Only a link is ever
    replaced: a directory at its path is someone's own files."""
    if link.exists() and not link.is_symlink():
        raise ModelError(f"refusing to replace non-symlink model path: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f"{LINK_STAGING}{link.name}-", dir=link.parent)
    )
    temporary = stage / "model"
    try:
        os.symlink(target, temporary, target_is_directory=True)
        os.replace(temporary, link)
    finally:
        temporary.unlink(missing_ok=True)
        stage.rmdir()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Install Splash runtime weights")
    parser.add_argument("--models", type=Path, default=MODELS)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--model",
        type=serve_options.parse_model_id,
        help="Hugging Face repository ID (owner/repo[:variant])",
    )
    source.add_argument(
        "--model-dir",
        dest="model_dir",
        type=parse_model_dir,
        metavar="DIRECTORY",
        help="local model directory: a Splash package, or an MLX/GGUF target "
        "(with --draft-model)",
    )
    parser.add_argument("--revision", help="optional upstream branch, tag or commit")
    parser.add_argument(
        "--draft-model",
        type=parse_draft_model,
        help="the DFlash2 draft's repository or local directory, in place of the "
        "family's released one (required for a family without one)",
    )
    parser.add_argument(
        "--mmproj",
        type=parse_mmproj,
        metavar="FILE",
        help="a local GGUF target's vision projector (a GGUF mmproj file), in "
        "place of the one the installer selects beside the target",
    )
    parser.add_argument(
        "--language-only",
        action="store_true",
        help="skip vision preparation and loading",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare").add_argument(
        "--json-progress",
        action="store_true",
        help="also report download progress as JSON lines on stdout",
    )
    commands.add_parser("verify").add_argument("--full", action="store_true")
    commands.add_parser("link", help="print the selection link")
    args = parser.parse_args(argv)
    if args.model_dir is not None and args.revision:
        # A local target directory has no Hub revision to follow. Whether it
        # also needs --draft-model depends on what it holds: a Splash runtime
        # package carries its own draft, an MLX or GGUF target names one.
        parser.error("--revision requires a Hugging Face model ID")
    if args.mmproj is not None:
        if args.model_dir is None:
            # A Hub model's projector is chosen from its repository's headers.
            parser.error("--mmproj requires --model-dir")
        if args.language_only:
            parser.error("--mmproj and --language-only are exclusive")
    return args


def main(argv=None):
    args = parse_args(argv)
    selection = Selection.of(
        args.models,
        args.model,
        directory=args.model_dir,
        revision=args.revision,
        language_only=args.language_only,
        draft_model=args.draft_model,
        mmproj=args.mmproj,
    )
    if args.command == "link":
        print(selection.link)
        return 0
    # The installers import this module, so it imports them once it exists.
    from . import assembly, hub, upstream

    try:
        # The launcher starts this with the stop signals blocked, so that one
        # sent while it starts is not lost; it arrives here.
        signal.pthread_sigmask(signal.SIG_UNBLOCK, (signal.SIGINT, signal.SIGTERM))
        if args.command == "prepare":
            if args.json_progress:
                hub.report_json_progress()
            if selection.directory is not None:
                upstream.prepare_local(selection)
            else:
                upstream.prepare(selection)
        else:
            kind = installation_kind(selection.link)
            if kind == PACKAGE:
                refuse_package(
                    args.model or args.model_dir,
                    read_json(selection.link / "manifest.json"),
                    selection.link,
                )
            if kind != ASSEMBLY:
                raise ModelError(
                    f"{args.model or args.model_dir} is not installed in {args.models}"
                )
            assembly.verify(selection.link, full=args.full)
            print(
                f"Splash model {args.model or args.model_dir} preflight passed "
                f"({'full' if args.full else 'quick'})."
            )
    except (ModelError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    # The other installer modules import this file by its module name. Run
    # that module, not this __main__ copy, so the ModelError they raise is the
    # one main() reports.
    import importlib

    installer = importlib.import_module("install.models")
    status = installer.main()
    if status == 130:
        # Interrupted, a download goes on in huggingface_hub's threads, which
        # the interpreter would wait for at exit (hub._interruptible): end
        # the process at once, its output written.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(status)
    raise SystemExit(status)
