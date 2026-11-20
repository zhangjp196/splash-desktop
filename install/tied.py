"""A tied MLX checkpoint's logits head, materialized at installation.

A checkpoint with `tie_word_embeddings` true stores no `lm_head`: its output
projection is the token table. The native engine reads a head, so this writes
the embedding's affine tensors under the MLX head name into a shard of its
own and a configuration that says the head is not tied. The assembly links
both; the user's checkpoint is not changed.
"""

from __future__ import annotations

import hashlib
import json
import struct

from . import hub, models
from .models import ModelError


def _shards(paths):
    return [path for name, path in paths.items() if name.endswith(".safetensors")]


def _find_tensor(paths, suffix):
    for path in paths:
        try:
            with path.open("rb") as stream:
                length = struct.unpack("<Q", stream.read(8))[0]
                header = json.loads(stream.read(length))
                start = 8 + length
        except (OSError, ValueError, struct.error):
            continue
        for name, entry in header.items():
            if name == "language_model.model.embed_tokens." + suffix:
                begin, end = entry["data_offsets"]
                return path, start + begin, end - begin, entry
    return None


def _shard_plan(paths):
    tensors = {}
    for suffix in ("weight", "scales", "biases"):
        found = _find_tensor(paths, suffix)
        if found is None:
            raise ModelError(f"a tied checkpoint has no embed_tokens.{suffix}")
        tensors[suffix] = found
    header = {}
    offset = 0
    for suffix, (_, _, length, entry) in tensors.items():
        header["language_model.lm_head." + suffix] = {
            "dtype": entry["dtype"],
            "shape": entry["shape"],
            "data_offsets": [offset, offset + length],
        }
        offset += length
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    return tensors, encoded, 8 + len(encoded) + offset


def _write_shard(tensors, encoded, destination):
    with destination.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        for path, start, length, _ in tensors.values():
            with path.open("rb") as source:
                source.seek(start)
                stream.write(source.read(length))
    return destination


def prepare(config_path, paths, models_root):
    """The assembly paths of a materialized tied head, or None for a config
    that is not tied. paths maps repository file names to local files."""
    config = json.loads(config_path.read_text())
    text = config.get("text_config", config)
    if not text.get("tie_word_embeddings") and not config.get("tie_word_embeddings"):
        return None
    shards = _shards(paths)
    if not shards:
        raise ModelError("a tied checkpoint has no safetensors shard")
    text["tie_word_embeddings"] = False
    config["tie_word_embeddings"] = False
    encoded = json.dumps(config, separators=(",", ":")).encode()
    tensors, header, shard_bytes = _shard_plan(shards)
    output_bytes = len(encoded) + shard_bytes
    key = hashlib.sha256(encoded).hexdigest()[:16]
    directory = models_root / ".metadata" / ("tied-" + key)
    volumes = hub.plan_space([(directory, output_bytes, (), output_bytes, ())])
    hub._report_plan(volumes)
    for volume in volumes:
        if volume["available"] is None:
            models.warn(
                f"could not check free space for {volume['path']}: "
                f"{volume['error']}; continuing without a disk space check"
            )
        elif volume["available"] < output_bytes:
            models.warn(
                f"tied-head preparation needs {output_bytes / 1e9:.2f} GB "
                f"on {volume['path']}, with {volume['available'] / 1e9:.2f} GB free; "
                "continuing with preparation"
            )
    directory.mkdir(parents=True, exist_ok=True)
    written = directory / "config.json"
    written.write_bytes(encoded)
    shard = _write_shard(tensors, header, directory / "lm_head.safetensors")
    return {
        "config.json": written,
        "target/config.json": written,
        "tokenizer/config.json": written,
        "target/lm_head.safetensors": shard,
    }
