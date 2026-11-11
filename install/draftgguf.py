"""Convert a DFlash2 draft published as GGUF into the BF16 safetensors the
engine loads at every start.

The runtime's draft loader reads a checkpoint released as `config.json` and
BF16 safetensors (runtime/model/DraftCheckpoint.cpp). A GGUF conversion of the
same checkpoint is data too: this module reads its BF16, F16 and F32 tensors,
renames them to the release's names, and writes one `model.safetensors` (and
`mask_embedding.safetensors` when the checkpoint ships that tensor). The
installation links the converted files under `draft/`; the source repository
and commit stay in `model.json` as `sources.draft`.

Only the types a draft is published in are converted: BF16, F16 and F32. A
quantized draft GGUF is refused by name rather than silently dequantized here;
that is a later provider, not an installation accident.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

from . import gguf
from .models import ModelError

# The converter's identity: a change to the mapping or the writing changes the
# derived entry's key (assembly's .metadata path), so an installed draft is
# regenerated rather than reused.
VERSION = 1
MASK_EMBEDDING = "mask_embedding"
# kind -> (block elements, block bytes). BF16, F16 and F32 are one element and
# their bytes; Q8_0 is llama.cpp's 32-element block, one half d and 32 int8
# codes.
SUPPORTED = {
    "BF16": (1, 2),
    "F16": (1, 2),
    "F32": (1, 4),
    "Q8_0": (32, 34),
}
GGUF_ALIGNMENT = 32


def tensor_name(name: str) -> str:
    """The release's name of a GGUF draft tensor: the `draft.` prefix removed,
    `blk.N` and `model.layers.N` written as `layers.N`, and a plain
    `selector.` as `candidate_selector.`."""
    if name.startswith("draft."):
        name = name[len("draft.") :]
    if (
        name.startswith("model.layers.")
        or name.startswith("model.fc.")
        or name.startswith("model.norm.")
        or name.startswith("model.hidden_norm.")
        or name.startswith("model.candidate_selector.")
    ):
        name = name[len("model.") :]
    if name.startswith("blk."):
        name = "layers." + name[len("blk.") :]
    if name.startswith("selector."):
        name = "candidate_selector." + name[len("selector.") :]
    if name == "candidate_selector.hidden_projection" or (
        name.startswith("candidate_selector.hidden_projection.")
        and not name.endswith(".weight")
    ):
        name += ".weight"
    return name


def _bf16(raw: bytes, kind: str) -> bytes:
    """raw values of kind as little-endian bf16 bits."""
    if kind == "BF16":
        return raw
    if kind == "F16":
        values = struct.unpack(f"<{len(raw) // 2}e", raw)
        return b"".join(_float_to_bf16(value) for value in values)
    if kind == "F32":
        values = struct.unpack(f"<{len(raw) // 4}f", raw)
        return b"".join(_float_to_bf16(value) for value in values)
    if kind == "Q8_0":
        out = bytearray()
        for at in range(0, len(raw), 34):
            block = raw[at : at + 34]
            if len(block) != 34:
                raise ModelError("truncated Q8_0 draft block")
            (scale,) = struct.unpack("<e", block[:2])
            for code in struct.unpack("<32b", block[2:]):
                out.extend(_float_to_bf16(scale * code))
        return bytes(out)
    raise ModelError(f"unsupported draft GGUF tensor type {kind}")


def _float_to_bf16(value: float) -> bytes:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    # Round to nearest, ties to even, as the loader's BF16 conversion does.
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return struct.pack("<H", (rounded >> 16) & 0xFFFF)


def _write_safetensors(path: Path, tensors) -> None:
    """tensors: (name, shape, bf16 bytes), in the order the caller wants."""
    header = {}
    data = bytearray()
    for name, shape, raw in tensors:
        begin = len(data)
        data.extend(raw)
        header[name] = {
            "dtype": "BF16",
            "shape": list(shape),
            "data_offsets": [begin, len(data)],
        }
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(data))


def _split_parts(sources):
    """The metadata of a split draft's parts, in split order, and each
    part's data start: a single file is one part."""
    headers = [gguf.Metadata(source, tensors=True) for source in sources]
    if len(headers) == 1:
        count = headers[0].values.get("split.count", 1)
        if count != 1:
            raise ModelError("a split draft GGUF is missing its other parts")
        return [(headers[0], sources[0])]
    count = headers[0].values.get("split.count")
    if type(count) is not int or count != len(headers):
        raise ModelError("the draft GGUF's split files do not match split.count")
    parts = []
    for number, (header, source) in enumerate(zip(headers, sources, strict=True)):
        if header.values.get("split.no") != number:
            raise ModelError(f"draft GGUF split file {number + 1} is out of order")
        parts.append((header, source))
    total = headers[0].values.get("split.tensors.count")
    tensors = {name for header, _ in parts for name in header.tensors}
    if total != len(tensors):
        raise ModelError("the draft GGUF's split files hold another number of tensors")
    return parts


def convert(source, directory: Path) -> dict[str, Path]:
    """Convert the GGUF at source, or a split draft's ordered parts, into
    directory, by the release's relative names: model.safetensors, and
    mask_embedding.safetensors when the file holds that tensor. The returned
    files are written once, atomically enough for the installer's entry
    staging (assembly._reuse_or_write)."""
    sources = [source] if isinstance(source, (str, Path)) else list(source)
    parts = _split_parts(sources)
    model = []
    mask = []
    seen = set()
    streams = []
    try:
        for header, path in parts:
            alignment = header.values.get("general.alignment", GGUF_ALIGNMENT)
            if type(alignment) is not int or alignment <= 0:
                raise ModelError("invalid GGUF alignment in the draft")
            streams.append(
                (
                    header,
                    path.open("rb"),
                    header.consumed + (-header.consumed % alignment),
                )
            )
        for header, stream, data_start in streams:
            for name, kind_id in header.tensors.items():
                kind = gguf.TENSOR_TYPES.get(kind_id)
                if kind not in SUPPORTED:
                    raise ModelError(
                        f"draft GGUF tensor {name} is {kind or f'type {kind_id}'}; "
                        "a draft GGUF must be BF16, F16, F32 or Q8_0"
                    )
                extents = header.extents[name]
                shape = tuple(reversed(extents))
                elements = 1
                for extent in shape:
                    elements *= extent
                block_elements, block_bytes = SUPPORTED[kind]
                if elements % block_elements:
                    raise ModelError(
                        f"draft GGUF tensor {name} is not whole {kind} blocks"
                    )
                size = elements // block_elements * block_bytes
                offset = data_start + header.offsets[name]
                stream.seek(offset)
                raw = stream.read(size)
                if len(raw) != size:
                    raise ModelError(f"truncated draft GGUF tensor {name}")
                mapped = tensor_name(name)
                if mapped in seen:
                    raise ModelError(f"duplicate draft tensor {mapped}")
                seen.add(mapped)
                converted = _bf16(raw, kind)
                if mapped == MASK_EMBEDDING:
                    mask.append((mapped, shape, converted))
                else:
                    model.append((mapped, shape, converted))
    finally:
        for _, stream, _ in streams:
            stream.close()
    if not model:
        raise ModelError("the draft GGUF holds no tensors")
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "model.safetensors"
    _write_safetensors(model_path, sorted(model))
    files = {"model.safetensors": model_path}
    if mask:
        mask_path = directory / "mask_embedding.safetensors"
        _write_safetensors(mask_path, mask)
        files["mask_embedding.safetensors"] = mask_path
    return files
