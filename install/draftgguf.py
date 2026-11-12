"""Convert a DFlash2 draft published as GGUF into the BF16 safetensors the
engine loads at every start.

The runtime's draft loader reads a checkpoint released as `config.json` and
BF16 safetensors (runtime/model/DraftCheckpoint.cpp). A GGUF conversion of the
same checkpoint is data too: this module reads its BF16, F16 and F32 tensors,
renames them to the release's names, and writes one `model.safetensors` (and
`mask_embedding.safetensors` when the checkpoint ships that tensor). The
installation links the converted files under `draft/`; the source repository
and commit stay in `model.json` as `sources.draft`.

The runtime's draft loader reads BF16 alone, so a quantized draft is
dequantized here, at install time, as ggml's own dequantizers unpack it: the
Q8_0 blocks and the K-quant super-blocks (Q4_K, Q5_K, Q6_K). A draft
published in another quantization is refused by name; that is a later
provider, not an installation accident.
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
# codes; the K-quants are super-blocks of QK_K = 256 weights, in the layouts
# ggml-common.h declares (Q4_K 144, Q5_K 176, Q6_K 210 bytes).
SUPPORTED = {
    "BF16": (1, 2),
    "F16": (1, 2),
    "F32": (1, 4),
    "Q8_0": (32, 34),
    "Q4_K": (256, 144),
    "Q5_K": (256, 176),
    "Q6_K": (256, 210),
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
    if kind == "Q4_K":
        return _q4_k(raw)
    if kind == "Q5_K":
        return _q5_k(raw)
    if kind == "Q6_K":
        return _q6_k(raw)
    raise ModelError(f"unsupported draft GGUF tensor type {kind}")


def _scale_min_k4(scales: bytes) -> tuple[list[int], list[int]]:
    """The eight 6-bit scales and mins of a Q4_K or Q5_K block's 12 bytes, as
    ggml's get_scale_min_k4 unpacks them: the low four sub-blocks' values in
    the low six bits of the first eight bytes, the high four's low nibble in
    the last four bytes and high bits in the first eight's top two."""
    d, m, md = scales[0:4], scales[4:8], scales[8:12]
    sc = [d[i] & 0x3F for i in range(4)]
    sc += [(md[i] & 0x0F) | ((d[i] >> 2) & 0x30) for i in range(4)]
    mn = [m[i] & 0x3F for i in range(4)]
    mn += [(md[i] >> 4) | ((m[i] >> 2) & 0x30) for i in range(4)]
    return sc, mn


def _q4_k(raw: bytes) -> bytes:
    """Q4_K's 144-byte super-blocks of 256 weights as bf16: two fp16 scales,
    twelve bytes of packed 6-bit scales and mins, and 256 four-bit codes that
    eight 32-weight sub-blocks share two apiece."""
    out = bytearray()
    for at in range(0, len(raw), 144):
        block = raw[at : at + 144]
        if len(block) != 144:
            raise ModelError("truncated Q4_K draft block")
        (d,) = struct.unpack_from("<e", block, 0)
        (dmin,) = struct.unpack_from("<e", block, 2)
        sc, mn = _scale_min_k4(block[4:16])
        qs = block[16:]
        for sub in range(8):
            dsc = d * sc[sub]
            dmn = dmin * mn[sub]
            shift = 4 * (sub % 2)
            for code in range(32):
                q = (qs[(sub // 2) * 32 + code] >> shift) & 0xF
                out.extend(_float_to_bf16(dsc * q - dmn))
    return bytes(out)


def _q5_k(raw: bytes) -> bytes:
    """Q5_K's 176-byte super-blocks of 256 weights as bf16: Q4_K's layout with
    one more bit per weight, in 32 bytes between the scales and the codes."""
    out = bytearray()
    for at in range(0, len(raw), 176):
        block = raw[at : at + 176]
        if len(block) != 176:
            raise ModelError("truncated Q5_K draft block")
        (d,) = struct.unpack_from("<e", block, 0)
        (dmin,) = struct.unpack_from("<e", block, 2)
        sc, mn = _scale_min_k4(block[4:16])
        qh = block[16:48]
        qs = block[48:]
        for sub in range(8):
            dsc = d * sc[sub]
            dmn = dmin * mn[sub]
            shift = 4 * (sub % 2)
            for code in range(32):
                q = (qs[(sub // 2) * 32 + code] >> shift) & 0xF
                # The fifth bit of element `code` of every sub-block lives in
                # byte `code`, one bit per sub-block (ggml's
                # dequantize_row_q5_K).
                h = (qh[code] >> sub) & 1
                out.extend(_float_to_bf16(dsc * (q + (h << 4)) - dmn))
    return bytes(out)


def _q6_k(raw: bytes) -> bytes:
    """Q6_K's 210-byte super-blocks of 256 weights as bf16: 128 bytes of low
    nibbles, 64 of high two-bit pairs, sixteen int8 scales and one fp16 scale;
    the six-bit code is the pair past 32."""
    out = bytearray()
    for at in range(0, len(raw), 210):
        block = raw[at : at + 210]
        if len(block) != 210:
            raise ModelError("truncated Q6_K draft block")
        ql = block[0:128]
        qh = block[128:192]
        scales = struct.unpack("<16b", block[192:208])
        (d,) = struct.unpack_from("<e", block, 208)
        for j in range(256):
            c, rem = j // 128, j % 128
            nib = (ql[c * 64 + rem % 64] >> (rem // 64 * 4)) & 0xF
            high = (qh[c * 32 + rem % 32] >> (rem // 32 * 2)) & 3
            q = (nib | (high << 4)) - 32
            out.extend(_float_to_bf16(d * scales[j // 16] * q))
    return bytes(out)


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
                        f"draft GGUF tensor {name} is {kind or f'type {kind_id}'}; a draft "
                        "GGUF must be BF16, F16, F32, Q8_0, Q4_K, Q5_K or Q6_K"
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
