#!/usr/bin/env python3
"""Independent torch CPU reference for Inkling2-Mini's vision projector.

This is the executable specification the native Metal graph is graded against
(``dev/tests/engine/inkling2_vision_parity_test.mm``). It reads the projector's
422 tensors straight out of the mmproj GGUF, binds them in torch on the CPU,
and re-derives the tower from the same pixels the GPU consumed: the 2x2
patchify of each 40x40 tile against the GGUF's CHW convolution weight, the CLS
row plus the bilinear (align_corners=False) resample of the learned 32x32
position table, the tower-wide pre-LN, 23 pre-norm blocks with separate Q/K/V,
image-wide non-causal attention, width-then-height interleaved axial RoPE and
per-channel layer scales, then the 2x2 merge and mm.1/mm.2.

The GGUF stores every F32 tensor only at values that are exactly a BF16 (the
loader refuses any other), so rounding each tensor once to the BF16 the engine
binds is the same arithmetic on both sides. Activations follow the engine's
own layout: the residual stream stays float32, every scratch activation is
rounded to BF16 where the kernels round it.

    python dev/tools/inkling2_vision_reference.py SOURCE OUTPUT_DIR
    add --source-format safetensors when SOURCE is an Inkling checkpoint directory

    --grid 1x1,1x2,2x2   the grids the parity test wrote
    --variant correct    the tower as the engine runs it, BF16 activations
    --variant fp32       the same tower with float32 activations throughout,
                         the ceiling the engine's BF16 activations round below
    --variant no-rope        a wrong tower that skips axial RoPE
    --variant cls-rope       a wrong tower that rotates CLS too, one-based
    --variant no-pre-ln      a wrong tower that drops the tower-wide pre-LN
    --variant merger-dim-major  a wrong merger packing that is dim-major
                             rather than patch-major, the layout confusion
                             CHW convolution weights invite

The wrong variants exist so that the parity test has discrimination: a test
whose reference agreed with the engine under a wrong assumption would report a
gap for the right tower and none for the wrong one.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from install import gguf  # noqa: E402

# ggml type -> (label, numpy element type); the projector's tensors are F32 or
# BF16, and its F32 values are exactly BF16 ones.
GGUF_TYPES = {0: ("F32", np.float32), 30: ("BF16", np.uint16)}

HIDDEN = 1024
INTERMEDIATE = 4096
LAYERS = 23
HEADS = 16
HEAD_DIM = 64
PATCH = 20
EXTENT = 40
POSITION_SIDE = 32
OUTPUT = 2304
NORM_EPS = 1e-5
ROPE_THETA = 10000.0
TENSOR_COUNT = 422

VARIANTS = ("correct", "fp32", "no-rope", "cls-rope", "no-pre-ln", "merger-dim-major")


def bf16_to_f32(words):
    """The float32 each BF16 bit pattern names."""
    return (np.asarray(words, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def round_bf16(values):
    """values rounded to the nearest BF16, ties to even, and back: the rounding
    a Metal ``bfloat`` conversion applies. Accepts and returns a numpy array or
    a torch float32 tensor, whichever it was given."""
    if isinstance(values, np.ndarray):
        bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
        rounded = (bits.astype(np.uint64) + 0x7FFF + ((bits >> 16) & 1)) >> 16
        return bf16_to_f32(rounded.astype(np.uint16))
    torch = _torch()
    # The logical shifts go through int64, where a right shift cannot sign
    # extend, and the widened 16 bits come back as float32 by concatenation.
    bits = (
        values.to(torch.float32).contiguous().view(torch.int32).to(torch.int64)
        & 0xFFFFFFFF
    )
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).to(torch.int32)
    return (rounded << 16).view(torch.float32)


def read_projector(path):
    """Every tensor of the projector as a torch float32 array in row-major
    order of the GGUF's reversed extents (the layout every GEMM below uses),
    each value the BF16 the engine's weight image holds."""
    import torch

    header = gguf.Metadata(path, tensors=True)
    if header.values.get("general.architecture") != "clip" or (
        header.values.get("clip.vision.projector_type") != "inkling2v"
    ):
        raise ValueError(f"{path} is not a clip inkling2v projector")
    if len(header.tensors) != TENSOR_COUNT:
        raise ValueError(
            f"the projector holds {len(header.tensors)} tensors, not {TENSOR_COUNT}"
        )
    # The data section starts at the first multiple of the alignment past the
    # header.
    alignment = header.values.get("general.alignment", 32)
    data = (header.consumed + alignment - 1) // alignment * alignment
    tensors = {}
    for name, kind in header.tensors.items():
        if kind not in GGUF_TYPES:
            raise ValueError(f"unsupported projector tensor type {kind}: {name}")
        label, dtype = GGUF_TYPES[kind]
        dims = header.extents[name]
        raw = np.fromfile(
            path,
            dtype=dtype,
            count=int(np.prod(dims)),
            offset=data + header.offsets[name],
        )
        if label == "BF16":
            values = bf16_to_f32(raw)
        elif np.any(raw.view(np.uint32) & 0xFFFF):
            raise ValueError(f"F32 projector tensor is not exactly BF16: {name}")
        else:
            values = raw
        tensors[name] = torch.from_numpy(
            np.ascontiguousarray(round_bf16(values.reshape(dims[::-1])))
        )
    return tensors


def safetensors_vision_names():
    """Source checkpoint names keyed by the canonical mmproj names."""
    names = {}

    def add(image_name, source_name):
        if image_name in names:
            raise ValueError(f"duplicate Inkling vision image tensor: {image_name}")
        names[image_name] = source_name

    def affine(image_stem, source_stem, bias=True):
        add(image_stem + ".weight", source_stem + ".weight")
        if bias:
            add(image_stem + ".bias", source_stem + ".bias")

    add("v.patch_embd.weight", "vision_tower.backbone.embeddings.patch_embedding.weight")
    add("v.class_embd", "vision_tower.backbone.embeddings.class_embedding")
    add("v.class_position_embd.weight", "vision_tower.backbone.embeddings.cls_position_embedding")
    add("v.position_embd.weight", "vision_tower.backbone.embeddings.position_embedding")
    affine("v.pre_ln", "vision_tower.backbone.ln_pre")
    for layer in range(LAYERS):
        image = f"v.blk.{layer}."
        source = f"vision_tower.backbone.layers.{layer}."
        affine(image + "ln1", source + "ln_1")
        affine(image + "attn_q", source + "self_attn.q_proj")
        affine(image + "attn_k", source + "self_attn.k_proj")
        affine(image + "attn_v", source + "self_attn.v_proj")
        affine(image + "attn_out", source + "self_attn.out_proj")
        affine(image + "ln2", source + "ln_2")
        affine(image + "ffn_up", source + "mlp.c_fc")
        affine(image + "ffn_down", source + "mlp.c_proj")
        add(image + "ls1.weight", source + "ls_1.gamma")
        add(image + "ls2.weight", source + "ls_2.gamma")
    affine("mm.1", "vision_tower.proj_in", bias=False)
    affine("mm.2", "vision_tower.proj_out", bias=False)
    if len(names) != TENSOR_COUNT:
        raise ValueError(f"mapped {len(names)} Safetensors vision tensors, expected {TENSOR_COUNT}")
    return names


def read_safetensors_vision(root):
    """Read the exact BF16 tower inventory from an Inkling checkpoint."""
    import torch
    from safetensors import safe_open

    root = Path(root)
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    names = safetensors_vision_names()
    visual_names = {name for name in weight_map if name.startswith("vision_tower.")}
    if len(visual_names) != TENSOR_COUNT or set(names.values()) != visual_names:
        raise ValueError("Safetensors checkpoint vision inventory does not match the 422-tensor Inkling tower")
    by_shard = {}
    for image_name, source_name in names.items():
        by_shard.setdefault(weight_map[source_name], []).append((image_name, source_name))
    tensors = {}
    for shard, entries in by_shard.items():
        with safe_open(root / shard, framework="pt", device="cpu") as source:
            for image_name, source_name in entries:
                value = source.get_tensor(source_name)
                if value.dtype != torch.bfloat16:
                    raise ValueError(f"{source_name} is {value.dtype}, expected BF16")
                if image_name == "v.patch_embd.weight":
                    # Transformers flattens each patch in HWC order; the
                    # native canonical image uses GGUF's CHW weight order.
                    value = value.reshape(HIDDEN, PATCH, PATCH, 3).permute(0, 3, 1, 2).reshape(HIDDEN, -1)
                tensors[image_name] = round_bf16(value.contiguous())
    return tensors


class Tower:
    """One projector bound in torch, encoding an image on the CPU."""

    def __init__(self, tensors, variant):
        import torch

        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant}")
        self.torch = torch
        self.tensors = tensors
        self.variant = variant
        self.fp32 = variant == "fp32"
        self.precision = torch.float32 if self.fp32 else torch.float64

    def _out(self, value):
        """value as one of the engine's BF16 stores writes it."""
        return value.to(self.torch.float32) if self.fp32 else round_bf16(value)

    def _weight(self, name):
        return self.tensors[name].to(self.precision)

    def _linear(self, name, source, bias=True):
        """round(source @ weight.T + bias) for one projection, its bias a GGUF
        tensor of the same stem or zero, as the merger's uncased rows."""
        weight = self._weight(name + ".weight")
        value = source.to(self.precision) @ weight.T
        if bias:
            value = value + self._weight(name + ".bias")
        return self._out(value)

    def _layer_norm(self, x, stem):
        """Row-wise LayerNorm in float64 statistics over the BF16 weights, the
        result rounded as the norm kernels store it."""
        centre = x.to(self.torch.float64)
        mean = centre.mean(dim=-1, keepdim=True)
        centred = centre - mean
        variance = (centred * centred).mean(dim=-1, keepdim=True)
        normalized = centred * self.torch.rsqrt(variance + NORM_EPS)
        if self.fp32:
            value = normalized.to(self.torch.float32) * self._weight(stem + ".weight")
            return value + self._weight(stem + ".bias")
        value = normalized * self._weight(stem + ".weight") + self._weight(
            stem + ".bias"
        )
        return self._out(value)

    def _positions(self, grid_height, grid_width):
        """The learned 32x32 position table resampled onto the image's patch
        grid, half-pixel, align_corners=False, clamped at the table's edge."""
        torch = self.torch
        rows = torch.arange(grid_height, dtype=torch.float64)
        cols = torch.arange(grid_width, dtype=torch.float64)
        fy = torch.clamp(
            (rows + 0.5) * POSITION_SIDE / grid_height - 0.5, 0.0, POSITION_SIDE - 1.0
        )
        fx = torch.clamp(
            (cols + 0.5) * POSITION_SIDE / grid_width - 0.5, 0.0, POSITION_SIDE - 1.0
        )
        table = self._weight("v.position_embd.weight").reshape(
            POSITION_SIDE, POSITION_SIDE, HIDDEN
        )
        table = table.to(torch.float64)
        y0 = torch.floor(fy).to(torch.int64)
        x0 = torch.floor(fx).to(torch.int64)
        y1 = torch.clamp(y0 + 1, max=POSITION_SIDE - 1)
        x1 = torch.clamp(x0 + 1, max=POSITION_SIDE - 1)
        wy = (fy - y0.to(torch.float64)).reshape(-1, 1, 1)
        wx = (fx - x0.to(torch.float64)).reshape(1, -1, 1)
        upper = table[y0][:, x0] * (1 - wx) + table[y0][:, x1] * wx
        lower = table[y1][:, x0] * (1 - wx) + table[y1][:, x1] * wx
        return self._out(upper * (1 - wy) + lower * wy).reshape(
            grid_height * grid_width, HIDDEN
        )

    @staticmethod
    def _patchify(pixels, tile_height, tile_width):
        """(tiles*40*40*3) tile-major HWC pixels as (4*tiles, 1200) rows, one
        per 20x20 patch in image-wide raster order, each row (channel, row,
        col): the flat vector the GGUF's CHW convolution weight multiplies."""
        patches = pixels.reshape(tile_height, tile_width, 2, PATCH, 2, PATCH, 3)
        # (tile row, its patch row, tile column, its patch column, y, x, c):
        # the image's patch grid in raster order, whatever order the tiles
        # arrive in.
        ordered = patches.permute(0, 2, 1, 4, 3, 5, 6).reshape(-1, PATCH, PATCH, 3)
        return ordered.permute(0, 3, 1, 2).reshape(-1, 3 * PATCH * PATCH)

    def _axial_rope(self, x, grid_height, grid_width, skip_cls):
        """Width-then-height interleaved axial RoPE over (tokens, hidden) rows:
        each head's adjacent dimension pairs rotate, pairs 0..15 by the column
        coordinate and 16..31 by the row, at theta^(-2f/32). A skipped CLS row
        is left unrotated, which is the identity rotation of its (0, 0)."""
        torch = self.torch
        tokens = x.shape[0]
        pairs = HEAD_DIM // 2
        frequency = torch.arange(pairs) % (HEAD_DIM // 4)
        inverse = ROPE_THETA ** (-2.0 * frequency.to(torch.float64) / (HEAD_DIM // 2))
        start = 1 if skip_cls else 0
        index = torch.arange(tokens)
        spatial = torch.clamp(index - start, min=0)
        coordinate = torch.where(
            torch.arange(pairs) < pairs // 2,
            (spatial % grid_width).reshape(-1, 1),
            (spatial // grid_width).reshape(-1, 1),
        )
        angle = (coordinate.to(torch.float64) + 1.0) * inverse.reshape(1, -1)
        cos = torch.cos(angle).reshape(tokens, 1, pairs)
        sin = torch.sin(angle).reshape(tokens, 1, pairs)
        rotated = x.reshape(tokens, HEADS, pairs, 2).to(torch.float64)
        first, second = rotated[..., 0], rotated[..., 1]
        value = torch.stack(
            (first * cos - second * sin, second * cos + first * sin), dim=-1
        )
        value = self._out(value).reshape(tokens, HIDDEN)
        if skip_cls:
            value = torch.where(
                (index == 0).reshape(-1, 1), self._out(x.to(torch.float64)), value
            )
        return value

    def _attention(self, query, key, value):
        """Image-wide, non-causal attention in float64, the context rounded as
        ``vision_attention`` stores it."""
        torch = self.torch
        tokens = query.shape[0]
        scale = 1.0 / np.sqrt(float(HEAD_DIM))

        def heads(x):
            return x.reshape(tokens, HEADS, HEAD_DIM).permute(1, 0, 2).to(torch.float64)

        scores = torch.matmul(heads(query), heads(key).transpose(1, 2)) * scale
        context = torch.matmul(torch.softmax(scores, dim=-1), heads(value))
        return self._out(context.permute(1, 0, 2).reshape(tokens, HIDDEN))

    @staticmethod
    def _gelu(value):
        """The exact (erf) GELU both GEMM epilogues apply, in float64."""
        return 0.5 * value * (1.0 + _torch().erf(value / np.sqrt(2.0)))

    def _residual(self, residual, branch, scale):
        """residual + branch * layer_scale in float32, as
        ``inkling2_vision_scaled_residual`` writes the stream."""
        value = residual.to(self.torch.float64) + branch.to(
            self.torch.float64
        ) * scale.to(self.torch.float64)
        return value.to(self.torch.float32)

    def encode(self, pixels, grid_height, grid_width):
        """The merger's (tiles, 2304) rows for one image, as the engine's
        output buffer holds them."""
        torch = self.torch
        tiles = grid_height * grid_width
        gh, gw = 2 * grid_height, 2 * grid_width
        tokens = 1 + 4 * tiles

        # Stem: patch embedding, then CLS plus its learned position, then each
        # patch's bilinear position, each sum rounded as the stem kernel
        # rounds it.
        patches = self._patchify(pixels, grid_height, grid_width)
        weight = self._weight("v.patch_embd.weight").reshape(HIDDEN, 3 * PATCH * PATCH)
        hidden = torch.zeros(tokens, HIDDEN, dtype=torch.float32)
        hidden[1:] = self._out(patches.to(torch.float64) @ weight.T.to(torch.float64))
        cls = self._weight("v.class_embd").to(torch.float64) + self._weight(
            "v.class_position_embd.weight"
        ).to(torch.float64)
        hidden[0] = self._out(cls)
        hidden[1:] = self._out(
            hidden[1:].to(torch.float64) + self._positions(gh, gw).to(torch.float64)
        )

        # The tower-wide pre-LN, whose BF16 result opens the residual stream.
        residual = hidden
        if self.variant != "no-pre-ln":
            residual = self._layer_norm(residual, "v.pre_ln")

        for layer in range(LAYERS):
            prefix = f"v.blk.{layer}."
            norm = self._layer_norm(residual, prefix + "ln1")
            query = self._linear(prefix + "attn_q", norm)
            key = self._linear(prefix + "attn_k", norm)
            value = self._linear(prefix + "attn_v", norm)
            if self.variant != "no-rope":
                skip_cls = self.variant != "cls-rope"
                query = self._axial_rope(query, gh, gw, skip_cls)
                key = self._axial_rope(key, gh, gw, skip_cls)
            projected = self._linear(
                prefix + "attn_out", self._attention(query, key, value)
            )
            residual = self._residual(
                residual, projected, self.tensors[prefix + "ls1.weight"]
            )

            norm = self._layer_norm(residual, prefix + "ln2")
            up = self._linear(prefix + "ffn_up", norm)
            up = self._out(self._gelu(up.to(torch.float64)))
            down = self._linear(prefix + "ffn_down", up)
            residual = self._residual(
                residual, down, self.tensors[prefix + "ls2.weight"]
            )

        return self._merger(residual, tiles, gh, gw)

    def _merger(self, residual, tiles, gh, gw):
        """Pack each tile's four patches into a 4*1024 row, mm.1 + erf GELU,
        mm.2. mm.1's columns are patch-major: patch * 1024 + dim."""
        torch = self.torch
        spatial = residual[1:].reshape(gh, gw, HIDDEN)
        grid_width = gw // 2
        packed = torch.zeros(tiles, 4 * HIDDEN, dtype=torch.float32)
        dim_major = self.variant == "merger-dim-major"
        for tile in range(tiles):
            tile_y, tile_x = tile // grid_width, tile % grid_width
            for patch in range(4):
                row = tile_y * 2 + patch // 2
                col = tile_x * 2 + patch % 2
                if dim_major:
                    packed[tile, patch::4] = spatial[row, col]
                else:
                    packed[tile, patch * HIDDEN : (patch + 1) * HIDDEN] = spatial[
                        row, col
                    ]
        packed = self._out(packed)
        up = self._linear("mm.1", packed, bias=False)
        up = self._out(self._gelu(up.to(torch.float64)))
        return self._out(self._linear("mm.2", up, bias=False).to(torch.float64))


def compare(actual, expected):
    """Max absolute error and RMS of one grid's rows."""
    torch = _torch()
    difference = actual.to(torch.float64) - expected.to(torch.float64)
    return float(difference.abs().max()), float(
        torch.sqrt((difference * difference).mean())
    )


def _torch():
    import torch

    return torch


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-format", choices=("gguf", "safetensors"), default="gguf")
    parser.add_argument("--grid", default="1x1,1x2,2x2")
    parser.add_argument("--variant", default="correct", choices=VARIANTS)
    parser.add_argument(
        "--write-expected",
        action="store_true",
        help="write expected_<grid>.f32 beside the GPU's output_<grid>.f32",
    )
    parser.add_argument(
        "--max-abs",
        type=float,
        default=None,
        help="fail when a grid's max absolute error exceeds this",
    )
    parser.add_argument(
        "--max-rms",
        type=float,
        default=None,
        help="fail when a grid's RMS error exceeds this",
    )
    parser.add_argument(
        "--discriminate",
        type=float,
        default=None,
        metavar="FLOOR",
        help="after the requested variant, require every wrong variant's worst "
        "max absolute error to exceed FLOOR: the proof that a wrong tower is "
        "told apart from the one the engine runs",
    )
    args = parser.parse_args(argv)

    torch = _torch()
    torch.set_grad_enabled(False)
    tensors = (
        read_projector(args.source)
        if args.source_format == "gguf"
        else read_safetensors_vision(args.source)
    )
    worst_max, worst_rms = examine(args.variant, tensors, args)
    if args.max_abs is not None and worst_max > args.max_abs:
        raise SystemExit(
            f"{args.variant}: worst max_abs_error {worst_max:.7f} exceeds {args.max_abs}"
        )
    if args.max_rms is not None and worst_rms > args.max_rms:
        raise SystemExit(
            f"{args.variant}: worst rms {worst_rms:.7f} exceeds {args.max_rms}"
        )
    if args.discriminate is not None:
        for variant in VARIANTS:
            if variant in (args.variant, "fp32"):
                continue
            maximum, _ = examine(variant, tensors, args, quiet=True)
            verdict = (
                "told apart" if maximum > args.discriminate else "INDISTINGUISHABLE"
            )
            print(f"{variant}: worst max_abs_error={maximum:.7f} {verdict}")
            if maximum <= args.discriminate:
                raise SystemExit(
                    f"{variant}: worst max_abs_error {maximum:.7f} is within "
                    f"{args.discriminate}; the parity test would not tell a wrong "
                    "tower from the engine's"
                )
    return 0


def examine(variant, tensors, args, quiet=False):
    """One variant over every grid: its worst max absolute error and RMS."""
    tower = Tower(tensors, variant)
    worst_max, worst_rms = 0.0, 0.0
    for label in args.grid.split(","):
        height, width = (int(part) for part in label.split("x"))
        pixels = torch_from(args.output / f"pixels_{label}.f32")
        actual = torch_from(args.output / f"output_{label}.f32")
        expected = tower.encode(pixels, height, width).reshape(-1).to(_torch().float32)
        if expected.numel() != actual.numel():
            raise SystemExit(
                f"{label}: GPU wrote {actual.numel()} values, reference {expected.numel()}"
            )
        maximum, rms = compare(actual, expected)
        worst_max, worst_rms = max(worst_max, maximum), max(worst_rms, rms)
        if not quiet:
            if args.write_expected:
                expected.numpy().tofile(args.output / f"expected_{label}.f32")
            print(
                f"grid {label} ({variant}): max_abs_error={maximum:.7f} rms={rms:.7f}"
            )
    if not quiet:
        print(f"{variant}: worst max_abs_error={worst_max:.7f} rms={worst_rms:.7f}")
    return worst_max, worst_rms


def torch_from(path):
    """The float32 values the parity test wrote to path."""
    return _torch().from_numpy(np.fromfile(path, dtype=np.float32).copy())


if __name__ == "__main__":
    sys.exit(main())
