"""The DFlash2 GGUF draft converter: names, types and the safetensors it
writes for the engine's draft loader."""

import json
import struct
import tempfile
import unittest
from pathlib import Path

from dev.tests import fixture_files
from dev.tests.installer_fixtures import GGML
from install import draftgguf
from install.models import ModelError


class DraftGgufTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def gguf(self, name, tensors):
        return fixture_files.write_gguf(
            self.root / name, {"general.alignment": 32}, tensors
        )

    def test_release_names(self):
        for source, expected in {
            "draft.fc.weight": "fc.weight",
            "draft.hidden_norm.weight": "hidden_norm.weight",
            "draft.layers.3.self_attn.q_proj.weight": "layers.3.self_attn.q_proj.weight",
            "draft.blk.3.mlp.down_proj.weight": "layers.3.mlp.down_proj.weight",
            "model.layers.3.input_layernorm.weight": "layers.3.input_layernorm.weight",
            "model.norm.weight": "norm.weight",
            "model.fc.weight": "fc.weight",
            "draft.candidate_selector.hidden_projection": "candidate_selector.hidden_projection.weight",
            "draft.candidate_selector.hidden_projection.weight": "candidate_selector.hidden_projection.weight",
            "draft.candidate_selector.predecessor_codebook": "candidate_selector.predecessor_codebook",
            "draft.mask_embedding": "mask_embedding",
        }.items():
            with self.subTest(source=source):
                self.assertEqual(draftgguf.tensor_name(source), expected)

    def test_bf16_conversion_rounds_like_the_loader(self):
        self.assertEqual(draftgguf._bf16(struct.pack("<e", 1.0), "F16"), b"\x80\x3f")
        self.assertEqual(draftgguf._bf16(struct.pack("<f", 1.0), "F32"), b"\x80\x3f")
        raw = b"\x00\x3f\x00\x40"
        self.assertEqual(draftgguf._bf16(raw, "BF16"), raw)

    def test_convert_writes_the_release_files(self):
        source = self.gguf(
            "draft.gguf",
            [
                (
                    "draft.fc.weight",
                    [2, 2],
                    GGML["BF16"],
                    struct.pack("<4H", 0, 1, 2, 3),
                ),
                (
                    "draft.mask_embedding",
                    [2],
                    GGML["F16"],
                    struct.pack("<2e", 1.0, -1.0),
                ),
            ],
        )
        files = draftgguf.convert(source, self.root / "out")
        self.assertEqual(
            set(files), {"model.safetensors", "mask_embedding.safetensors"}
        )
        weights = (self.root / "out" / "model.safetensors").read_bytes()
        header_bytes = struct.unpack("<Q", weights[:8])[0]
        header = json.loads(weights[8 : 8 + header_bytes])
        self.assertEqual(header["fc.weight"]["shape"], [2, 2])
        self.assertEqual(header["fc.weight"]["dtype"], "BF16")

    def test_a_q8_0_draft_is_dequantized_to_bf16(self):
        block = struct.pack("<e", 1.0) + struct.pack("<32b", *([1] * 32))
        source = self.gguf(
            "draft.gguf",
            [("draft.fc.weight", [32], GGML["Q8_0"], block)],
        )
        files = draftgguf.convert(source, self.root / "out")
        raw = files["model.safetensors"].read_bytes()
        header_bytes = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8 : 8 + header_bytes])
        self.assertEqual(header["fc.weight"]["shape"], [32])
        self.assertEqual(header["fc.weight"]["dtype"], "BF16")

    def test_a_split_draft_is_converted_in_order(self):
        one = self.gguf(
            "draft-00001-of-00002.gguf",
            [("draft.fc.weight", [2], GGML["BF16"], struct.pack("<2H", 1, 2))],
        )
        # The second part's metadata is written separately, as a split
        # release does: each part carries split.count, split.no and the total
        # tensor count.
        two = fixture_files.write_gguf(
            self.root / "draft-00002-of-00002.gguf",
            {
                "general.alignment": 32,
                "split.count": 2,
                "split.no": 1,
                "split.tensors.count": 2,
            },
            [
                (
                    "draft.norm.weight",
                    [2],
                    GGML["BF16"],
                    struct.pack("<2H", 3, 4),
                )
            ],
        )
        # Rewrite the first part with the split keys too.
        one = fixture_files.write_gguf(
            self.root / "draft-00001-of-00002.gguf",
            {
                "general.alignment": 32,
                "split.count": 2,
                "split.no": 0,
                "split.tensors.count": 2,
            },
            [
                (
                    "draft.fc.weight",
                    [2],
                    GGML["BF16"],
                    struct.pack("<2H", 1, 2),
                )
            ],
        )
        files = draftgguf.convert([one, two], self.root / "out")
        raw = files["model.safetensors"].read_bytes()
        header_bytes = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8 : 8 + header_bytes])
        self.assertEqual(set(header), {"fc.weight", "norm.weight"})

    # Scales whose every 6-bit value unpacks to 1 and 0: the low four
    # sub-blocks' scale in the first four bytes' low bits, the high four's in
    # the last four's low nibble, and every min zero.
    UNIT_SCALES = bytes([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])

    def test_a_q4_k_draft_is_dequantized_in_nibble_order(self):
        # d = 1, dmin = 0, every scale 1 and min 0: the value is the code, so
        # the 256 outputs are the codes in ggml's order - sub-block s takes
        # chunk s/2's low nibbles when s is even and its high ones when odd.
        qs = bytes((value >> 4) | ((value & 0xF) << 4) for value in range(128))
        block = struct.pack("<ee", 1.0, 0.0) + self.UNIT_SCALES + qs
        source = self.gguf(
            "draft.gguf", [("draft.fc.weight", [256], GGML["Q4_K"], block)]
        )
        files = draftgguf.convert(source, self.root / "out")
        raw = files["model.safetensors"].read_bytes()
        header_bytes = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8 : 8 + header_bytes])
        self.assertEqual(header["fc.weight"]["dtype"], "BF16")
        data = raw[8 + header_bytes :]
        values = struct.unpack(f"<{len(data) // 2}H", data)
        for sub in range(8):
            for code_index in range(32):
                index = sub * 32 + code_index
                byte = (sub // 2) * 32 + code_index
                want = (qs[byte] >> (4 * (sub % 2))) & 0xF
                got = struct.unpack("<f", struct.pack("<I", values[index] << 16))[0]
                self.assertEqual(got, want, f"sub {sub} element {code_index}")

    def test_a_q5_k_draft_adds_the_fifth_bit(self):
        # d = 1, dmin = 0, every scale 1 and min 0: the value is the code, so
        # the outputs are the four-bit codes with qh's bit as the fifth.
        qs = bytes((value >> 4) | ((value & 0xF) << 4) for value in range(128))
        qh = bytes(((index // 8) + index) % 256 for index in range(32))
        block = struct.pack("<ee", 1.0, 0.0) + self.UNIT_SCALES + qh + qs
        source = self.gguf(
            "draft.gguf", [("draft.fc.weight", [256], GGML["Q5_K"], block)]
        )
        files = draftgguf.convert(source, self.root / "out")
        raw = files["model.safetensors"].read_bytes()
        header_bytes = struct.unpack("<Q", raw[:8])[0]
        data = raw[8 + header_bytes :]
        values = struct.unpack(f"<{len(data) // 2}H", data)
        for sub in range(8):
            for code_index in range(32):
                index = sub * 32 + code_index
                byte = (sub // 2) * 32 + code_index
                nib = (qs[byte] >> (4 * (sub % 2))) & 0xF
                h = (qh[code_index] >> sub) & 1
                got = struct.unpack("<f", struct.pack("<I", values[index] << 16))[0]
                self.assertEqual(got, nib + (h << 4), f"sub {sub} element {code_index}")

    def test_a_q6_k_draft_scales_by_the_sub_block(self):
        # d = 1, every scale 1: the value is the six-bit code past 32. The
        # codes are laid out so each weight's parts are hand-placed.
        ql = bytes(range(128))
        qh = bytes(range(64))
        scales = bytes([1] * 16)
        block = ql + qh + scales + struct.pack("<e", 1.0)
        source = self.gguf(
            "draft.gguf", [("draft.fc.weight", [256], GGML["Q6_K"], block)]
        )
        files = draftgguf.convert(source, self.root / "out")
        raw = files["model.safetensors"].read_bytes()
        header_bytes = struct.unpack("<Q", raw[:8])[0]
        data = raw[8 + header_bytes :]
        values = struct.unpack(f"<{len(data) // 2}H", data)
        for j in range(256):
            c, rem = j // 128, j % 128
            nib = (ql[c * 64 + rem % 64] >> (rem // 64 * 4)) & 0xF
            high = (qh[c * 32 + rem % 32] >> (rem // 32 * 2)) & 3
            want = (nib | (high << 4)) - 32
            got = struct.unpack("<f", struct.pack("<I", values[j] << 16))[0]
            self.assertEqual(got, want, f"weight {j}")

    def test_k_quants_match_a_numpy_reference(self):
        # The reference is ggml's dequantization written with numpy, the shape
        # the gguf package's quants.py implements: vectorized, so it shares no
        # code with this module's byte loops.
        numpy = __import__("numpy")
        QK_K = 256

        def scales_of(block):
            d, m, md = (
                block[0:4].view(numpy.uint8),
                block[4:8].view(numpy.uint8),
                block[8:12].view(numpy.uint8),
            )
            sc = numpy.concatenate([d & 0x3F, (md & 0x0F) | ((d >> 2) & 0x30)])
            mn = numpy.concatenate([m & 0x3F, (md >> 4) | ((m >> 2) & 0x30)])
            return sc.astype(numpy.float64), mn.astype(numpy.float64)

        # Each reference computes in double, in the order the converter does
        # (the sub-block's scale first, then the code, then the minimum), so
        # the two agree bit for bit once both are rounded to bf16.
        def q4_k(block):
            d = block[0:2].view(numpy.float16).astype(numpy.float64)[0]
            dmin = block[2:4].view(numpy.float16).astype(numpy.float64)[0]
            sc, mn = scales_of(block[4:16])
            qs = numpy.frombuffer(block[16:], dtype=numpy.uint8)
            nibbles = (qs.reshape(4, 1, 32) >> numpy.array([0, 4]).reshape(2, 1)) & 0xF
            q = nibbles.reshape(8, 32).astype(numpy.float64)
            dsc, dmn = d * sc, dmin * mn
            return (dsc.reshape(8, 1) * q - dmn.reshape(8, 1)).reshape(QK_K)

        def q5_k(block):
            d = block[0:2].view(numpy.float16).astype(numpy.float64)[0]
            dmin = block[2:4].view(numpy.float16).astype(numpy.float64)[0]
            sc, mn = scales_of(block[4:16])
            qh = numpy.frombuffer(block[16:48], dtype=numpy.uint8)
            qs = numpy.frombuffer(block[48:], dtype=numpy.uint8)
            nibbles = (qs.reshape(4, 1, 32) >> numpy.array([0, 4]).reshape(2, 1)) & 0xF
            bits = (qh.reshape(1, 32) >> numpy.arange(8).reshape(8, 1)) & 1
            q = (nibbles.reshape(8, 32) | (bits << 4)).astype(numpy.float64)
            dsc, dmn = d * sc, dmin * mn
            return (dsc.reshape(8, 1) * q - dmn.reshape(8, 1)).reshape(QK_K)

        def q6_k(block):
            d = block[208:210].view(numpy.float16).astype(numpy.float64)[0]
            scales = numpy.frombuffer(block[192:208], dtype=numpy.int8).astype(
                numpy.float64
            )
            ql = numpy.frombuffer(block[0:128], dtype=numpy.uint8)
            qh = numpy.frombuffer(block[128:192], dtype=numpy.uint8)
            nibbles = (ql.reshape(2, 1, 64) >> numpy.array([0, 4]).reshape(2, 1)) & 0xF
            pairs = (
                qh.reshape(2, 1, 32) >> numpy.array([0, 2, 4, 6]).reshape(4, 1)
            ) & 3
            q = (nibbles.reshape(8, 32) | (pairs.reshape(8, 32) << 4)).astype(
                numpy.int64
            ) - 32
            return (d * scales.reshape(16, 1) * q.reshape(16, 16)).reshape(QK_K)

        rng = numpy.random.default_rng(7)
        for kind, dequantize, block_bytes in (
            ("Q4_K", q4_k, 144),
            ("Q5_K", q5_k, 176),
            ("Q6_K", q6_k, 210),
        ):
            with self.subTest(kind=kind):
                blocks = numpy.frombuffer(
                    bytes(
                        rng.integers(
                            0, 256, block_bytes * 3, dtype=numpy.uint8
                        ).tobytes()
                    ),
                    dtype=numpy.uint8,
                )
                want = numpy.concatenate(
                    [
                        dequantize(blocks[at : at + block_bytes])
                        for at in (0, block_bytes, 2 * block_bytes)
                    ]
                ).reshape(-1)
                source = self.gguf(
                    "draft.gguf",
                    [("draft.fc.weight", [3 * QK_K], GGML[kind], blocks.tobytes())],
                )
                files = draftgguf.convert(source, self.root / f"out-{kind}")
                raw = files["model.safetensors"].read_bytes()
                header_bytes = struct.unpack("<Q", raw[:8])[0]
                data = numpy.frombuffer(raw[8 + header_bytes :], dtype=numpy.uint16)
                # Both implementations compute in double and round to bf16,
                # so they agree bit for bit once the reference is rounded.
                bits = want.astype(numpy.float32).view(numpy.uint32)
                rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
                self.assertTrue(
                    numpy.array_equal(data, (rounded & 0xFFFF).astype(numpy.uint16)),
                    f"{kind}: the converter disagrees with the reference",
                )

    def test_a_quantized_draft_is_refused_naming_its_type(self):
        source = self.gguf(
            "draft.gguf",
            [("draft.fc.weight", [256], GGML["Q2_K"], b"\0" * 84)],
        )
        with self.assertRaisesRegex(ModelError, "Q2_K"):
            draftgguf.convert(source, self.root / "out")
