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

    def test_a_quantized_draft_is_refused_naming_its_type(self):
        source = self.gguf(
            "draft.gguf",
            [("draft.fc.weight", [256], GGML["Q4_K"], b"\0" * 144)],
        )
        with self.assertRaisesRegex(ModelError, "Q4_K"):
            draftgguf.convert(source, self.root / "out")
