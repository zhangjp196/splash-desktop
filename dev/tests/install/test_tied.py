import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from install import tied


class TiedPreparationTest(unittest.TestCase):
    def test_disk_plan_covers_generated_head_and_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.safetensors"
            entries = {
                "language_model.model.embed_tokens.weight": ("U32", [1, 64], b"w" * 4),
                "language_model.model.embed_tokens.scales": ("BF16", [1, 1], b"s" * 2),
                "language_model.model.embed_tokens.biases": ("BF16", [1, 1], b"b" * 2),
            }
            header, bodies, offset = {}, [], 0
            for name, (dtype, shape, payload) in entries.items():
                header[name] = {
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [offset, offset + len(payload)],
                }
                bodies.append(payload)
                offset += len(payload)
            encoded = json.dumps(header, separators=(",", ":")).encode()
            encoded += b" " * ((8 - len(encoded) % 8) % 8)
            source.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(bodies))
            config = root / "config.json"
            config.write_text(json.dumps({"tie_word_embeddings": True}))
            models_root = root / "models"
            models_root.mkdir()

            with mock.patch.object(tied.hub, "_report_plan") as report, mock.patch.object(
                tied.hub, "plan_space", wraps=tied.hub.plan_space
            ) as plan_space:
                files = tied.prepare(config, {source.name: source}, models_root)

            self.assertEqual(report.call_count, 1)
            planned = plan_space.call_args.args[0][0]
            actual = sum(path.stat().st_size for path in set(files.values()))
            self.assertEqual(planned[1], actual)
            self.assertEqual(planned[3], actual)
            self.assertTrue(files["target/lm_head.safetensors"].is_file())


if __name__ == "__main__":
    unittest.main()
