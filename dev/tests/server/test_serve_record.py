import json
import os
import tempfile
import unittest
from pathlib import Path

from server import serve_record


class ServeRecordTests(unittest.TestCase):
    def test_writes_each_phase_and_the_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "serve-8000.json"
            serve_record.starting(path)
            self.assertEqual(
                json.loads(path.read_text()),
                {"pid": os.getpid(), "phase": "starting"},
            )
            serve_record.entered(path, "loading")
            self.assertEqual(
                json.loads(path.read_text()),
                {"pid": os.getpid(), "phase": "loading"},
            )
            serve_record.ready(path)
            self.assertEqual(
                json.loads(path.read_text()),
                {"pid": os.getpid(), "phase": "ready"},
            )
            failure = serve_record.failure(serve_record.PORT, "port 8000 is in use")
            serve_record.failed(path, failure)
            self.assertEqual(
                json.loads(path.read_text()),
                {"pid": os.getpid(), "phase": "failed", "failure": failure},
            )

    def test_a_record_that_cannot_be_written_raises(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "missing" / "serve-8000.json"
            with self.assertRaisesRegex(
                serve_record.NotRecorded, "cannot write the serve record"
            ):
                serve_record.starting(path)
            # Best effort writes keep the start going.
            serve_record.entered(path, "loading")


if __name__ == "__main__":
    unittest.main()
