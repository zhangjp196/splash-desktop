import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dev.tests.installer_fixtures import (
    DENSE,
    DRAFT_COMMIT,
    MOE,
    cached_snapshot,
    draft_dir,
    fixture,
    loadable_tensors,
    mlx_target,
    write_gguf,
)
from dev.tests.server_fixtures import keep_stop_signals, server_arguments
from install import hub, launcher
from server import server as api


class LocalServeTests(unittest.TestCase):
    """A local target reaches the server through the installer's prepare, which
    upstream.prepare_local runs: the command a Hub model's installation runs,
    with --model-dir in place of --model."""

    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.enterContext(mock.patch.dict(os.environ))
        os.environ.pop("SPLASH_PORT", None)
        os.environ["HF_HUB_CACHE"] = str(self.root / "cache")
        keep_stop_signals(self)
        self.enterContext(mock.patch.object(launcher, "RUNTIME_DIR", self.root / "run"))
        self.enterContext(
            mock.patch.object(launcher.paths, "MODELS", self.root / "models")
        )
        self.enterContext(mock.patch.object(launcher, "_check_port"))
        self.enterContext(mock.patch.object(launcher, "_version", return_value="test"))

    def gguf(self):
        headers = self.root / "headers"
        headers.mkdir(exist_ok=True)
        values = fixture(native=True)
        return write_gguf(
            self.root / "my model.gguf",
            values,
            loadable_tensors(values, headers).items(),
        )

    def serve(self, target, *options):
        """launcher.main's result, the prepare it ran and the exec it made, with
        the runtime build and the device check stubbed (hosted runners expose a
        virtual GPU) and the installation left real."""
        with (
            mock.patch.object(launcher, "_ensure_runtime") as setup,
            mock.patch.object(launcher, "_run_held", return_value=0) as run,
            mock.patch.object(launcher.os, "execve") as execute,
        ):
            result = launcher.main(["serve", "--model-dir", str(target), *options])
        setup.assert_called_once_with()
        return result, run, execute

    def test_local_gguf_is_prepared_by_splash_python_and_served_without_the_hub(self):
        target = self.gguf()
        draft = draft_dir(self.root / "my draft", MOE)
        result, run, execute = self.serve(
            target, "--draft-model", str(draft), "--max-context", "8K"
        )
        self.assertIsNone(result)
        command = run.call_args.args[0]
        self.assertEqual(
            (command[0], command[1]),
            (str(launcher.paths.PYTHON), str(launcher.ROOT / "install/models.py")),
        )
        parsed = launcher.model_artifacts.parse_args(command[2:])
        self.assertEqual(
            (parsed.command, parsed.model_dir, parsed.draft_model),
            ("prepare", str(target.resolve()), str(draft.resolve())),
        )
        execute.assert_called_once()
        binary, served, environment = execute.call_args.args
        self.assertEqual(binary, str(launcher.paths.PYTHON))
        with mock.patch.dict(os.environ, environment):
            arguments = api.parse_args(server_arguments(served))
        # The API model ID comes from the target's own name.
        self.assertEqual(arguments.model, "my-model")
        self.assertEqual(arguments.max_context, 8192)
        # The server serves the selection link the prepare published, under the
        # models root.
        self.assertTrue(
            str(server_arguments(served)[0]).startswith(
                str(self.root / "models" / ".selections")
            )
        )

    def test_local_gguf_filename_symlink_and_projector_are_preserved(self):
        blob = self.gguf().rename(self.root / "blob-digest")
        target = self.root / "snapshot" / "named-model.gguf"
        target.parent.mkdir()
        target.symlink_to(blob)
        projector = self.root / "my projector.gguf"
        projector.write_bytes(b"GGUF")
        draft = draft_dir(self.root / "draft", MOE)
        # A relative --model-dir is resolved against the working directory.
        previous = os.getcwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)
        result, run, _ = self.serve(
            "snapshot/named-model.gguf",
            "--draft-model",
            str(draft),
            "--mmproj",
            "my projector.gguf",
        )
        self.assertIsNone(result)
        parsed = launcher.model_artifacts.parse_args(run.call_args.args[0][2:])
        # The file's name survives: resolving it would replace its .gguf name.
        self.assertEqual(parsed.model_dir, str(target))
        self.assertEqual(parsed.mmproj, str(projector))

    def test_local_mlx_directory_and_language_only_are_forwarded(self):
        target = mlx_target(self.root / "MLX model", DENSE)
        draft = draft_dir(self.root / "draft", DENSE)
        result, run, _ = self.serve(
            target, "--draft-model", str(draft), "--language-only"
        )
        self.assertIsNone(result)
        parsed = launcher.model_artifacts.parse_args(run.call_args.args[0][2:])
        self.assertEqual(parsed.model_dir, str(target))
        self.assertTrue(parsed.language_only)

    def test_local_gguf_directory_uses_the_cached_default_draft(self):
        target = self.gguf()
        cache = self.root / "cache"
        cached_snapshot(
            cache, MOE.draft_repo, DRAFT_COMMIT, lambda p: draft_dir(p, MOE)
        )
        refs = cache / hub.folder_name(MOE.draft_repo) / "refs"
        refs.mkdir(parents=True)
        (refs / "main").write_text(DRAFT_COMMIT)
        result, run, _ = self.serve(target.parent)
        self.assertIsNone(result)
        parsed = launcher.model_artifacts.parse_args(run.call_args.args[0][2:])
        # No draft named: the family's cached one stands in at prepare time.
        self.assertIsNone(parsed.draft_model)
        self.assertEqual(parsed.model_dir, str(target.parent.resolve()))

    def test_conflicting_local_options_are_refused_before_setup(self):
        mmproj = self.root / "tower.gguf"
        mmproj.write_bytes(b"GGUF")
        target = self.root / "model"
        target.mkdir()
        for options, reason in (
            (
                ["--model", "owner/repo", "--mmproj", str(mmproj)],
                "--mmproj requires --model-dir",
            ),
            (
                ["--model-dir", str(target), "--revision", "main"],
                "--revision requires --model, not --model-dir",
            ),
            (
                [
                    "--model-dir",
                    str(target),
                    "--mmproj",
                    str(mmproj),
                    "--language-only",
                ],
                "--mmproj and --language-only are exclusive",
            ),
        ):
            with (
                self.subTest(options=options),
                mock.patch.object(launcher, "_ensure_runtime") as setup,
                mock.patch("sys.stderr", new_callable=io.StringIO) as error,
                self.assertRaises(SystemExit) as refused,
            ):
                launcher.main(["serve", *options])
            self.assertEqual(refused.exception.code, 2)
            self.assertIn(reason, error.getvalue())
            setup.assert_not_called()

    def test_a_local_target_and_a_model_id_are_exclusive(self):
        """--model names the API model of a Hub target; a local one derives its
        ID from the directory's name."""
        target = self.root / "model"
        target.mkdir()
        with (
            mock.patch.object(launcher, "_ensure_runtime") as setup,
            mock.patch("sys.stderr", new_callable=io.StringIO),
            self.assertRaises(SystemExit) as refused,
        ):
            launcher.main(
                ["serve", "--model", "owner/repo", "--model-dir", str(target)]
            )
        self.assertEqual(refused.exception.code, 2)
        setup.assert_not_called()

    def test_failed_local_preparation_does_not_start_the_server(self):
        target = self.root / "model"
        target.mkdir()
        with (
            mock.patch.object(launcher, "_ensure_runtime"),
            mock.patch.object(launcher, "_run_held", return_value=1),
            mock.patch.object(launcher.os, "execve") as execute,
            mock.patch("sys.stderr", new_callable=io.StringIO) as error,
        ):
            self.assertEqual(launcher.main(["serve", "--model-dir", str(target)]), 1)
        self.assertIn("model download or verification failed", error.getvalue())
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
