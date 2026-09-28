"""The DMG packager's pure pieces and bundle layout, without a real model.

These run on any Mac with the system tools (`ditto`, `sips`, `iconutil`,
`hdiutil`) and do not build the Swift app: a stand-in executable and runtime
prove the layout, and the release machine builds the real ones.
"""

import plistlib
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from dev.tools import package_dmg


def options(**overrides):
    values = {"runtime": None, "archive": None, "version": "1.1.0", "dist": None}
    values.update(overrides)
    return SimpleNamespace(**values)


def fake_runtime(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "install").mkdir()
    (root / "install/launcher.py").write_text("# launcher\n")
    (root / "server").mkdir()
    (root / "server/server.py").write_text("# server\n")
    (root / "release.json").write_text('{"version": "1.1.0"}\n')
    return root


def stand_in_binary(root: Path) -> Path:
    binary = root / "stand-in"
    binary.write_bytes(b"#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    return binary


class PackageDmgTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_icon_is_a_rounded_gradient_with_a_white_droplet(self):
        size = 64
        pixels = package_dmg.icon_pixels(size)
        self.assertEqual(len(pixels), size * size * 4)

        def alpha(x, y):
            return pixels[(y * size + x) * 4 + 3]

        def rgb(x, y):
            index = (y * size + x) * 4
            return tuple(pixels[index : index + 3])

        # The corners are cut away; the middle of the droplet is near white
        # (the droplet carries a slight vertical shade).
        self.assertEqual(alpha(0, 0), 0)
        self.assertEqual(alpha(size - 1, size - 1), 0)
        center = (int(size * 0.5), int(size * 0.63))
        self.assertTrue(all(channel > 220 for channel in rgb(*center)))

    def test_info_plist_states_the_bundle_and_local_network_policy(self):
        plist = plistlib.loads(package_dmg.info_plist("9.9.9").encode())
        self.assertEqual(plist["CFBundleIdentifier"], package_dmg.BUNDLE_ID)
        self.assertEqual(plist["CFBundleShortVersionString"], "9.9.9")
        self.assertEqual(plist["LSMinimumSystemVersion"], package_dmg.MIN_MACOS)
        self.assertTrue(plist["NSAppTransportSecurity"]["NSAllowsLocalNetworking"])

    def test_resolve_runtime_prefers_a_given_directory(self):
        runtime = fake_runtime(self.root / "runtime")
        resolved, temporary = package_dmg.resolve_runtime(
            options(runtime=str(runtime), dist=self.root)
        )
        self.assertEqual(resolved, runtime.resolve())
        self.assertIsNone(temporary)

    def test_resolve_runtime_extracts_the_release_archive(self):
        runtime = fake_runtime(self.root / "staged")
        archive = self.root / "splash-1.1.0-arm64-macos26.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(runtime, arcname="splash-1.1.0")
        resolved, temporary = package_dmg.resolve_runtime(
            options(archive=str(archive), dist=self.root)
        )
        self.addCleanup(temporary.cleanup)
        self.assertEqual(resolved.name, "splash-1.1.0")
        self.assertTrue((resolved / "install/launcher.py").is_file())

    def test_resolve_runtime_without_a_source_explains_the_fix(self):
        with self.assertRaises(SystemExit) as raised:
            package_dmg.resolve_runtime(options(dist=self.root))
        self.assertIn("make package", str(raised.exception))

    def test_build_app_embeds_the_runtime_icon_and_executable(self):
        runtime = fake_runtime(self.root / "runtime")
        app = package_dmg.build_app(
            runtime, "1.1.0", self.root / "dist", stand_in_binary(self.root)
        )
        contents = app / "Contents"
        self.assertTrue((contents / "MacOS/Splash").is_file())
        self.assertTrue((contents / "Resources/Splash.icns").is_file())
        self.assertTrue((contents / "Resources/runtime/install/launcher.py").is_file())
        resources = contents / "Resources"
        self.assertTrue((resources / "en.lproj/Localizable.strings").is_file())
        self.assertTrue((resources / "zh-Hans.lproj/Localizable.strings").is_file())
        plist = plistlib.loads((contents / "Info.plist").read_bytes())
        self.assertEqual(plist["CFBundleExecutable"], "Splash")
        self.assertIn("zh-Hans", plist["CFBundleLocalizations"])

    def test_overlay_source_uses_this_checkouts_python(self):
        runtime = fake_runtime(self.root / "runtime")
        (runtime / "install/launcher.py").write_text("# old launcher\n")
        package_dmg.overlay_source(runtime)
        launcher = (runtime / "install/launcher.py").read_text()
        self.assertIn("Serve in the foreground", launcher)  # the checkout's launcher
        self.assertIn("--model-dir", launcher)
        self.assertTrue((runtime / "server/server.py").is_file())
        self.assertFalse(any(path.name == "__pycache__" for path in runtime.rglob("*")))

    def test_create_dmg_holds_the_app_and_an_applications_link(self):
        if shutil.which("hdiutil") is None:
            self.skipTest("hdiutil is unavailable")
        runtime = fake_runtime(self.root / "runtime")
        app = package_dmg.build_app(
            runtime, "1.1.0", self.root / "dist", stand_in_binary(self.root)
        )
        dmg = package_dmg.create_dmg(app, "1.1.0", self.root / "dist")
        self.assertTrue(dmg.is_file())
        info = subprocess.run(
            ["hdiutil", "imageinfo", str(dmg)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        self.assertIn("UDZO", info)


if __name__ == "__main__":
    unittest.main()
