"""The heat-and-noise ladder is written down twice: as the server's policy
table (`serve_options.THERMAL_POLICIES`, which `--thermal-policy` names) and as
the app's rung table (`macos/Sources/SplashApp/ThermalPane.swift`, which draws
the control and sends the duty). A build that changed one and not the other
would label a rung with a duty it does not set, so they are read back and
compared, the app's own labels and the server's names included."""

import re
import unittest

from dev.tools import package_dmg
from server import serve_options

ROOT = package_dmg.ROOT
SWIFT = ROOT / "macos/Sources/SplashApp/ThermalPane.swift"
MODEL = ROOT / "macos/Sources/SplashApp/AppModel.swift"

# The app's rungs, as `case .quiet: return 60` in the duty switch.
RUNG = re.compile(r"case \.(\w+): return (\d+(?:\.\d+)?)")


def app_rungs():
    """The rung each name carries in the app, in the order it lists them."""
    found = RUNG.findall(SWIFT.read_text())
    return [(name, float(duty)) for name, duty in found]


class ThermalLadderTests(unittest.TestCase):
    def test_the_app_and_the_server_agree_on_every_rung(self):
        ladder = app_rungs()
        self.assertEqual(
            [name for name, _ in ladder], list(serve_options.THERMAL_POLICIES)
        )
        for name, duty in ladder:
            with self.subTest(rung=name):
                self.assertEqual(
                    duty, serve_options.THERMAL_POLICIES[name].prefill_duty
                )

    def test_every_rung_has_a_label_in_both_languages(self):
        for lang in ("en", "zh-Hans"):
            text = (
                ROOT / f"macos/Resources/{lang}.lproj/Localizable.strings"
            ).read_text()
            for name, _ in app_rungs():
                with self.subTest(language=lang, rung=name):
                    self.assertIn(f'"thermal.{name}" = ', text)

    def test_the_top_rung_is_the_default_without_pacing(self):
        # The rung the app starts on, and the server's own default, are the
        # same one: a Mac that asks for nothing is not paced.
        top = app_rungs()[0][0]
        self.assertEqual(top, serve_options.DEFAULT_THERMAL_POLICY)
        self.assertEqual(
            serve_options.THERMAL_POLICIES[top].prefill_duty,
            serve_options.DEFAULT_PREFILL_DUTY,
        )
        swift = MODEL.read_text()
        self.assertIn("var thermalRung: ThermalRung = .auto", swift)


if __name__ == "__main__":
    unittest.main()
