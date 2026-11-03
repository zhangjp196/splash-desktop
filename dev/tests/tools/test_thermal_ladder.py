"""The heat-and-noise ladder is written down twice: as the server's policy
table (`serve_options.THERMAL_POLICIES`, which `--thermal-policy` names) and as
the app's rung table (`macos/Sources/SplashApp/ThermalPane.swift`, which draws
the control and sends the duty). A build that changed one and not the other
would label a rung with a duty it does not set, so they are read back and
compared, the app's own labels and the server's names included.

The app's `smart` rung is not in the server's table: it holds no fixed duty, so
the app computes one from the fans and temperatures and pushes it through
`POST /v1/thermal`. What is checked for it is the other way round: it is marked
as a rung without a fixed duty, and its three levels and their labels exist.
"""

import re
import unittest

from dev.tools import package_dmg
from server import serve_options

ROOT = package_dmg.ROOT
SWIFT = ROOT / "macos/Sources/SplashApp/ThermalPane.swift"
MODEL = ROOT / "macos/Sources/SplashApp/AppModel.swift"

# The app's rungs, as `case .quiet: return 60` in the duty switch.
RUNG = re.compile(r"case \.(\w+): return (\d+(?:\.\d+)?)")
# The rung without a fixed duty, as `case .smart: return nil`.
SMART = re.compile(r"case \.(\w+): return nil")
# The smart mode's levels: the `case high, medium, low` line of its enum.
SMART_LEVEL = re.compile(r"enum SmartLevel[^{]*\{(?:.|\n)*?\n\s*case ([^\n]+)")


def thermal_rung_source():
    """The `enum ThermalRung` declaration alone, without the smart mode's."""
    swift = SWIFT.read_text()
    return swift[swift.index("enum ThermalRung"):swift.index("enum SmartLevel")]


def app_rungs():
    """The rung each name carries in the app, in the order it lists them."""
    return [(name, float(duty)) for name, duty in RUNG.findall(thermal_rung_source())]


def duty_less_rungs():
    """The rungs the app marks as having no fixed duty."""
    return SMART.findall(thermal_rung_source())


def smart_levels():
    """The smart mode's levels, in the order its enum lists them."""
    found = SMART_LEVEL.search(SWIFT.read_text())
    if found is None:
        return []
    return [name.strip() for name in found.group(1).split(",")]


def strings(lang):
    """One localization table, as its key lines."""
    return (ROOT / f"macos/Resources/{lang}.lproj/Localizable.strings").read_text()


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

    def test_the_smart_rung_holds_no_fixed_duty(self):
        # The app draws it, but the server has no policy by that name: what
        # reaches the engine from it is a computed duty, not a rung.
        self.assertEqual(duty_less_rungs(), ["smart"])
        self.assertNotIn("smart", serve_options.THERMAL_POLICIES)

    def test_the_smart_rung_has_three_levels_labeled_in_both_languages(self):
        self.assertEqual(smart_levels(), ["high", "medium", "low"])
        for lang in ("en", "zh-Hans"):
            text = strings(lang)
            for level in smart_levels():
                with self.subTest(language=lang, level=level):
                    self.assertIn(f'"thermal.smart.{level}" = ', text)

    def test_every_rung_has_a_label_in_both_languages(self):
        rungs = [name for name, _ in app_rungs()] + duty_less_rungs()
        for lang in ("en", "zh-Hans"):
            text = strings(lang)
            for name in rungs:
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
        self.assertIn("var thermalRung: ThermalRung = .auto", MODEL.read_text())


if __name__ == "__main__":
    unittest.main()
