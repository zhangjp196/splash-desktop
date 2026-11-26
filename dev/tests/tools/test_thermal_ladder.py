"""The heat-and-noise control now has exactly three choices, drawn in
`macos/Sources/SplashApp/ThermalPane.swift`: `smartOne`, `smartTwo` and `auto`.
Both smart modes adapt their prefill duty from live fan, temperature and macOS
pressure readings and push it through `POST /v1/thermal`; `auto` leaves pacing
to macOS and names the server's own default policy.

The old fixed ladder and its `FixedLevel` / `SmartLevel` sub-enums are gone: a
build that still drew a fixed rung, or that sent a `--thermal-policy`, would
reintroduce a strategy the UI no longer offers. These checks read the sources
back so that the modes, their labels, their ceilings and the launch arguments
stay in step.
"""

import re
import unittest

from dev.tools import package_dmg

ROOT = package_dmg.ROOT
SWIFT = ROOT / "macos/Sources/SplashApp/ThermalPane.swift"
MODEL = ROOT / "macos/Sources/SplashApp/AppModel.swift"

# The three cases of the mode enum, as `case smartOne, smartTwo, auto`.
MODE_CASES = re.compile(r"enum ThermalRung[^{]*\{(?:.|\n)*?\n\s*case ([^\n]+)")
# A mode's cool ceiling, as `case .smartOne: return 72`.
CEILING = re.compile(r"case \.(\w+): return (\d+(?:\.\d+)?)")


def mode_names():
    found = MODE_CASES.search(SWIFT.read_text())
    if found is None:
        return []
    return [name.strip() for name in found.group(1).split(",")]


def ceilings():
    swift = SWIFT.read_text()
    block = swift[swift.index("var ceiling"): swift.index("var floor")]
    return [(name, float(value)) for name, value in CEILING.findall(block)]


def strings(lang):
    return (ROOT / f"macos/Resources/{lang}.lproj/Localizable.strings").read_text()


class ThermalModesTests(unittest.TestCase):
    def test_exactly_three_modes(self):
        self.assertEqual(mode_names(), ["smartOne", "smartTwo", "auto"])

    def test_old_fixed_and_level_enums_are_gone(self):
        swift = SWIFT.read_text()
        self.assertNotIn("enum FixedLevel", swift)
        self.assertNotIn("enum SmartLevel", swift)
        self.assertNotIn("case .fixed", swift)
        self.assertNotIn("--thermal-policy", MODEL.read_text())

    def test_smart_ceilings_rise_toward_system_default(self):
        values = dict(ceilings())
        self.assertLess(values["smartOne"], values["smartTwo"])
        self.assertLess(values["smartTwo"], values["auto"])
        self.assertEqual(values["auto"], 100.0)

    def test_every_mode_has_title_and_caption_in_both_languages(self):
        keys = ("thermal.auto", "thermal.auto.caption",
                "thermal.smart.one", "thermal.smart.one.caption",
                "thermal.smart.two", "thermal.smart.two.caption")
        for lang in ("en", "zh-Hans"):
            text = strings(lang)
            for key in keys:
                with self.subTest(language=lang, key=key):
                    self.assertIn(f'"{key}" = ', text)

    def test_retired_labels_are_removed(self):
        for lang in ("en", "zh-Hans"):
            text = strings(lang)
            for key in ("thermal.fixed", "thermal.balanced", "thermal.moderate",
                        "thermal.quiet", "thermal.smart.level", "thermal.smart.band"):
                with self.subTest(language=lang, key=key):
                    self.assertNotIn(f'"{key}" = ', text)


if __name__ == "__main__":
    unittest.main()
