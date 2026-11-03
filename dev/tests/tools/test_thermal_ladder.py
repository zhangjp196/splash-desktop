"""The heat-and-noise ladder is written down twice: as the server's policy
table (`serve_options.THERMAL_POLICIES`, which `--thermal-policy` names) and as
the app's mode and level tables (`macos/Sources/SplashApp/ThermalPane.swift`,
which draw the control and send the duty). A build that changed one and not
the other would label a level with a duty it does not set, so they are read
back and compared, the app's own labels and the server's names included.

The app offers three modes: `auto`, the server's own default; `smart`, which
holds no fixed duty, so the app computes one from the fans and temperatures
and pushes it through `POST /v1/thermal`; and `fixed`, which runs steady at
one of the ladder's rungs. The fixed mode's levels (`FixedLevel`) are the
ladder's rungs, and they alone match the server's policies one for one: what
the level's label says is what --thermal-policy names and what the duty sends.
"""

import re
import unittest

from dev.tools import package_dmg
from server import serve_options

ROOT = package_dmg.ROOT
SWIFT = ROOT / "macos/Sources/SplashApp/ThermalPane.swift"
MODEL = ROOT / "macos/Sources/SplashApp/AppModel.swift"

# A mode or level's fixed duty, as `case .balanced: return 60` in a duty switch.
DUTY = re.compile(r"case \.(\w+): return (\d+(?:\.\d+)?)")
# A mode the app marks as having no fixed duty, as `case .smart: return nil`.
NO_DUTY = re.compile(r"case \.(\w+): return nil")
# The levels of a mode, as the `case balanced, moderate, quiet` line of its enum.
MODE_LEVELS = re.compile(
    r"enum (\w+)[^{]*\{(?:.|\n)*?\n\s*case ([^\n]+)"
)


def enum_source(start, end):
    """One enum's declaration alone, from its name to the next enum's."""
    swift = SWIFT.read_text()
    return swift[swift.index(start):swift.index(end)]


def mode_duties():
    """The duty each mode carries in itself, in the order it lists them."""
    return [(name, float(duty))
            for name, duty in DUTY.findall(enum_source("enum ThermalRung", "enum FixedLevel"))]


def duty_less_modes():
    """The modes the app marks as having no fixed duty of their own."""
    return NO_DUTY.findall(enum_source("enum ThermalRung", "enum FixedLevel"))


def enum_levels(name, end):
    """The levels a mode's enum lists, in the order it lists them."""
    found = MODE_LEVELS.search(enum_source(f"enum {name}", end))
    if found is None:
        return []
    return [level.strip() for level in found.group(2).split(",")]


def fixed_level_duties():
    """The duty each level of the fixed mode carries, in the order it lists them."""
    return [(name, float(duty))
            for name, duty in DUTY.findall(enum_source("enum FixedLevel", "enum SmartLevel"))]


def strings(lang):
    """One localization table, as its key lines."""
    return (ROOT / f"macos/Resources/{lang}.lproj/Localizable.strings").read_text()


class ThermalLadderTests(unittest.TestCase):
    def test_the_fixed_levels_match_the_server_policies(self):
        # The fixed mode's levels are the ladder's rungs: what --thermal-policy
        # names is what the level sends, one for one, in the same order.
        policies = [name for name in serve_options.THERMAL_POLICIES
                    if name != serve_options.DEFAULT_THERMAL_POLICY]
        self.assertEqual([name for name, _ in fixed_level_duties()], policies)
        for name, duty in fixed_level_duties():
            with self.subTest(level=name):
                self.assertEqual(
                    duty, serve_options.THERMAL_POLICIES[name].prefill_duty
                )
        self.assertEqual(
            enum_levels("FixedLevel", "enum SmartLevel"),
            [name for name, _ in fixed_level_duties()],
        )

    def test_the_modes_without_fixed_duties(self):
        # The app draws them, but the server has no policy by either name:
        # what reaches the engine from `smart` is a computed duty, and what
        # reaches it from `fixed` is the chosen level's.
        self.assertEqual(duty_less_modes(), ["smart", "fixed"])
        for name in duty_less_modes():
            self.assertNotIn(name, serve_options.THERMAL_POLICIES)

    def test_the_smart_mode_has_three_levels_labeled_in_both_languages(self):
        levels = enum_levels("SmartLevel", "struct ThermalPane")
        self.assertEqual(levels, ["high", "medium", "low"])
        for lang in ("en", "zh-Hans"):
            text = strings(lang)
            for level in levels:
                with self.subTest(language=lang, level=level):
                    self.assertIn(f'"thermal.smart.{level}" = ', text)

    def test_every_mode_and_level_has_a_label_in_both_languages(self):
        names = ([name for name, _ in mode_duties()]
                 + duty_less_modes()
                 + [name for name, _ in fixed_level_duties()])
        for lang in ("en", "zh-Hans"):
            text = strings(lang)
            for name in names:
                with self.subTest(language=lang, name=name):
                    self.assertIn(f'"thermal.{name}" = ', text)

    def test_the_system_default_is_the_default_in_both(self):
        # The mode the app starts on, and the server's own default, are the
        # same one: a Mac that asks for nothing is not paced.
        self.assertEqual(mode_duties()[0][0], serve_options.DEFAULT_THERMAL_POLICY)
        self.assertEqual(mode_duties()[0][1], serve_options.DEFAULT_PREFILL_DUTY)
        self.assertIn("var thermalRung: ThermalRung = .auto", MODEL.read_text())


if __name__ == "__main__":
    unittest.main()
