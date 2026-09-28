#!/usr/bin/env python3
"""Build Splash.app from a staged runtime and package it into a disk image.

The app is a thin SwiftUI control panel; the runtime it drives is the
directory `make package` stages (install/, server/, engine/, python/ and
release.json). This tool copies that directory into the app bundle's
Resources, so the DMG is self-contained: nothing is downloaded on first run
except the model the user chooses.

The runtime comes from --runtime DIRECTORY, from --archive FILE, or from the
release archive `dist/splash-<version>-arm64-macos26.tar.gz` `make package`
writes. Building the app needs Xcode's Swift toolchain; assembling the DMG
needs `hdiutil` and `iconutil`, both of macOS.
"""

from __future__ import annotations

import argparse
import math
import shutil
import struct
import subprocess
import tempfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP_NAME = "Splash"
BUNDLE_ID = "ai.inco.splash"
# The engine refuses older systems; the app follows it.
MIN_MACOS = "26.4"
MACOS_BUILD = Path("macos")
ICON_SIZES = (16, 32, 128, 256, 512)


def run(command, **kwargs):
    subprocess.run([str(part) for part in command], check=True, **kwargs)


# --- placeholder icon ------------------------------------------------------


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def _write_png(path: Path, size: int, pixels: bytes):
    raw = bytearray()
    stride = size * 4
    for y in range(size):
        raw.append(0)  # filter: none
        raw += pixels[y * stride : (y + 1) * stride]
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + _chunk(b"IEND", b"")
    )


def _clamp(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def _rounded_box(x: float, y: float, half: float, radius: float) -> float:
    qx = abs(x) - half + radius
    qy = abs(y) - half + radius
    return min(max(qx, qy), 0.0) + math.hypot(max(qx, 0.0), max(qy, 0.0)) - radius


def icon_pixels(size: int) -> bytes:
    """A rounded square with a diagonal gradient and a white droplet."""
    center = size / 2
    radius = size * 0.2237
    # Droplet geometry, in fractions of the icon.
    drop_x = size * 0.5
    drop_y = size * 0.63
    drop_r = size * 0.185
    apex = size * 0.21
    top = (0x2F, 0x6B, 0xF6)
    bottom = (0x18, 0xC7, 0xC1)
    pixels = bytearray(size * size * 4)
    for y in range(size):
        for x in range(size):
            px, py = x + 0.5 - center, y + 0.5 - center
            cover = _clamp(0.5 - _rounded_box(px, py, center, radius))
            index = (y * size + x) * 4
            if cover <= 0.0:
                continue
            mix = (x + y) / (2 * (size - 1))
            red = top[0] + (bottom[0] - top[0]) * mix
            green = top[1] + (bottom[1] - top[1]) * mix
            blue = top[2] + (bottom[2] - top[2]) * mix
            # The droplet: its circular body plus the cone that caps it.
            dx, dy = x + 0.5 - drop_x, y + 0.5 - drop_y
            inside_circle = math.hypot(dx, dy) <= drop_r
            inside_cone = apex <= y + 0.5 <= drop_y and abs(dx) <= (
                drop_r * (y + 0.5 - apex) / (drop_y - apex)
            )
            droplet = 1.0 if (inside_circle or inside_cone) else 0.0
            shade = 1.0 - 0.12 * drop_y / size
            red = red * (1 - droplet) + 0xFF * droplet * shade
            green = green * (1 - droplet) + 0xFF * droplet * shade
            blue = blue * (1 - droplet) + 0xFF * droplet * shade
            pixels[index : index + 4] = bytes(
                (
                    int(red + 0.5),
                    int(green + 0.5),
                    int(blue + 0.5),
                    int(cover * 255 + 0.5),
                )
            )
    return bytes(pixels)


def build_icon(workdir: Path) -> Path:
    """A Splash.icns beside the iconset, from the generated master."""
    master = workdir / "icon-1024.png"
    _write_png(master, 1024, icon_pixels(1024))
    iconset = workdir / "Splash.iconset"
    iconset.mkdir(parents=True, exist_ok=True)
    for size in ICON_SIZES:
        for scale in (1, 2):
            target = size * scale
            name = f"icon_{size}x{size}" + ("@2x" if scale == 2 else "") + ".png"
            if target == 1024:
                shutil.copyfile(master, iconset / name)
            else:
                run(
                    [
                        "sips",
                        "-z",
                        str(target),
                        str(target),
                        str(master),
                        "--out",
                        str(iconset / name),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
    icns = workdir / "Splash.icns"
    run(["iconutil", "-c", "icns", str(iconset), "-o", str(icns)])
    return icns


# --- app bundle ------------------------------------------------------------


def info_plist(version: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>{APP_NAME}</string>
    <key>CFBundleDisplayName</key><string>{APP_NAME}</string>
    <key>CFBundleIdentifier</key><string>{BUNDLE_ID}</string>
    <key>CFBundleExecutable</key><string>{APP_NAME}</string>
    <key>CFBundleIconFile</key><string>{APP_NAME}</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>{version}</string>
    <key>CFBundleVersion</key><string>{version}</string>
    <key>LSMinimumSystemVersion</key><string>{MIN_MACOS}</string>
    <key>LSApplicationCategoryType</key><string>public.app-category.developer-tools</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSHumanReadableCopyright</key><string>Apache-2.0; see THIRD_PARTY_NOTICES in the runtime.</string>
    <key>NSAppTransportSecurity</key>
    <dict>
        <key>NSAllowsLocalNetworking</key><true/>
        <key>NSAllowsArbitraryLoadsInWebContent</key><true/>
    </dict>
    <key>CFBundleDevelopmentRegion</key><string>en</string>
    <key>CFBundleLocalizations</key>
    <array><string>en</string><string>zh-Hans</string></array>
</dict>
</plist>
"""


def build_app(
    runtime: Path, version: str, destination: Path, swift_binary: Path
) -> Path:
    app = destination / f"{APP_NAME}.app"
    if app.exists():
        shutil.rmtree(app)
    contents = app / "Contents"
    macos = contents / "MacOS"
    resources = contents / "Resources"
    macos.mkdir(parents=True)
    resources.mkdir(parents=True)
    shutil.copy2(swift_binary, macos / APP_NAME)
    (contents / "Info.plist").write_text(info_plist(version))
    with tempfile.TemporaryDirectory() as temporary:
        icon = build_icon(Path(temporary))
        shutil.copy2(icon, resources / f"{APP_NAME}.icns")
    # The interface follows the system language through these tables.
    localizations = ROOT / MACOS_BUILD / "Resources"
    if localizations.is_dir():
        for lproj in localizations.iterdir():
            if lproj.is_dir():
                shutil.copytree(lproj, resources / lproj.name)
    # ditto preserves the interpreter's symlinks and permissions.
    run(["ditto", runtime, resources / "runtime"])
    return app


def overlay_source(runtime: Path) -> None:
    """Replace the runtime's install/ and server/ with this checkout's, so a
    published engine serves the working tree's installer and server code
    without a local rebuild."""
    for folder in ("install", "server"):
        source = ROOT / folder
        if not source.is_dir():
            continue
        destination = runtime / folder
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(
            source,
            destination,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )


def overlay_engine(runtime: Path) -> None:
    """Replace the runtime's engine binary and metallib with the locally built
    ones (build/splash and build/splash.metallib), when the working tree's
    native code differs from the published runtime's."""
    engine = runtime / "engine"
    for name in ("splash", "splash.metallib"):
        built = ROOT / "build" / name
        if built.is_file():
            shutil.copy2(built, engine / name)
            print(f"overlaid engine/{name} from the local build")
        else:
            raise SystemExit(f"missing {built}; build the engine first")


def build_swift(package: Path) -> Path:
    run(["swift", "build", "--package-path", package, "-c", "release"])
    binary = package / ".build/release" / APP_NAME
    if not binary.is_file():
        raise SystemExit(f"swift build produced no {binary}")
    return binary


# --- runtime ---------------------------------------------------------------


def resolve_runtime(args) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    if args.runtime:
        return Path(args.runtime).resolve(), None
    archive = (
        Path(args.archive)
        if args.archive
        else args.dist / f"splash-{args.version}-arm64-macos26.tar.gz"
    )
    if not archive.is_file():
        raise SystemExit(
            f"no runtime: {archive} is missing and no --runtime was given; "
            "run 'make package RELEASE_VERSION="
            f"{args.version}' first, or pass --runtime DIRECTORY"
        )
    import tarfile

    temporary = tempfile.TemporaryDirectory(prefix="splash-runtime-")
    with tarfile.open(archive) as tar:
        tar.extractall(temporary.name, filter="data")
    roots = list(Path(temporary.name).iterdir())
    if len(roots) != 1 or not roots[0].is_dir():
        raise SystemExit(f"unexpected layout in {archive}")
    return roots[0], temporary


def create_dmg(app: Path, version: str, destination: Path) -> Path:
    dmg = destination / f"Splash-{version}.dmg"
    dmg.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="splash-dmg-") as temporary:
        stage = Path(temporary)
        shutil.copytree(app, stage / app.name, symlinks=True)
        (stage / "Applications").symlink_to("/Applications")
        run(
            [
                "hdiutil",
                "create",
                "-volname",
                f"{APP_NAME} {version}",
                "-srcfolder",
                stage,
                "-ov",
                "-format",
                "UDZO",
                dmg,
            ]
        )
    return dmg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--dist", type=Path, default=ROOT / "dist")
    parser.add_argument("--runtime", help="a staged runtime directory")
    parser.add_argument("--archive", help="the runtime tar.gz from make package")
    parser.add_argument(
        "--no-build", action="store_true", help="reuse the existing swift build"
    )
    parser.add_argument("--app-only", action="store_true", help="skip the DMG")
    parser.add_argument(
        "--overlay-source",
        action="store_true",
        help="use this checkout's install/ and server/ in the embedded runtime",
    )
    parser.add_argument(
        "--overlay-engine",
        action="store_true",
        help="use this checkout's build/splash and splash.metallib as the engine",
    )
    args = parser.parse_args(argv)
    # A relative --dist (or the default ROOT/dist, already absolute) is
    # resolved so the archive default below matches the output directory.
    args.dist = args.dist.resolve()
    args.dist.mkdir(parents=True, exist_ok=True)

    runtime, temporary = resolve_runtime(args)
    try:
        if args.overlay_source:
            overlay_source(runtime)
        if args.overlay_engine:
            overlay_engine(runtime)
        binary = (
            MACOS_BUILD / ".build/release" / APP_NAME
            if args.no_build
            else build_swift(ROOT / MACOS_BUILD)
        )
        if not binary.is_file():
            raise SystemExit(f"missing {binary}; build the app first")
        app = build_app(runtime, args.version, args.dist, binary)
        print(f"Built {app}")
        if not args.app_only:
            dmg = create_dmg(app, args.version, args.dist)
            print(f"Built {dmg}")
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    main()
