#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."
root=$PWD
version=${1:?usage: package_rust_dmg.sh VERSION}
mode=${2:-}
if [[ ! $version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Expected a numeric macOS bundle version (major.minor.patch)" >&2
    exit 2
fi

destination="$root/dist/Splash-Desktop-Rust-$version.dmg"
mkdir -p "$root/dist"
work=$(mktemp -d "$root/dist/.splash-rust.XXXXXX")
trap 'rm -rf "$work"' EXIT
mkdir "$work/image"
app="$work/image/Splash Desktop.app"
contents="$app/Contents"
resources="$contents/Resources"
runtime="$resources/runtime"
mkdir -p "$contents/MacOS" "$runtime/engine"

MACOSX_DEPLOYMENT_TARGET=26.4 cargo build --locked --release --manifest-path server-rs/Cargo.toml
swift build --package-path macos -c release ${SWIFT_BUILD_SYSTEM:+--build-system "$SWIFT_BUILD_SYSTEM"}
swift_products=$(swift build --package-path macos -c release ${SWIFT_BUILD_SYSTEM:+--build-system "$SWIFT_BUILD_SYSTEM"} --show-bin-path)
install -m 755 "$swift_products/SplashDesktop" "$contents/MacOS/SplashDesktop"
for bundle in "$swift_products"/*.bundle; do
    if [[ -d $bundle ]]; then ditto "$bundle" "$resources/${bundle##*/}"; fi
done
for localization in macos/Resources/*.lproj; do
    ditto "$localization" "$resources/${localization##*/}"
done

install -m 755 server-rs/target/release/splash-server "$runtime/engine/splash-server"
install -m 755 build/splash "$runtime/engine/splash"
install -m 644 build/splash.metallib "$runtime/engine/splash.metallib"
if [[ -f build/host-step.metallib ]]; then
    install -m 644 build/host-step.metallib "$runtime/engine/host-step.metallib"
fi
install -m 644 LICENSE THIRD_PARTY_NOTICES install/suggested-models.tsv "$runtime/"

iconset="$work/SplashDesktop.iconset"
mkdir "$iconset"
for size in 16 32 128 256 512; do
    sips -z "$size" "$size" macos/Resources/AppIcon.png \
        --out "$iconset/icon_${size}x${size}.png" >/dev/null
    double=$((size * 2))
    sips -z "$double" "$double" macos/Resources/AppIcon.png \
        --out "$iconset/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$iconset" -o "$resources/SplashDesktop.icns"

cat > "$contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
    <key>CFBundleName</key><string>Splash Desktop</string>
    <key>CFBundleDisplayName</key><string>Splash Desktop</string>
    <key>CFBundleIdentifier</key><string>ai.inco.splash</string>
    <key>CFBundleExecutable</key><string>SplashDesktop</string>
    <key>CFBundleIconFile</key><string>SplashDesktop</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>$version</string>
    <key>CFBundleVersion</key><string>$version</string>
    <key>CFBundleDevelopmentRegion</key><string>en</string>
    <key>CFBundleLocalizations</key><array><string>en</string><string>zh-Hans</string></array>
    <key>LSMinimumSystemVersion</key><string>26.4</string>
    <key>LSApplicationCategoryType</key><string>public.app-category.developer-tools</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSAppTransportSecurity</key><dict>
        <key>NSAllowsLocalNetworking</key><true/>
        <key>NSAllowsArbitraryLoadsInWebContent</key><true/>
    </dict>
</dict></plist>
PLIST

plutil -lint "$contents/Info.plist"
codesign --force --sign - "$runtime/engine/splash-server"
codesign --force --sign - "$runtime/engine/splash"
codesign --force --sign - "$app"
codesign --verify --deep --strict "$app"
"$contents/MacOS/SplashDesktop" --print-runtime
if [[ $mode == --app-only ]]; then
    destination="$root/dist/Splash Desktop Rust.app"
    rm -rf "$destination"
    mv "$app" "$destination"
    echo "$destination"
    exit 0
fi
ln -s /Applications "$work/image/Applications"

image="$work/${destination##*/}"
hdiutil create -volname "Splash Desktop Rust $version" -srcfolder "$work/image" -format UDZO "$image"
hdiutil verify "$image"
(cd "$work" && shasum -a 256 "${image##*/}" > "${image##*/}.sha256")
mv -f "$image" "$destination"
mv -f "$image.sha256" "$destination.sha256"
echo "$destination"
