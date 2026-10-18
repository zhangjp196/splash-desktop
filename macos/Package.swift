// swift-tools-version: 5.10
import PackageDescription

// The macOS control panel for Splash Desktop. It owns no model or engine code:
// it locates the runtime the DMG embeds (Resources/runtime, or
// SPLASH_RUNTIME_DIR), launches `install/launcher.py serve`, shows its log and
// status, and renders the server's own chat page in a web view.
//
// The target is SplashDesktop, so the executable inside the bundle is
// SplashDesktop while the bundle, its display name and the volume read
// "Splash Desktop" (dev/tools/package_dmg.py).
let package = Package(
    name: "SplashDesktop",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "SplashDesktop",
            path: "Sources/SplashApp"
        )
    ]
)
