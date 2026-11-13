// swift-tools-version: 5.10
import PackageDescription

// The macOS control panel for Splash Desktop. It owns no model or engine code:
// it locates the runtime the DMG embeds (Resources/runtime, or
// SPLASH_RUNTIME_DIR), launches `engine/splash-server serve`, shows its log and
// status, and renders the server's own chat page in a web view.
//
// The target is SplashDesktop, so the executable inside the bundle is
// SplashDesktop while the bundle, its display name and the volume read
// "Splash Desktop" (dev/tools/package_dmg.py).
let package = Package(
    name: "SplashDesktop",
    platforms: [.macOS(.v14)],
    dependencies: [
        // The terminal drawer's view over a pseudo-terminal, MIT licensed,
        // pinned as the app's other dependencies are.
        .package(url: "https://github.com/migueldeicaza/SwiftTerm", exact: "1.20.0"),
    ],
    targets: [
        .executableTarget(
            name: "SplashDesktop",
            dependencies: [.product(name: "SwiftTerm", package: "SwiftTerm")],
            path: "Sources/SplashApp"
        )
    ]
)
