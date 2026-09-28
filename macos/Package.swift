// swift-tools-version: 5.10
import PackageDescription

// The macOS control panel for Splash. It owns no model or engine code: it
// locates the runtime the DMG embeds (Resources/runtime, or
// SPLASH_RUNTIME_DIR), launches `install/launcher.py serve`, shows its log and
// status, and renders the server's own chat page in a web view.
let package = Package(
    name: "Splash",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "Splash",
            path: "Sources/SplashApp"
        )
    ]
)
