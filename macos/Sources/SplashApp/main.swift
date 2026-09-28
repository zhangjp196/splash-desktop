import Foundation

// A single executable backs both the windowed app and a small headless probe.
// The packaging and tests use `--print-runtime` to check that the app finds
// the runtime the DMG embeds; otherwise SwiftUI owns the process.
if CommandLine.arguments.contains("--print-runtime") {
    let runtime = Runtime.root
    print(runtime?.path ?? "")
    if let runtime {
        print("python: \(Runtime.python?.path ?? "missing")")
        print("launcher: \(Runtime.launcher?.path ?? "missing")")
        print("catalog: \(Runtime.catalog.count) model(s)")
        _ = runtime
        exit(0)
    }
    exit(1)
}

SplashApp.main()
