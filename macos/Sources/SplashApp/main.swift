import AppKit
import Foundation
import SwiftUI


// The headless probes — the UI render and the settings and attach selfchecks —
// live in `SelfCheck.swift` beside the app they exercise; each flag runs its
// probe and exits the process, so the windowed app below is never reached.
if CommandLine.arguments.contains("--render-ui") { SelfCheck.renderUI() }

// A single executable backs both the windowed app and a small headless probe.
// `--print-runtime` checks that the app finds the runtime the DMG embeds;
// otherwise SwiftUI owns the process.
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

if CommandLine.arguments.contains("--probe-python-launch") {
    guard let python = Runtime.python, let launcher = Runtime.launcher else { exit(1) }
    let process = Process()
    process.executableURL = python
    let probeArguments = Array(CommandLine.arguments.dropFirst(2))
    process.arguments = [launcher.path] + (probeArguments.isEmpty ? ["serve", "--help"] : probeArguments)
    var environment = ProcessInfo.processInfo.environment
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    process.environment = environment
    do {
        try process.run()
        process.waitUntilExit()
        exit(process.terminationStatus)
    } catch {
        fputs("Python launcher probe failed: \\(error.localizedDescription)\\n", stderr)
        exit(1)
    }
}

if CommandLine.arguments.contains("--selfcheck-settings") { SelfCheck.settings() }
if CommandLine.arguments.contains("--selfcheck-attach") { SelfCheck.attach() }

SplashApp.main()
