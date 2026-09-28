import Foundation

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

// Exercises `detectRunningServer`, then `stop`: with a reachable /status and
// a serve-<port>.lock the app must present the leftover server as running
// and stop it (SIGINT), idling when /status goes silent.
if CommandLine.arguments.contains("--selfcheck-attach") {
    var attached = false
    var stopped = false
    let semaphore = DispatchSemaphore(value: 0)
    MainActor.assumeIsolated {
        let app = AppModel.shared
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 700_000_000)
            attached = app.phase == .ready && app.isRunning && app.foreignPID != nil
            if attached {
                app.stop()
                let deadline = Date().addingTimeInterval(20)
                while app.phase == .stopping && Date() < deadline {
                    try? await Task.sleep(nanoseconds: 200_000_000)
                }
                stopped = app.phase == .idle && !app.isRunning
            }
            semaphore.signal()
        }
    }
    // The task and its awaits run on the main actor, so pump the main run
    // loop instead of blocking the thread.
    while semaphore.wait(timeout: .now()) == .timedOut {
        RunLoop.current.run(mode: .default, before: Date().addingTimeInterval(0.05))
    }
    print(
        attached && stopped
            ? "attach selfcheck: PASS"
            : "attach selfcheck: FAIL (attached=\(attached) stopped=\(stopped))"
    )
    exit((attached && stopped) ? 0 : 1)
}

SplashApp.main()