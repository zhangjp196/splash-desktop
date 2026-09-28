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

// Exercises the settings persistence without a window: what is saved must come
// back unchanged, a second save must replace rather than accumulate, a value
// the launcher would reject must fall back to its first-launch default, and a
// model built after an edit must open on it. The database is a temporary file
// named through SPLASH_SETTINGS_DB, so the user's own settings are untouched.
if CommandLine.arguments.contains("--selfcheck-settings") {
    let path = NSTemporaryDirectory() + "splash-settings-selfcheck.db"
    func discard() {
        for suffix in ["", "-wal", "-shm"] {
            try? FileManager.default.removeItem(atPath: path + suffix)
        }
    }
    discard()
    setenv("SPLASH_SETTINGS_DB", path, 1)
    let store = SettingsStore()

    // A fresh store answers with the values the app opens with.
    var passed = store.load() == ServerSettings.default

    // Every setting must survive a round trip, and a second save replaces the
    // first instead of leaving it behind.
    var edited = ServerSettings.default
    edited.modelMode = .local
    edited.modelID = "selfcheck/model"
    edited.modelDirectory = "/tmp/selfcheck"
    edited.draftDirectory = "/tmp/selfcheck-draft"
    edited.port = 9123
    edited.languageOnly = true
    edited.kvFormat = "bf16"
    edited.maxMemory = "28G"
    edited.maxContext = "100K"
    edited.maxCacheDisk = "0"
    edited.idleOffloadSeconds = "30"
    edited.residencySeconds = "45"
    edited.maxRequestSize = "128M"
    edited.apiKey = "selfcheck-key"
    edited.servedNames = "one, two"
    edited.reasoningEffort = "high"
    passed = store.save(edited) && store.load() == edited

    var replaced = edited
    replaced.port = 8000
    replaced.apiKey = ""
    passed = store.save(replaced) && store.load() == replaced

    // Values the launcher refuses must not survive into the next launch.
    var broken = replaced
    broken.kvFormat = "fp8"
    broken.reasoningEffort = "extreme"
    broken.port = 70_000
    _ = store.save(broken)
    let repaired = store.load()
    passed = passed
        && repaired.kvFormat == ServerSettings.First.kvFormat
        && repaired.reasoningEffort == ""
        && repaired.port == ServerSettings.First.port

    // The model: what one launch saves is what the next one opens on, and the
    // defaults button returns it to the state the app first opened with. The
    // per-field debounce the form drives is a view path and is left to the UI.
    MainActor.assumeIsolated {
        _ = store.save(edited)
        let reloaded = AppModel()
        passed = passed
            && reloaded.settings == edited
            && reloaded.settingsFingerprint == edited.fingerprint
        reloaded.restoreDefaultSettings()
        let restored = AppModel()
        passed = passed
            && restored.settings == ServerSettings.default
            && store.load() == ServerSettings.default
            && restored.maxCacheDisk == ServerSettings.First.maxCacheDisk
    }

    discard()
    print(passed ? "settings selfcheck: PASS" : "settings selfcheck: FAIL")
    exit(passed ? 0 : 1)
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