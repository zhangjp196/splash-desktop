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
    edited.mmproj = "/tmp/selfcheck-mmproj.gguf"
    edited.port = 9123
    edited.lanAccess = true
    edited.allowedHosts = "mymac.local, tunnel.example"
    edited.languageOnly = true
    edited.kvFormat = "bf16"
    edited.neuralEngine = false
    edited.maxMemory = "28G"
    edited.maxContext = "100K"
    edited.expertMemory = "16G"
    edited.pleMemory = "2G"
    edited.sparseKvMemory = "1200M"
    edited.verifyRows = "all"
    edited.residencySeconds = "45"
    edited.diskCache = false
    edited.maxCacheDisk = "64G"
    edited.maxCacheDiskWrites = "200G"
    edited.cacheEvictionPreset = "custom"
    edited.cacheEvictionPolicy = "lfu"
    edited.cacheEvictionProtect = 3
    edited.maxRequestSize = "128M"
    edited.apiKey = "selfcheck-key"
    edited.servedNames = "one, two"
    edited.reasoningEffort = "high"
    edited.maxConcurrency = 24
    passed = store.save(edited) && store.load() == edited
    // The model selection is stored again: the rows name the mode, the target
    // and the local folders, so the next launch opens on the same target.
    passed = passed
        && edited.rows.contains { $0.key == "modelMode" && $0.value == AppModel.ModelMode.local.rawValue }
        && edited.rows.contains { $0.key == "modelDirectory" && $0.value == "/tmp/selfcheck" }
        && edited.rows.contains { $0.key == "mmproj" && $0.value == "/tmp/selfcheck-mmproj.gguf" }
        && edited.rows.contains { $0.key == "cacheEvictionPreset" && $0.value == "custom" }
        && edited.resolvedCacheEviction.policy == "lfu"
        && edited.resolvedCacheEviction.protect == 3

    var replaced = edited
    replaced.port = 8000
    replaced.apiKey = ""
    passed = store.save(replaced) && store.load() == replaced

    // Values the launcher refuses must not survive into the next launch.
    var broken = replaced
    broken.kvFormat = "fp8"
    broken.reasoningEffort = "extreme"
    broken.port = 70_000
    broken.maxConcurrency = 0
    broken.expertMemory = "banana"
    broken.pleMemory = "-2G"
    broken.sparseKvMemory = "0"
    broken.verifyRows = "every"
    broken.cacheEvictionPreset = "turbo"
    broken.cacheEvictionPolicy = "random"
    broken.cacheEvictionProtect = 9
    _ = store.save(broken)
    let repaired = store.load()
    passed = passed
        && repaired.kvFormat == ServerSettings.First.kvFormat
        && repaired.reasoningEffort == ""
        && repaired.port == ServerSettings.First.port
        && repaired.maxConcurrency == ServerSettings.First.maxConcurrency
        && repaired.expertMemory == ServerSettings.default.expertMemory
        && repaired.pleMemory == ServerSettings.default.pleMemory
        && repaired.sparseKvMemory == ServerSettings.default.sparseKvMemory
        && repaired.verifyRows == ServerSettings.default.verifyRows
        && repaired.cacheEvictionPreset == ServerSettings.First.cacheEvictionPreset
        && repaired.cacheEvictionPolicy == ServerSettings.First.cacheEvictionPolicy
        && repaired.cacheEvictionProtect == ServerSettings.First.cacheEvictionProtect

    // A configuration stored before a setting existed still decodes: a missing
    // key keeps its first-launch value instead of dropping the user's saved
    // settings. A target an earlier build stored decodes with it.
    let legacy = Data(#"{"modelMode":"upstream","modelID":"legacy/model","maxContext":"64K"}"#.utf8)
    let decoded = try? JSONDecoder().decode(ServerSettings.self, from: legacy)
    passed = passed
        && decoded?.modelID == "legacy/model"
        && decoded?.modelMode == .upstream
        && decoded?.maxContext == "64K"
        && decoded?.maxConcurrency == ServerSettings.First.maxConcurrency
        && decoded?.diskCache == ServerSettings.First.diskCache
        && decoded?.maxCacheDiskWrites == ServerSettings.First.maxCacheDiskWrites
        && decoded?.cacheEvictionPreset == ServerSettings.First.cacheEvictionPreset
        && decoded?.cacheEvictionPolicy == ServerSettings.First.cacheEvictionPolicy
        && decoded?.cacheEvictionProtect == ServerSettings.First.cacheEvictionProtect
        && decoded?.lanAccess == ServerSettings.default.lanAccess
        && decoded?.allowedHosts == ServerSettings.default.allowedHosts
    // A configuration that turned the SSD cache off through its quota keeps
    // that meaning after the switch arrives.
    let quotaOff = Data(#"{"maxMemory":"20G","maxCacheDisk":"0"}"#.utf8)
    let quotaDecoded = try? JSONDecoder().decode(ServerSettings.self, from: quotaOff)
    passed = passed && quotaDecoded?.diskCache == false

    // A row set an earlier build wrote carries the switch's old default, which
    // nobody chose: it reads as the new default, off. The rows this build writes
    // carry their version, so a choice made from then on is kept.
    var legacyRows = ServerSettings.default.rows.filter { $0.key != "settingsVersion" }
    legacyRows.removeAll { $0.key == "diskCache" }
    legacyRows.append((key: "diskCache", value: "1"))
    let migrated = ServerSettings(rows: Dictionary(uniqueKeysWithValues: legacyRows))
    passed = passed
        && migrated.diskCache == false
        && migrated.settingsVersion == ServerSettings.currentVersion

    // The configuration library the models pane lists: a saved configuration
    // and the active one come back the next launch, so the pane opens on what
    // it showed. The library follows SPLASH_SETTINGS_DB into a suite of its
    // own, so this never touches the user's real configurations.
    let library = LibraryStore()
    var named = ModelConfig(settings: edited)
    named.name = "Selfcheck"
    let spare = ModelConfig(settings: ServerSettings.default)
    library.saveConfigs([named, spare])
    library.saveActive(spare.id)
    let loadedConfigs = library.loadConfigs()
    passed = passed
        && loadedConfigs.map(\.id) == [named.id, spare.id]
        && loadedConfigs.first?.displayName == "Selfcheck"
        && library.loadActive() == spare.id
    library.saveConfigs([])
    library.saveActive(nil)

    // The model: what one launch saves is what the next one opens on, and the
    // defaults button returns it to the state the app first opened with. The
    // per-field debounce the form drives is a view path and is left to the UI.
    MainActor.assumeIsolated {
        _ = store.save(edited)
        let reloaded = AppModel()
        passed = passed
            && reloaded.settings == edited
            && reloaded.settingsFingerprint == edited.fingerprint
        // What the form holds is what the command carries: the merge's
        // streamed stores and the verify policy included.
        let launch = reloaded.launchArguments(launcherPath: "/tmp/launcher.py")
        func carries(_ arguments: [String], _ flag: String, _ value: String) -> Bool {
            guard let index = arguments.firstIndex(of: flag) else { return false }
            return arguments.indices.contains(index + 1) && arguments[index + 1] == value
        }
        func carries(_ flag: String, _ value: String) -> Bool {
            carries(launch, flag, value)
        }
        passed = passed
            && carries("--model-dir", "/tmp/selfcheck")
            && carries("--draft-model", "/tmp/selfcheck-draft")
            && carries("--expert-memory", "16G")
            && carries("--ple-memory", "2G")
            && carries("--sparse-kv-memory", "1200M")
            && carries("--verify-rows", "all")
            && carries("--max-context", "100K")
            // Text only loads no tower, so this configuration names no
            // projector even though the form still holds one; turning vision
            // back on puts the same path back on the command.
            && !launch.contains("--mmproj")
            && launch.contains("--language-only")
            // The switch is off in this configuration, so the command turns
            // the disk tier off instead of passing the quota or write budget;
            // nothing here asks for a cache that survives the server.
            && carries("--max-cache-disk", "0")
            && !launch.contains("--persistent-cache")
            && !launch.contains("--max-cache-disk-writes")
        reloaded.languageOnly = false
        let vision = reloaded.launchArguments(launcherPath: "/tmp/launcher.py")
        passed = passed
            && vision.contains("--mmproj")
            && vision.contains("/tmp/selfcheck-mmproj.gguf")
            && !vision.contains("--language-only")
        // An upstream target is the model ID the panel holds, and its draft is
        // the installer's choice.
        reloaded.modelMode = .upstream
        reloaded.modelID = "mlx-community/selfcheck-model"
        let upstream = reloaded.launchArguments(launcherPath: "/tmp/launcher.py")
        passed = passed
            && carries(upstream, "--model", "mlx-community/selfcheck-model")
            && !upstream.contains("--model-dir")
        // Every KV format the panel offers reaches the launcher, and the
        // server's own default stays off the command.
        for format in ServerSettings.kvFormats {
            reloaded.kvFormat = format
            let arguments = reloaded.launchArguments(launcherPath: "/tmp/launcher.py")
            passed = passed
                && (format == "int8"
                    ? !arguments.contains("--kv-format")
                    : carries(arguments, "--kv-format", format))
        }
        // Turning the SSD cache on is what keeps a coding conversation's prefix
        // across a restart: the command asks for the persistent cache and
        // passes the quota and the day's write budget beside it.
        reloaded.diskCache = true
        let cached = reloaded.launchArguments(launcherPath: "/tmp/launcher.py")
        passed = passed
            && cached.contains("--persistent-cache")
            && carries(cached, "--max-cache-disk", edited.maxCacheDisk)
            && carries(cached, "--max-cache-disk-writes", edited.maxCacheDiskWrites)
        // A custom eviction preset passes the policy and its protected tier;
        // the default level passes neither, leaving the engine's own order.
        reloaded.cacheEvictionPreset = "custom"
        reloaded.cacheEvictionPolicy = "largest"
        reloaded.cacheEvictionProtect = 4
        let evicted = reloaded.launchArguments(launcherPath: "/tmp/launcher.py")
        passed = passed
            && carries(evicted, "--cache-eviction", "largest")
            && carries(evicted, "--cache-eviction-protect", "4")
        reloaded.cacheEvictionPreset = "light"
        let defaultEviction = reloaded.launchArguments(launcherPath: "/tmp/launcher.py")
        passed = passed
            && !defaultEviction.contains("--cache-eviction")
            && !defaultEviction.contains("--cache-eviction-protect")
        reloaded.restoreDefaultSettings()
        let restored = AppModel()
        let defaultLaunch = restored.launchArguments(launcherPath: "/tmp/launcher.py")
        passed = passed
            && restored.settings == ServerSettings.default
            && store.load() == ServerSettings.default
            // The SSD cache is off on a first launch: the command turns the
            // disk tier off and keeps every cached prefix in memory, and the
            // panel's memory and context defaults are on the command so the
            // limit it shows is the one the engine runs with.
            && carries(defaultLaunch, "--max-cache-disk", "0")
            && !defaultLaunch.contains("--persistent-cache")
            && carries(defaultLaunch, "--max-memory", ServerSettings.First.maxMemory)
            && carries(defaultLaunch, "--max-context", ServerSettings.First.maxContext)
            // The default preset is level 2: the plain policy with the reuse
            // protection threshold on the command.
            && ServerSettings.default.cacheEvictionPreset == "balanced"
            && carries(defaultLaunch, "--cache-eviction-protect", "2")
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