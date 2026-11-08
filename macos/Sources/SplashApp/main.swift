import AppKit
import Foundation
import SwiftUI


// Renders the configuration editor and the live board to PNG for a UI
// check without opening a window. The app's other probes are headless too.
if CommandLine.arguments.contains("--render-ui") {
    MainActor.assumeIsolated {
        @MainActor func write<V: View>(_ view: V, width: CGFloat, height: CGFloat, _ name: String) {
            _ = NSApplication.shared
            let hosting = NSHostingView(
                rootView: view.frame(width: width, height: height).preferredColorScheme(.light)
            )
            hosting.frame = NSRect(x: 0, y: 0, width: width, height: height)
            hosting.appearance = NSAppearance(named: .aqua)
            hosting.layoutSubtreeIfNeeded()
            guard let bitmap = hosting.bitmapImageRepForCachingDisplay(in: hosting.bounds) else {
                print("render \(name): FAIL")
                return
            }
            hosting.cacheDisplay(in: hosting.bounds, to: bitmap)
            guard let png = bitmap.representation(using: .png, properties: [:]) else {
                print("render \(name): FAIL")
                return
            }
            try? png.write(to: URL(fileURLWithPath: "/tmp/\(name).png"))
            print("render \(name): \(png.count) bytes")
        }

        var sample = ServerSettings()
        sample.modelID = "mlx-community/Qwen3.8-27B-4bit"
        sample.maxMemory = "28G"
        sample.maxContext = "204800"
        sample.expertMemory = "16G"
        sample.pleMemory = "2G"
        sample.sparseKvMemory = "1200M"
        sample.apiKey = "sk-local"
        sample.servedNames = "gpt-4o, qwen"
        sample.diskCache = true
        sample.cacheDir = "~/Library/Caches/Splash/prefix-cache"
        sample.maxCacheDisk = "16G"
        sample.maxCacheDiskWrites = "32G"
        sample.cacheIdleSpill = "30"
        sample.requestTimeout = "30m"
        sample.thinkingTokenBudget = "4096"
        sample.decodeShare = "0.5"
        for section in SettingsSection.allCases {
            write(
                SettingsEditor(settings: .constant(sample), baseline: sample, section: section),
                width: 840,
                height: 620,
                "settings-\(section.rawValue)"
            )
        }

        let model = AppModel.shared
        model.phase = .ready
        model.live = LiveStatus(
            updated: true, ready: true, modelID: "mlx-community/Qwen3.8-27B-4bit",
            contextTokens: 204800, decodeTokensPerSecond: 42.6,
            prefillTokensPerSecond: 318.4, draftAcceptanceRate: 0.72,
            ttftP50Milliseconds: 210, ttftP95Milliseconds: 480, ttftSamples: 121,
            itlP50Milliseconds: 28, itlP95Milliseconds: 41, itlSamples: 4096,
            submitted: 128, completed: 121, failed: 2, cancelled: 5,
            pending: 3, pendingLimit: 24, currentBytes: 18_300_000_000,
            peakBytes: 21_200_000_000, physicalBytes: 17_100_000_000,
            peakPhysicalBytes: 19_400_000_000,
            governorLimitBytes: 24_000_000_000, configuredMemoryLimitBytes: 28_000_000_000,
            cacheHitRate: 0.68, kvDiskHitTokens: 96_000, kvDiskHitRate: 0.42,
            reusedTokens: 128_000, stateDiskHits: 37, waiting: 2,
            waitingMemory: 1, waitingConcurrency: 1, diskUsedBytes: 4_200_000_000,
            diskCapacityBytes: 16_000_000_000, idleOffloadPasses: 3,
            idleOffloadBytes: 2_400_000_000, capacityFailures: 1, metalFailures: 0,
            fansAvailable: true, fanSpeeds: [1342, 3197],
            fanMinimums: [1350, 1458], fanMaximums: [5349, 5777],
            dutyPercent: 60, thermalScale: 0.75, pacedWaits: 3,
            pacedWaitMilliseconds: 1234.5
        )
        model.statusUpdatedAt = Date()
        model.decodeHistory = (0..<60).map { 30 + sin(Double($0) / 5) * 12 }
        model.prefillHistory = (0..<60).map { 280 + cos(Double($0) / 7) * 80 }
        model.acceptanceHistory = (0..<60).map { 68 + sin(Double($0) / 9) * 8 }
        write(LivePane().environmentObject(model), width: 1200, height: 900, "live")
        exit(0)
    }
}

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

    // A fresh store answers with the values the app opens with: the idle
    // unload is on for ten minutes.
    var passed = store.load() == ServerSettings.default
        && ServerSettings.default.unloadIdleWeights
        && ServerSettings.default.residencySeconds == "600"
        && !ServerSettings.default.responsivePrefill
        && ServerSettings.default.prefillSliceMilliseconds == "1000"

    // Every setting must survive a round trip, and a second save replaces the
    // first instead of leaving it behind.
    var edited = ServerSettings.default
    edited.modelMode = .local
    edited.modelID = "selfcheck/model"
    edited.modelDirectory = "/tmp/selfcheck"
    edited.draftDirectory = "/tmp/selfcheck-draft"
    edited.draftMode = "folder"
    edited.mmproj = "/tmp/selfcheck-mmproj.gguf"
    edited.port = 9123
    edited.lanAccess = true
    edited.allowedHosts = "mymac.local, tunnel.example"
    edited.allowedOrigins = "tauri://localhost, https://example.com"
    edited.languageOnly = true
    edited.kvFormat = "bf16"
    edited.stateFormat = "bf16"
    edited.neuralEngine = false
    edited.maxMemory = "28G"
    edited.maxContext = "100K"
    edited.expertMemory = "16G"
    edited.expertMemoryGrow = true
    edited.pleMemory = "2G"
    edited.sparseKvMemory = "1200M"
    edited.verifyRows = "all"
    edited.residencySeconds = "45"
    edited.residencyKeepAlive = "5m"
    edited.responsivePrefill = true
    edited.prefillSliceMilliseconds = "750"
    edited.cacheIdleSpill = "2m"
    edited.cacheIdleEvict = false
    edited.cacheDir = "/tmp/splash-cache"
    edited.diskCache = false
    edited.maxCacheDisk = "64G"
    edited.maxCacheDiskWrites = "200G"
    edited.cacheEvictionPreset = "custom"
    edited.cacheEvictionPolicy = "lfu"
    edited.cacheEvictionProtect = 3
    edited.maxRequestSize = "128M"
    edited.requestTimeout = "30m"
    edited.maxImagePixels = "1048576"
    edited.decodeShare = "0.75"
    edited.allowIdleSleep = true
    edited.thinkingTokenBudget = "4096"
    edited.thinkingEndText = "That is the answer."
    edited.announceServedName = true
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

    // The old ten-second residency was the app's default, not a choice: it
    // moves to this build's ten minutes, with the idle unload on. A value this
    // build wrote is the user's own and is kept.
    var legacyResidency = ServerSettings.default
    legacyResidency.settingsVersion = "3"
    legacyResidency.residencySeconds = ServerSettings.legacyResidencySeconds
    passed = passed
        && ServerSettings.migrateLegacyResidency(&legacyResidency, storedVersion: "3")
        && legacyResidency.residencySeconds == ServerSettings.First.residencySeconds
        && legacyResidency.unloadIdleWeights == ServerSettings.First.unloadIdleWeights
        && legacyResidency.settingsVersion == ServerSettings.currentVersion
    var chosenResidency = ServerSettings.default
    chosenResidency.settingsVersion = ServerSettings.currentVersion
    chosenResidency.residencySeconds = ServerSettings.legacyResidencySeconds
    passed = passed
        && !ServerSettings.migrateLegacyResidency(
            &chosenResidency, storedVersion: ServerSettings.currentVersion)
        && chosenResidency.residencySeconds == ServerSettings.legacyResidencySeconds

    // A row set from before the slice's switch carries no row, so the value
    // nobody chose reads as off rather than as this build's first-launch
    // default: an update cannot turn slicing on behind the user's back.
    var preSliceRows = ServerSettings.default.rows.filter { $0.key != "responsivePrefill" }
    preSliceRows.removeAll { $0.key == "settingsVersion" }
    preSliceRows.append((key: "settingsVersion", value: "3"))
    passed = passed
        && ServerSettings(rows: Dictionary(uniqueKeysWithValues: preSliceRows))
            .responsivePrefill == false

    // The download source is the app's own setting, and the selfcheck's suite
    // keeps it out of the user's real preferences.
    var source = SourceSettings()
    source.baseDirectory = "/tmp/hub-cache"
    source.hubEndpoint = "https://mirror.example/v1"
    source.save()
    passed = passed
        && SourceSettings.load() == source
        && source.environment == ["HF_HUB_CACHE": "/tmp/hub-cache",
                                  "HF_ENDPOINT": "https://mirror.example/v1"]

    // An unnamed configuration still names itself after the model it serves,
    // which is what the editor opens on and what its cleared field falls back
    // to; one with no model at all reads as untitled.
    var unnamed = ModelConfig(settings: ServerSettings.default)
    unnamed.settings.modelMode = .local
    unnamed.settings.modelDirectory = "/tmp/models/Qwen3.8-27B-GSQ-RCO-GGUF"
    var untitled = ModelConfig(settings: ServerSettings.default)
    untitled.settings.modelDirectory = ""
    passed = passed
        && unnamed.derivedName == "Qwen3.8-27B-GSQ-RCO-GGUF"
        && unnamed.displayName == "Qwen3.8-27B-GSQ-RCO-GGUF"
        && untitled.derivedName == nil
        && untitled.displayName == L10n.string("config.untitled")

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
    // A saved configuration from before the switch still opens on this
    // build's idle defaults instead of reloading the weights after 10 s.
    var legacyConfig = ServerSettings.default
    legacyConfig.settingsVersion = "3"
    legacyConfig.residencySeconds = ServerSettings.legacyResidencySeconds
    library.saveConfigs([ModelConfig(settings: legacyConfig)])
    let migratedConfig = library.loadConfigs().first?.settings
    passed = passed
        && migratedConfig?.residencySeconds == ServerSettings.First.residencySeconds
        && migratedConfig?.unloadIdleWeights == ServerSettings.First.unloadIdleWeights
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
            && carries("--expert-memory-grow", "on")
            && carries("--ple-memory", "2G")
            && carries("--sparse-kv-memory", "1200M")
            && carries("--verify-rows", "all")
            && carries("--max-context", "100K")
            && carries("--idle-release", "45")
            && carries("--residency-keepalive", "5m")
            && carries("--prefill-slice-ms", "750")
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
            // The eviction of an idle cache with no disk tier to spill to is
            // the engine's default; only the switch that turns it off is
            // passed.
            && carries("--cache-idle-evict", "off")
        // Turning the idle unload off keeps the weights resident.
        reloaded.unloadIdleWeights = false
        let resident = reloaded.launchArguments(launcherPath: "/tmp/launcher.py")
        passed = passed && carries(resident, "--idle-release", "off")
        reloaded.unloadIdleWeights = true
        // Turning responsive prefill off leaves the engine's chunk size.
        reloaded.responsivePrefill = false
        let chunked = reloaded.launchArguments(launcherPath: "/tmp/launcher.py")
        passed = passed && !chunked.contains("--prefill-slice-ms")
        reloaded.responsivePrefill = true
        reloaded.languageOnly = false
        let vision = reloaded.launchArguments(launcherPath: "/tmp/launcher.py")
        passed = passed
            && vision.contains("--mmproj")
            && vision.contains("/tmp/selfcheck-mmproj.gguf")
            && !vision.contains("--language-only")
        // An MLX target has no projector row of its own: the command carries
        // none for one, whatever the form still holds.
        reloaded.localKind = "mlx"
        let mlx = reloaded.launchArguments(launcherPath: "/tmp/launcher.py")
        passed = passed && !mlx.contains("--mmproj")
        reloaded.localKind = "gguf"
        // A Hub draft is named by its repository ID instead of a directory,
        // on either kind of target.
        reloaded.draftMode = "hub"
        reloaded.draftID = "org/dfash2-draft"
        // A folder outranks the stored ID, so the ID's own step clears it.
        reloaded.draftDirectory = ""
        let hubDraft = reloaded.launchArguments(launcherPath: "/tmp/launcher.py")
        passed = passed && carries(hubDraft, "--draft-model", "org/dfash2-draft")
        reloaded.draftMode = "folder"
        reloaded.draftDirectory = "/tmp/selfcheck-draft"
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