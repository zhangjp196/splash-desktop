import AppKit
import Darwin
import Foundation

/// What the live panel shows, taken from `/status` on a one-second loop.
struct LiveStatus: Equatable {
    var updated = false
    var ready = false
    var modelID: String?
    var contextTokens: Int?
    var decodeTokensPerSecond: Double?
    var prefillTokensPerSecond: Double?
    var draftAcceptanceRate: Double?
    var submitted = 0
    var completed = 0
    var failed = 0
    var cancelled = 0
    var pending = 0
    var pendingLimit = 0
    var currentBytes: UInt64?
    var peakBytes: UInt64?
    /// Host memory the engine actually occupies; the allocated bytes above
    /// are address space whose weight pages the host reclaims when idle.
    var physicalBytes: UInt64?
    var peakPhysicalBytes: UInt64?
    var cacheHitRate: Double?
    var kvDiskHitTokens: Int?
    /// The share of the tokens requests reused that came back from the disk
    /// tier: `kv_disk_hit_rate`, the token-level reading `hit_rate` cannot
    /// give, since it counts requests.
    var kvDiskHitRate: Double?
    /// What requests resumed from, the denominator of the rate above.
    var reusedTokens: Int?
    /// States a lookup found on the disk tier (`state.disk_hits`).
    var stateDiskHits: Int?
    var waiting = 0
    var waitingMemory = 0
    var waitingConcurrency = 0
    var suspended = 0
    var diskUsedBytes: UInt64?
    var diskCapacityBytes: UInt64?
    var idleOffloadPasses = 0
    var idleOffloadBytes: UInt64?
    var restarts = 0
    var capacityFailures = 0
    var metalFailures = 0
    /// The engine's fans as `/status` reports them, read-only: what the Mac is
    /// doing about the heat, which is what the thermal rungs trade against.
    var fansAvailable = false
    var fanSpeeds: [Int] = []
    var fanMinimums: [Int] = []
    var fanMaximums: [Int] = []
    /// The sensors the engine read, as it reported them, hottest first.
    var temperatures: [TemperatureReading] = []
    /// The prefill pacer: the duty the engine holds, the backoff the Mac's
    /// thermal state applies to it, and what the pacing has cost in waits.
    var dutyPercent: Double?
    var thermalScale: Double?
    var pacedWaits: Int?
    var pacedWaitMilliseconds: Double?
}

/// One temperature sensor's reading: the engine reads them from the SMC, so
/// they are what the fans themselves answer to.
struct TemperatureReading: Equatable, Identifiable {
    var sensor: String
    var celsius: Double
    var id: String { sensor }
}

/// The four temperatures the panel shows, each aggregated from the SMC keys
/// that belong to one part of the Mac: the CPU clusters (`Tp`, `Te`), the GPU
/// (`Tg`), the IO fabric (`Tf`), and the storage (`Ts`). A Mac that reports
/// none of a group's keys leaves that row without a reading.
enum TemperatureGroup: String, CaseIterable, Identifiable {
    case cpu, gpu, io, disk

    var id: String { rawValue }

    var title: String { L10n.string("thermal.sensor." + rawValue) }

    func contains(_ sensor: String) -> Bool {
        switch self {
        case .cpu: return sensor.hasPrefix("Tp") || sensor.hasPrefix("Te")
        case .gpu: return sensor.hasPrefix("Tg")
        case .io: return sensor.hasPrefix("Tf")
        case .disk: return sensor.hasPrefix("Ts")
        }
    }

    /// The hottest sensor of this group, or nil when the Mac reports none.
    static func peak(_ readings: [TemperatureReading], in group: TemperatureGroup) -> Double? {
        readings.lazy
            .filter { group.contains($0.sensor) }
            .map(\.celsius)
            .max()
    }

    /// The group whose hottest sensor is the hottest of the sample.
    static func hottest(
        _ readings: [TemperatureReading]
    ) -> (group: TemperatureGroup, celsius: Double)? {
        allCases
            .compactMap { group in peak(readings, in: group).map { (group, $0) } }
            .max { $0.celsius < $1.celsius }
    }
}

/// One `/status` fetch and the model id `/v1/models` reports, both decoded in
/// the background so the live panel never blocks the main thread.
struct LiveSnapshot {
    let status: [String: Any]
    let modelID: String?
}

/// Fetches the engine's `/status` and the served model id in the background.
actor LiveStatusFetcher {
    func fetch(port: Int) async -> LiveSnapshot? {
        guard let status = await json(at: "/status", port: port) else { return nil }
        var modelID: String?
        if let models = await json(at: "/v1/models", port: port),
           let data = models["data"] as? [[String: Any]],
           let first = data.first {
            modelID = first["id"] as? String
        }
        return LiveSnapshot(status: status, modelID: modelID)
    }

    private func json(at path: String, port: Int) async -> [String: Any]? {
        guard let url = URL(string: "http://127.0.0.1:\(port)\(path)") else { return nil }
        var request = URLRequest(url: url)
        request.timeoutInterval = 2
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200,
                  let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { return nil }
            return object
        } catch {
            return nil
        }
    }
}

/// Owns the server subprocess and the settings the control panel edits.
/// The app never loads a model itself: it starts `install/launcher.py serve`
/// from the embedded runtime, streams its output, watchs `/status`, and
/// opens the conversation in the web browser.
@MainActor
final class AppModel: ObservableObject {
    /// One instance for the window, the menu bar and the app delegate, so a
    /// server attached on launch is the one the delegate stops on quit.
    @MainActor static let shared = AppModel()

    enum Phase: Equatable {
        case idle
        case starting
        case ready
        case stopping
        case failed(String)
    }

    /// How the launcher is asked for a target: an upstream MLX or GGUF model
    /// whose draft the installer selects, or a local directory that names its
    /// target and draft (or the family's).
    enum ModelMode: String, CaseIterable, Identifiable, Codable {
        case upstream, local
        var id: String { rawValue }
    }

    @Published var phase: Phase = .idle
    @Published var log = ""
    @Published var contextTokens: Int?
    @Published var live = LiveStatus()

    // The first-launch values live in ServerSettings; the store replaces them
    // below before the window is shown.
    @Published var modelMode: ModelMode = .upstream
    @Published var modelID = ServerSettings.First.modelID
    @Published var modelDirectory = ""
    @Published var draftDirectory = ""
    @Published var port = ServerSettings.First.port
    @Published var lanAccess = false
    @Published var allowedHosts = ""
    @Published var languageOnly = false
    @Published var maxMemory = ""
    @Published var maxContext = ""
    @Published var expertMemory = ""
    @Published var pleMemory = ""
    @Published var sparseKvMemory = ""
    @Published var verifyRows = "auto"
    @Published var kvFormat = ServerSettings.First.kvFormat
    @Published var neuralEngine = true
    @Published var apiKey = ""
    @Published var servedNames = ""
    @Published var residencySeconds = ServerSettings.First.residencySeconds
    @Published var diskCache = ServerSettings.First.diskCache
    @Published var maxCacheDisk = ServerSettings.First.maxCacheDisk
    @Published var maxCacheDiskWrites = ServerSettings.First.maxCacheDiskWrites
    @Published var maxRequestSize = ""
    @Published var reasoningEffort = ""
    @Published var maxConcurrency = ServerSettings.First.maxConcurrency

    // The model library: the configurations the app has served, persisted as
    // JSON beside the settings.
    @Published var configs: [ModelConfig] = []
    @Published var activeConfigID: UUID?

    /// The interface language follows the system unless the user fixes it.
    enum InterfaceLanguage: String, CaseIterable, Identifiable {
        case followSystem, chinese, english
        var id: String { rawValue }
    }

    @Published var language: InterfaceLanguage = .followSystem

    private static let languageKey = "splash.interface-language"

    /// How loud the Mac may get while it serves: the server's own ladder
    /// (server/serve_options.py THERMAL_POLICIES), kept as one preference of
    /// the app rather than as part of a model configuration, because how loud
    /// a Mac may get is a property of the Mac. A server the app starts is
    /// given the rung, and a running engine's duty is changed through
    /// /v1/thermal, so a change applies without a restart.
    @Published var thermalRung: ThermalRung = .auto
    /// What the engine answered a change with that it would not take, shown on
    /// the thermal page until the next change goes through.
    @Published var thermalError: String?

    private static let thermalKey = "splash.thermal-rung"

    private var process: Process?
    private var poll: Task<Void, Never>?
    private var stopping = false
    /// True while stored settings are being applied, so the assignments do not
    /// read as edits and schedule a save of what was just loaded.
    private var restoring = false
    private var pendingSave: Task<Void, Never>?
    private let store = SettingsStore()
    private let libraryStore = LibraryStore()
    // A server the app found already running on the configured port (its PID
    // from the launcher's serve-<port>.lock), shown and stoppable like its own.
    var foreignPID: pid_t?
    private var offlineCount = 0
    private let statusFetcher = LiveStatusFetcher()

    let catalogIDs = Runtime.catalog

    init() {
        if let raw = UserDefaults.standard.string(forKey: Self.languageKey),
           let stored = InterfaceLanguage(rawValue: raw) {
            language = stored
        }
        if let raw = UserDefaults.standard.string(forKey: Self.thermalKey),
           let stored = ThermalRung(rawValue: raw) {
            thermalRung = stored
        }
        applyLanguage()
        let stored = store.hasStoredSettings
        applySettings(store.load())
        // The first launch records what the app opened with, so the settings
        // are on disk before the first edit rather than appearing from nowhere.
        if !stored { _ = saveSettings() }
        configs = libraryStore.loadConfigs()
        activeConfigID = libraryStore.loadActive()
        if configs.isEmpty {
            let seed = ModelConfig(settings: settings)
            configs = [seed]
            activeConfigID = seed.id
            libraryStore.saveConfigs(configs)
            libraryStore.saveActive(seed.id)
        }
        // A server left over from a previous session is detected and taken
        // over asynchronously.
        Task { @MainActor in
            await detectRunningServer()
        }
    }

    /// Keep the chosen interface language in effect and remembered.
    func applyLanguage() {
        switch language {
        case .followSystem: L10n.languageOverride = nil
        case .chinese: L10n.languageOverride = "zh-Hans"
        case .english: L10n.languageOverride = "en"
        }
        UserDefaults.standard.set(language.rawValue, forKey: Self.languageKey)
    }

    // MARK: Settings

    /// The settings the form currently shows, the value SQLite keeps.
    var settings: ServerSettings {
        var settings = ServerSettings()
        settings.modelMode = modelMode
        settings.modelID = modelID
        settings.modelDirectory = modelDirectory
        settings.draftDirectory = draftDirectory
        settings.port = port
        settings.lanAccess = lanAccess
        settings.allowedHosts = allowedHosts
        settings.languageOnly = languageOnly
        settings.kvFormat = kvFormat
        settings.neuralEngine = neuralEngine
        settings.maxMemory = maxMemory
        settings.maxContext = maxContext
        settings.expertMemory = expertMemory
        settings.pleMemory = pleMemory
        settings.sparseKvMemory = sparseKvMemory
        settings.verifyRows = verifyRows
        settings.residencySeconds = residencySeconds
        settings.diskCache = diskCache
        settings.maxCacheDisk = maxCacheDisk
        settings.maxCacheDiskWrites = maxCacheDiskWrites
        settings.maxRequestSize = maxRequestSize
        settings.apiKey = apiKey
        settings.servedNames = servedNames
        settings.reasoningEffort = reasoningEffort
        settings.maxConcurrency = maxConcurrency
        return settings
    }

    /// Changes when any setting does, so a view can autosave on a real edit
    /// rather than on every keystroke of the field being typed in.
    var settingsFingerprint: String { settings.fingerprint }

    /// Show the settings and write them out. The autosave calls this after a
    /// pause in typing; quitting and the defaults button call it directly.
    @discardableResult
    func saveSettings() -> Bool {
        pendingSave?.cancel()
        pendingSave = nil
        return store.save(settings)
    }

    /// Autosave, coalescing a burst of edits into one write.
    func scheduleSettingsSave() {
        pendingSave?.cancel()
        pendingSave = Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: 600_000_000)
            guard !Task.isCancelled else { return }
            self?.saveSettings()
        }
    }

    /// Return the form to the values the app opened with, leaving a running
    /// server alone: these are the next launch's settings, not this one's.
    func restoreDefaultSettings() {
        applySettings(.default)
        saveSettings()
        activeConfigID = nil
        libraryStore.saveActive(nil)
    }

    private func applySettings(_ settings: ServerSettings) {
        restoring = true
        defer { restoring = false }
        modelMode = settings.modelMode
        modelID = settings.modelID
        modelDirectory = settings.modelDirectory
        draftDirectory = settings.draftDirectory
        port = settings.port
        lanAccess = settings.lanAccess
        allowedHosts = settings.allowedHosts
        languageOnly = settings.languageOnly
        kvFormat = settings.kvFormat
        neuralEngine = settings.neuralEngine
        maxMemory = settings.maxMemory
        maxContext = settings.maxContext
        expertMemory = settings.expertMemory
        pleMemory = settings.pleMemory
        sparseKvMemory = settings.sparseKvMemory
        verifyRows = settings.verifyRows
        residencySeconds = settings.residencySeconds
        diskCache = settings.diskCache
        maxCacheDisk = settings.maxCacheDisk
        maxCacheDiskWrites = settings.maxCacheDiskWrites
        maxRequestSize = settings.maxRequestSize
        apiKey = settings.apiKey
        servedNames = settings.servedNames
        reasoningEffort = settings.reasoningEffort
        maxConcurrency = settings.maxConcurrency
    }

    /// Every edited field calls this, so one place decides when to write.
    func settingsEdited() {
        guard !restoring else { return }
        scheduleSettingsSave()
    }

    // MARK: Heat and noise

    /// The duty the running engine holds, once /status has reported one.
    var engineDutyPercent: Double? { live.updated ? live.dutyPercent : nil }

    /// The rung the engine holds, when it holds one of the ladder's: nil while
    /// no server runs, and for a duty between rungs, as a server started
    /// outside the app with --prefill-duty leaves.
    var engineRung: ThermalRung? { engineDutyPercent.flatMap(ThermalRung.init(duty:)) }

    /// Choose a rung: remember it for the next start, and take a running
    /// engine's duty to it now.
    func setThermalRung(_ rung: ThermalRung) {
        thermalRung = rung
        UserDefaults.standard.set(rung.rawValue, forKey: Self.thermalKey)
        thermalError = nil
        Task { @MainActor [weak self] in
            await self?.pushThermalDuty(rung.duty)
        }
    }

    /// Changes the duty of the running engine in place, through /v1/thermal.
    /// Nothing to do while no server is ready: the rung is the next start's
    /// too, which is what its command line carries.
    private func pushThermalDuty(_ duty: Double) async {
        guard isRunning, phase == .ready,
              let url = URL(string: "http://127.0.0.1:\(port)/v1/thermal") else { return }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 8
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if !apiKey.isEmpty {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        request.httpBody = try? JSONSerialization.data(
            withJSONObject: ["prefill_duty_percent": duty]
        )
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            if let http = response as? HTTPURLResponse, http.statusCode != 200 {
                thermalError = refusalMessage(data)
                    ?? L10n.format("thermal.error.status", http.statusCode)
            }
        } catch {
            thermalError = error.localizedDescription
        }
    }

    /// The server's own words for a refusal, when it wrote any.
    private func refusalMessage(_ data: Data) -> String? {
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let error = object["error"] as? [String: Any],
              let message = error["message"] as? String
        else { return nil }
        return message
    }

    // MARK: Model configurations

    /// Store a configuration and make it the one the server starts with.
    func save(_ config: ModelConfig) {
        if let index = configs.firstIndex(where: { $0.id == config.id }) {
            configs[index] = config
        } else {
            configs.append(config)
        }
        libraryStore.saveConfigs(configs)
        apply(config)
    }

    /// Make a configuration current: the form and the next start follow it.
    func apply(_ config: ModelConfig) {
        activeConfigID = config.id
        libraryStore.saveActive(config.id)
        applySettings(config.settings)
        saveSettings()
    }

    func removeConfig(_ config: ModelConfig) {
        configs.removeAll { $0.id == config.id }
        libraryStore.saveConfigs(configs)
        if activeConfigID == config.id {
            activeConfigID = nil
            libraryStore.saveActive(nil)
        }
    }

    /// A new, empty configuration to edit: the defaults, with no model chosen
    /// yet, so adding one starts from scratch instead of copying the current.
    func newConfig() -> ModelConfig {
        var defaults = ServerSettings.default
        defaults.modelID = ""
        defaults.modelDirectory = ""
        defaults.draftDirectory = ""
        return ModelConfig(settings: defaults)
    }

    // MARK: Derived state

    var isRunning: Bool {
        (process?.isRunning ?? false) || foreignPID != nil || phase == .ready
    }
    var isBusy: Bool { phase == .starting || phase == .stopping }

    // The chat is the server's own web page, opened in the browser; the API
    // address stays one keystroke away for tools that talk to it directly.
    var chatURL: URL? { URL(string: "http://127.0.0.1:\(port)/") }
    var apiBaseURL: String { "http://127.0.0.1:\(port)/v1" }

    func copyToClipboard(_ text: String) {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)
    }

    var canStart: Bool {
        guard !isBusy, !isRunning, Runtime.python != nil, Runtime.launcher != nil else {
            return false
        }
        switch modelMode {
        case .upstream:
            return !modelID.trimmingCharacters(in: .whitespaces).isEmpty
        case .local:
            // An MLX or GGUF target without a named draft has its family's
            // DFlash2 draft downloaded by the installer.
            return !modelDirectory.isEmpty
        }
    }

    var statusText: String {
        switch phase {
        case .idle: return L10n.string("status.stopped")
        case .starting: return L10n.string("status.starting")
        case .ready: return L10n.format("status.ready", port)
        case .stopping: return L10n.string("status.stopping")
        case .failed(let reason): return L10n.format("status.failed", reason)
        }
    }

    var menuBarSymbol: String {
        switch phase {
        case .ready: return "bolt.horizontal.circle.fill"
        case .starting, .stopping, .idle: return "bolt.horizontal.circle"
        case .failed: return "exclamationmark.triangle"
        }
    }

    // MARK: Lifecycle

    /// The command `serve` runs with, from the settings the panel holds:
    /// the launcher first, then each flag this configuration sets. The
    /// headless selfcheck reads it too, so a setting the panel added is also
    /// one the command carries.
    func launchArguments(launcherPath: String) -> [String] {
        var arguments = [launcherPath, "serve"]
        switch modelMode {
        case .upstream:
            arguments += ["--model", modelID.trimmingCharacters(in: .whitespaces)]
        case .local:
            arguments += ["--model-dir", modelDirectory]
            let draft = draftDirectory.trimmingCharacters(in: .whitespaces)
            if !draft.isEmpty { arguments += ["--draft-model", draft] }
        }
        arguments += ["--port", String(port)]
        // Loopback unless the configuration asks for the LAN; the extra Host
        // names are passed either way, since a tunnel or proxy reaches the
        // loopback listener under a name of its own.
        if lanAccess { arguments += ["--host", "0.0.0.0"] }
        for name in allowedHosts.split(separator: ",") {
            let trimmed = name.trimmingCharacters(in: .whitespaces)
            if !trimmed.isEmpty { arguments += ["--allowed-host", trimmed] }
        }
        // The panel owns the admission limit: it equals the server's own
        // default until edited, so passing it keeps the two in step.
        arguments += ["--queue-size", String(maxConcurrency)]
        if languageOnly { arguments.append("--language-only") }
        if !maxMemory.isEmpty { arguments += ["--max-memory", maxMemory] }
        if !maxContext.isEmpty { arguments += ["--max-context", maxContext] }
        // The stores the merge added: a MoE target's routed experts, a
        // Flash-Next target's PLE table, and the KV blocks read back from the
        // SSD. Empty leaves each to the server's automatic plan.
        for (flag, value) in [
            ("--expert-memory", expertMemory),
            ("--ple-memory", pleMemory),
            ("--sparse-kv-memory", sparseKvMemory),
        ] {
            let trimmed = value.trimmingCharacters(in: .whitespaces)
            if !trimmed.isEmpty { arguments += [flag, trimmed] }
        }
        // `auto` is the server's own default; `all` makes a draft's greedy and
        // seeded outputs repeat.
        if verifyRows != "auto" { arguments += ["--verify-rows", verifyRows] }
        if kvFormat != "int8" { arguments += ["--kv-format", kvFormat] }
        if !neuralEngine { arguments.append("--disable-ane") }
        if !apiKey.isEmpty { arguments += ["--api-key", apiKey] }
        for name in servedNames.split(separator: ",") {
            let trimmed = name.trimmingCharacters(in: .whitespaces)
            if !trimmed.isEmpty { arguments += ["--served-model-name", trimmed] }
        }
        // The SSD cache: on by default, its quota the server's own automatic
        // one unless the panel set a size, and a daily write budget that paces
        // it against the drive's life. The tier lives for the session: the
        // panel never asks for a persistent cache, so a restart starts from an
        // empty one. The switch turns the disk tier off outright
        // (`--max-cache-disk 0`), keeping cached prefixes in memory only; the
        // engine's idle spill is its own timer and needs no argument. A quota
        // of `0` still means off.
        let cacheQuota = maxCacheDisk.trimmingCharacters(in: .whitespaces)
        let cacheOn = diskCache && cacheQuota != "0"
        if cacheOn {
            if !cacheQuota.isEmpty { arguments += ["--max-cache-disk", cacheQuota] }
            let cacheWrites = maxCacheDiskWrites.trimmingCharacters(in: .whitespaces)
            if !cacheWrites.isEmpty {
                arguments += ["--max-cache-disk-writes", cacheWrites]
            }
        } else {
            arguments += ["--max-cache-disk", "0"]
        }
        if let seconds = Int(residencySeconds), seconds >= 1, seconds <= 86_400 {
            arguments += ["--idle-release", String(seconds)]
        }
        if !maxRequestSize.isEmpty { arguments += ["--max-request-size", maxRequestSize] }
        if !reasoningEffort.isEmpty {
            arguments += ["--default-reasoning-effort", reasoningEffort]
        }
        // How loud the Mac may get: the top rung passes nothing, leaving the
        // Mac's own thermal management in charge, which is the server's
        // default too. A change made while the server runs goes to the engine
        // through /v1/thermal instead.
        if thermalRung != .auto {
            arguments += ["--thermal-policy", thermalRung.rawValue]
        }
        return arguments
    }

    func start() {
        guard canStart else { return }
        guard let python = Runtime.python, let launcher = Runtime.launcher else {
            phase = .failed(L10n.string("error.no_runtime"))
            return
        }
        log = ""
        contextTokens = nil
        live = LiveStatus()
        stopping = false
        let arguments = launchArguments(launcherPath: launcher.path)

        // The engine reports settings a model cannot use ("--ple-memory
        // ignored: ... has no PLE table to stream"), and the log keeps those
        // lines; naming the command here makes a start that kept or dropped a
        // setting visible without reading the whole log. The API key is the
        // one argument the log must not carry.
        logLaunch(arguments)

        let process = Process()
        process.executableURL = python
        process.arguments = arguments
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONUNBUFFERED"] = "1"
        // Keep the signed bundle sealed: an import that writes a .pyc inside
        // it would add files the signature does not cover.
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["TRANSFORMERS_VERBOSITY"] = "error"
        process.environment = environment
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor in self?.append(text) }
        }
        process.terminationHandler = { [weak self] finished in
            Task { @MainActor in self?.exited(finished) }
        }

        do {
            try process.run()
        } catch {
            append("error: \(error.localizedDescription)\n")
            phase = .failed(error.localizedDescription)
            return
        }
        self.process = process
        phase = .starting
        pollStatus()
    }

    func stop() {
        if let process, process.isRunning {
            phase = .stopping
            stopping = true
            // The launcher execs into the server, so one SIGINT shuts it down.
            process.interrupt()
            poll?.cancel()
            poll = nil
            Task { @MainActor in
                try? await Task.sleep(nanoseconds: 8_000_000_000)
                if process.isRunning {
                    self.append(L10n.string("error.server_did_not_stop") + "\n")
                    process.terminate()
                }
            }
        } else if let pid = foreignPID ?? servePID(port: port) {
            foreignPID = pid
            phase = .stopping
            stopping = true
            kill(pid, SIGINT)
            // The polling loop watches for /status going silent and idles;
            // a hard kill guards a server that ignores SIGINT.
            Task { @MainActor in
                try? await Task.sleep(nanoseconds: 8_000_000_000)
                if let current = self.foreignPID, current == pid, kill(pid, 0) == 0 {
                    self.append(L10n.string("error.server_did_not_stop") + "\n")
                    kill(pid, SIGKILL)
                }
            }
        } else {
            phase = .idle
            foreignPID = nil
        }
    }

    func clearLog() {
        log = ""
    }

    func openInBrowser() {
        guard let url = chatURL else { return }
        NSWorkspace.shared.open(url)
    }

    // MARK: Attach and quit

    /// A server a previous session left running is taken over on launch: the
    /// header shows it ready with a working Stop, and the live panel reads it.
    func detectRunningServer() async {
        guard let snapshot = await statusFetcher.fetch(port: port),
              snapshot.status["maximum_context_tokens"] is Int else { return }
        foreignPID = servePID(port: port)
        if let tokens = snapshot.status["maximum_context_tokens"] as? Int {
            contextTokens = tokens
        }
        append(L10n.format("log.attached", foreignPID ?? 0) + "\n")
        phase = .ready
        pollStatus()
    }


    /// The launched command, with the API key's value held back: what a start
    /// passed to the server, for the log beside the engine's own notices.
    private func logLaunch(_ arguments: [String]) {
        var shown: [String] = []
        var redact = false
        for argument in arguments {
            if redact {
                shown.append("***")
                redact = false
                continue
            }
            shown.append(argument)
            redact = argument == "--api-key"
        }
        append("Launch: " + shown.joined(separator: " ") + "\n")
    }

    /// The PID the launcher records in `serve-<port>.lock`, or nil.
    private func servePID(port: Int) -> pid_t? {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let lock = home.appendingPathComponent(
            "Library/Application Support/Splash/runtime/serve-\(port).lock"
        )
        guard let data = try? Data(contentsOf: lock),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let pid = object["pid"] as? Int, pid > 0
        else { return nil }
        return pid_t(pid)
    }

    /// Stop every tracked server before the app leaves, without waiting on a
    /// second sleep: the app delegate calls this synchronously on quit.
    func stopOnQuit() {
        // Flush a debounced edit before the process goes away, so quitting in
        // the middle of typing still keeps what was typed.
        _ = saveSettings()
        poll?.cancel()
        poll = nil
        stopping = true
        if let process, process.isRunning {
            process.interrupt()
            let deadline = Date().addingTimeInterval(6)
            while process.isRunning && Date() < deadline {
                Thread.sleep(forTimeInterval: 0.05)
            }
            if process.isRunning { process.terminate() }
            self.process = nil
        }
        if let pid = foreignPID {
            kill(pid, SIGINT)
            let deadline = Date().addingTimeInterval(6)
            while kill(pid, 0) == 0 && Date() < deadline {
                Thread.sleep(forTimeInterval: 0.05)
            }
            if kill(pid, 0) == 0 { kill(pid, SIGKILL) }
            foreignPID = nil
        }
    }

    // MARK: Internals

    private func exited(_ finished: Process) {
        process = nil
        poll?.cancel()
        poll = nil
        if stopping {
            stopping = false
            phase = .idle
            return
        }
        let status = finished.terminationStatus
        if status == 0 {
            phase = .idle
        } else {
            phase = .failed(L10n.format("error.exited", status))
        }
    }

    private func append(_ text: String) {
        log += text
        // Bound the transcript; a long prefill emits progress lines forever.
        if log.count > 200_000 {
            log.removeFirst(log.count - 160_000)
        }
    }

    private func pollStatus() {
        poll?.cancel()
        poll = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                await self?.refreshStatus()
                try? await Task.sleep(nanoseconds: 1_000_000_000)
            }
        }
    }

    private func refreshStatus() async {
        guard let snapshot = await statusFetcher.fetch(port: port) else {
            offlineCount += 1
            // A foreign server we asked to stop has gone silent: done.
            if offlineCount >= 3, phase == .stopping {
                phase = .idle
                stopping = false
                foreignPID = nil
                poll?.cancel()
                poll = nil
            }
            return
        }
        offlineCount = 0
        applyStatus(snapshot)
    }

    private func applyStatus(_ snapshot: LiveSnapshot) {
        let object = snapshot.status
        var status = LiveStatus()
        status.modelID = snapshot.modelID
        status.ready = (object["ready"] as? Bool) ?? false
        if let tokens = object["maximum_context_tokens"] as? Int {
            status.contextTokens = tokens
            contextTokens = tokens
            if phase == .starting { phase = .ready }
        }
        if let metrics = object["metrics"] as? [String: Any] {
            status.decodeTokensPerSecond = metrics["decode_tokens_per_second"] as? Double
            status.prefillTokensPerSecond = metrics["prefill_tokens_per_second"] as? Double
            status.draftAcceptanceRate = metrics["draft_acceptance_rate"] as? Double
            status.capacityFailures = metrics["capacity_failures"] as? Int ?? 0
            status.metalFailures = metrics["metal_failures"] as? Int ?? 0
        }
        if let requests = object["requests"] as? [String: Any] {
            status.submitted = requests["submitted"] as? Int ?? 0
            status.completed = requests["completed"] as? Int ?? 0
            status.failed = requests["failed"] as? Int ?? 0
            status.cancelled = requests["cancelled"] as? Int ?? 0
        }
        if let transport = object["transport"] as? [String: Any] {
            status.pending = transport["pending"] as? Int ?? 0
            status.pendingLimit = transport["pending_limit"] as? Int ?? 0
            status.restarts = transport["restarts"] as? Int ?? 0
        }
        if let memory = object["memory_actual"] as? [String: Any] {
            status.currentBytes = memory["current_bytes"] as? UInt64
            status.peakBytes = memory["peak_bytes"] as? UInt64
            status.physicalBytes = memory["physical_bytes"] as? UInt64
            status.peakPhysicalBytes = memory["peak_physical_bytes"] as? UInt64
        }
        if let cache = object["cache"] as? [String: Any] {
            status.cacheHitRate = cache["hit_rate"] as? Double
            status.kvDiskHitTokens = cache["kv_disk_hit_tokens"] as? Int
            status.kvDiskHitRate = cache["kv_disk_hit_rate"] as? Double
            status.reusedTokens = cache["reused_tokens"] as? Int
        }
        if let state = object["state"] as? [String: Any] {
            status.stateDiskHits = state["disk_hits"] as? Int
        }
        if let admission = object["admission"] as? [String: Any] {
            status.waiting = admission["waiting"] as? Int ?? 0
            status.waitingMemory = admission["waiting_memory"] as? Int ?? 0
            status.waitingConcurrency = admission["waiting_concurrency"] as? Int ?? 0
            status.suspended = admission["suspended"] as? Int ?? 0
        }
        if let disk = object["disk"] as? [String: Any] {
            status.diskUsedBytes = disk["used_bytes"] as? UInt64
            status.diskCapacityBytes = disk["capacity_bytes"] as? UInt64
        }
        if let offload = object["idle_offload"] as? [String: Any] {
            status.idleOffloadPasses = offload["passes"] as? Int ?? 0
            status.idleOffloadBytes = offload["bytes"] as? UInt64
        }
        if let fans = object["fans"] as? [String: Any] {
            status.fansAvailable = fans["available"] as? Bool ?? false
            status.fanSpeeds = fans["speeds_rpm"] as? [Int] ?? []
            status.fanMinimums = fans["minimum_rpm"] as? [Int] ?? []
            status.fanMaximums = fans["maximum_rpm"] as? [Int] ?? []
            let readings = fans["temperatures"] as? [[String: Any]] ?? []
            status.temperatures = readings
                .compactMap { entry in
                    guard let sensor = entry["sensor"] as? String,
                          let celsius = entry["celsius"] as? Double
                    else { return nil }
                    return TemperatureReading(sensor: sensor, celsius: celsius)
                }
                .sorted { $0.celsius > $1.celsius }
        }
        if let throttle = object["throttle"] as? [String: Any] {
            status.dutyPercent = throttle["duty_percent"] as? Double
            status.thermalScale = throttle["thermal_scale"] as? Double
            status.pacedWaits = throttle["paced_waits"] as? Int
            status.pacedWaitMilliseconds = throttle["paced_wait_ms"] as? Double
        }
        status.updated = true
        live = status
    }
}
