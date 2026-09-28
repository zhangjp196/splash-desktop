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
    var cacheHitRate: Double?
    var kvDiskHitTokens: Int?
    var stateHitTokens: Int?
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

    /// How the launcher is asked for a target. A Splash package carries its
    /// own DFlash2 draft; an upstream MLX or GGUF model's draft is selected
    /// by the installer; a local directory names one unless it is a package.
    enum ModelMode: String, CaseIterable, Identifiable {
        case splash, upstream, local
        var id: String { rawValue }
    }

    @Published var phase: Phase = .idle
    @Published var log = ""
    @Published var contextTokens: Int?
    @Published var live = LiveStatus()

    @Published var modelMode: ModelMode = .upstream
    @Published var modelID = "mlx-community/Qwen3.8-27B-4bit"
    @Published var modelDirectory = ""
    @Published var draftDirectory = ""
    @Published var port = 8000
    @Published var languageOnly = false
    @Published var maxMemory = ""
    @Published var maxContext = ""
    @Published var kvFormat = "int8"
    @Published var apiKey = ""
    @Published var servedNames = ""
    @Published var maxCacheDisk = ""
    @Published var idleOffloadSeconds = "10"
    @Published var maxRequestSize = ""
    @Published var reasoningEffort = ""

    /// The interface language follows the system unless the user fixes it.
    enum InterfaceLanguage: String, CaseIterable, Identifiable {
        case followSystem, chinese, english
        var id: String { rawValue }
    }

    @Published var language: InterfaceLanguage = .followSystem

    private static let languageKey = "splash.interface-language"

    private var process: Process?
    private var poll: Task<Void, Never>?
    private var stopping = false
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
        applyLanguage()
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

    // MARK: Derived state

    var isRunning: Bool {
        (process?.isRunning ?? false) || foreignPID != nil || phase == .ready
    }
    var isBusy: Bool { phase == .starting || phase == .stopping }
    var chatURL: URL? { URL(string: "http://127.0.0.1:\(port)/") }

    var canStart: Bool {
        guard !isBusy, !isRunning, Runtime.python != nil, Runtime.launcher != nil else {
            return false
        }
        switch modelMode {
        case .splash, .upstream:
            return !modelID.trimmingCharacters(in: .whitespaces).isEmpty
        case .local:
            // A Splash package needs no draft; an MLX or GGUF target without
            // one is refused by the launcher with a clear line in the log.
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
        var arguments = [launcher.path, "serve"]
        switch modelMode {
        case .splash, .upstream:
            arguments += ["--model", modelID.trimmingCharacters(in: .whitespaces)]
        case .local:
            arguments += ["--model-dir", modelDirectory]
            let draft = draftDirectory.trimmingCharacters(in: .whitespaces)
            if !draft.isEmpty { arguments += ["--draft-model", draft] }
        }
        arguments += ["--port", String(port)]
        if languageOnly && modelMode != .splash { arguments.append("--language-only") }
        if !maxMemory.isEmpty { arguments += ["--max-memory", maxMemory] }
        if !maxContext.isEmpty { arguments += ["--max-context", maxContext] }
        if kvFormat != "int8" { arguments += ["--kv-format", kvFormat] }
        if !apiKey.isEmpty { arguments += ["--api-key", apiKey] }
        for name in servedNames.split(separator: ",") {
            let trimmed = name.trimmingCharacters(in: .whitespaces)
            if !trimmed.isEmpty { arguments += ["--served-model-name", trimmed] }
        }
        if !maxCacheDisk.isEmpty { arguments += ["--max-cache-disk", maxCacheDisk] }
        if let seconds = Int(idleOffloadSeconds), seconds > 0, seconds <= 86_400 {
            arguments += ["--idle-offload-seconds", String(seconds)]
        }
        if !maxRequestSize.isEmpty { arguments += ["--max-request-size", maxRequestSize] }
        if !reasoningEffort.isEmpty {
            arguments += ["--default-reasoning-effort", reasoningEffort]
        }

        let process = Process()
        process.executableURL = python
        process.arguments = arguments
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONUNBUFFERED"] = "1"
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

    func openInBrowser() {
        guard let url = chatURL else { return }
        NSWorkspace.shared.open(url)
    }

    func clearLog() {
        log = ""
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
        }
        if let cache = object["cache"] as? [String: Any] {
            status.cacheHitRate = cache["hit_rate"] as? Double
            status.kvDiskHitTokens = cache["kv_disk_hit_tokens"] as? Int
            status.stateHitTokens = cache["state_hit_tokens"] as? Int
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
        status.updated = true
        live = status
    }
}
