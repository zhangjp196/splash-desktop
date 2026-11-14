import AppKit
import Combine
import Darwin
import Foundation

/// Owns the server subprocess and the settings the control panel edits.
/// The app starts the bundled Rust server, streams its output, watches `/status`, and
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
    /// The server's transcript, one entry per line. Lines rather than one
    /// growing string: the log pane renders rows lazily, so a chunk of
    /// output appends instead of copying and re-laying-out the whole
    /// transcript, and the retention cap below counts rows. A reader that
    /// wants the old single block joins the lines.
    @Published var logLines: [String] = []
    /// The line still being written: what arrived since the last newline.
    /// Published beside the lines so a carriage-return progress bar grows
    /// in place instead of waiting for a newline that may be seconds away.
    @Published var logPendingLine = ""
    /// Characters the retained transcript holds, for the pane's size
    /// readout. Kept incrementally so the header never walks the lines.
    @Published var logCharacterCount = 0
    // The per-second live board is the engine's: the fields live on
    // `liveEngine` and are forwarded here, so the views keep reading the app
    // model they observe. The bridge in init re-announces each engine change
    // on this object's `objectWillChange`, so a poll wakes exactly the
    // observers it woke before the split.

    /// The poll, the parsed `/status`, the histories and the client scan.
    /// See `LiveEngine`.
    let liveEngine = LiveEngine()
    /// The menu bar's walking-pace snapshot of this model. The menu bar
    /// reads it, not this object, so a poll no longer rebuilds the menu
    /// host's items once a second (`MenuBarModel`).
    let menuBar = MenuBarModel()
    /// The live board's reading, polled once a second by the engine.
    var live: LiveStatus {
        get { liveEngine.live }
        set { liveEngine.live = newValue }
    }
    var contextTokens: Int? {
        get { liveEngine.contextTokens }
        set { liveEngine.contextTokens = newValue }
    }
    var decodeHistory: [Double] {
        get { liveEngine.decodeHistory }
        set { liveEngine.decodeHistory = newValue }
    }
    var prefillHistory: [Double] {
        get { liveEngine.prefillHistory }
        set { liveEngine.prefillHistory = newValue }
    }
    var acceptanceHistory: [Double] {
        get { liveEngine.acceptanceHistory }
        set { liveEngine.acceptanceHistory = newValue }
    }
    var memoryHistory: [Double] {
        get { liveEngine.memoryHistory }
        set { liveEngine.memoryHistory = newValue }
    }
    var recentEvents: [RequestEvent] {
        get { liveEngine.recentEvents }
        set { liveEngine.recentEvents = newValue }
    }
    var connectedClients: [ConnectedClient] {
        get { liveEngine.connectedClients }
        set { liveEngine.connectedClients = newValue }
    }
    var stage: RequestStage {
        get { liveEngine.stage }
        set { liveEngine.stage = newValue }
    }
    var readySince: Date? {
        get { liveEngine.readySince }
        set { liveEngine.readySince = newValue }
    }
    /// Deliberately off the published surface: the header badge reads it on a
    /// clock of its own. See `LiveEngine.statusUpdatedAt`.
    var statusUpdatedAt: Date? {
        get { liveEngine.statusUpdatedAt }
        set { liveEngine.statusUpdatedAt = newValue }
    }

    // The first-launch values live in ServerSettings; the store replaces them
    // below before the window is shown.
    @Published var source = SourceSettings.load()
    /// The download source is the app's own, so an edit stores it here rather
    /// than in a configuration.
    func saveSource() { source.save() }
    /// The model-owned share of the settings: what one configuration chooses
    /// about the model it serves. Everything the whole server shares lives in
    /// `global` below.
    @Published var modelMode: ModelMode = .upstream
    @Published var modelID = ServerSettings.First.modelID
    @Published var modelDirectory = ""
    @Published var localKind = ServerSettings.First.localKind
    @Published var draftDirectory = ""
    @Published var draftMode = ServerSettings.First.draftMode
    @Published var draftID = ""
    @Published var speculation = ServerSettings.First.speculation
    @Published var requireDraft = ServerSettings.First.requireDraft
    @Published var mmproj = ""
    @Published var languageOnly = false
    @Published var maxMemory = ServerSettings.First.maxMemory
    @Published var maxContext = ServerSettings.First.maxContext
    @Published var expertMemory = ""
    @Published var pleMemory = ""
    @Published var sparseKvMemory = ""
    @Published var verifyRows = "auto"
    /// How the engine runs this model: precision, admission, prefill pacing,
    /// and the thinking defaults a request may use. These belong to the
    /// configuration because they are chosen for the model it serves.
    @Published var kvFormat = ServerSettings.First.kvFormat
    @Published var stateFormat = ServerSettings.First.stateFormat
    @Published var neuralEngine = true
    @Published var maxConcurrency = ServerSettings.First.maxConcurrency
    @Published var decodeShare = ""
    @Published var responsivePrefill = ServerSettings.First.responsivePrefill
    @Published var prefillSliceMilliseconds = ServerSettings.First.prefillSliceMilliseconds
    @Published var reasoningEffort = ""
    @Published var thinkingTokenBudget = ""
    @Published var thinkingEndText = ""

    /// The settings every configuration shares, edited on the Settings page
    /// and stored with the rest of the settings. A configuration keeps a copy
    /// for a complete `ServerSettings`, but a start always takes this value.
    @Published var global = GlobalSettings()

    // The model library: the configurations the app has served, persisted as
    // JSON beside the settings.
    @Published var configs: [ModelConfig] = []
    @Published var activeConfigID: UUID?
    /// The configuration the running server was started with. It stays put
    /// while another configuration is edited, so "in use" never follows the
    /// editor; a model that is merely selected is not running anything.
    @Published private(set) var runningConfigID: UUID?

    /// The interface language follows the system unless the user fixes it.
    enum InterfaceLanguage: String, CaseIterable, Identifiable {
        case followSystem, chinese, english
        var id: String { rawValue }

        /// The name the toolbar menu and the Settings page show. The two
        /// language names stay in their own language, as language menus do.
        var title: String {
            switch self {
            case .followSystem: return L10n.string("language.follow")
            case .chinese: return "中文"
            case .english: return "English"
            }
        }
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
    /// How hard the smart mode works: the most prefill may keep the GPU busy
    /// when the Mac is cool, before the fans and temperatures pull it down.
    @Published var smartLevel: SmartLevel = .medium
    /// The level the fixed mode runs at: one of the ladder's rungs, the share
    /// of wall time prefill may keep the GPU busy, held steady while the fixed
    /// mode is chosen.
    @Published var fixedLevel: FixedLevel = .balanced
    /// What the engine answered a change with that it would not take, shown on
    /// the thermal page until the next change goes through.
    @Published var thermalError: String?

    private static let thermalKey = "splash.thermal-rung"
    private static let smartLevelKey = "splash.smart-level"
    private static let fixedLevelKey = "splash.fixed-level"

    // The process, the stopping flag and the last pushed duty are internal:
    // the engine reads the process's pid for the client scan, clears the flag
    // when a foreign server goes silent, and feeds the duty the smart mode
    // pushed.
    var process: Process?
    var stopping = false
    /// The last duty the smart mode pushed, so the poll pushes only when the
    /// fans and temperatures have moved it, not once a second forever.
    var lastSmartDuty: Double?
    /// True while stored settings are being applied, so the assignments do not
    /// read as edits and schedule a save of what was just loaded.
    private var restoring = false
    private var pendingSave: Task<Void, Never>?
    private let store = SettingsStore()
    private let libraryStore = LibraryStore()
    // A server the app found already running on the configured port (its PID
    // from the launcher's serve-<port>.lock), shown and stoppable like its own.
    var foreignPID: pid_t?
    /// Re-announces each of the engine's publishes on this object, so the
    /// views observing the app model see the poll's changes as before.
    private var liveEngineSubscription: AnyCancellable?
    /// Re-announces this object's publishes on the menu bar's snapshot, so
    /// the menu bar reads a walking pace of changes instead of every one.
    private var menuBarSubscription: AnyCancellable?

    let catalogIDs = Runtime.catalog

    init() {
        liveEngine.attach(to: self)
        liveEngineSubscription = liveEngine.objectWillChange.sink { [weak self] _ in
            self?.objectWillChange.send()
        }
        menuBarSubscription = objectWillChange.sink { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.menuBar.scheduleRefresh()
            }
        }
        if let raw = UserDefaults.standard.string(forKey: Self.languageKey),
           let stored = InterfaceLanguage(rawValue: raw) {
            language = stored
        }
        let hasStoredSettings = store.hasStoredSettings
        // The fixed level first: a thermal preference stored before the modes
        // replaced the ladder named one of its rungs, and that rung becomes
        // the same level of the fixed mode rather than losing the choice to
        // the default. The migrated pair is written back, so it runs once.
        if let raw = UserDefaults.standard.string(forKey: Self.fixedLevelKey),
           let stored = FixedLevel(rawValue: raw) {
            fixedLevel = stored
        }
        let storedThermal = UserDefaults.standard.string(forKey: Self.thermalKey)
        if let raw = storedThermal {
            if let stored = ThermalRung(rawValue: raw) {
                thermalRung = stored
            } else if let rung = FixedLevel(rawValue: raw) {
                thermalRung = .fixed
                fixedLevel = rung
                UserDefaults.standard.set(rung.rawValue, forKey: Self.fixedLevelKey)
                UserDefaults.standard.set(ThermalRung.fixed.rawValue, forKey: Self.thermalKey)
            }
        } else if hasStoredSettings {
            // An existing install that never chose a mode keeps the system
            // default it has been running; only a fresh install opens on the
            // smart mode, so an update does not change a running Mac's heat
            // and speed behind the user's back.
            thermalRung = .auto
            UserDefaults.standard.set(ThermalRung.auto.rawValue, forKey: Self.thermalKey)
        } else {
            // A fresh install opens on Smart/High: full speed while the Mac is
            // cool, backing off as the fans and sensors climb.
            thermalRung = .smart
            if UserDefaults.standard.string(forKey: Self.smartLevelKey) == nil {
                smartLevel = .high
                UserDefaults.standard.set(SmartLevel.high.rawValue, forKey: Self.smartLevelKey)
            }
            UserDefaults.standard.set(ThermalRung.smart.rawValue, forKey: Self.thermalKey)
        }
        if let raw = UserDefaults.standard.string(forKey: Self.smartLevelKey),
           let stored = SmartLevel(rawValue: raw) {
            smartLevel = stored
        }
        applyLanguage()
        // The store keeps the whole command's settings, so it is the first
        // source for both shares. The active configuration then names the
        // model share, while the app-wide share stays what the store holds.
        let stored = store.load()
        applyGlobalSettings(stored)
        applyModelSettings(stored)
        // The first launch records what the app opened with, so the settings
        // are on disk before the first edit rather than appearing from nowhere.
        if !hasStoredSettings { _ = saveSettings() }
        configs = libraryStore.loadConfigs()
        activeConfigID = libraryStore.loadActive()
        if configs.isEmpty {
            let seed = ModelConfig(settings: settings)
            configs = [seed]
            activeConfigID = seed.id
            libraryStore.saveConfigs(configs)
            libraryStore.saveActive(seed.id)
        } else if let active = configs.first(where: { $0.id == activeConfigID }) {
            applyModelSettings(active.settings)
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

    /// The settings the form currently shows, the value SQLite keeps: the
    /// model's own fields followed by the app-wide share that every launch
    /// takes from the Settings page.
    var settings: ServerSettings {
        var settings = ServerSettings()
        settings.modelMode = modelMode
        settings.modelID = modelID
        settings.modelDirectory = modelDirectory
        settings.localKind = localKind
        settings.draftDirectory = draftDirectory
        settings.draftMode = draftMode
        settings.draftID = draftID
        settings.speculation = speculation
        settings.requireDraft = requireDraft
        settings.mmproj = mmproj
        settings.languageOnly = languageOnly
        settings.maxMemory = maxMemory
        settings.maxContext = maxContext
        settings.expertMemory = expertMemory
        settings.pleMemory = pleMemory
        settings.sparseKvMemory = sparseKvMemory
        settings.verifyRows = verifyRows
        settings.kvFormat = kvFormat
        settings.stateFormat = stateFormat
        settings.neuralEngine = neuralEngine
        settings.maxConcurrency = maxConcurrency
        settings.decodeShare = decodeShare
        settings.responsivePrefill = responsivePrefill
        settings.prefillSliceMilliseconds = prefillSliceMilliseconds
        settings.reasoningEffort = reasoningEffort
        settings.thinkingTokenBudget = thinkingTokenBudget
        settings.thinkingEndText = thinkingEndText
        global.applying(to: &settings)
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
        applyGlobalSettings(.default)
        applyModelSettings(.default)
        saveSettings()
        activeConfigID = nil
        libraryStore.saveActive(nil)
    }

    /// Return only the app-wide settings to their first-launch values, leaving
    /// the model the user picked and every configuration alone. This is the
    /// Settings page's reset: a model configuration has no reset of its own.
    func restoreDefaultGlobalSettings() {
        applyGlobalSettings(.default)
        saveSettings()
    }

    /// Take the app-wide share of a complete configuration. The store is the
    /// first source at launch, and Reset returns this share to its defaults
    /// without touching the model the user picked.
    func applyGlobalSettings(_ settings: ServerSettings) {
        global = GlobalSettings(settings)
    }

    /// Take the model-owned share of a complete configuration, leaving the
    /// app-wide share as the Settings page last left it.
    private func applyModelSettings(_ settings: ServerSettings) {
        restoring = true
        defer { restoring = false }
        modelMode = settings.modelMode
        modelID = settings.modelID
        modelDirectory = settings.modelDirectory
        localKind = settings.localKind
        draftDirectory = settings.draftDirectory
        draftMode = settings.draftMode
        draftID = settings.draftID
        speculation = settings.speculation
        requireDraft = settings.requireDraft
        mmproj = settings.mmproj
        languageOnly = settings.languageOnly
        maxMemory = settings.maxMemory
        maxContext = settings.maxContext
        expertMemory = settings.expertMemory
        pleMemory = settings.pleMemory
        sparseKvMemory = settings.sparseKvMemory
        verifyRows = settings.verifyRows
        kvFormat = settings.kvFormat
        stateFormat = settings.stateFormat
        neuralEngine = settings.neuralEngine
        maxConcurrency = settings.maxConcurrency
        decodeShare = settings.decodeShare
        responsivePrefill = settings.responsivePrefill
        prefillSliceMilliseconds = settings.prefillSliceMilliseconds
        reasoningEffort = settings.reasoningEffort
        thinkingTokenBudget = settings.thinkingTokenBudget
        thinkingEndText = settings.thinkingEndText
    }

    /// Every edited app-wide field calls this, so one place decides when to
    /// write. A model configuration is saved by its own editor instead.
    func settingsEdited() {
        guard !restoring else { return }
        scheduleSettingsSave()
    }

    // MARK: Heat and noise

    /// The duty the running engine holds, once /status has reported one.
    var engineDutyPercent: Double? { live.updated ? live.dutyPercent : nil }

    /// The mode the engine holds, when it holds one of the ladder's: nil while
    /// no server runs, and for a duty between the fixed levels, as a server
    /// started outside the app with --prefill-duty leaves. A duty one of the
    /// fixed levels holds marks the fixed mode; a top duty marks the system
    /// default.
    var engineRung: ThermalRung? {
        guard let duty = engineDutyPercent else { return nil }
        if FixedLevel(duty: duty) != nil { return .fixed }
        if abs(duty - 100) < 0.5 { return .auto }
        return nil
    }

    /// Choose a mode: remember it for the next start, and take a running
    /// engine's duty to it now. `smart` has no fixed duty: it is pushed each
    /// second from the fans and temperatures, so a change to its level is
    /// remembered and applied on the next poll. `fixed` takes its level's
    /// duty.
    func setThermalRung(_ rung: ThermalRung) {
        thermalRung = rung
        UserDefaults.standard.set(rung.rawValue, forKey: Self.thermalKey)
        thermalError = nil
        Task { @MainActor [weak self] in
            guard let self else { return }
            switch rung {
            case .auto: await pushThermalDuty(rung.duty ?? 100)
            case .smart: await pushSmartDuty()
            case .fixed: await pushThermalDuty(fixedLevel.duty)
            }
        }
    }

    /// Remember the smart level and, while the smart mode is chosen and a
    /// server runs, take the engine's duty to it now.
    func setSmartLevel(_ level: SmartLevel) {
        smartLevel = level
        UserDefaults.standard.set(level.rawValue, forKey: Self.smartLevelKey)
        thermalError = nil
        Task { @MainActor [weak self] in
            guard let self, thermalRung == .smart else { return }
            await pushSmartDuty()
        }
    }

    /// Remember the fixed level and, while the fixed mode is chosen and a
    /// server runs, take the engine's duty to it now.
    func setFixedLevel(_ level: FixedLevel) {
        fixedLevel = level
        UserDefaults.standard.set(level.rawValue, forKey: Self.fixedLevelKey)
        thermalError = nil
        Task { @MainActor [weak self] in
            guard let self, thermalRung == .fixed else { return }
            await pushThermalDuty(level.duty)
        }
    }

    /// The duty the smart mode asks for now: its level's ceiling while the fans
    /// are still inside the band that level allows and the sensors are cool,
    /// falling toward its floor as the fans leave that band or the hottest
    /// sensor climbs. The fans are the Mac's own answer to the heat, so a Mac
    /// already spinning them is asked to feed less work; a Mac without fans
    /// (a fanless Air) is judged by its sensors alone. The result is clamped to
    /// the engine's range, so it is always one the engine will take.
    func smartDuty() -> Double {
        let live = self.live
        let level = smartLevel
        // How far the fans are past what this level allows, 0 inside the band
        // and 1 at their top speed.
        let fanShare: Double
        if live.fansAvailable, let maxRpm = live.fanMaximums.max(), maxRpm > 0 {
            let fastest = live.fanSpeeds.max() ?? 0
            fanShare = min(1.0, Double(fastest) / Double(maxRpm))
        } else {
            fanShare = 0.0
        }
        let band = level.fanShare
        let fanHead = min(1.0, max(0.0, (fanShare - band) / max(0.01, 1.0 - band)))
        // The sensors: cool below 65 °C, at their hottest the runtime reads to
        // at 90 °C.
        let heatHead: Double
        if let hottest = TemperatureGroup.hottest(live.temperatures)?.celsius {
            heatHead = min(1.0, max(0.0, (hottest - 65.0) / 25.0))
        } else {
            heatHead = 0.0
        }
        // Whichever of the two is further along holds the duty down, so a hot
        // Mac with lazy fans is paced as hard as a cool one with loud fans.
        let backoff = max(fanHead, heatHead)
        let duty = level.ceiling - (level.ceiling - level.floor) * backoff
        return min(100.0, max(10.0, duty))
    }

    /// Changes the running engine's duty to what the smart mode asks for now.
    /// Nothing to do while no server runs: the level is the next start's too.
    private func pushSmartDuty() async {
        guard thermalRung == .smart else { return }
        await pushThermalDuty(smartDuty())
    }

    /// Changes the duty of the running engine in place, through /v1/thermal.
    /// Nothing to do while no server is ready: the rung is the next start's
    /// too, which is what its command line carries.
    func pushThermalDuty(_ duty: Double) async {
        guard isRunning, phase == .ready,
              let url = URL(string: "http://127.0.0.1:\(global.port)/v1/thermal") else { return }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 8
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if !global.apiKey.isEmpty {
            request.setValue("Bearer \(global.apiKey)", forHTTPHeaderField: "Authorization")
        }
        request.httpBody = try? JSONSerialization.data(
            withJSONObject: ["prefill_duty_percent": duty]
        )
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            if let http = response as? HTTPURLResponse, http.statusCode != 200 {
                thermalError = refusalMessage(data)
                    ?? L10n.format("thermal.error.status", http.statusCode)
                lastSmartDuty = nil
            } else {
                // A change that goes through clears a refusal an earlier
                // push got while the server was busy.
                thermalError = nil
                lastSmartDuty = duty
            }
        } catch {
            thermalError = error.localizedDescription
            lastSmartDuty = nil
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

    /// Store a configuration and make it the one the server starts with. The
    /// app-wide share is copied in so the saved configuration stays a complete
    /// command; a start still reads the Settings page's current value.
    func save(_ config: ModelConfig) {
        var saved = config
        global.applying(to: &saved.settings)
        if let index = configs.firstIndex(where: { $0.id == saved.id }) {
            configs[index] = saved
        } else {
            configs.append(saved)
        }
        libraryStore.saveConfigs(configs)
        apply(saved)
    }

    /// Make a configuration current: the model share follows it, while the
    /// app-wide share stays where the Settings page left it.
    func apply(_ config: ModelConfig) {
        activeConfigID = config.id
        libraryStore.saveActive(config.id)
        applyModelSettings(config.settings)
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
        defaults.mmproj = ""
        global.applying(to: &defaults)
        return ModelConfig(settings: defaults)
    }

    // MARK: Derived state

    var isRunning: Bool {
        (process?.isRunning ?? false) || foreignPID != nil || phase == .ready
    }
    var isBusy: Bool { phase == .starting || phase == .stopping }

    // The chat is the server's own web page. The app embeds it in embed mode,
    // where the page keeps only the conversation; the browser gets the full
    // page, history sidebar and all.
    var chatURL: URL? { URL(string: "http://127.0.0.1:\(global.port)/") }
    var chatEmbedURL: URL? { URL(string: "http://127.0.0.1:\(global.port)/?embed=1") }
    var apiBaseURL: String { "http://127.0.0.1:\(global.port)/v1" }

    var activeConfig: ModelConfig? {
        configs.first { $0.id == activeConfigID }
    }

    /// The configuration behind the server that is actually running, or nil
    /// when the server was attached from outside the app.
    var runningConfig: ModelConfig? {
        configs.first { $0.id == runningConfigID }
    }

    func copyToClipboard(_ text: String) {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)
        NotificationCenter.default.post(
            name: .splashToast,
            object: L10n.string("toast.copied")
        )
    }

    var canStart: Bool {
        guard !isBusy, !isRunning, Runtime.server != nil else {
            return false
        }
        switch modelMode {
        case .upstream:
            return !modelID.trimmingCharacters(in: .whitespaces).isEmpty
        case .local:
            return !modelDirectory.isEmpty
        }
    }

    var statusText: String {
        switch phase {
        case .idle: return L10n.string("status.stopped")
        case .starting: return L10n.string("status.starting")
        case .ready: return L10n.format("status.ready", global.port)
        case .stopping: return L10n.string("status.stopping")
        case .failed(let reason): return L10n.format("status.failed", reason)
        }
    }

    var menuBarSymbol: String {
        switch phase {
        case .ready:
            switch stage {
            case .idle: return "bolt.horizontal.circle.fill"
            case .queued: return "hourglass.circle.fill"
            case .prefilling: return "arrow.down.circle.fill"
            case .decoding: return "bolt.circle.fill"
            }
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
        let profile = ModelProfile.of(modelID: modelID, directory: modelDirectory)
        var arguments = [launcherPath, "serve"]
        switch modelMode {
        case .upstream:
            arguments += ["--model", modelID.trimmingCharacters(in: .whitespaces)]
        case .local:
            arguments += ["--model-dir", modelDirectory]
            // Text only loads no tower, so it names no projector: the launcher
            // refuses the two together, and the form keeps the path for the
            // configuration that turns vision back on. Only a GGUF target has
            // one; an MLX target is vision or text as its own files say.
            let projector = mmproj.trimmingCharacters(in: .whitespaces)
            if localKind == "gguf" && !projector.isEmpty && !languageOnly && profile.vision {
                arguments += ["--mmproj", projector]
            }
        }
        // Whether decode may propose tokens, and whether a missing draft is
        // fatal. `auto` and `dflash2` use the draft the configuration names.
        if speculation != "auto" {
            arguments += ["--speculation", speculation]
        }
        if requireDraft {
            arguments.append("--require-draft")
        }
        // The DFlash2 draft is a local folder, like any other model the panel
        // serves. A repository ID an older configuration saved still names
        // its own download until an edit moves it to a folder.
        let draft = draftDirectory.trimmingCharacters(in: .whitespaces)
        if !draft.isEmpty {
            arguments += ["--draft-model", draft]
        } else if draftMode == "hub" {
            let stored = draftID.trimmingCharacters(in: .whitespaces)
            if !stored.isEmpty { arguments += ["--draft-model", stored] }
        }
        arguments += ["--port", String(global.port)]
        // Loopback unless the Settings page asks for the LAN; the extra Host
        // names are passed either way, since a tunnel or proxy reaches the
        // loopback listener under a name of its own.
        if global.lanAccess { arguments += ["--host", "0.0.0.0"] }
        for name in global.allowedHosts.split(separator: ",") {
            let trimmed = name.trimmingCharacters(in: .whitespaces)
            if !trimmed.isEmpty { arguments += ["--allowed-host", trimmed] }
        }
        for origin in global.allowedOrigins.split(separator: ",") {
            let trimmed = origin.trimmingCharacters(in: .whitespaces)
            if !trimmed.isEmpty { arguments += ["--allowed-origin", trimmed] }
        }
        // The panel owns the admission limit: it equals the server's own
        // default until edited, so passing it keeps the two in step.
        arguments += ["--queue-size", String(maxConcurrency)]
        if languageOnly && profile.vision { arguments.append("--language-only") }
        if !maxMemory.isEmpty { arguments += ["--max-memory", maxMemory] }
        if !maxContext.isEmpty { arguments += ["--max-context", maxContext] }
        // The stores the merge added: a MoE target's routed experts, a
        // Flash-Next target's PLE table, and the KV blocks read back from the
        // SSD. Empty leaves each to the server's automatic plan, and a model
        // that cannot use one passes nothing even if another model set it.
        for (flag, value, supported) in [
            ("--expert-memory", expertMemory, profile.experts),
            ("--ple-memory", pleMemory, profile.ple),
            ("--sparse-kv-memory", sparseKvMemory, profile.sparseKv),
        ] where supported {
            let trimmed = value.trimmingCharacters(in: .whitespaces)
            if !trimmed.isEmpty { arguments += [flag, trimmed] }
        }
        // `auto` is the server's own default; `all` makes a draft's greedy and
        // seeded outputs repeat.
        if verifyRows != "auto" { arguments += ["--verify-rows", verifyRows] }
        if kvFormat != "int8" { arguments += ["--kv-format", kvFormat] }
        if stateFormat != "fp32" { arguments += ["--state-format", stateFormat] }
        if !neuralEngine { arguments.append("--disable-ane") }
        if !global.apiKey.isEmpty { arguments += ["--api-key", global.apiKey] }
        for name in global.servedNames.split(separator: ",") {
            let trimmed = name.trimmingCharacters(in: .whitespaces)
            if !trimmed.isEmpty { arguments += ["--served-model-name", trimmed] }
        }
        // The SSD cache: on by default, its quota the server's own automatic
        // one unless the panel set a size, and a daily write budget that paces
        // it against the drive's life. A coding conversation keeps its prefix
        // across an app restart through the persistent cache, so the next
        // launch does not prefill the system prompt and history again. The
        // switch turns the disk tier off outright (`--max-cache-disk 0`),
        // keeping cached prefixes in memory only; the engine's idle spill is
        // its own timer and needs no argument. A quota of `0` still means off.
        let cacheQuota = global.maxCacheDisk.trimmingCharacters(in: .whitespaces)
        let cacheOn = global.diskCache && cacheQuota != "0"
        let idleSpill = global.cacheIdleSpill.trimmingCharacters(in: .whitespaces)
        if !idleSpill.isEmpty { arguments += ["--cache-idle-spill", idleSpill] }
        // A disk tier that is off leaves the pass nothing to spill to; the
        // switch says whether it evicts the idle prefixes instead. On, the
        // engine's own default, passes nothing.
        if !global.cacheIdleEvict { arguments += ["--cache-idle-evict", "off"] }
        if cacheOn {
            arguments.append("--persistent-cache")
            if !cacheQuota.isEmpty { arguments += ["--max-cache-disk", cacheQuota] }
            let directory = global.cacheDir.trimmingCharacters(in: .whitespaces)
            if !directory.isEmpty { arguments += ["--cache-dir", directory] }
            let cacheWrites = global.maxCacheDiskWrites.trimmingCharacters(in: .whitespaces)
            if !cacheWrites.isEmpty {
                arguments += ["--max-cache-disk-writes", cacheWrites]
            }
        } else {
            arguments += ["--max-cache-disk", "0"]
        }
        // How the prefix cache picks what a reclaim gives up. The default
        // preset passes nothing, leaving the engine's own LRU; the custom one
        // passes its policy and protected tier.
        let eviction = ServerSettings.cacheEviction(
            preset: global.cacheEvictionPreset, policy: global.cacheEvictionPolicy,
            protect: global.cacheEvictionProtect)
        if eviction.policy != "lru" {
            arguments += ["--cache-eviction", eviction.policy]
        }
        if eviction.protect != 0 {
            arguments += ["--cache-eviction-protect", String(eviction.protect)]
        }
        // The idle unload switch: off keeps the weights resident, on releases
        // them after the duration below. An empty or unparsable duration uses
        // the first-launch ten minutes rather than leaving the engine's timer
        // at a value the panel does not show.
        if global.unloadIdleWeights {
            let stored = Int(global.residencySeconds)
            let seconds = stored.map { (1...86_400).contains($0) ? $0 : 600 } ?? 600
            arguments += ["--idle-release", String(seconds)]
        } else {
            arguments += ["--idle-release", "off"]
        }
        // The buffers' own keep-alive, apart from the weights' release:
        // empty follows --idle-release, the engine's behavior without the
        // option.
        let keepAlive = global.residencyKeepAlive.trimmingCharacters(in: .whitespaces)
        if !keepAlive.isEmpty { arguments += ["--residency-keepalive", keepAlive] }
        if !global.maxRequestSize.isEmpty {
            arguments += ["--max-request-size", global.maxRequestSize]
        }
        let timeout = global.requestTimeout.trimmingCharacters(in: .whitespaces)
        if !timeout.isEmpty { arguments += ["--request-timeout", timeout] }
        let pixels = global.maxImagePixels.trimmingCharacters(in: .whitespaces)
        if !pixels.isEmpty && profile.vision {
            arguments += ["--max-image-pixels", pixels]
        }
        let share = decodeShare.trimmingCharacters(in: .whitespaces)
        if !share.isEmpty { arguments += ["--decode-share", share] }
        // A long prompt otherwise runs one prefill command for seconds. The
        // slice leaves gaps for other GPU work; the server and CLI keep the
        // engine's chunk size unless this switch asks for it.
        if responsivePrefill {
            let stored = Int(prefillSliceMilliseconds)
            let milliseconds = stored.map { (1...10_000).contains($0) ? $0 : 1_000 } ?? 1_000
            arguments += ["--prefill-slice-ms", String(milliseconds)]
        }
        if global.allowIdleSleep { arguments.append("--allow-idle-sleep") }
        if !reasoningEffort.isEmpty && profile.thinking {
            arguments += ["--default-reasoning-effort", reasoningEffort]
        }
        let budget = thinkingTokenBudget.trimmingCharacters(in: .whitespaces)
        if !budget.isEmpty && profile.thinking {
            arguments += ["--thinking-token-budget", budget]
        }
        let endText = thinkingEndText.trimmingCharacters(in: .whitespaces)
        if !endText.isEmpty && profile.thinking {
            arguments += ["--thinking-end-text", endText]
        }
        if global.announceServedName &&
            !global.servedNames.trimmingCharacters(in: .whitespaces).isEmpty {
            arguments.append("--announce-served-name")
        }
        // How loud the Mac may get: the system default passes nothing, leaving
        // the Mac's own thermal management in charge, which is the server's
        // default too. `smart` starts at its level's ceiling, and the poll
        // takes it to what the fans and temperatures ask for each second.
        // `fixed` starts at its level, one of the ladder's rungs, which the
        // server names as a policy. A change made while the server runs goes
        // to the engine through /v1/thermal instead.
        if thermalRung == .smart {
            arguments += ["--prefill-duty", String(Int(smartLevel.ceiling))]
        } else if thermalRung == .fixed {
            arguments += ["--thermal-policy", fixedLevel.rawValue]
        }
        return arguments
    }

    func start() {
        guard canStart else { return }
        guard let server = Runtime.server else {
            phase = .failed(L10n.string("error.no_runtime"))
            return
        }
        resetLog()
        contextTokens = nil
        live = LiveStatus()
        decodeHistory = []
        prefillHistory = []
        acceptanceHistory = []
        memoryHistory = []
        recentEvents = []
        connectedClients = []
        liveEngine.clientTick = 0
        stage = .idle
        readySince = nil
        statusUpdatedAt = nil
        runningConfigID = nil
        stopping = false
        let arguments = launchArguments(launcherPath: server.path)

        // The engine reports settings a model cannot use ("--ple-memory
        // ignored: ... has no PLE table to stream"), and the log keeps those
        // lines; naming the command here makes a start that kept or dropped a
        // setting visible without reading the whole log. The API key is the
        // one argument the log must not carry.
        logLaunch(arguments)

        let process = Process()
        process.executableURL = server
        var serverArguments = Array(arguments.dropFirst())
        var environment = ProcessInfo.processInfo.environment
        if let index = serverArguments.firstIndex(of: "--api-key") {
            environment["SPLASH_API_KEY"] = serverArguments[index + 1]
            serverArguments.removeSubrange(index...index + 1)
        }
        if environment["SPLASH_PDFIUM_PATH"] == nil,
           let pdfium = Runtime.pdfiumLibrary {
            environment["SPLASH_PDFIUM_PATH"] = pdfium.path
        }
        process.arguments = serverArguments
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
        runningConfigID = activeConfigID
        liveEngine.pollStatus()
    }

    func stop() {
        if let process, process.isRunning {
            phase = .stopping
            stopping = true
            // The launcher execs into the server, so one SIGINT shuts it down.
            process.interrupt()
            liveEngine.stopPolling()
            Task { @MainActor in
                try? await Task.sleep(nanoseconds: 8_000_000_000)
                if process.isRunning {
                    self.append(L10n.string("error.server_did_not_stop") + "\n")
                    process.terminate()
                }
            }
        } else if let pid = foreignPID ?? servePID(port: global.port) {
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
            settle()
        }
    }

    func clearLog() {
        resetLog()
    }

    func openInBrowser() {
        guard let url = chatURL else { return }
        NSWorkspace.shared.open(url)
    }

    // MARK: Attach and quit

    /// A server a previous session left running is taken over on launch: the
    /// header shows it ready with a working Stop, and the live panel reads it.
    func detectRunningServer() async {
        guard let snapshot = await liveEngine.statusFetcher.fetch(port: global.port),
              snapshot.status["maximum_context_tokens"] is Int else { return }
        foreignPID = servePID(port: global.port)
        if let tokens = snapshot.status["maximum_context_tokens"] as? Int {
            contextTokens = tokens
        }
        append(L10n.format("log.attached", foreignPID ?? 0) + "\n")
        phase = .ready
        liveEngine.pollStatus()
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

    /// The PID the launcher records in `serve-<port>.lock`, or nil. The
    /// engine reads it too, to leave the server's own connections out of the
    /// client scan.
    func servePID(port: Int) -> pid_t? {
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
        liveEngine.stopPolling()
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
        liveEngine.stopPolling()
        settle()
        if stopping {
            stopping = false
            phase = .idle
            return
        }
        let status = finished.terminationStatus
        if status == 0 {
            phase = .idle
        } else if let error = logLines.last(where: { $0.hasPrefix("splash-server: ") }) {
            phase = .failed(String(error.dropFirst("splash-server: ".count)))
        } else {
            phase = .failed(L10n.format("error.exited", status))
        }
    }

    // The transcript's plumbing. Chunks land in a private buffer and the
    // published values are refreshed at most once per flush window: the
    // launcher can emit many chunks a second while loading a model, and
    // every publish wakes each observer of the model, so a burst shares
    // one refresh. The window is a trailing debounce scheduled by the
    // first chunk of its burst, which also means the last chunk is always
    // flushed — an idle log publishes nothing and loses nothing.

    /// How many lines the transcript keeps. Rendering is lazy now, so the
    /// cap is for memory and scroll-back sanity, not for draw cost: two
    /// thousand lines is several times the launcher's chattiest load
    /// output and far past what anyone scrolls back to read.
    private static let logLineLimit = 2_000
    /// A carriage-return progress line is rewritten in place rather than
    /// newline-terminated, so left alone it would grow by a frame per
    /// refresh. Past this length the oldest frames are dropped.
    private static let logPendingLineLimit = 4_000
    /// The flush window, in nanoseconds: chunks inside it share a publish.
    private static let logFlushWindow: UInt64 = 100_000_000

    /// Complete lines not yet published.
    private var logBuffer: [String] = []
    /// What arrived since the last newline, from any chunk.
    private var logPartial = ""
    /// Whether the buffer holds anything the published values lack.
    private var logDirty = false
    /// Whether a flush is already scheduled for the running window.
    private var logFlushScheduled = false

    private func append(_ text: String) {
        // A chunk may end mid-line: its tail joins the line the previous
        // chunk left open, and the newline that closes it may arrive in a
        // later chunk. Everything up to the last newline is complete.
        let incoming = logPartial + text
        logPartial = ""
        if let cut = incoming.lastIndex(of: "\n") {
            let head = incoming[incoming.startIndex..<cut]
            logPartial = String(incoming[incoming.index(after: cut)...])
            // A lone newline closes an empty line, and blank lines are
            // kept: the transcript should read as the server wrote it.
            if head.isEmpty {
                logBuffer.append("")
            } else {
                logBuffer.append(
                    contentsOf: head.split(separator: "\n", omittingEmptySubsequences: false)
                        .map(String.init)
                )
            }
        } else {
            logPartial = incoming
        }
        if logPartial.count > Self.logPendingLineLimit {
            logCharacterCount -= logPartial.count - Self.logPendingLineLimit
            logPartial = String(logPartial.suffix(Self.logPendingLineLimit))
        }
        logCharacterCount += text.count
        if logBuffer.count > Self.logLineLimit {
            let dropped = logBuffer.count - Self.logLineLimit
            // The newline each dropped line ended with leaves with it.
            logCharacterCount -= logBuffer[..<dropped].reduce(0) { $0 + $1.count + 1 }
            logBuffer.removeFirst(dropped)
        }
        logDirty = true
        scheduleLogFlush()
    }

    private func scheduleLogFlush() {
        guard !logFlushScheduled else { return }
        logFlushScheduled = true
        Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: Self.logFlushWindow)
            guard let self else { return }
            self.logFlushScheduled = false
            self.flushLog()
        }
    }

    /// Publish what the buffer holds. Runs only when a chunk arrived since
    /// the last flush, so an idle log never wakes the views.
    private func flushLog() {
        guard logDirty else { return }
        logDirty = false
        logLines = logBuffer
        logPendingLine = logPartial
    }

    /// Empty the transcript: a start begins a new server's output, and the
    /// pane's trash button asks for the same on demand.
    private func resetLog() {
        logBuffer.removeAll()
        logPartial = ""
        logCharacterCount = 0
        logDirty = true
        flushLog()
    }

    /// A stopped server has no live stage and no uptime, but keeps the last
    /// minute of history so the dashboard does not blank between runs. The
    /// engine clears what is its own; this adds the configuration the running
    /// server was started with.
    func settle() {
        liveEngine.settle()
        runningConfigID = nil
    }

    /// Re-read the connection table on demand, for the clients panel's refresh
    /// button.
    func refreshConnectedClients() {
        liveEngine.refreshConnectedClients()
    }

    /// The time the current server has been ready, in words for the sidebar.
    var uptimeText: String? {
        guard let readySince else { return nil }
        let seconds = max(0, Int(Date().timeIntervalSince(readySince)))
        if seconds < 60 { return "\(seconds)s" }
        if seconds < 3600 {
            return String(format: "%dm %02ds", seconds / 60, seconds % 60)
        }
        return String(format: "%dh %02dm", seconds / 3600, (seconds % 3600) / 60)
    }

}
