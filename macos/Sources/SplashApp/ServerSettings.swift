import Foundation

/// The control panel's settings as one value: the model-owned share a
/// configuration edits, the app-wide share the Settings page edits, and what
/// a launch reads. SQLite keeps the whole value between launches. `default` is
/// the state the app is in when it is first opened, which is the state the
/// defaults button restores.
struct ServerSettings: Codable, Equatable {
    /// The first-launch values, named so the store, the form and the headless
    /// selfcheck all read them instead of repeating the literals.
    enum First {
        static let modelID = "mlx-community/Qwen3.8-27B-4bit"
        static let localKind = "gguf"
        static let draftMode = "none"
        // Speculation is auto by default: a compatible DFlash2 draft when one
        // resolves, and the target alone when it does not.
        static let speculation = "auto"
        static let requireDraft = false
        static let port = 8000
        static let kvFormat = "int8"
        // The recurrent state: fp32 is the server's default; bf16 halves it.
        static let stateFormat = "fp32"
        // The idle spill is the engine's own timer now and no longer follows
        // the residency below; the residency releases the weights alone.
        // Ten minutes keeps a chat responsive between pauses; the switch
        // beside it can keep the weights resident instead.
        static let residencySeconds = "600"
        static let unloadIdleWeights = true
        // Empty follows --idle-release; a duration such as 5m, or off, sets
        // how long the buffers stay wired on their own, apart from the
        // weights' release.
        static let residencyKeepAlive = ""
        // Slice an unopposed prefill command at about a second: the GPU gets
        // gaps between commands, at a measured prefill cost of about 4%, and
        // 13% at half a second. That buys responsiveness rather than
        // throughput, so it is off by default, as the CLI and server keep it,
        // and the panel's switch turns it on. The slice time stands ready for
        // that switch.
        static let responsivePrefill = false
        static let prefillSliceMilliseconds = "1000"
        // The SSD cache: off by default; the user turns it on in the panel.
        // When on, within a 16 GiB quota and a 32 GiB a day write budget.
        static let diskCache = false
        static let maxCacheDisk = "16G"
        static let maxCacheDiskWrites = "32G"
        // Whether an idle pass with no writable disk tier evicts the cached
        // prefixes. On is the behavior serving without a disk cache has always
        // had; off keeps them, so the next request's prompt is not computed
        // again, and the memory they hold is the price.
        static let cacheIdleEvict = true
        // The engine's memory and context defaults, shown in the panel.
        static let maxMemory = "28G"
        static let maxContext = "204800"
        // The admission limit: requests running and waiting at once. The
        // server's own default, so the panel and the engine open on the same
        // number.
        static let maxConcurrency = 8
        // The cache eviction presets: level 2 is the default, and protects
        // objects that prove reused; level 1 keeps the engine's plain LRU as
        // it was, and level 3 orders the protected tier by use count. The
        // policy and threshold are what a custom preset edits.
        static let cacheEvictionPreset = "balanced"
        static let cacheEvictionPolicy = "lru"
        static let cacheEvictionProtect = 2
    }

    /// The reasoning levels the server accepts; the empty string is the model
    /// default and is not one of them.
    static let reasoningEfforts = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
    static let kvFormats = ["int8", "int4", "k8v4", "bf16"]
    /// What a local target is picked as: a GGUF file (its shards beside it)
    /// with an optional vision projector, or an MLX model directory.
    static let localKinds = ["gguf", "mlx"]
    /// Where the DFlash2 draft comes from: nothing, a local draft directory,
    /// or a Hugging Face repository ID.
    static let draftModes = ["none", "folder", "hub"]
    /// How decode proposes tokens: automatic (a compatible DFlash2 draft,
    /// else none), none (autoregressive), or DFlash2 explicitly.
    static let speculationModes = ["auto", "dflash2"]
    static let stateFormats = ["fp32", "bf16"]
    static let verifyRowsOptions = ["auto", "all"]
    /// The eviction presets by level, and the policies a custom one picks.
    static let cacheEvictionPresets = ["light", "balanced", "strong", "custom"]
    static let cacheEvictionPolicies = ["lru", "fifo", "lfu", "largest"]

    /// Local models are the panel's own: an online model is fetched by the
    /// downloader first, and then chosen as a local target. The upstream case
    /// stays readable so a configuration the older online mode saved keeps its
    /// repository ID and still starts.
    var modelMode = AppModel.ModelMode.local
    var modelID = First.modelID
    var modelDirectory = ""
    /// Which local target the panel asks for (`localKinds`): `gguf` picks a
    /// file, the first of several, whose shards and projector lie beside it;
    /// `mlx` picks a model directory.
    var localKind = First.localKind
    var draftDirectory = ""
    /// Where the DFlash2 draft comes from (`draftModes`), and the repository
    /// ID it names when the mode is `hub`.
    var draftMode = First.draftMode
    var draftID = ""
    /// How decode proposes tokens (`speculationModes`), and whether a start
    /// refuses to run without the requested draft (`--require-draft`).
    var speculation = First.speculation
    var requireDraft = First.requireDraft
    /// A local GGUF target's vision projector (`--mmproj`): a GGUF file
    /// wherever it lies, or empty for the one beside the target.
    var mmproj = ""
    var port = First.port
    /// Listen on every IPv4 interface (`--host 0.0.0.0`) so other devices on
    /// the network can reach the server. Off keeps it on loopback.
    var lanAccess = false
    /// Extra HTTP Host names to accept (`--allowed-host`), comma-separated:
    /// the Mac's `.local` name, a DNS name, or a tunnel or proxy hostname.
    var allowedHosts = ""
    /// Extra browser origins whose pages may call the API
    /// (`--allowed-origin`), comma-separated; `*` allows any.
    var allowedOrigins = ""
    var languageOnly = false
    var kvFormat = First.kvFormat
    /// How a hybrid model keeps its recurrent state (`--state-format`): fp32,
    /// or bf16 for half the memory and twice as many cached states.
    var stateFormat = First.stateFormat
    var neuralEngine = true
    var maxMemory = First.maxMemory
    var maxContext = First.maxContext
    /// Memory of the stores the engine streams from the SSD: a MoE target's
    /// routed experts (`--expert-memory`), Qwen3.8-Flash-Next's PLE table
    /// (`--ple-memory`), and the KV blocks read back for requests whose KV
    /// lives on the SSD (`--sparse-kv-memory`). Empty is the server's
    /// automatic plan, `off` loads the store whole, and a size such as `16G`
    /// caps it.
    var expertMemory = ""
    var pleMemory = ""
    var sparseKvMemory = ""
    /// Whether a draft's decode cycles verify every row (`all`) or let the
    /// verify budget choose (`auto`): `all` repeats its output, `auto` is
    /// faster but its sampled tokens vary between runs.
    var verifyRows = "auto"
    /// Whether idle weights are released after `residencySeconds`. Off passes
    /// `--idle-release off`, so the weights stay resident until the engine
    /// stops or `POST /v1/memory/release` asks for them.
    var unloadIdleWeights = First.unloadIdleWeights
    /// Seconds without a request before the engine releases the weights
    /// (`--idle-release`), when `unloadIdleWeights` is on.
    var residencySeconds = First.residencySeconds
    /// How long the buffers stay wired without work (`--residency-keepalive`),
    /// apart from the weights' release: a duration, empty to follow
    /// `--idle-release`, or `off` to keep them wired while the engine runs.
    var residencyKeepAlive = First.residencyKeepAlive
    /// Whether long prefill commands are sliced (`--prefill-slice-ms`) so the
    /// GPU is free between them for other work and the system.
    var responsivePrefill = First.responsivePrefill
    /// The slice's target wall time in milliseconds, 1 to 10,000. A smaller
    /// value leaves the GPU more often, at a higher prefill cost.
    var prefillSliceMilliseconds = First.prefillSliceMilliseconds
    /// When the engine writes the cached prefixes to disk and releases their
    /// memory (`--cache-idle-spill`): a duration, empty for the server's 30
    /// seconds, `off` to keep them. Its own timer, apart from residency.
    var cacheIdleSpill = ""
    /// Whether a pass that finds no writable disk tier evicts the cached
    /// prefixes (`--cache-idle-evict off` keeps them in memory).
    var cacheIdleEvict = First.cacheIdleEvict
    /// Where the persistent SSD cache keeps its files (`--cache-dir`);
    /// empty is the server's own location.
    var cacheDir = ""
    /// Whether the engine may use the SSD for cached KV pages and states at
    /// all. Off keeps every cached prefix in memory and passes
    /// `--max-cache-disk 0`, which turns the disk tier off. On, the tier is
    /// persistent: a restart keeps the conversation prefixes.
    var diskCache = First.diskCache
    /// The SSD cache's quota. The default is a size; empty follows the
    /// volume's free space up to 64 GiB, and `0` turns the cache off.
    var maxCacheDisk = First.maxCacheDisk
    /// The most the SSD cache writes a day: the default paces the drive, and
    /// empty is no limit.
    var maxCacheDiskWrites = First.maxCacheDiskWrites
    /// The cache eviction preset: `light` (level 1, the LRU as it was),
    /// `balanced` (level 2, the default), `strong` (level 3), or `custom`,
    /// which uses the policy and threshold below.
    var cacheEvictionPreset = First.cacheEvictionPreset
    /// The policy a custom preset runs: `lru`, `fifo`, `lfu` or `largest`.
    var cacheEvictionPolicy = First.cacheEvictionPolicy
    /// The touches that put an object in the protected tier (0 to 8, 0
    /// disables it).
    var cacheEvictionProtect = First.cacheEvictionProtect
    var maxRequestSize = ""
    /// How long a queued or in-flight request may run (`--request-timeout`):
    /// a duration such as `30m`, empty for no limit.
    var requestTimeout = ""
    /// The largest resized image (`--max-image-pixels`), 65,536 to 4,194,304.
    var maxImagePixels = ""
    /// Decode time owed per unit of prefill time (`--decode-share`), empty
    /// for the server's 0.5.
    var decodeShare = ""
    /// Let the Mac sleep while requests run (`--allow-idle-sleep`).
    var allowIdleSleep = false
    /// Thinking tokens a request that sets no budget thinks
    /// (`--thinking-token-budget`), empty for no cap.
    var thinkingTokenBudget = ""
    /// Text that closes thinking where its budget runs out
    /// (`--thinking-end-text`), empty for the model's own.
    var thinkingEndText = ""
    /// Report the first served name in API responses and /v1/models
    /// (`--announce-served-name`); needs a served name.
    var announceServedName = false
    var apiKey = ""
    var servedNames = ""
    var reasoningEffort = ""
    /// Requests admitted at once, running or waiting.
    var maxConcurrency = First.maxConcurrency
    /// The row set this value was stored by, so a value an earlier build wrote
    /// can be told from one this build wrote. See `currentVersion`.
    var settingsVersion = ServerSettings.currentVersion

    /// The first-launch state, before anything is edited or stored.
    init() {}

    /// What the panel shows before anything is edited or stored.
    static let `default` = ServerSettings()

    /// The policy and protected tier a start passes to the engine: the
    /// preset's own, or a custom preset's.
    var resolvedCacheEviction: (policy: String, protect: Int) {
        Self.cacheEviction(preset: cacheEvictionPreset, policy: cacheEvictionPolicy,
                           protect: cacheEvictionProtect)
    }

    static func cacheEviction(preset: String, policy: String,
                              protect: Int) -> (policy: String, protect: Int) {
        switch preset {
        case "balanced": return ("lru", 2)
        case "strong": return ("lfu", 2)
        case "custom": return (policy, protect)
        default: return ("lru", 0)
        }
    }

    // The stored JSON is decoded field by field: a value a later build added,
    // or one a hand edit corrupted, keeps its first-launch default instead of
    // failing the whole decode and dropping the user's configurations.
    private enum CodingKeys: String, CodingKey {
        case modelMode, modelID, modelDirectory, localKind
        case draftMode, draftDirectory, draftID, speculation, requireDraft, mmproj, port
        case lanAccess, allowedHosts, allowedOrigins
        case languageOnly, kvFormat, stateFormat, neuralEngine, maxMemory, maxContext
        case unloadIdleWeights, residencySeconds, residencyKeepAlive, responsivePrefill
        case prefillSliceMilliseconds
        case cacheIdleSpill, cacheIdleEvict, cacheDir
        case maxRequestSize, requestTimeout, maxImagePixels, decodeShare
        case allowIdleSleep, thinkingTokenBudget, thinkingEndText, announceServedName
        case apiKey, servedNames
        case reasoningEffort, maxConcurrency
        case diskCache, maxCacheDisk, maxCacheDiskWrites
        case cacheEvictionPreset, cacheEvictionPolicy, cacheEvictionProtect
        case expertMemory, pleMemory, sparseKvMemory, verifyRows
        case settingsVersion
    }

    /// The row set this build writes, so a stored one can be told from those
    /// an earlier build wrote. `3` is the row set that carries the model
    /// selection again, after `2` dropped it: `2` turned the SSD cache off by
    /// default, and a row set without a version carries the switch's old
    /// default (`on`), which was not a choice the user made, so it reads as the
    /// new default — off. A row set that carries a version keeps the switch.
    /// `4` carries the idle-unload switch and moves the old ten-second
    /// residency to the new ten-minute default.
    static let currentVersion = "4"

    /// The residency the app used to write by default. A value of exactly this
    /// from an older row set was the old default rather than a choice, and it
    /// made every pause of ten seconds reload the whole model; migration moves
    /// it to the new default. A value this build writes is kept as it is.
    static let legacyResidencySeconds = "10"

    /// Moves a stored old-default residency to this build's default and
    /// records the new row set. Returns whether anything changed.
    /// `storedVersion` is the row set the value came from, nil for one written
    /// before row sets carried a version.
    @discardableResult
    static func migrateLegacyResidency(_ settings: inout ServerSettings,
                                       storedVersion: String?) -> Bool {
        guard settings.residencySeconds == legacyResidencySeconds,
              storedVersion != currentVersion else { return false }
        settings.unloadIdleWeights = First.unloadIdleWeights
        settings.residencySeconds = First.residencySeconds
        settings.settingsVersion = currentVersion
        return true
    }

    init(from decoder: Decoder) throws {
        self.init()
        let container = try decoder.container(keyedBy: CodingKeys.self)
        func value<T: Decodable>(_ type: T.Type, _ key: CodingKeys) -> T? {
            try? container.decodeIfPresent(T.self, forKey: key)
        }
        modelMode = value(AppModel.ModelMode.self, .modelMode) ?? modelMode
        modelID = value(String.self, .modelID) ?? modelID
        modelDirectory = value(String.self, .modelDirectory) ?? modelDirectory
        localKind = value(String.self, .localKind) ?? localKind
        draftDirectory = value(String.self, .draftDirectory) ?? draftDirectory
        draftMode = value(String.self, .draftMode) ?? draftMode
        draftID = value(String.self, .draftID) ?? draftID
        if let mode = value(String.self, .speculation),
           Self.speculationModes.contains(mode) {
            speculation = mode
        }
        requireDraft = value(Bool.self, .requireDraft) ?? requireDraft
        mmproj = value(String.self, .mmproj) ?? mmproj
        port = value(Int.self, .port) ?? port
        lanAccess = value(Bool.self, .lanAccess) ?? lanAccess
        allowedHosts = value(String.self, .allowedHosts) ?? allowedHosts
        allowedOrigins = value(String.self, .allowedOrigins) ?? allowedOrigins
        languageOnly = value(Bool.self, .languageOnly) ?? languageOnly
        kvFormat = value(String.self, .kvFormat) ?? kvFormat
        stateFormat = value(String.self, .stateFormat) ?? stateFormat
        neuralEngine = value(Bool.self, .neuralEngine) ?? neuralEngine
        maxMemory = value(String.self, .maxMemory) ?? maxMemory
        maxContext = value(String.self, .maxContext) ?? maxContext
        expertMemory = value(String.self, .expertMemory) ?? expertMemory
        pleMemory = value(String.self, .pleMemory) ?? pleMemory
        sparseKvMemory = value(String.self, .sparseKvMemory) ?? sparseKvMemory
        verifyRows = value(String.self, .verifyRows) ?? verifyRows
        unloadIdleWeights = value(Bool.self, .unloadIdleWeights) ?? unloadIdleWeights
        residencySeconds = value(String.self, .residencySeconds) ?? residencySeconds
        residencyKeepAlive = value(String.self, .residencyKeepAlive) ?? residencyKeepAlive
        responsivePrefill = value(Bool.self, .responsivePrefill) ?? responsivePrefill
        prefillSliceMilliseconds =
            value(String.self, .prefillSliceMilliseconds) ?? prefillSliceMilliseconds
        cacheIdleSpill = value(String.self, .cacheIdleSpill) ?? cacheIdleSpill
        cacheIdleEvict = value(Bool.self, .cacheIdleEvict) ?? cacheIdleEvict
        cacheDir = value(String.self, .cacheDir) ?? cacheDir
        // A configuration stored before the switch existed follows its quota:
        // a quota of `0` already meant "no disk cache", and one that named a
        // size meant the cache was on. One that named neither gets the
        // first-launch value, which is off.
        let storedDiskCache = value(Bool.self, .diskCache)
        let storedQuota = value(String.self, .maxCacheDisk)
        maxCacheDisk = storedQuota ?? maxCacheDisk
        diskCache = storedDiskCache
            ?? (storedQuota.map { $0.trimmingCharacters(in: .whitespaces) != "0" }
                ?? First.diskCache)
        maxCacheDiskWrites = value(String.self, .maxCacheDiskWrites) ?? maxCacheDiskWrites
        if let preset = value(String.self, .cacheEvictionPreset),
           Self.cacheEvictionPresets.contains(preset) {
            cacheEvictionPreset = preset
        }
        if let policy = value(String.self, .cacheEvictionPolicy),
           Self.cacheEvictionPolicies.contains(policy) {
            cacheEvictionPolicy = policy
        }
        if let protect = value(Int.self, .cacheEvictionProtect),
           (0...8).contains(protect) {
            cacheEvictionProtect = protect
        }
        maxRequestSize = value(String.self, .maxRequestSize) ?? maxRequestSize
        requestTimeout = value(String.self, .requestTimeout) ?? requestTimeout
        maxImagePixels = value(String.self, .maxImagePixels) ?? maxImagePixels
        decodeShare = value(String.self, .decodeShare) ?? decodeShare
        allowIdleSleep = value(Bool.self, .allowIdleSleep) ?? allowIdleSleep
        thinkingTokenBudget = value(String.self, .thinkingTokenBudget) ?? thinkingTokenBudget
        thinkingEndText = value(String.self, .thinkingEndText) ?? thinkingEndText
        announceServedName = value(Bool.self, .announceServedName) ?? announceServedName
        apiKey = value(String.self, .apiKey) ?? apiKey
        servedNames = value(String.self, .servedNames) ?? servedNames
        reasoningEffort = value(String.self, .reasoningEffort) ?? reasoningEffort
        maxConcurrency = value(Int.self, .maxConcurrency) ?? maxConcurrency
        settingsVersion = value(String.self, .settingsVersion) ?? settingsVersion
    }

    /// A digest that changes exactly when the settings do, so the autosave
    /// writes once per edit instead of once per keystroke.
    var fingerprint: String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard let data = try? encoder.encode(self) else { return "" }
        return data.base64EncodedString()
    }

    /// Row sets this build can read a choice from: the one it writes, and the
    /// ones before it, which already turned the SSD cache off by default.
    static let knownVersions: Set<String> = ["2", "3", currentVersion]

    /// The settings as the store's key/value rows, in a fixed order.
    var rows: [(key: String, value: String)] {
        [
            ("settingsVersion", Self.currentVersion),
            ("modelMode", modelMode.rawValue),
            ("modelID", modelID),
            ("modelDirectory", modelDirectory),
            ("localKind", localKind),
            ("draftDirectory", draftDirectory),
            ("draftMode", draftMode),
            ("draftID", draftID),
            ("speculation", speculation),
            ("requireDraft", requireDraft ? "1" : "0"),
            ("mmproj", mmproj),
            ("port", String(port)),
            ("lanAccess", lanAccess ? "1" : "0"),
            ("allowedHosts", allowedHosts),
            ("allowedOrigins", allowedOrigins),
            ("languageOnly", languageOnly ? "1" : "0"),
            ("kvFormat", kvFormat),
            ("stateFormat", stateFormat),
            ("neuralEngine", neuralEngine ? "1" : "0"),
            ("maxMemory", maxMemory),
            ("maxContext", maxContext),
            ("expertMemory", expertMemory),
            ("pleMemory", pleMemory),
            ("sparseKvMemory", sparseKvMemory),
            ("verifyRows", verifyRows),
            ("unloadIdleWeights", unloadIdleWeights ? "1" : "0"),
            ("residencySeconds", residencySeconds),
            ("residencyKeepAlive", residencyKeepAlive),
            ("responsivePrefill", responsivePrefill ? "1" : "0"),
            ("prefillSliceMilliseconds", prefillSliceMilliseconds),
            ("cacheIdleSpill", cacheIdleSpill),
            ("cacheIdleEvict", cacheIdleEvict ? "1" : "0"),
            ("cacheDir", cacheDir),
            ("diskCache", diskCache ? "1" : "0"),
            ("maxCacheDisk", maxCacheDisk),
            ("maxCacheDiskWrites", maxCacheDiskWrites),
            ("cacheEvictionPreset", cacheEvictionPreset),
            ("cacheEvictionPolicy", cacheEvictionPolicy),
            ("cacheEvictionProtect", String(cacheEvictionProtect)),
            ("maxRequestSize", maxRequestSize),
            ("requestTimeout", requestTimeout),
            ("maxImagePixels", maxImagePixels),
            ("decodeShare", decodeShare),
            ("allowIdleSleep", allowIdleSleep ? "1" : "0"),
            ("thinkingTokenBudget", thinkingTokenBudget),
            ("thinkingEndText", thinkingEndText),
            ("announceServedName", announceServedName ? "1" : "0"),
            ("apiKey", apiKey),
            ("servedNames", servedNames),
            ("reasoningEffort", reasoningEffort),
            ("maxConcurrency", String(maxConcurrency)),
        ]
    }

    /// Rebuilds the settings from stored rows. A key that is absent keeps its
    /// first-launch value, and one that is unparsable or out of the range the
    /// launcher accepts does too, so a hand-edited database cannot make the
    /// next launch start a server that refuses to run.
    init(rows: [String: String]) {
        self.init()
        if let value = rows["settingsVersion"] { settingsVersion = value }
        if let raw = rows["modelMode"], let mode = AppModel.ModelMode(rawValue: raw) {
            modelMode = mode
        }
        if let value = rows["modelID"] { modelID = value }
        if let value = rows["modelDirectory"] { modelDirectory = value }
        if let value = rows["localKind"], Self.localKinds.contains(value) { localKind = value }
        if let value = rows["draftDirectory"] { draftDirectory = value }
        if let value = rows["draftMode"], Self.draftModes.contains(value) { draftMode = value }
        if let value = rows["draftID"] { draftID = value }
        if let value = rows["speculation"], Self.speculationModes.contains(value) {
            speculation = value
        }
        if let value = rows["requireDraft"] { requireDraft = value == "1" }
        if let value = rows["mmproj"] { mmproj = value }
        if let value = Self.number(rows["port"], in: 0...65_535) { port = value }
        if let value = rows["lanAccess"] { lanAccess = value == "1" }
        if let value = rows["allowedHosts"] { allowedHosts = value }
        if let value = rows["allowedOrigins"] { allowedOrigins = value }
        if let value = rows["languageOnly"] { languageOnly = value == "1" }
        if let value = rows["kvFormat"], Self.kvFormats.contains(value) { kvFormat = value }
        if let value = rows["stateFormat"], Self.stateFormats.contains(value) {
            stateFormat = value
        }
        if let value = rows["neuralEngine"] { neuralEngine = value == "1" }
        if let value = rows["maxMemory"] { maxMemory = value }
        if let value = rows["maxContext"] { maxContext = value }
        if let value = Self.streamedSetting(rows["expertMemory"]) { expertMemory = value }
        if let value = Self.streamedSetting(rows["pleMemory"]) { pleMemory = value }
        if let value = Self.streamedSetting(rows["sparseKvMemory"]) { sparseKvMemory = value }
        if let value = rows["verifyRows"], Self.verifyRowsOptions.contains(value) {
            verifyRows = value
        }
        if let value = rows["unloadIdleWeights"] { unloadIdleWeights = value == "1" }
        if let value = rows["residencyKeepAlive"] { residencyKeepAlive = value }
        if let value = rows["residencySeconds"],
           Self.number(value, in: 1...86_400) != nil {
            residencySeconds = value
        }
        // The slice's switch arrived with row set 4. A row set from before it
        // carries no row, and the value nobody chose must not turn the slice
        // on behind the user's back, as the SSD cache's row does not: it reads
        // as off. A row set this build writes carries its version, so a choice
        // made from then on is kept.
        if let value = rows["responsivePrefill"] {
            responsivePrefill = value == "1"
        } else if rows["settingsVersion"] != Self.currentVersion {
            responsivePrefill = false
        }
        if let value = Self.number(rows["prefillSliceMilliseconds"], in: 1...10_000) {
            prefillSliceMilliseconds = String(value)
        }
        if let value = rows["maxCacheDisk"] { maxCacheDisk = value }
        if let value = rows["cacheIdleEvict"] { cacheIdleEvict = value == "1" }
        // The switch's own row is a choice only once a build that knew it wrote
        // the rows with a version. A row set from before that carries the old
        // default (on) for every install, which is not a choice anyone made, so
        // it reads as the new default instead of turning the cache back on; the
        // next save writes the version, and from then on the row is the user's.
        if let version = rows["settingsVersion"], Self.knownVersions.contains(version),
           let value = rows["diskCache"] {
            diskCache = value == "1"
        }
        if let value = rows["cacheIdleSpill"], Self.duration(value) != nil {
            cacheIdleSpill = value
        }
        if let value = rows["cacheDir"] { cacheDir = value }
        if let value = rows["maxCacheDiskWrites"] { maxCacheDiskWrites = value }
        if let value = rows["cacheEvictionPreset"],
           Self.cacheEvictionPresets.contains(value) {
            cacheEvictionPreset = value
        }
        if let value = rows["cacheEvictionPolicy"],
           Self.cacheEvictionPolicies.contains(value) {
            cacheEvictionPolicy = value
        }
        if let value = Self.number(rows["cacheEvictionProtect"], in: 0...8) {
            cacheEvictionProtect = value
        }
        if let value = rows["maxRequestSize"] { maxRequestSize = value }
        if let value = rows["requestTimeout"], Self.duration(value) != nil {
            requestTimeout = value
        }
        if let value = Self.number(rows["maxImagePixels"], in: 65_536...4_194_304) {
            maxImagePixels = String(value)
        }
        if let value = rows["decodeShare"], Self.decimal(value) != nil {
            decodeShare = value
        }
        if let value = rows["allowIdleSleep"] { allowIdleSleep = value == "1" }
        if let value = Self.number(rows["thinkingTokenBudget"], in: 0...1_000_000) {
            thinkingTokenBudget = String(value)
        }
        if let value = rows["thinkingEndText"] { thinkingEndText = value }
        if let value = rows["announceServedName"] { announceServedName = value == "1" }
        if let value = rows["apiKey"] { apiKey = value }
        if let value = rows["servedNames"] { servedNames = value }
        if let value = rows["reasoningEffort"],
           value.isEmpty || Self.reasoningEfforts.contains(value) {
            reasoningEffort = value
        }
        // The launcher refuses a queue size of zero or less, so one that
        // cannot run falls back to the first-launch limit.
        if let value = Self.number(rows["maxConcurrency"], in: 1...4096) {
            maxConcurrency = value
        }
    }

    /// A streamed store's setting: empty (the server's automatic plan),
    /// `auto`, `off`, or a positive size such as `16G`, `1200M` or `0.75G`.
    /// Anything else falls back to the first-launch value, so a hand-edited
    /// database cannot make the next launch start a server that refuses to.
    private static func streamedSetting(_ raw: String?) -> String? {
        guard let raw else { return nil }
        let value = raw.trimmingCharacters(in: .whitespaces)
        if value.isEmpty || value.caseInsensitiveCompare("auto") == .orderedSame
            || value.caseInsensitiveCompare("off") == .orderedSame {
            return value.isEmpty ? "" : value.lowercased()
        }
        var number = Substring(value)
        if let last = value.last, "KMGT".contains(last) {
            number = value.dropLast()
        }
        guard let amount = Double(number), amount > 0 else { return nil }
        return value
    }

    /// A duration as the server takes one: seconds, or a number with an s, m
    /// or h suffix; `off` for no limit.
    static func duration(_ raw: String?) -> String? {
        guard let raw else { return nil }
        let value = raw.trimmingCharacters(in: .whitespaces).lowercased()
        if value.isEmpty || value == "off" { return value }
        let digits = value.last.map { "smh".contains($0) } == true ? value.dropLast() : value[...]
        guard !digits.isEmpty, digits.allSatisfy(\.isNumber) else { return nil }
        return value
    }

    /// A nonnegative decimal, such as the decode share.
    static func decimal(_ raw: String?) -> String? {
        guard let raw else { return nil }
        let value = raw.trimmingCharacters(in: .whitespaces)
        guard let number = Double(value), number.isFinite, number >= 0 else { return nil }
        return value
    }

    private static func number(_ raw: String?, in range: ClosedRange<Int>) -> Int? {
        guard let raw, let value = Int(raw), range.contains(value) else { return nil }
        return value
    }
}
