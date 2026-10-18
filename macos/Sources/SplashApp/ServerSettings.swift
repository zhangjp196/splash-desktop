import Foundation

/// The control panel's settings as one value: what the form edits, what SQLite
/// keeps between launches, and what Restore Defaults returns to. `default` is
/// the state the app is in when it is first opened, which is the state the
/// defaults button restores.
struct ServerSettings: Codable, Equatable {
    /// The first-launch values, named so the store, the form and the headless
    /// selfcheck all read them instead of repeating the literals.
    enum First {
        static let modelID = "mlx-community/Qwen3.8-27B-4bit"
        static let port = 8000
        static let kvFormat = "int8"
        // The idle spill is the engine's own timer now and no longer follows
        // the residency below; the residency releases the weights alone.
        static let residencySeconds = "10"
        // The SSD cache: kept across restarts, so a restart, an upgrade or a
        // crash does not cost every conversation its prefix; its quota the
        // engine's own automatic one (up to 128 GiB, following the volume);
        // and a daily write budget that paces it against the drive's life.
        // The budget is what a desktop app should default to: it paces, and
        // past it the cache keeps the states conversations resume from.
        static let persistentCache = true
        static let maxCacheDisk = ""
        static let maxCacheDiskWrites = "50G"
        // The admission limit: requests running and waiting at once. The
        // server's own default, so the panel and the engine open on the same
        // number.
        static let maxConcurrency = 16
    }

    /// The reasoning levels the server accepts; the empty string is the model
    /// default and is not one of them.
    static let reasoningEfforts = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
    static let kvFormats = ["int8", "bf16"]

    var modelMode = AppModel.ModelMode.upstream
    var modelID = First.modelID
    var modelDirectory = ""
    var draftDirectory = ""
    var port = First.port
    var languageOnly = false
    var kvFormat = First.kvFormat
    var neuralEngine = true
    var maxMemory = ""
    var maxContext = ""
    var residencySeconds = First.residencySeconds
    /// Keep the SSD cache across restarts (`--persistent-cache`).
    var persistentCache = First.persistentCache
    /// The SSD cache's quota: empty follows the volume's free space, up to
    /// 128 GiB; `0` turns the cache off.
    var maxCacheDisk = First.maxCacheDisk
    /// The most the SSD cache writes a day: empty is no limit.
    var maxCacheDiskWrites = First.maxCacheDiskWrites
    var maxRequestSize = ""
    var apiKey = ""
    var servedNames = ""
    var reasoningEffort = ""
    /// Requests admitted at once, running or waiting.
    var maxConcurrency = First.maxConcurrency

    /// The first-launch state, before anything is edited or stored.
    init() {}

    /// What the panel shows before anything is edited or stored.
    static let `default` = ServerSettings()

    // The stored JSON is decoded field by field: a value a later build added,
    // or one a hand edit corrupted, keeps its first-launch default instead of
    // failing the whole decode and dropping the user's configurations.
    private enum CodingKeys: String, CodingKey {
        case modelMode, modelID, modelDirectory, draftDirectory, port
        case languageOnly, kvFormat, neuralEngine, maxMemory, maxContext
        case residencySeconds, maxRequestSize, apiKey, servedNames
        case reasoningEffort, maxConcurrency
        case persistentCache, maxCacheDisk, maxCacheDiskWrites
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
        draftDirectory = value(String.self, .draftDirectory) ?? draftDirectory
        port = value(Int.self, .port) ?? port
        languageOnly = value(Bool.self, .languageOnly) ?? languageOnly
        kvFormat = value(String.self, .kvFormat) ?? kvFormat
        neuralEngine = value(Bool.self, .neuralEngine) ?? neuralEngine
        maxMemory = value(String.self, .maxMemory) ?? maxMemory
        maxContext = value(String.self, .maxContext) ?? maxContext
        residencySeconds = value(String.self, .residencySeconds) ?? residencySeconds
        persistentCache = value(Bool.self, .persistentCache) ?? persistentCache
        maxCacheDisk = value(String.self, .maxCacheDisk) ?? maxCacheDisk
        maxCacheDiskWrites = value(String.self, .maxCacheDiskWrites) ?? maxCacheDiskWrites
        maxRequestSize = value(String.self, .maxRequestSize) ?? maxRequestSize
        apiKey = value(String.self, .apiKey) ?? apiKey
        servedNames = value(String.self, .servedNames) ?? servedNames
        reasoningEffort = value(String.self, .reasoningEffort) ?? reasoningEffort
        maxConcurrency = value(Int.self, .maxConcurrency) ?? maxConcurrency
    }

    /// A digest that changes exactly when the settings do, so the autosave
    /// writes once per edit instead of once per keystroke.
    var fingerprint: String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard let data = try? encoder.encode(self) else { return "" }
        return data.base64EncodedString()
    }

    /// The settings as the store's key/value rows, in a fixed order.
    var rows: [(key: String, value: String)] {
        [
            ("modelMode", modelMode.rawValue),
            ("modelID", modelID),
            ("modelDirectory", modelDirectory),
            ("draftDirectory", draftDirectory),
            ("port", String(port)),
            ("languageOnly", languageOnly ? "1" : "0"),
            ("kvFormat", kvFormat),
            ("neuralEngine", neuralEngine ? "1" : "0"),
            ("maxMemory", maxMemory),
            ("maxContext", maxContext),
            ("residencySeconds", residencySeconds),
            ("persistentCache", persistentCache ? "1" : "0"),
            ("maxCacheDisk", maxCacheDisk),
            ("maxCacheDiskWrites", maxCacheDiskWrites),
            ("maxRequestSize", maxRequestSize),
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
        if let raw = rows["modelMode"], let mode = AppModel.ModelMode(rawValue: raw) {
            modelMode = mode
        }
        if let value = rows["modelID"] { modelID = value }
        if let value = rows["modelDirectory"] { modelDirectory = value }
        if let value = rows["draftDirectory"] { draftDirectory = value }
        if let value = Self.number(rows["port"], in: 0...65_535) { port = value }
        if let value = rows["languageOnly"] { languageOnly = value == "1" }
        if let value = rows["kvFormat"], Self.kvFormats.contains(value) { kvFormat = value }
        if let value = rows["neuralEngine"] { neuralEngine = value == "1" }
        if let value = rows["maxMemory"] { maxMemory = value }
        if let value = rows["maxContext"] { maxContext = value }
        if let value = rows["residencySeconds"],
           Self.number(value, in: 1...86_400) != nil {
            residencySeconds = value
        }
        if let value = rows["persistentCache"] { persistentCache = value == "1" }
        if let value = rows["maxCacheDisk"] { maxCacheDisk = value }
        if let value = rows["maxCacheDiskWrites"] { maxCacheDiskWrites = value }
        if let value = rows["maxRequestSize"] { maxRequestSize = value }
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

    private static func number(_ raw: String?, in range: ClosedRange<Int>) -> Int? {
        guard let raw, let value = Int(raw), range.contains(value) else { return nil }
        return value
    }
}
