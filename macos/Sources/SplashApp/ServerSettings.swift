import Foundation

/// The control panel's settings as one value: what the form edits, what SQLite
/// keeps between launches, and what Restore Defaults returns to. `default` is
/// the state the app is in when it is first opened, which is the state the
/// defaults button restores.
struct ServerSettings: Codable, Equatable {
    /// The first-launch values, named so the store, the form and the headless
    /// selfcheck all read them instead of repeating the literals.
    enum First {
        static let port = 8000
        static let kvFormat = "int8"
        // The idle spill is the engine's own timer now and no longer follows
        // the residency below; the residency releases the weights alone.
        static let residencySeconds = "10"
        // The SSD cache: off by default; the user turns it on in the panel.
        // When on, within a 16 GiB quota and a 32 GiB a day write budget.
        static let diskCache = false
        static let maxCacheDisk = "16G"
        static let maxCacheDiskWrites = "32G"
        // The engine's memory and context defaults, shown in the panel.
        static let maxMemory = "28G"
        static let maxContext = "204800"
        // The admission limit: requests running and waiting at once. The
        // server's own default, so the panel and the engine open on the same
        // number.
        static let maxConcurrency = 8
    }

    /// The reasoning levels the server accepts; the empty string is the model
    /// default and is not one of them.
    static let reasoningEfforts = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
    static let kvFormats = ["int8", "int4", "bf16"]
    static let verifyRowsOptions = ["auto", "all"]

    var port = First.port
    /// Listen on every IPv4 interface (`--host 0.0.0.0`) so other devices on
    /// the network can reach the server. Off keeps it on loopback.
    var lanAccess = false
    /// Extra HTTP Host names to accept (`--allowed-host`), comma-separated:
    /// the Mac's `.local` name, a DNS name, or a tunnel or proxy hostname.
    var allowedHosts = ""
    var languageOnly = false
    var kvFormat = First.kvFormat
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
    var residencySeconds = First.residencySeconds
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
    var maxRequestSize = ""
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

    // The stored JSON is decoded field by field: a value a later build added,
    // or one a hand edit corrupted, keeps its first-launch default instead of
    // failing the whole decode and dropping the user's configurations.
    private enum CodingKeys: String, CodingKey {
        case port
        case lanAccess, allowedHosts
        case languageOnly, kvFormat, neuralEngine, maxMemory, maxContext
        case residencySeconds, maxRequestSize, apiKey, servedNames
        case reasoningEffort, maxConcurrency
        case diskCache, maxCacheDisk, maxCacheDiskWrites
        case expertMemory, pleMemory, sparseKvMemory, verifyRows
        case settingsVersion
    }

    /// The row set this build writes, so a stored one can be told from those
    /// an earlier build wrote. `2` is the row set that turned the SSD cache off
    /// by default: a row set without it carries the switch's old default (`on`),
    /// which was not a choice the user made, so it reads as the new default —
    /// off. A row set that carries it keeps what the switch was set to.
    static let currentVersion = "2"

    init(from decoder: Decoder) throws {
        self.init()
        let container = try decoder.container(keyedBy: CodingKeys.self)
        func value<T: Decodable>(_ type: T.Type, _ key: CodingKeys) -> T? {
            try? container.decodeIfPresent(T.self, forKey: key)
        }
        port = value(Int.self, .port) ?? port
        lanAccess = value(Bool.self, .lanAccess) ?? lanAccess
        allowedHosts = value(String.self, .allowedHosts) ?? allowedHosts
        languageOnly = value(Bool.self, .languageOnly) ?? languageOnly
        kvFormat = value(String.self, .kvFormat) ?? kvFormat
        neuralEngine = value(Bool.self, .neuralEngine) ?? neuralEngine
        maxMemory = value(String.self, .maxMemory) ?? maxMemory
        maxContext = value(String.self, .maxContext) ?? maxContext
        expertMemory = value(String.self, .expertMemory) ?? expertMemory
        pleMemory = value(String.self, .pleMemory) ?? pleMemory
        sparseKvMemory = value(String.self, .sparseKvMemory) ?? sparseKvMemory
        verifyRows = value(String.self, .verifyRows) ?? verifyRows
        residencySeconds = value(String.self, .residencySeconds) ?? residencySeconds
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
        maxRequestSize = value(String.self, .maxRequestSize) ?? maxRequestSize
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

    /// The settings as the store's key/value rows, in a fixed order.
    var rows: [(key: String, value: String)] {
        [
            ("settingsVersion", Self.currentVersion),
            ("port", String(port)),
            ("lanAccess", lanAccess ? "1" : "0"),
            ("allowedHosts", allowedHosts),
            ("languageOnly", languageOnly ? "1" : "0"),
            ("kvFormat", kvFormat),
            ("neuralEngine", neuralEngine ? "1" : "0"),
            ("maxMemory", maxMemory),
            ("maxContext", maxContext),
            ("expertMemory", expertMemory),
            ("pleMemory", pleMemory),
            ("sparseKvMemory", sparseKvMemory),
            ("verifyRows", verifyRows),
            ("residencySeconds", residencySeconds),
            ("diskCache", diskCache ? "1" : "0"),
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
        if let value = rows["settingsVersion"] { settingsVersion = value }
        if let value = Self.number(rows["port"], in: 0...65_535) { port = value }
        if let value = rows["lanAccess"] { lanAccess = value == "1" }
        if let value = rows["allowedHosts"] { allowedHosts = value }
        if let value = rows["languageOnly"] { languageOnly = value == "1" }
        if let value = rows["kvFormat"], Self.kvFormats.contains(value) { kvFormat = value }
        if let value = rows["neuralEngine"] { neuralEngine = value == "1" }
        if let value = rows["maxMemory"] { maxMemory = value }
        if let value = rows["maxContext"] { maxContext = value }
        if let value = Self.streamedSetting(rows["expertMemory"]) { expertMemory = value }
        if let value = Self.streamedSetting(rows["pleMemory"]) { pleMemory = value }
        if let value = Self.streamedSetting(rows["sparseKvMemory"]) { sparseKvMemory = value }
        if let value = rows["verifyRows"], Self.verifyRowsOptions.contains(value) {
            verifyRows = value
        }
        if let value = rows["residencySeconds"],
           Self.number(value, in: 1...86_400) != nil {
            residencySeconds = value
        }
        if let value = rows["maxCacheDisk"] { maxCacheDisk = value }
        // The switch's own row is a choice only once this build has written the
        // rows with its version. A row set from before that carries the old
        // default (on) for every install, which is not a choice anyone made, so
        // it reads as the new default instead of turning the cache back on; the
        // next save writes the version, and from then on the row is the user's.
        if rows["settingsVersion"] == Self.currentVersion, let value = rows["diskCache"] {
            diskCache = value == "1"
        }
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

    private static func number(_ raw: String?, in range: ClosedRange<Int>) -> Int? {
        guard let raw, let value = Int(raw), range.contains(value) else { return nil }
        return value
    }
}
