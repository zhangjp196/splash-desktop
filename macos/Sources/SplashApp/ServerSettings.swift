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
        // The SSD cache tier is on by default, so the idle offload below has
        // somewhere to write; 0 is how the user turns it off.
        static let maxCacheDisk = "16G"
        static let idleOffloadSeconds = "10"
        static let residencySeconds = "10"
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
    var maxMemory = ""
    var maxContext = ""
    var maxCacheDisk = First.maxCacheDisk
    var idleOffloadSeconds = First.idleOffloadSeconds
    var residencySeconds = First.residencySeconds
    var maxRequestSize = ""
    var apiKey = ""
    var servedNames = ""
    var reasoningEffort = ""

    /// The first-launch state, before anything is edited or stored.
    init() {}

    /// What the panel shows before anything is edited or stored.
    static let `default` = ServerSettings()

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
            ("maxMemory", maxMemory),
            ("maxContext", maxContext),
            ("maxCacheDisk", maxCacheDisk),
            ("idleOffloadSeconds", idleOffloadSeconds),
            ("residencySeconds", residencySeconds),
            ("maxRequestSize", maxRequestSize),
            ("apiKey", apiKey),
            ("servedNames", servedNames),
            ("reasoningEffort", reasoningEffort),
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
        if let value = rows["maxMemory"] { maxMemory = value }
        if let value = rows["maxContext"] { maxContext = value }
        if let value = rows["maxCacheDisk"] { maxCacheDisk = value }
        if let value = rows["idleOffloadSeconds"],
           Self.number(value, in: 0...86_400) != nil {
            idleOffloadSeconds = value
        }
        if let value = rows["residencySeconds"],
           Self.number(value, in: 1...86_400) != nil {
            residencySeconds = value
        }
        if let value = rows["maxRequestSize"] { maxRequestSize = value }
        if let value = rows["apiKey"] { apiKey = value }
        if let value = rows["servedNames"] { servedNames = value }
        if let value = rows["reasoningEffort"],
           value.isEmpty || Self.reasoningEfforts.contains(value) {
            reasoningEffort = value
        }
    }

    private static func number(_ raw: String?, in range: ClosedRange<Int>) -> Int? {
        guard let raw, let value = Int(raw), range.contains(value) else { return nil }
        return value
    }
}
