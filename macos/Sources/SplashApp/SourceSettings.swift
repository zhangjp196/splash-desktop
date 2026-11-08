import Foundation

/// Where a Hub download lands and which Hub it asks. This is the app's own
/// setting rather than a configuration's: one Mac has one download cache and
/// one reachable Hub, and every configuration shares them. The panel's
/// downloader and the launcher pass them on as `HF_HUB_CACHE` and
/// `HF_ENDPOINT`, so the shell's own Hugging Face settings keep working when
/// they are empty.
struct SourceSettings: Equatable {
    /// The directory downloads land in (`HF_HUB_CACHE`). Empty follows the Hugging
    /// Face default, the cache under the home folder.
    var baseDirectory = ""
    /// The Hub address to ask (`HF_ENDPOINT`), such as a mirror or a proxy.
    /// Empty asks huggingface.co.
    var hubEndpoint = ""

    private enum Keys {
        static let baseDirectory = "splash.source.base-directory"
        static let hubEndpoint = "splash.source.hub-endpoint"
    }

    /// The defaults this setting lives in: while `SPLASH_SETTINGS_DB`
    /// redirects the settings SQLite, the headless selfcheck reads and writes
    /// its own suite instead of the user's real preferences.
    static var defaults: UserDefaults {
        ProcessInfo.processInfo.environment["SPLASH_SETTINGS_DB"] != nil
            ? (UserDefaults(suiteName: "splash.selfcheck") ?? .standard)
            : .standard
    }

    static func load(from defaults: UserDefaults? = nil) -> SourceSettings {
        let defaults = defaults ?? Self.defaults
        var settings = SourceSettings()
        if let value = defaults.string(forKey: Keys.baseDirectory), !value.isEmpty {
            settings.baseDirectory = value
        }
        if let value = defaults.string(forKey: Keys.hubEndpoint), !value.isEmpty {
            settings.hubEndpoint = value
        }
        return settings
    }

    func save(to defaults: UserDefaults? = nil) {
        let defaults = defaults ?? Self.defaults
        defaults.set(baseDirectory, forKey: Keys.baseDirectory)
        defaults.set(hubEndpoint, forKey: Keys.hubEndpoint)
    }

    /// What a download or a serve that may download gets: the base folder and
    /// the Hub address, each only when it is set.
    var environment: [String: String] {
        var values: [String: String] = [:]
        let base = baseDirectory.trimmingCharacters(in: .whitespaces)
        if !base.isEmpty {
            values["HF_HUB_CACHE"] = (base as NSString).expandingTildeInPath
        }
        let endpoint = hubEndpoint.trimmingCharacters(in: .whitespaces)
        if !endpoint.isEmpty { values["HF_ENDPOINT"] = endpoint }
        return values
    }

    /// Whether both settings are empty, so the panel can say the Hugging Face
    /// defaults are in use.
    var isDefault: Bool {
        baseDirectory.trimmingCharacters(in: .whitespaces).isEmpty
            && hubEndpoint.trimmingCharacters(in: .whitespaces).isEmpty
    }

    /// The base folder as the panel shows it: the setting, or the Hugging Face
    /// default whose path the launcher knows.
    var effectiveBaseDescription: String {
        let base = baseDirectory.trimmingCharacters(in: .whitespaces)
        return base.isEmpty ? "$HF_HOME/hub" : (base as NSString).expandingTildeInPath
    }
}
