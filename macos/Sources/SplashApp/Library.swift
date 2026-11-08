import Foundation

/// A saved configuration the models pane adds, edits and switches to: a name
/// the user can set and the whole `ServerSettings` the server starts with.
struct ModelConfig: Codable, Identifiable, Equatable {
    var id = UUID()
    var name = ""
    var settings: ServerSettings

    init(settings: ServerSettings) { self.settings = settings }

    /// Whether a local target's folder is gone: a configuration is kept, as
    /// the user may point it at the model again, but it cannot start.
    var missingFiles: Bool {
        guard settings.modelMode == AppModel.ModelMode.local else { return false }
        let directory = settings.modelDirectory.trimmingCharacters(in: .whitespaces)
        return !directory.isEmpty && !FileManager.default.fileExists(atPath: directory)
    }

    /// The name derived from the model it serves, when it serves one: what the
    /// editor opens an unnamed configuration on, and what an empty name falls
    /// back to.
    var derivedName: String? {
        switch settings.modelMode {
        case .upstream:
            let identifier = settings.modelID.trimmingCharacters(in: .whitespaces)
            return identifier.isEmpty ? nil : identifier
        case .local:
            let directory = settings.modelDirectory.trimmingCharacters(in: .whitespaces)
            return directory.isEmpty ? nil : (directory as NSString).lastPathComponent
        }
    }

    /// The name, or one derived from the model it serves.
    var displayName: String {
        let trimmed = name.trimmingCharacters(in: .whitespaces)
        if !trimmed.isEmpty { return trimmed }
        return derivedName ?? L10n.string("config.untitled")
    }
}

/// Persists the configurations and the active one's id as JSON in
/// UserDefaults, beside the settings SQLite keeps.
final class LibraryStore {
    private let defaults: UserDefaults
    private static let configsKey = "splash.configs"
    private static let activeKey = "splash.active-config"

    /// `SPLASH_SETTINGS_DB` redirects the settings SQLite for the headless
    /// selfcheck; the library follows it into a suite of its own, so the
    /// selfcheck never touches the user's real configurations.
    init() {
        if ProcessInfo.processInfo.environment["SPLASH_SETTINGS_DB"] != nil {
            defaults = UserDefaults(suiteName: "splash.selfcheck") ?? .standard
        } else {
            defaults = .standard
        }
    }

    func loadConfigs() -> [ModelConfig] {
        guard let data = defaults.data(forKey: Self.configsKey),
              var configs = try? JSONDecoder().decode([ModelConfig].self, from: data)
        else { return [] }
        // A configuration saved by an earlier build carries the old
        // ten-second residency. Move it to this build's default and write the
        // library back once, so switching to it does not reload the whole
        // model after every pause. The raw rows are read too: a value stored
        // before row sets carried a version has none to decode.
        let stored = (try? JSONSerialization.jsonObject(with: data)) as? [[String: Any]]
        var changed = false
        for index in configs.indices {
            let version = (stored?[index]["settings"] as? [String: Any])?["settingsVersion"] as? String
            if ServerSettings.migrateLegacyResidency(
                &configs[index].settings, storedVersion: version) {
                changed = true
            }
        }
        if changed { saveConfigs(configs) }
        return configs
    }

    func saveConfigs(_ configs: [ModelConfig]) { encode(configs, Self.configsKey) }

    func loadActive() -> UUID? {
        defaults.string(forKey: Self.activeKey).flatMap(UUID.init(uuidString:))
    }
    func saveActive(_ id: UUID?) {
        if let id { defaults.set(id.uuidString, forKey: Self.activeKey) }
        else { defaults.removeObject(forKey: Self.activeKey) }
    }

    private func decode<T: Decodable>(_ type: T.Type, _ key: String) -> T? {
        guard let data = defaults.data(forKey: key) else { return nil }
        return try? JSONDecoder().decode(type, from: data)
    }

    private func encode<T: Encodable>(_ value: T, _ key: String) {
        if let data = try? JSONEncoder().encode(value) { defaults.set(data, forKey: key) }
    }
}
