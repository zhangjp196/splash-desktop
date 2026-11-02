import Foundation

/// A saved configuration the models pane adds, edits and switches to: a name
/// the user can set and the whole `ServerSettings` the server starts with.
struct ModelConfig: Codable, Identifiable, Equatable {
    var id = UUID()
    var name = ""
    var settings: ServerSettings

    init(settings: ServerSettings) { self.settings = settings }

    /// The name, or one derived from the model it serves.
    var displayName: String {
        let trimmed = name.trimmingCharacters(in: .whitespaces)
        if !trimmed.isEmpty { return trimmed }
        switch settings.modelMode {
        case .upstream:
            return settings.modelID.isEmpty ? L10n.string("config.untitled") : settings.modelID
        case .local:
            return settings.modelDirectory.isEmpty
                ? L10n.string("config.untitled")
                : (settings.modelDirectory as NSString).lastPathComponent
        }
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

    func loadConfigs() -> [ModelConfig] { decode([ModelConfig].self, Self.configsKey) ?? [] }
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
