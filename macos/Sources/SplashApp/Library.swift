import AppKit
import Foundation

/// What a directory holds: an MLX checkpoint, a GGUF, a legacy Splash
/// package, or something else. Read from the file names alone.
enum ModelKind: String, Codable {
    case mlx, gguf, splash, unknown

    var label: String {
        switch self {
        case .mlx: return "MLX"
        case .gguf: return "GGUF"
        case .splash: return L10n.string("kind.splash")
        case .unknown: return L10n.string("kind.unknown")
        }
    }
}

/// A saved model directory the models pane lists and manages: what it holds
/// (kind, whether it has a vision tower) and, once measured, its size.
struct ModelDirectory: Codable, Identifiable, Equatable {
    var path: String
    var kind: String
    var hasVision: Bool
    var sizeBytes: Int64?

    var id: String { path }
    var name: String { URL(fileURLWithPath: path).lastPathComponent }
    var kindValue: ModelKind { ModelKind(rawValue: kind) ?? .unknown }
}

/// A model configuration the app has started, for one-click switching: the
/// fields that decide which model serves and how.
struct ModelCombo: Codable, Identifiable, Equatable {
    var id = UUID()
    var mode: String
    var modelID: String
    var modelDirectory: String
    var draftDirectory: String
    var languageOnly: Bool
    var kvFormat: String

    var modeValue: AppModel.ModelMode { AppModel.ModelMode(rawValue: mode) ?? .upstream }
}

/// Persists the recent model combinations and the directory list as JSON in
/// UserDefaults, beside the settings SQLite keeps.
final class LibraryStore {
    private let defaults = UserDefaults.standard
    private static let combosKey = "splash.recent-combos"
    private static let directoriesKey = "splash.model-directories"

    func loadCombos() -> [ModelCombo] { decode([ModelCombo].self, Self.combosKey) ?? [] }
    func saveCombos(_ combos: [ModelCombo]) { encode(combos, Self.combosKey) }
    func loadDirectories() -> [ModelDirectory] { decode([ModelDirectory].self, Self.directoriesKey) ?? [] }
    func saveDirectories(_ directories: [ModelDirectory]) { encode(directories, Self.directoriesKey) }

    private func decode<T: Decodable>(_ type: T.Type, _ key: String) -> T? {
        guard let data = defaults.data(forKey: key) else { return nil }
        return try? JSONDecoder().decode(type, from: data)
    }

    private func encode<T: Encodable>(_ value: T, _ key: String) {
        if let data = try? JSONEncoder().encode(value) { defaults.set(data, forKey: key) }
    }
}

/// What a directory holds, from its file names alone; no weights are read.
func inspectModelDirectory(_ path: String) -> (kind: ModelKind, hasVision: Bool) {
    let names = (try? FileManager.default.contentsOfDirectory(atPath: path)) ?? []
    let lower = names.map { $0.lowercased() }
    let hasVision =
        lower.contains("vision")
        || lower.contains { $0.hasPrefix("mmproj") }
        || lower.contains("preprocessor_config.json")
    if names.contains("manifest.json") { return (.splash, hasVision) }
    if lower.contains(where: { $0.hasSuffix(".gguf") }) { return (.gguf, hasVision) }
    if names.contains("config.json"), lower.contains(where: { $0.hasSuffix(".safetensors") }) {
        return (.mlx, hasVision)
    }
    if lower.contains(where: { $0.hasSuffix(".safetensors") }) { return (.mlx, hasVision) }
    return (.unknown, hasVision)
}

/// The bytes a directory tree holds, or nil when it cannot be read. Metadata
/// only, so a 40 GB model is measured in a moment.
func modelDirectorySize(_ path: String) -> Int64? {
    let url = URL(fileURLWithPath: path)
    guard let enumerator = FileManager.default.enumerator(
        at: url,
        includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey],
        options: [.skipsHiddenFiles]
    ) else { return nil }
    var total: Int64 = 0
    for case let file as URL in enumerator {
        let values = try? file.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
        if values?.isRegularFile == true { total += Int64(values?.fileSize ?? 0) }
    }
    return total
}

/// A byte count as the finder writes it, for the directory list.
func formatBytes(_ bytes: Int64) -> String {
    let formatter = ByteCountFormatter()
    formatter.countStyle = .file
    return formatter.string(fromByteCount: bytes)
}
