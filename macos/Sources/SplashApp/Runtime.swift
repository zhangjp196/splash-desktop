import Foundation

/// The Python server runtime embedded under Contents/Resources/runtime.
enum Runtime {
    /// `SPLASH_RUNTIME_DIR` overrides the embedded runtime for development.
    static var root: URL? {
        let environment = ProcessInfo.processInfo.environment["SPLASH_RUNTIME_DIR"]
        let candidates = [environment.map { URL(fileURLWithPath: $0, isDirectory: true) },
                          embedded]
            .compactMap { $0 }
        return candidates.first(where: isRuntime)
    }

    private static var embedded: URL? {
        guard let resources = Bundle.main.resourceURL else { return nil }
        return resources.appendingPathComponent("runtime", isDirectory: true)
    }

    private static func isRuntime(_ directory: URL) -> Bool {
        FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("install/launcher.py").path
        ) && FileManager.default.isExecutableFile(
            atPath: directory.appendingPathComponent("python/bin/python3").path
        )
    }

    static var python: URL? {
        root?.appendingPathComponent("python/bin/python3")
    }

    static var launcher: URL? {
        root?.appendingPathComponent("install/launcher.py")
    }

    /// The models bundled for the picker, from `suggested-models.tsv`: one row
    /// per ID, with the model's name, the Mac memory its variant is meant for,
    /// its download size and a few words for choosing it.
    static var catalog: [SuggestedModel] {
        guard let root else { return [] }
        let file = root.appendingPathComponent("install/suggested-models.tsv")
        guard let text = try? String(contentsOf: file, encoding: .utf8) else { return [] }
        var seen = Set<String>()
        var models: [SuggestedModel] = []
        for line in text.split(separator: "\n") {
            let fields = line.split(separator: "\t", omittingEmptySubsequences: false)
            guard let first = fields.first else { continue }
            let identifier = first.trimmingCharacters(in: .whitespaces)
            guard !identifier.isEmpty, !identifier.hasPrefix("#"),
                  seen.insert(identifier).inserted
            else { continue }
            func field(_ index: Int) -> String {
                guard index < fields.count else { return "" }
                return fields[index].trimmingCharacters(in: .whitespaces)
            }
            models.append(SuggestedModel(
                id: identifier,
                name: field(1).isEmpty ? identifier : field(1),
                memoryGB: Int(field(2)) ?? 0,
                sizeGB: Int(field(3)) ?? 0,
                about: field(4)
            ))
        }
        return models
    }
}

/// A model `suggested-models.tsv` suggests, and what choosing it tells the user.
struct SuggestedModel: Identifiable, Hashable {
    /// The ID `splash serve --model` takes.
    let id: String
    /// The model's name, shared by its variants.
    let name: String
    /// The smallest Mac memory in GB the variant is meant for.
    let memoryGB: Int
    /// The download in whole GB: the model, its vision projector if it has
    /// one, and its DFlash2 draft.
    let sizeGB: Int
    /// A few words for choosing among the models.
    let about: String

    /// The ID's variant: a GGUF file's quantization after its colon, or how an
    /// MLX repository was converted; empty where the ID says neither.
    var variant: String {
        if let colon = id.firstIndex(of: ":") {
            return String(id[id.index(after: colon)...])
        }
        for marker in ["4bit", "6bit", "8bit", "bf16", "fp16"] where id.hasSuffix("-" + marker) {
            return marker
        }
        return ""
    }

    /// Whether the weights are a GGUF file or an MLX conversion.
    var kind: ModelKind {
        let lower = id.lowercased()
        return lower.contains("gguf") ? .gguf : .mlx
    }

    enum ModelKind: String, CaseIterable, Identifiable {
        // MLX first, as the models pane lists its kinds.
        case mlx
        case gguf

        var id: String { rawValue }

        /// The name the pane shows, as the model settings name the two.
        var title: String {
            L10n.string(rawValue == "gguf" ? "models.kind.gguf" : "models.kind.mlx")
        }
    }
}
