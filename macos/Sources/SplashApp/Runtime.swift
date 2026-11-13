import Foundation

/// The Rust server and native engine embedded under Contents/Resources/runtime.
enum Runtime {
    /// `SPLASH_RUNTIME_DIR` overrides the embedded runtime, for development
    /// against a source checkout or another build.
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
        let server = directory.appendingPathComponent("engine/splash-server")
        return FileManager.default.isExecutableFile(atPath: server.path)
    }

    static var server: URL? {
        root?.appendingPathComponent("engine/splash-server")
    }

    /// The model IDs the suggested list bundles, for the picker: the first
    /// column of `suggested-models.tsv`, comments and layout aside.
    static var catalog: [String] {
        guard let root else { return [] }
        let file = root.appendingPathComponent("suggested-models.tsv")
        guard let text = try? String(contentsOf: file, encoding: .utf8) else { return [] }
        var seen = Set<String>()
        var identifiers: [String] = []
        for line in text.split(separator: "\n") {
            guard let field = line.split(separator: "\t", maxSplits: 1).first else { continue }
            let candidate = field.trimmingCharacters(in: .whitespaces)
            if !candidate.isEmpty, !candidate.hasPrefix("#"), seen.insert(candidate).inserted {
                identifiers.append(candidate)
            }
        }
        return identifiers
    }
}
