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

    /// The model IDs bundled for the picker, from `suggested-models.tsv`.
    static var catalog: [String] {
        guard let root else { return [] }
        let file = root.appendingPathComponent("install/suggested-models.tsv")
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
