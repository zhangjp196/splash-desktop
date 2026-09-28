import Foundation

/// The embedded runtime the app drives: the directory `make package` stages,
/// as the DMG puts it under Contents/Resources/runtime. Nothing here knows a
/// model format; the launcher does.
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
        let launcher = directory.appendingPathComponent("install/launcher.py")
        return FileManager.default.fileExists(atPath: launcher.path)
    }

    /// The standalone interpreter packaged beside the launcher, falling back
    /// to a source checkout's virtual environment for development.
    static var python: URL? {
        guard let root else { return nil }
        let packaged = root.appendingPathComponent("python/bin/python3")
        if FileManager.default.isExecutableFile(atPath: packaged.path) {
            return packaged
        }
        let venv = root.appendingPathComponent(".venv/bin/python")
        if FileManager.default.isExecutableFile(atPath: venv.path) {
            return venv
        }
        return nil
    }

    static var launcher: URL? {
        root?.appendingPathComponent("install/launcher.py")
    }

    /// The model IDs the completion catalog bundles, for the picker.
    static var catalog: [String] {
        guard let root else { return [] }
        let directory = root.appendingPathComponent("install/completions")
        var seen = Set<String>()
        var identifiers: [String] = []
        for name in ["suggested-models.txt", "official-models.txt"] {
            let file = directory.appendingPathComponent(name)
            guard let text = try? String(contentsOf: file, encoding: .utf8) else { continue }
            for line in text.split(separator: "\n") {
                let candidate = line.trimmingCharacters(in: .whitespaces)
                if !candidate.isEmpty, !candidate.hasPrefix("#"), seen.insert(candidate).inserted {
                    identifiers.append(candidate)
                }
            }
        }
        return identifiers
    }
}
