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

    /// Where the launcher writes `serve-PORT.lock` and `serve-PORT.json`
    /// (`install/paths.py`): the app's Application Support when packaged,
    /// the checkout's build directory in development.
    static var serveDirectory: URL? {
        guard let root else { return nil }
        if FileManager.default.fileExists(atPath: root.appendingPathComponent("release.json").path) {
            return FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("Splash/runtime", isDirectory: true)
        }
        return root.appendingPathComponent("build/runtime", isDirectory: true)
    }

    /// What the launcher or the server last wrote of a start on `port`
    /// (`server/serve_record.py`), nil before one has.
    static func serveRecord(port: Int) -> ServeRecord? {
        guard let directory = serveDirectory else { return nil }
        let file = directory.appendingPathComponent("serve-\(port).json")
        guard let data = try? Data(contentsOf: file) else { return nil }
        return try? JSONDecoder().decode(ServeRecord.self, from: data)
    }

    /// The model IDs the suggested list bundles, for the picker: the first
    /// column of `install/suggested-models.tsv`, comments and layout aside.
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

/// What the launcher or the server writes of a start, as `server/serve_record.py`
/// lays it out: the process, the phase of its start, and, for a start that
/// failed, why.
struct ServeRecord: Decodable, Equatable {
    enum Phase: String, Decodable {
        case starting, loading, ready, failed
    }

    struct Failure: Decodable, Equatable {
        let kind: String
        let message: String
    }

    let pid: Int32
    let phase: Phase
    let failure: Failure?
}
