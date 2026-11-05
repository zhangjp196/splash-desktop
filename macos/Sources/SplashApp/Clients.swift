import Foundation

/// One process with an established connection to the inference port, as the
/// app shows it in the live board. The PID is the identity: one process may
/// hold several sockets (a chat window and a background stream), which are
/// folded into `connections`.
struct ConnectedClient: Identifiable, Equatable {
    enum Kind: String {
        case agent, code, chat, terminal, other

        var sortOrder: Int {
            switch self {
            case .agent: return 0
            case .chat: return 1
            case .code: return 2
            case .terminal: return 3
            case .other: return 4
            }
        }
    }

    let pid: Int32
    let name: String
    let command: String
    let connections: Int
    let kind: Kind

    var id: String { "\(pid)" }
}

/// Reads the local TCP table for the inference port. `lsof` is the one tool on
/// macOS that names both ends of a connection without a privileged helper; it
/// is run off the main actor and its output is parsed into clients.
actor ClientScanner {
    func scan(port: Int, excluding: Set<Int32>) async -> [ConnectedClient] {
        await Task.detached(priority: .utility) {
            Self.scanBlocking(port: port, excluding: excluding)
        }.value
    }

    private static func scanBlocking(
        port: Int,
        excluding: Set<Int32>
    ) -> [ConnectedClient] {
        let path = "/usr/sbin/lsof"
        guard FileManager.default.isExecutableFile(atPath: path) else { return [] }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: path)
        process.arguments = [
            "-nP",
            "-iTCP:\(port)",
            "-sTCP:ESTABLISHED",
            "-F", "pcn",
        ]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
        } catch {
            return []
        }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard let text = String(data: data, encoding: .utf8), !text.isEmpty else {
            return []
        }

        // The server's own accepted sockets are established too; the listener
        // is the process to leave out of the client list.
        var excluded = excluding
        excluded.formUnion(listenerPIDs(port: port))

        var current: Int32?
        var names: [Int32: String] = [:]
        var counts: [Int32: Int] = [:]
        for raw in text.split(separator: "\n") {
            let line = String(raw)
            guard let field = line.first else { continue }
            switch field {
            case "p":
                current = Int32(line.dropFirst())
            case "c":
                if let pid = current {
                    names[pid] = String(line.dropFirst())
                }
            case "n":
                if let pid = current {
                    counts[pid, default: 0] += 1
                }
            default:
                break
            }
        }

        return names.compactMap { pid, command -> ConnectedClient? in
            guard !excluded.contains(pid), let connections = counts[pid] else {
                return nil
            }
            let (name, kind) = classify(command)
            return ConnectedClient(
                pid: pid,
                name: name,
                command: command,
                connections: connections,
                kind: kind
            )
        }
        .sorted {
            if $0.kind.sortOrder != $1.kind.sortOrder {
                return $0.kind.sortOrder < $1.kind.sortOrder
            }
            return $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending
        }
    }

    /// The name a person recognizes, and the icon family to give it. Commands
    /// are matched case-insensitively and in the order that avoids a generic
    /// match swallowing a specific one (`opencode` before `code`).
    private static func classify(_ command: String) -> (String, ConnectedClient.Kind) {
        let lower = command.lowercased()
        let rules: [(String, String, ConnectedClient.Kind)] = [
            ("opencode", "OpenCode", .agent),
            ("claude", "Claude", .agent),
            ("codex", "Codex", .agent),
            ("dsh", "DeepSeek Harness", .agent),
            ("hermes", "Hermes", .agent),
            ("cursor", "Cursor", .code),
            ("xcode", "Xcode", .code),
            ("chatgpt", "ChatGPT", .chat),
            ("webkit", "Splash Chat", .chat),
            ("splashdesktop", "Splash", .chat),
            ("iterm", "iTerm", .terminal),
            ("terminal", "Terminal", .terminal),
            ("warp", "Warp", .terminal),
            ("python", "Python", .code),
            ("node", "Node", .code),
            ("code", "VS Code", .code),
            ("chrome", "Chrome", .other),
            ("safari", "Safari", .other),
            ("postman", "Postman", .code),
            ("curl", "curl", .terminal),
        ]
        for (needle, name, kind) in rules where lower.contains(needle) {
            return (name, kind)
        }
        let trimmed = command.trimmingCharacters(in: .whitespaces)
        let display = trimmed.isEmpty ? "Unknown" : trimmed.prefix(1).uppercased() + trimmed.dropFirst()
        return (display, .other)
    }

    /// The process listening on the inference port, read separately from the
    /// established sockets so the engine is never listed as its own client.
    private static func listenerPIDs(port: Int) -> Set<Int32> {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/sbin/lsof")
        process.arguments = [
            "-nP",
            "-iTCP:\(port)",
            "-sTCP:LISTEN",
            "-F", "p",
        ]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
        } catch {
            return []
        }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard let text = String(data: data, encoding: .utf8) else { return [] }
        var pids = Set<Int32>()
        for line in text.split(separator: "\n") where line.hasPrefix("p") {
            if let pid = Int32(line.dropFirst()) {
                pids.insert(pid)
            }
        }
        return pids
    }
}
