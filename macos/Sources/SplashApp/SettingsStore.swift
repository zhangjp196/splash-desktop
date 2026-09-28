import Foundation
import SQLite3

// SQLITE_TRANSIENT is a C macro; Swift sees the destructor type only.
private let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

/// Persists the control panel's settings in SQLite, at
/// ~/Library/Application Support/<bundle id>/settings.db, so the next launch
/// opens on the last run's configuration. One key/value row per setting and
/// `ServerSettings` as the schema, so a new setting needs no migration.
/// Uses the system libsqlite3, so the Swift package stays free of external
/// dependencies.
final class SettingsStore {
    private var db: OpaquePointer?

    /// `SPLASH_SETTINGS_DB` points the store at another file, which the
    /// headless selfcheck uses to stay out of the user's real settings.
    init(path override: String? = nil) {
        let explicit = override ?? ProcessInfo.processInfo.environment["SPLASH_SETTINGS_DB"]
        if let explicit, !explicit.isEmpty {
            open(at: explicit)
            return
        }
        guard let folder = Self.dataDirectory() else { return }
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        open(at: folder.appendingPathComponent("settings.db").path)
    }

    deinit {
        sqlite3_close(db)
    }

    /// The stored settings, or the first-launch values when the database is
    /// new, unreadable or empty. A store that cannot open still answers, so a
    /// missing application-support folder never blocks the panel.
    func load() -> ServerSettings {
        ServerSettings(rows: storedRows())
    }

    /// Whether anything has been stored yet, so the first launch can record
    /// the values it opened with instead of leaving the table empty.
    var hasStoredSettings: Bool {
        !storedRows().isEmpty
    }

    /// Replaces the stored settings with `settings` in one transaction, so a
    /// save is either the whole previous run's configuration or none of it.
    @discardableResult
    func save(_ settings: ServerSettings) -> Bool {
        guard let db else { return false }
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(db, Self.upsert, -1, &statement, nil) == SQLITE_OK else {
            return false
        }
        defer { sqlite3_finalize(statement) }
        guard sqlite3_exec(db, Self.begin, nil, nil, nil) == SQLITE_OK else { return false }
        for (key, value) in settings.rows {
            sqlite3_reset(statement)
            sqlite3_clear_bindings(statement)
            sqlite3_bind_text(statement, 1, key, -1, SQLITE_TRANSIENT)
            sqlite3_bind_text(statement, 2, value, -1, SQLITE_TRANSIENT)
            guard sqlite3_step(statement) == SQLITE_DONE else {
                sqlite3_exec(db, Self.rollback, nil, nil, nil)
                return false
            }
        }
        return sqlite3_exec(db, Self.commit, nil, nil, nil) == SQLITE_OK
    }

    // MARK: Internals

    private func open(at path: String) {
        var connection: OpaquePointer?
        guard sqlite3_open(path, &connection) == SQLITE_OK else { return }
        db = connection
        // The panel writes on the main thread while the engine serves; a
        // concurrent reader should wait rather than fail with SQLITE_BUSY.
        sqlite3_busy_timeout(db, 1_000)
        sqlite3_exec(db, Self.schema, nil, nil, nil)
    }

    private func storedRows() -> [String: String] {
        guard let db else { return [:] }
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(db, Self.selectAll, -1, &statement, nil) == SQLITE_OK else {
            return [:]
        }
        defer { sqlite3_finalize(statement) }
        var rows: [String: String] = [:]
        while sqlite3_step(statement) == SQLITE_ROW {
            guard let key = sqlite3_column_text(statement, 0),
                  let value = sqlite3_column_text(statement, 1)
            else { continue }
            rows[String(cString: key)] = String(cString: value)
        }
        return rows
    }

    private static func dataDirectory() -> URL? {
        guard let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first else {
            return nil
        }
        let identifier = Bundle.main.bundleIdentifier ?? "ai.inco.splash"
        return base.appendingPathComponent(identifier)
    }

    private static let schema = """
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY NOT NULL,
            value TEXT NOT NULL DEFAULT ''
        );
        """

    private static let selectAll = "SELECT key, value FROM settings;"
    private static let upsert = """
        INSERT INTO settings (key, value) VALUES (?1, ?2)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value;
        """
    private static let begin = "BEGIN IMMEDIATE;"
    private static let commit = "COMMIT;"
    private static let rollback = "ROLLBACK;"
}
