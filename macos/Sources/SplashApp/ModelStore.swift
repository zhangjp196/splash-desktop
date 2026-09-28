import Foundation
import SQLite3

// SQLITE_TRANSIENT is a C macro; Swift sees the destructor type only.
private let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

/// One saved model entry: the source the launcher is asked for plus the
/// server settings, persisted next to the app.
struct StoredModel: Identifiable, Equatable {
    var id: Int64?
    var name: String
    var mode: String
    var modelID = ""
    var modelDirectory = ""
    var draftDirectory = ""
    var port = 8000
    var languageOnly = false
    var kvFormat = "int8"
    var maxMemory = ""
    var maxContext = ""
    var maxCacheDisk = ""
    var maxRequestSize = ""
    var apiKey = ""
    var servedNames = ""
    var reasoningEffort = ""
    var createdAt = Date()
    var updatedAt = Date()

    var isNew: Bool { id == nil }
}

/// The SQLite store for the model library, at
/// ~/Library/Application Support/<bundle id>/models.db. Uses the system
/// libsqlite3, so the Swift package stays free of external dependencies.
final class ModelStore {
    private var db: OpaquePointer?

    init() {
        if let override = ProcessInfo.processInfo.environment["SPLASH_MODELS_DB"], !override.isEmpty {
            var connection: OpaquePointer?
            guard sqlite3_open(override, &connection) == SQLITE_OK else { return }
            db = connection
            sqlite3_exec(db, Self.schema, nil, nil, nil)
            return
        }
        guard let folder = Self.dataDirectory() else { return }
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        var connection: OpaquePointer?
        let path = folder.appendingPathComponent("models.db").path
        guard sqlite3_open(path, &connection) == SQLITE_OK else { return }
        db = connection
        sqlite3_exec(db, Self.schema, nil, nil, nil)
    }

    deinit {
        sqlite3_close(db)
    }

    // MARK: Queries

    func models() -> [StoredModel] {
        guard let db else { return [] }
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(db, Self.selectAll, -1, &statement, nil) == SQLITE_OK else {
            return []
        }
        defer { sqlite3_finalize(statement) }
        var result: [StoredModel] = []
        while sqlite3_step(statement) == SQLITE_ROW {
            result.append(read(statement))
        }
        return result
    }

    @discardableResult
    func upsert(_ model: StoredModel) -> Int64? {
        guard let db else { return nil }
        if let id = model.id {
            guard update(id, model) else { return nil }
            return id
        }
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(db, Self.insert, -1, &statement, nil) == SQLITE_OK else {
            return nil
        }
        defer { sqlite3_finalize(statement) }
        bind(model, statement, at: 1)
        guard sqlite3_step(statement) == SQLITE_DONE else { return nil }
        return sqlite3_last_insert_rowid(db)
    }

    func delete(id: Int64) {
        guard let db else { return }
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(db, Self.deleteRow, -1, &statement, nil) == SQLITE_OK else {
            return
        }
        defer { sqlite3_finalize(statement) }
        sqlite3_bind_int64(statement, 1, id)
        sqlite3_step(statement)
    }

    // MARK: Helpers

    private func update(_ id: Int64, _ model: StoredModel) -> Bool {
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(db, Self.update, -1, &statement, nil) == SQLITE_OK else {
            return false
        }
        defer { sqlite3_finalize(statement) }
        bind(model, statement, at: 1)
        sqlite3_bind_int64(statement, 18, id)
        return sqlite3_step(statement) == SQLITE_DONE
    }

    private func bind(_ model: StoredModel, _ statement: OpaquePointer?, at first: Int32) {
        sqlite3_bind_text(statement, first, model.name, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(statement, first + 1, model.mode, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(statement, first + 2, model.modelID, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(statement, first + 3, model.modelDirectory, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(statement, first + 4, model.draftDirectory, -1, SQLITE_TRANSIENT)
        sqlite3_bind_int64(statement, first + 5, Int64(model.port))
        sqlite3_bind_int(statement, first + 6, model.languageOnly ? 1 : 0)
        sqlite3_bind_text(statement, first + 7, model.kvFormat, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(statement, first + 8, model.maxMemory, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(statement, first + 9, model.maxContext, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(statement, first + 10, model.maxCacheDisk, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(statement, first + 11, model.maxRequestSize, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(statement, first + 12, model.apiKey, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(statement, first + 13, model.servedNames, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(statement, first + 14, model.reasoningEffort, -1, SQLITE_TRANSIENT)
        sqlite3_bind_double(statement, first + 15, model.createdAt.timeIntervalSince1970)
        sqlite3_bind_double(statement, first + 16, model.updatedAt.timeIntervalSince1970)
    }

    private func read(_ statement: OpaquePointer?) -> StoredModel {
        StoredModel(
            id: sqlite3_column_int64(statement, 0),
            name: text(statement, 1),
            mode: text(statement, 2),
            modelID: text(statement, 3),
            modelDirectory: text(statement, 4),
            draftDirectory: text(statement, 5),
            port: Int(sqlite3_column_int64(statement, 6)),
            languageOnly: sqlite3_column_int(statement, 7) != 0,
            kvFormat: text(statement, 8),
            maxMemory: text(statement, 9),
            maxContext: text(statement, 10),
            maxCacheDisk: text(statement, 11),
            maxRequestSize: text(statement, 12),
            apiKey: text(statement, 13),
            servedNames: text(statement, 14),
            reasoningEffort: text(statement, 15),
            createdAt: Date(timeIntervalSince1970: sqlite3_column_double(statement, 16)),
            updatedAt: Date(timeIntervalSince1970: sqlite3_column_double(statement, 17))
        )
    }

    private func text(_ statement: OpaquePointer?, _ index: Int32) -> String {
        guard let value = sqlite3_column_text(statement, index) else { return "" }
        return String(cString: value)
    }

    private static func dataDirectory() -> URL? {
        guard let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first else {
            return nil
        }
        let identifier = Bundle.main.bundleIdentifier ?? "ai.inco.splash"
        return base.appendingPathComponent(identifier)
    }

    private static let schema = """
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            mode TEXT NOT NULL,
            model_id TEXT NOT NULL DEFAULT '',
            model_directory TEXT NOT NULL DEFAULT '',
            draft_directory TEXT NOT NULL DEFAULT '',
            port INTEGER NOT NULL DEFAULT 8000,
            language_only INTEGER NOT NULL DEFAULT 0,
            kv_format TEXT NOT NULL DEFAULT 'int8',
            max_memory TEXT NOT NULL DEFAULT '',
            max_context TEXT NOT NULL DEFAULT '',
            max_cache_disk TEXT NOT NULL DEFAULT '',
            max_request_size TEXT NOT NULL DEFAULT '',
            api_key TEXT NOT NULL DEFAULT '',
            served_names TEXT NOT NULL DEFAULT '',
            reasoning_effort TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        """

    private static let selectAll = """
        SELECT id, name, mode, model_id, model_directory, draft_directory, port,
               language_only, kv_format, max_memory, max_context, max_cache_disk,
               max_request_size, api_key, served_names, reasoning_effort,
               created_at, updated_at
        FROM models ORDER BY updated_at DESC, id DESC;
        """

    private static let insert = """
        INSERT INTO models (name, mode, model_id, model_directory, draft_directory,
            port, language_only, kv_format, max_memory, max_context, max_cache_disk,
            max_request_size, api_key, served_names, reasoning_effort, created_at, updated_at)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17);
        """

    private static let update = """
        UPDATE models SET name=?1, mode=?2, model_id=?3, model_directory=?4,
            draft_directory=?5, port=?6, language_only=?7, kv_format=?8,
            max_memory=?9, max_context=?10, max_cache_disk=?11, max_request_size=?12,
            api_key=?13, served_names=?14, reasoning_effort=?15, created_at=?16,
            updated_at=?17
        WHERE id=?18;
        """

    private static let deleteRow = "DELETE FROM models WHERE id=?1;"
}