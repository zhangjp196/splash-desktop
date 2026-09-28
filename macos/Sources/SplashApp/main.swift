import Foundation

// A single executable backs both the windowed app and a small headless probe.
// The packaging and tests use `--print-runtime` to check that the app finds
// the runtime the DMG embeds; `--selfcheck-models` exercises the SQLite model
// library; otherwise SwiftUI owns the process.
if CommandLine.arguments.contains("--print-runtime") {
    let runtime = Runtime.root
    print(runtime?.path ?? "")
    if let runtime {
        print("python: \(Runtime.python?.path ?? "missing")")
        print("launcher: \(Runtime.launcher?.path ?? "missing")")
        print("catalog: \(Runtime.catalog.count) model(s)")
        _ = runtime
        exit(0)
    }
    exit(1)
}

if CommandLine.arguments.contains("--selfcheck-models") {
    let path = "/tmp/" + ProcessInfo.processInfo.globallyUniqueString + "-models.db"
    setenv("SPLASH_MODELS_DB", path, 1)
    let store = ModelStore()
    var ok = true
    var sample = StoredModel(name: "Check Model", mode: "upstream", modelID: "owner/repo")
    guard let id = store.upsert(sample) else {
        print("model store selfcheck: FAIL (insert)")
        exit(1)
    }
    sample.id = id
    if !store.models().contains(where: { $0.id == id && $0.name == "Check Model" }) {
        print("model store selfcheck: FAIL (fetch)")
        ok = false
    }
    var edited = sample
    edited.name = "Renamed"
    edited.port = 8123
    store.upsert(edited)
    if !store.models().contains(where: { $0.name == "Renamed" && $0.port == 8123 }) {
        print("model store selfcheck: FAIL (update)")
        ok = false
    }
    store.delete(id: id)
    if store.models().contains(where: { $0.id == id }) {
        print("model store selfcheck: FAIL (delete)")
        ok = false
    }
    try? FileManager.default.removeItem(atPath: path)
    print(ok ? "model store selfcheck: PASS" : "model store selfcheck: FAIL (delete-verify)")
    exit(ok ? 0 : 1)
}

SplashApp.main()
