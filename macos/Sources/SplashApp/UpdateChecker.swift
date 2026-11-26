import Foundation

@MainActor
final class UpdateChecker: ObservableObject {
    static let shared = UpdateChecker()
    static let releaseURL = URL(string: "https://github.com/zhangjp196/splash-desktop/releases/latest")!
    private static let interval: TimeInterval = 24 * 60 * 60
    private let defaults = UserDefaults.standard
    private var task: Task<Void, Never>?

    @Published private(set) var statusKey = "update.status.idle"
    @Published private(set) var latestVersion: String?
    @Published private(set) var isChecking = false

    private init() {}

    func checkIfDue(enabled: Bool) {
        guard enabled, Date().timeIntervalSince(defaults.object(forKey: "updates.lastCheck") as? Date ?? .distantPast) >= Self.interval else { return }
        check(force: false)
    }

    func checkManually() {
        check(force: true)
    }

    private func check(force: Bool) {
        guard !isChecking else { return }
        if !force, Date().timeIntervalSince(defaults.object(forKey: "updates.lastCheck") as? Date ?? .distantPast) < Self.interval { return }
        isChecking = true
        statusKey = "update.status.checking"
        defaults.set(Date(), forKey: "updates.lastCheck")
        task = Task {
            defer { isChecking = false }
            do {
                var request = URLRequest(url: Self.releaseURL)
                request.setValue("SplashDesktop", forHTTPHeaderField: "User-Agent")
                request.timeoutInterval = 15
                let (_, response) = try await URLSession.shared.data(for: request)
                guard let response = response as? HTTPURLResponse,
                      (200..<300).contains(response.statusCode),
                      let finalURL = response.url,
                      let tag = Self.latestTag(from: finalURL),
                      let current = Self.versionComponents(Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String),
                      let latest = Self.versionComponents(tag) else {
                    statusKey = "update.status.unavailable"
                    return
                }
                latestVersion = tag
                statusKey = latest.lexicographicallyPrecedes(current) || latest == current
                    ? "update.status.current" : "update.status.available"
            } catch {
                statusKey = "update.status.failed"
            }
        }
    }

    private static func latestTag(from url: URL) -> String? {
        guard url.scheme?.lowercased() == "https",
              url.host?.lowercased() == "github.com",
              url.user == nil,
              url.password == nil,
              url.port == nil else { return nil }
        let components = url.pathComponents
        guard components.count == 6,
              components[1] == "zhangjp196",
              components[2] == "splash-desktop",
              components[3] == "releases",
              components[4] == "tag" else { return nil }
        return components[5]
    }

    private static func versionComponents(_ value: String?) -> [Int]? {
        guard var value, !value.isEmpty else { return nil }
        if value.first == "v" || value.first == "V" { value.removeFirst() }
        let parts = value.split(separator: ".", omittingEmptySubsequences: false)
        guard !parts.isEmpty else { return nil }
        var numbers: [Int] = []
        for part in parts {
            guard !part.isEmpty,
                  part.utf8.allSatisfy({ (48...57).contains($0) }),
                  let number = Int(part) else { return nil }
            numbers.append(number)
        }
        while numbers.last == 0 && numbers.count > 1 { numbers.removeLast() }
        return numbers
    }
}
