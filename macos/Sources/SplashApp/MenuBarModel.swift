import SwiftUI

/// The menu bar's own reading of the app, refreshed at a walking pace.
///
/// The live board polls the engine once a second, and the bridge in
/// `AppModel` re-announces every poll on the app model, so the menu bar's
/// content — and with it the menu host's items — used to be rebuilt once a
/// second for as long as the server ran. A re-entrant menu update in that
/// churn can recurse until the thread's stack is gone and the app dies
/// (SplashDesktop-2026-11-12-190816.ips, `MenuBehavior
/// .menuHostDidChangeMenuItems` cycling into `ViewGraph.updateOutputs`).
/// The menu bar now reads this snapshot instead of the app model: a phase
/// change lands at once, the metrics at most every `pace` seconds, and a
/// refresh reads the app model only after its published values have landed.
@MainActor
final class MenuBarModel: ObservableObject {
    /// The metrics refresh at most this often. A phase change refreshes at
    /// once: the Start and Stop buttons must answer immediately.
    static let pace: TimeInterval = 5

    @Published private(set) var phase: AppModel.Phase = .idle
    @Published private(set) var stage: RequestStage = .idle
    @Published private(set) var isRunning = false
    @Published private(set) var canStart = false
    @Published private(set) var statusText = ""
    @Published private(set) var symbol = "bolt.horizontal.circle"
    @Published private(set) var decodeTokensPerSecond: Double?
    @Published private(set) var prefillTokensPerSecond: Double?
    @Published private(set) var pending = 0
    @Published private(set) var decodeHistory: [Double] = []
    @Published private(set) var physicalBytes: UInt64?
    @Published private(set) var memoryLimitBytes: UInt64?
    @Published private(set) var uptimeText: String?

    /// When the metrics last walked. The distant past makes the first
    /// refresh read them.
    private var lastMetrics = Date.distantPast
    /// Whether a deferred refresh is already queued, so a burst of app
    /// model changes schedules one read, not one per change.
    private var refreshPending = false

    /// Read the app model on a later turn of the main actor, where its
    /// published values have landed. Cheap to call often: a pending read
    /// coalesces the ones that arrive meanwhile.
    func scheduleRefresh() {
        guard !refreshPending else { return }
        refreshPending = true
        Task { @MainActor [weak self] in
            self?.refreshPending = false
            self?.refresh()
        }
    }

    private func refresh() {
        let model = AppModel.shared
        let live = model.live
        // The phase's faces: what the status line, the beacon, the stage
        // label and the menu bar icon read. A change here publishes at once,
        // so the Start and Stop buttons and the icon answer immediately.
        let status = model.statusText
        let icon = model.menuBarSymbol
        let running = model.isRunning
        let startable = model.canStart
        let phaseChanged = status != statusText || icon != symbol
            || running != isRunning || startable != canStart
        if phase != model.phase {
            phase = model.phase
        }
        if stage != model.stage {
            stage = model.stage
        }
        if statusText != status { statusText = status }
        if symbol != icon { symbol = icon }
        if isRunning != running { isRunning = running }
        if canStart != startable { canStart = startable }
        // The metrics walk at the pace; a phase change brings them along,
        // so a fresh dashboard reads whole.
        guard phaseChanged || Date().timeIntervalSince(lastMetrics) >= Self.pace
        else { return }
        lastMetrics = Date()
        if decodeTokensPerSecond != live.decodeTokensPerSecond {
            decodeTokensPerSecond = live.decodeTokensPerSecond
        }
        if prefillTokensPerSecond != live.prefillTokensPerSecond {
            prefillTokensPerSecond = live.prefillTokensPerSecond
        }
        if pending != live.pending { pending = live.pending }
        if decodeHistory != model.decodeHistory {
            decodeHistory = model.decodeHistory
        }
        if physicalBytes != live.physicalBytes {
            physicalBytes = live.physicalBytes
        }
        let limit = live.configuredMemoryLimitBytes ?? live.governorLimitBytes
        if memoryLimitBytes != limit { memoryLimitBytes = limit }
        if uptimeText != model.uptimeText { uptimeText = model.uptimeText }
    }
}
