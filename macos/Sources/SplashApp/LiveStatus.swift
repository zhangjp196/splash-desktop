import Foundation

/// The value types the live board is drawn from: what `/status` decoded into,
/// the stage inferred between polls, the request outcomes the board remembers,
/// and the temperature readings the fans answer to. They live apart from the
/// app model because the panels take them as plain inputs — the equatable
/// cards compare these values, not the model that holds them.

/// What the live panel shows, taken from `/status` on a one-second loop.
struct LiveStatus: Equatable {
    var updated = false
    var ready = false
    var modelID: String?
    var contextTokens: Int?
    var decodeTokensPerSecond: Double?
    var prefillTokensPerSecond: Double?
    var draftAcceptanceRate: Double?
    /// The installed speculation plan the engine runs: provider, proposal
    /// rows, and why a requested one fell back to none.
    var speculationProvider: String?
    var speculationRows: Int?
    var speculationFallback: String?
    /// The engine's request latency histograms, in milliseconds. `samples` is
    /// the window the percentiles cover; zero means the engine has not
    /// completed a request since it started.
    var ttftP50Milliseconds: Double?
    var ttftP95Milliseconds: Double?
    var ttftSamples = 0
    var itlP50Milliseconds: Double?
    var itlP95Milliseconds: Double?
    var itlSamples = 0
    /// The fastest the engine has been seen to run since it started: the peak
    /// behind each rate above, kept by the panel because /status reports the
    /// engine's lifetime average, not its best batch.
    var peakDecodeTokensPerSecond: Double?
    var peakPrefillTokensPerSecond: Double?
    var peakDraftAcceptanceRate: Double?
    var submitted = 0
    var completed = 0
    var failed = 0
    var cancelled = 0
    var pending = 0
    var pendingLimit = 0
    var currentBytes: UInt64?
    var peakBytes: UInt64?
    /// Host memory the engine actually occupies; the allocated bytes above
    /// are address space whose weight pages the host reclaims when idle.
    var physicalBytes: UInt64?
    var peakPhysicalBytes: UInt64?
    /// The budget the memory governor holds the engine to, and the ceiling the
    /// panel named for it (`--max-memory`): nil while the budget is automatic.
    var governorLimitBytes: UInt64?
    var configuredMemoryLimitBytes: UInt64?
    /// The weights' idle release: whether their memory is released now,
    /// whether a request is taking it back, and the timer that does it
    /// (nil for `--idle-release off`).
    var weightsKnown = false
    var weightsReleased = false
    var weightsRestoring = false
    var weightsRestores = 0
    var idleReleaseSeconds: Double?
    var cacheHitRate: Double?
    var kvDiskHitTokens: Int?
    /// The share of the tokens requests reused that came back from the disk
    /// tier: `kv_disk_hit_rate`, the token-level reading `hit_rate` cannot
    /// give, since it counts requests.
    var kvDiskHitRate: Double?
    /// What requests resumed from, the denominator of the rate above.
    var reusedTokens: Int?
    /// States a lookup found on the disk tier (`state.disk_hits`).
    var stateDiskHits: Int?
    var waiting = 0
    var waitingMemory = 0
    var waitingConcurrency = 0
    var suspended = 0
    var diskUsedBytes: UInt64?
    var diskCapacityBytes: UInt64?
    var idleOffloadPasses = 0
    var idleOffloadBytes: UInt64?
    var restarts = 0
    var capacityFailures = 0
    var metalFailures = 0
    /// The engine's fans as `/status` reports them, read-only: what the Mac is
    /// doing about the heat, which is what the thermal rungs trade against.
    var fansAvailable = false
    var fanSpeeds: [Int] = []
    var fanMinimums: [Int] = []
    var fanMaximums: [Int] = []
    /// The sensors the engine read, as it reported them, hottest first.
    var temperatures: [TemperatureReading] = []
    /// The prefill pacer: the duty the engine holds, the backoff the Mac's
    /// thermal state applies to it, and what the pacing has cost in waits.
    var dutyPercent: Double?
    var thermalScale: Double?
    var pacedWaits: Int?
    var pacedWaitMilliseconds: Double?
}

/// What the engine appears to be doing right now, inferred from the counters
/// between two `/status` polls: a submitted request lights prefill, a completed
/// batch lights decode, a pending queue lights the queue, and a quiet engine
/// settles to idle. The polling interval is one second, so the view holds a
/// stage for a moment after its edge instead of flickering.
enum RequestStage: String, CaseIterable, Identifiable {
    case idle, queued, prefilling, decoding

    var id: String { rawValue }

    var title: String { L10n.string("stage." + rawValue) }

    var systemImage: String {
        switch self {
        case .idle: return "pause.fill"
        case .queued: return "hourglass"
        case .prefilling: return "arrow.down.to.line"
        case .decoding: return "bolt.fill"
        }
    }
}

/// One request outcome the live board remembers for a few minutes, so the
/// dashboard has a short history of what happened rather than only totals.
struct RequestEvent: Identifiable, Equatable {
    enum Kind: String {
        case completed, failed, cancelled
    }

    let id = UUID()
    let date: Date
    let kind: Kind
    let count: Int
    let tokensPerSecond: Double?
}

/// One temperature sensor's reading: the engine reads them from the SMC, so
/// they are what the fans themselves answer to.
struct TemperatureReading: Equatable, Identifiable {
    var sensor: String
    var celsius: Double
    var id: String { sensor }
}

/// The four temperatures the panel shows, each aggregated from the SMC keys
/// that belong to one part of the Mac: the CPU clusters (`Tp`, `Te`), the GPU
/// (`Tg`), the IO fabric (`Tf`), and the storage (`Ts`). A Mac that reports
/// none of a group's keys leaves that row without a reading.
enum TemperatureGroup: String, CaseIterable, Identifiable {
    case cpu, gpu, io, disk

    var id: String { rawValue }

    var title: String { L10n.string("thermal.sensor." + rawValue) }

    func contains(_ sensor: String) -> Bool {
        switch self {
        case .cpu: return sensor.hasPrefix("Tp") || sensor.hasPrefix("Te")
        case .gpu: return sensor.hasPrefix("Tg")
        case .io: return sensor.hasPrefix("Tf")
        case .disk: return sensor.hasPrefix("Ts")
        }
    }

    /// The hottest sensor of this group, or nil when the Mac reports none.
    static func peak(_ readings: [TemperatureReading], in group: TemperatureGroup) -> Double? {
        readings.lazy
            .filter { group.contains($0.sensor) }
            .map(\.celsius)
            .max()
    }

    /// The group whose hottest sensor is the hottest of the sample.
    static func hottest(
        _ readings: [TemperatureReading]
    ) -> (group: TemperatureGroup, celsius: Double)? {
        allCases
            .compactMap { group in peak(readings, in: group).map { (group, $0) } }
            .max { $0.celsius < $1.celsius }
    }
}
