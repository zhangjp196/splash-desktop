import Combine
import Darwin
import Foundation

/// One `/status` fetch and the model id `/v1/models` reports, both decoded in
/// the background so the live panel never blocks the main thread.
struct LiveSnapshot {
    let status: [String: Any]
    let modelID: String?
}

/// Fetches the engine's `/status` and the served model id in the background.
actor LiveStatusFetcher {
    func fetch(port: Int) async -> LiveSnapshot? {
        guard let status = await json(at: "/status", port: port) else { return nil }
        var modelID: String?
        if let models = await json(at: "/v1/models", port: port),
           let data = models["data"] as? [[String: Any]],
           let first = data.first {
            modelID = first["id"] as? String
        }
        return LiveSnapshot(status: status, modelID: modelID)
    }

    private func json(at path: String, port: Int) async -> [String: Any]? {
        guard let url = URL(string: "http://127.0.0.1:\(port)\(path)") else { return nil }
        var request = URLRequest(url: url)
        request.timeoutInterval = 2
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200,
                  let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { return nil }
            return object
        } catch {
            return nil
        }
    }
}

/// The app model's per-second half: the live board's reading, the minute of
/// history behind it, the inferred stage and the client scan, polled from
/// `/status` once a second while a server runs. It is its own observable so
/// the app model carries the settings and the server's lifecycle without the
/// poll's machinery; the app model forwards the fields the views read and
/// re-announces each change on its own `objectWillChange`, so a poll wakes
/// exactly the observers it woke before the split.
///
/// The engine reaches back into the app model for what is the app's, not the
/// poll's: the server's phase and process, the port to poll, and the thermal
/// pacing a reading feeds. The reference is weak because the app owns the
/// engine, so the pair is freed together.
@MainActor
final class LiveEngine: ObservableObject {
    /// The app model this engine polls for. Set once by the app's init,
    /// before any poll can run.
    private weak var app: AppModel?

    func attach(to app: AppModel) {
        self.app = app
    }

    @Published var live = LiveStatus()
    @Published var contextTokens: Int?

    // The dashboard's short memory: the last minute of the rates the live
    // board draws as sparklines. The engine reports only its current reading,
    // so the panel keeps the shape of the minute itself.
    @Published var decodeHistory: [Double] = []
    @Published var prefillHistory: [Double] = []
    @Published var acceptanceHistory: [Double] = []
    @Published var memoryHistory: [Double] = []
    /// The last few completed, failed and cancelled requests, newest last.
    @Published var recentEvents: [RequestEvent] = []
    /// Processes with an established connection to the inference port, scanned
    /// every few seconds while the server is up.
    @Published var connectedClients: [ConnectedClient] = []
    /// What the engine looks to be doing right now.
    @Published var stage: RequestStage = .idle
    /// When the current server first answered as ready, for the uptime line.
    @Published var readySince: Date?
    /// When the latest status poll landed, whatever it changed. Deliberately
    /// not `@Published`: an idle engine is polled every second and reports the
    /// same reading, and a published date here would wake every observer of
    /// the model each second for a timestamp only the header badge reads. The
    /// badge re-reads this on its own one-second TimelineView instead.
    var statusUpdatedAt: Date?

    private var poll: Task<Void, Never>?
    private var offlineCount = 0
    /// How long the inferred stage may sit at its last edge before a quiet
    /// engine settles back to idle.
    private var stageHoldUntil = Date.distantPast
    /// Read by the app model's attach probe, which fetches one snapshot
    /// without starting the loop.
    let statusFetcher = LiveStatusFetcher()
    private let clientScanner = ClientScanner()
    /// Counts status polls so the connection table is read every fourth
    /// second rather than every one. The app's start resets it with the rest
    /// of the board.
    var clientTick = 0

    /// Starts the one-second loop, cancelling any loop a previous server left
    /// running.
    func pollStatus() {
        poll?.cancel()
        poll = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                await self?.refreshStatus()
                try? await Task.sleep(nanoseconds: 1_000_000_000)
            }
        }
    }

    /// Stops the loop: a server being stopped or already gone has no status
    /// to read.
    func stopPolling() {
        poll?.cancel()
        poll = nil
    }

    /// A stopped server has no live stage and no uptime, but keeps the last
    /// minute of history so the dashboard does not blank between runs. The
    /// app model's own settle adds what is still its own.
    func settle() {
        stage = .idle
        readySince = nil
        statusUpdatedAt = nil
        stageHoldUntil = .distantPast
        connectedClients = []
        clientTick = 0
    }

    private func refreshStatus() async {
        guard let app else { return }
        guard let snapshot = await statusFetcher.fetch(port: app.global.port) else {
            offlineCount += 1
            // A foreign server we asked to stop has gone silent: done.
            if offlineCount >= 3, app.phase == .stopping {
                app.phase = .idle
                app.stopping = false
                app.foreignPID = nil
                stopPolling()
                app.settle()
            }
            return
        }
        offlineCount = 0
        applyStatus(snapshot)
        if app.phase == .ready {
            clientTick += 1
            if clientTick % 4 == 1 {
                await refreshClients()
            }
        }
    }

    /// Re-read the connection table on demand, for the clients panel's refresh
    /// button.
    func refreshConnectedClients() {
        Task { @MainActor [weak self] in
            await self?.refreshClients()
        }
    }

    private func refreshClients() async {
        guard let app else { return }
        guard app.phase == .ready else {
            // An empty board is the common answer here; writing it onto an
            // already empty board would publish for nothing.
            if !connectedClients.isEmpty { connectedClients = [] }
            return
        }
        var excluding = Set<Int32>()
        excluding.insert(getpid())
        if let pid = app.process?.processIdentifier { excluding.insert(pid) }
        if let pid = app.foreignPID { excluding.insert(pid) }
        if let pid = app.servePID(port: app.global.port) { excluding.insert(pid) }
        let scanned = await clientScanner.scan(port: app.global.port, excluding: excluding)
        guard app.phase == .ready else { return }
        if scanned != connectedClients {
            connectedClients = scanned
        }
    }

    /// The rate of the engine's most recent completed batch of one kind, when
    /// it has one: `/status` reports it as `current_*_batch`.
    private func batchRate(_ value: Any?) -> Double? {
        guard let batch = value as? [String: Any],
              (batch["valid"] as? Bool) ?? false,
              let rate = batch["tokens_per_second"] as? Double,
              rate > 0
        else { return nil }
        return rate
    }

    /// The share of the draft tokens a completed decode batch had accepted:
    /// `current_decode_batch` reports both counts, so a single good batch is a
    /// peak the lifetime ratio averages away.
    private func batchAcceptance(_ value: Any?) -> Double? {
        guard let batch = value as? [String: Any],
              (batch["valid"] as? Bool) ?? false,
              let drafted = batch["drafted_tokens"] as? Int, drafted > 0,
              let accepted = batch["accepted_draft_tokens"] as? Int
        else { return nil }
        return Double(accepted) / Double(drafted)
    }

    /// The larger of a fresh reading and what has been seen before.
    private func peak(_ value: Double?, _ previous: Double?) -> Double? {
        switch (value, previous) {
        case (nil, nil): return nil
        case (let value?, nil): return value
        case (nil, let previous?): return previous
        case (let value?, let previous?): return max(value, previous)
        }
    }

    /// Keep one minute of a reading, so the board draws its shape as well as
    /// its current value. A metric the engine has not reported yet enters as
    /// zero rather than widening the y-axis with a stale guess.
    private func record(
        _ value: Double?,
        in path: ReferenceWritableKeyPath<LiveEngine, [Double]>
    ) {
        var series = self[keyPath: path]
        series.append(max(0, value ?? 0))
        if series.count > 60 {
            series.removeFirst(series.count - 60)
        }
        self[keyPath: path] = series
    }

    /// The share of the memory budget in use, for the ring and its history.
    private func memoryFraction(_ status: LiveStatus) -> Double? {
        let limit = status.configuredMemoryLimitBytes ?? status.governorLimitBytes
        guard let used = status.physicalBytes, let limit, limit > 0 else { return nil }
        return min(1, Double(used) / Double(limit))
    }

    /// Turn the counter deltas into the live board's short activity feed. The
    /// first sample of a server is skipped: it reports totals from before the
    /// panel started watching, which are not events of this session.
    private func recordEvents(from previous: LiveStatus, to next: LiveStatus) {
        guard previous.updated else { return }
        func append(_ kind: RequestEvent.Kind, _ count: Int, rate: Double? = nil) {
            guard count > 0 else { return }
            recentEvents.append(
                RequestEvent(
                    date: Date(),
                    kind: kind,
                    count: count,
                    tokensPerSecond: rate
                )
            )
        }
        append(.completed, next.completed - previous.completed,
               rate: next.decodeTokensPerSecond)
        append(.failed, next.failed - previous.failed)
        append(.cancelled, next.cancelled - previous.cancelled)
        if recentEvents.count > 8 {
            recentEvents.removeFirst(recentEvents.count - 8)
        }
    }

    /// Read the counters that changed since the last poll, with a short hold so
    /// a live stage does not blink between one-second samples. The engine's
    /// `submitted - completed - failed - cancelled` is what is in flight; a
    /// rising prefill rate names prompt ingestion, a rising decode rate names
    /// generation, and a queue is a request waiting before it starts.
    private func advanceStage(from previous: LiveStatus, to next: LiveStatus) {
        let now = Date()
        let active = max(0, next.submitted - next.completed - next.failed - next.cancelled)
        if active > 0 {
            let prefillRose = (next.prefillTokensPerSecond ?? 0) > (previous.prefillTokensPerSecond ?? 0) + 0.5
            let decodeRose = (next.decodeTokensPerSecond ?? 0) > (previous.decodeTokensPerSecond ?? 0) + 0.5
            if prefillRose {
                setStage(.prefilling, hold: 3)
            } else if decodeRose || stage == .decoding {
                setStage(.decoding, hold: 3)
            } else {
                setStage(.prefilling, hold: 3)
            }
        } else if next.pending > 0 || next.waiting > 0 {
            setStage(.queued)
        } else if now < stageHoldUntil {
            return
        } else {
            setStage(.idle)
        }
    }

    private func setStage(_ new: RequestStage, hold: TimeInterval = 0) {
        if stage != new {
            stage = new
        }
        if hold > 0 {
            stageHoldUntil = Date().addingTimeInterval(hold)
        }
    }

    private func applyStatus(_ snapshot: LiveSnapshot) {
        guard let app else { return }
        let object = snapshot.status
        var status = LiveStatus()
        status.modelID = snapshot.modelID
        status.ready = (object["ready"] as? Bool) ?? false
        if let tokens = object["maximum_context_tokens"] as? Int {
            status.contextTokens = tokens
            // The same figure every poll is the common case; republishing it
            // would wake every observer of the model for a number nobody
            // watched change.
            if contextTokens != tokens { contextTokens = tokens }
            if app.phase == .starting { app.phase = .ready }
        }
        if let speculation = object["speculation"] as? [String: Any] {
            status.speculationProvider = speculation["provider"] as? String
            status.speculationRows = speculation["proposal_rows"] as? Int
            status.speculationFallback = speculation["fallback_reason"] as? String
        }
        if let metrics = object["metrics"] as? [String: Any] {
            status.decodeTokensPerSecond = metrics["decode_tokens_per_second"] as? Double
            status.prefillTokensPerSecond = metrics["prefill_tokens_per_second"] as? Double
            status.draftAcceptanceRate = metrics["draft_acceptance_rate"] as? Double
            if let ttft = metrics["ttft_ms"] as? [String: Any] {
                status.ttftP50Milliseconds = ttft["p50"] as? Double
                status.ttftP95Milliseconds = ttft["p95"] as? Double
                status.ttftSamples = ttft["samples"] as? Int ?? 0
            }
            if let itl = metrics["itl_ms"] as? [String: Any] {
                status.itlP50Milliseconds = itl["p50"] as? Double
                status.itlP95Milliseconds = itl["p95"] as? Double
                status.itlSamples = itl["samples"] as? Int ?? 0
            }
            status.capacityFailures = metrics["capacity_failures"] as? Int ?? 0
            status.metalFailures = metrics["metal_failures"] as? Int ?? 0
            // The peaks: the fastest completed batch the engine has reported
            // since the panel started watching it, falling back to the lifetime
            // averages while no batch has completed. They are what the Mac can
            // reach, which the averages beside them hide.
            let decode = batchRate(metrics["current_decode_batch"])
                ?? status.decodeTokensPerSecond
            let prefill = batchRate(metrics["current_prefill_batch"])
                ?? status.prefillTokensPerSecond
            let acceptance = batchAcceptance(metrics["current_decode_batch"])
                ?? status.draftAcceptanceRate
            status.peakDecodeTokensPerSecond = peak(decode, live.peakDecodeTokensPerSecond)
            status.peakPrefillTokensPerSecond = peak(prefill, live.peakPrefillTokensPerSecond)
            status.peakDraftAcceptanceRate = peak(acceptance, live.peakDraftAcceptanceRate)
        }
        if let requests = object["requests"] as? [String: Any] {
            status.submitted = requests["submitted"] as? Int ?? 0
            status.completed = requests["completed"] as? Int ?? 0
            status.failed = requests["failed"] as? Int ?? 0
            status.cancelled = requests["cancelled"] as? Int ?? 0
        }
        if let transport = object["transport"] as? [String: Any] {
            status.pending = transport["pending"] as? Int ?? 0
            status.pendingLimit = transport["pending_limit"] as? Int ?? 0
            status.restarts = transport["restarts"] as? Int ?? 0
        }
        if let memory = object["memory_actual"] as? [String: Any] {
            status.currentBytes = memory["current_bytes"] as? UInt64
            status.peakBytes = memory["peak_bytes"] as? UInt64
            status.physicalBytes = memory["physical_bytes"] as? UInt64
            status.peakPhysicalBytes = memory["peak_physical_bytes"] as? UInt64
        }
        if let governor = object["memory_governor"] as? [String: Any] {
            status.governorLimitBytes = governor["limit_bytes"] as? UInt64
        }
        if let weights = object["weights"] as? [String: Any] {
            status.weightsKnown = true
            status.weightsReleased = weights["released"] as? Bool ?? false
            status.weightsRestoring = weights["restoring"] as? Bool ?? false
            status.weightsRestores = weights["restores"] as? Int ?? 0
            // null in /status is no idle release at all (--idle-release off).
            if let seconds = weights["idle_release_seconds"] as? Double {
                status.idleReleaseSeconds = seconds
            } else if let seconds = weights["idle_release_seconds"] as? Int {
                status.idleReleaseSeconds = Double(seconds)
            }
        }
        if let plan = object["memory_plan"] as? [String: Any],
           let budget = plan["budget"] as? [String: Any] {
            let configured = budget["configured_memory_limit_bytes"] as? UInt64
            status.configuredMemoryLimitBytes = (configured ?? 0) > 0 ? configured : nil
        }
        if let cache = object["cache"] as? [String: Any] {
            status.cacheHitRate = cache["hit_rate"] as? Double
            status.kvDiskHitTokens = cache["kv_disk_hit_tokens"] as? Int
            status.kvDiskHitRate = cache["kv_disk_hit_rate"] as? Double
            status.reusedTokens = cache["reused_tokens"] as? Int
        }
        if let state = object["state"] as? [String: Any] {
            status.stateDiskHits = state["disk_hits"] as? Int
        }
        if let admission = object["admission"] as? [String: Any] {
            status.waiting = admission["waiting"] as? Int ?? 0
            status.waitingMemory = admission["waiting_memory"] as? Int ?? 0
            status.waitingConcurrency = admission["waiting_concurrency"] as? Int ?? 0
            status.suspended = admission["suspended"] as? Int ?? 0
        }
        if let disk = object["disk"] as? [String: Any] {
            status.diskUsedBytes = disk["used_bytes"] as? UInt64
            // The tier's own quota is what a used figure is a share of; the
            // volume's capacity is the size of the whole disk, not of the
            // cache, so a cache that names no quota shows no bar.
            let quota = disk["limit_bytes"] as? UInt64
            status.diskCapacityBytes = (quota ?? 0) > 0 ? quota : nil
        }
        if let offload = object["idle_offload"] as? [String: Any] {
            status.idleOffloadPasses = offload["passes"] as? Int ?? 0
            status.idleOffloadBytes = offload["bytes"] as? UInt64
        }
        if let fans = object["fans"] as? [String: Any] {
            status.fansAvailable = fans["available"] as? Bool ?? false
            status.fanSpeeds = fans["speeds_rpm"] as? [Int] ?? []
            status.fanMinimums = fans["minimum_rpm"] as? [Int] ?? []
            status.fanMaximums = fans["maximum_rpm"] as? [Int] ?? []
            let readings = fans["temperatures"] as? [[String: Any]] ?? []
            status.temperatures = readings
                .compactMap { entry in
                    guard let sensor = entry["sensor"] as? String,
                          let celsius = entry["celsius"] as? Double
                    else { return nil }
                    return TemperatureReading(sensor: sensor, celsius: celsius)
                }
                .sorted { $0.celsius > $1.celsius }
        }
        if let throttle = object["throttle"] as? [String: Any] {
            status.dutyPercent = throttle["duty_percent"] as? Double
            status.thermalScale = throttle["thermal_scale"] as? Double
            status.pacedWaits = throttle["paced_waits"] as? Int
            status.pacedWaitMilliseconds = throttle["paced_wait_ms"] as? Double
        }
        status.updated = true
        let previous = live
        statusUpdatedAt = Date()
        if status.ready {
            if readySince == nil { readySince = Date() }
        } else if readySince != nil {
            // A published property announces even a write of the value it
            // already holds, so the nil is written only when it is news.
            readySince = nil
        }
        // Nothing moved since the last poll — an idle engine reports the same
        // counters, rates and sensors second after second — so stop here: the
        // board, its histories and the stage keep exactly what they show, and
        // the header badge keeps its fresh reading on a clock of its own, a
        // TimelineView that re-reads `statusUpdatedAt` each second without a
        // publish from here. A quiet server then costs that one small badge a
        // second instead of a full pane re-render.
        guard status != previous else { return }
        record(status.decodeTokensPerSecond, in: \.decodeHistory)
        record(status.prefillTokensPerSecond, in: \.prefillHistory)
        record((status.draftAcceptanceRate ?? 0) * 100, in: \.acceptanceHistory)
        record(memoryFraction(status), in: \.memoryHistory)
        recordEvents(from: previous, to: status)
        advanceStage(from: previous, to: status)
        live = status
        // The smart mode is the app's own pacer: each poll it reads the fans
        // and temperatures the engine just reported and takes the engine's
        // duty to them, so the Mac stays as loud as it needs to be and no
        // louder. A push only when the reading has moved the duty, so the
        // engine is not asked to change a duty it already holds.
        if app.thermalRung == .smart, app.phase == .ready {
            let duty = app.smartDuty()
            if let last = app.lastSmartDuty, abs(last - duty) < 1.0 {
                return
            }
            app.lastSmartDuty = duty
            Task { @MainActor [weak app] in
                await app?.pushThermalDuty(duty)
            }
        } else {
            app.lastSmartDuty = nil
        }
    }
}
