import AppKit
import SwiftUI

/// A live view of the engine's `/status`: throughput, request counts, Metal
/// memory, caches and admission, refreshed once a second while it serves.
/// The board keeps a minute of each rate, so the shape of the load is visible
/// beside the number, and the lifecycle strip says what the engine is doing
/// right now rather than leaving it to a caption.
struct LivePane: View {
    @EnvironmentObject private var model: AppModel
    @State private var tab: LiveTab = .overview

    var body: some View {
        let live = model.live
        VStack(alignment: .leading, spacing: 14) {
            VStack(alignment: .leading, spacing: 14) {
                header(live)
                if live.updated { tabBar }
            }
            .frame(maxWidth: Theme.contentWidth, alignment: .leading)

            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    if !live.updated {
                        PaneEmpty(
                            title: L10n.string("live.empty.title"),
                            message: L10n.string("live.empty.description"),
                            systemImage: "waveform.path.ecg",
                            actionTitle: model.canStart ? L10n.string("start") : nil,
                            action: model.canStart ? { model.start() } : nil
                        )
                        .frame(minHeight: 380)
                    } else {
                        tabContent(live)
                            .id(tab)
                            .transition(.opacity.combined(with: .offset(y: 4)))
                    }
                }
                .padding(.bottom, Theme.pagePadding)
                .frame(maxWidth: Theme.contentWidth, alignment: .leading)
                .frame(maxWidth: .infinity, alignment: .topLeading)
            }
        }
        .padding(.horizontal, Theme.pagePadding)
        .padding(.top, Theme.pagePadding)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(Color.clear)
        .animation(.spring(response: 0.4, dampingFraction: 0.85), value: model.stage)
        .animation(.easeOut(duration: 0.16), value: tab)
    }

    /// The board's inner tabs: one glance at everything, then the requests,
    /// the memory and storage, and the heat, each with the panels that belong
    /// to it rather than one long page.
    private enum LiveTab: String, CaseIterable, Identifiable {
        case overview, requests, memory, thermal

        var id: String { rawValue }
        var title: String { L10n.string("live.tab.\(rawValue)") }
        var summary: String { L10n.string("live.tab.\(rawValue).help") }

        var systemImage: String {
            switch self {
            case .overview: return "rectangle.grid.2x2.fill"
            case .requests: return "tray.full.fill"
            case .memory: return "memorychip.fill"
            case .thermal: return "fan.fill"
            }
        }

        var tint: Color {
            switch self {
            case .overview: return SplashPalette.mint
            case .requests: return SplashPalette.violet
            case .memory: return SplashPalette.blue
            case .thermal: return SplashPalette.amber
            }
        }
    }

    private var tabBar: some View {
        HStack(spacing: 6) {
            ForEach(LiveTab.allCases) { item in
                LiveTabButton(
                    title: item.title,
                    systemImage: item.systemImage,
                    tint: item.tint,
                    selected: tab == item,
                    badge: item == .requests && model.live.pending > 0
                        ? model.live.pending : nil
                ) {
                    withAnimation(.easeOut(duration: 0.16)) { tab = item }
                }
                .help(item.summary)
            }
            Spacer(minLength: 0)
        }
        .padding(5)
        .glassPanel(radius: 14)
    }

    @ViewBuilder
    private func tabContent(_ live: LiveStatus) -> some View {
        switch tab {
        case .overview:
            VStack(alignment: .leading, spacing: 20) {
                LifecycleCard(
                    live: live,
                    stage: model.stage,
                    updated: model.statusUpdatedAt
                )
                heroes(live)
                HStack(spacing: 12) {
                    MemoryPanel(live: live, history: model.memoryHistory)
                    RateChartPanel(
                        decode: model.decodeHistory,
                        prefill: model.prefillHistory
                    )
                }
                .frame(height: 212)
                ClientsPanel(
                    clients: model.connectedClients,
                    endpoint: model.apiBaseURL
                ) {
                    model.refreshConnectedClients()
                }
            }
        case .requests:
            VStack(alignment: .leading, spacing: 20) {
                ActivityPanel(
                    events: model.recentEvents,
                    completed: live.completed,
                    uptime: model.uptimeText
                )
                MetricSection(
                    title: L10n.string("live.serving"),
                    systemImage: "power",
                    cards: serving(live),
                    tint: SplashPalette.blue
                )
                MetricSection(
                    title: L10n.string("live.requests"),
                    systemImage: "tray.full.fill",
                    cards: requests(live),
                    tint: SplashPalette.violet
                )
                MetricSection(
                    title: L10n.string("live.latency"),
                    systemImage: "timer",
                    cards: latency(live),
                    tint: SplashPalette.violet
                )
            }
        case .memory:
            VStack(alignment: .leading, spacing: 20) {
                HStack(spacing: 12) {
                    MemoryPanel(live: live, history: model.memoryHistory)
                    RateChartPanel(
                        decode: model.decodeHistory,
                        prefill: model.prefillHistory
                    )
                }
                .frame(height: 212)
                MetricSection(
                    title: L10n.string("live.memory"),
                    systemImage: "memorychip.fill",
                    cards: memory(live),
                    tint: SplashPalette.mint
                )
                MetricSection(
                    title: L10n.string("live.caches"),
                    systemImage: "externaldrive.fill",
                    cards: caches(live),
                    tint: SplashPalette.cyan
                )
            }
        case .thermal:
            VStack(alignment: .leading, spacing: 20) {
                MetricSection(
                    title: L10n.string("live.thermal"),
                    systemImage: "fan.fill",
                    cards: thermal(live),
                    tint: SplashPalette.amber
                )
            }
        }
    }

    private func heroes(_ live: LiveStatus) -> some View {
        HStack(spacing: 12) {
            HeroCard(
                title: L10n.string("live.decode"),
                value: rate(live.decodeTokensPerSecond),
                unit: L10n.string("live.caption.tokens_per_second"),
                systemImage: "bolt.fill",
                tint: SplashPalette.mint,
                note: peakNote(
                    "live.caption.peak_output",
                    live.peakDecodeTokensPerSecond
                ),
                history: model.decodeHistory
            )
            HeroCard(
                title: L10n.string("live.prefill"),
                value: rate(live.prefillTokensPerSecond),
                unit: L10n.string("live.caption.tokens_per_second"),
                systemImage: "arrow.down.to.line",
                tint: SplashPalette.cyan,
                note: peakNote(
                    "live.caption.peak_input",
                    live.peakPrefillTokensPerSecond
                ),
                history: model.prefillHistory
            )
            HeroCard(
                title: L10n.string("live.draft_acceptance"),
                value: percent(live.draftAcceptanceRate),
                unit: L10n.string("live.caption.draft_acceptance"),
                systemImage: "checkmark.seal.fill",
                tint: SplashPalette.violet,
                progress: live.draftAcceptanceRate,
                note: peakNote(
                    "live.caption.peak_acceptance",
                    live.peakDraftAcceptanceRate,
                    percent: true
                ),
                history: model.acceptanceHistory,
                chartMax: 100
            )
        }
    }

    /// The title and the one live badge: what the engine is doing, and when the
    /// reading behind it was taken.
    private func header(_ live: LiveStatus) -> some View {
        PaneHeader(
            title: L10n.string("view.live"),
            subtitle: L10n.string("live.subtitle"),
            eyebrow: L10n.string("live.eyebrow")
        ) {
            if live.updated {
                HStack(spacing: 9) {
                    StatusBeacon(color: stageColor(model.stage), active: true)
                    VStack(alignment: .leading, spacing: 1) {
                        Text(model.stage.title.uppercased())
                            .font(.system(size: 10, weight: .bold))
                            .tracking(1.1)
                            .foregroundStyle(stageColor(model.stage))
                        if let updated = model.statusUpdatedAt {
                            HStack(spacing: 3) {
                                Text(L10n.string("live.updated"))
                                Text(updated, style: .time)
                            }
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundStyle(SplashPalette.faint)
                        }
                    }
                }
                .padding(.horizontal, 13)
                .padding(.vertical, 8)
                .glassPanel(radius: 12, tint: stageColor(model.stage))
            }
        }
    }

    /// "max 123 tokens/s", or nothing while the panel has not seen a reading.
    private func peakNote(_ key: String, _ value: Double?, percent: Bool = false) -> String {
        guard let value, value > 0 else { return "" }
        return L10n.format(key, percent ? self.percent(value) : rate(value))
    }

    private func serving(_ live: LiveStatus) -> [MetricCard] {
        let ready = live.ready
        let state = ready
            ? L10n.string("live.ready")
            : (model.phase == .starting ? L10n.string("live.starting") : L10n.string("live.not_ready"))
        let restarts = live.restarts > 0 ? L10n.format("live.restarts", live.restarts) : ""
        return [
            MetricCard(
                title: L10n.string("live.model"), value: live.modelID ?? "—",
                systemImage: "cube.fill", tint: SplashPalette.violet
            ),
            MetricCard(
                title: L10n.string("live.state"), value: state, caption: restarts,
                systemImage: "power", tint: ready ? SplashPalette.mint : SplashPalette.amber
            ),
            MetricCard(
                title: L10n.string("live.context"), value: tokens(live.contextTokens),
                caption: contextCaption,
                systemImage: "text.alignleft", tint: SplashPalette.blue
            ),
            MetricCard(
                title: L10n.string("live.speculation"),
                value: speculationLabel(live.speculationProvider),
                caption: speculationCaption(live),
                systemImage: "sparkles", tint: SplashPalette.violet
            ),
        ]
    }

    private func speculationLabel(_ provider: String?) -> String {
        switch provider {
        case "dflash2": return "DFlash2"
        case "none": return L10n.string("speculation.target_only")
        case "dspark": return "DSpark"
        default: return provider ?? "—"
        }
    }

    /// The rows a cycle verifies and, when a requested draft fell back, why.
    private func speculationCaption(_ live: LiveStatus) -> String {
        if let fallback = live.speculationFallback, !fallback.isEmpty {
            return L10n.format("live.caption.speculation_fallback", fallback)
        }
        var parts: [String] = []
        if let rows = live.speculationRows {
            parts.append(L10n.format("live.caption.speculation_rows", rows))
        }
        return parts.joined(separator: " · ")
    }

    /// The context the engine serves, with the panel's own setting beside it:
    /// the card's figure is what the engine opened with, which is the smaller
    /// of that setting and what the model's memory allows.
    private var contextCaption: String {
        let configured = model.maxContext.trimmingCharacters(in: .whitespaces)
        guard !configured.isEmpty else { return "" }
        return L10n.format("live.caption.context_setting", configured)
    }

    /// The request counters as a few blocks rather than a dozen: what was asked
    /// and what came of it, what is queued, what is held back. Each of the two
    /// failure and waiting kinds is folded into the block it belongs to, with
    /// its own share as the caption, so the board stays readable while no count
    /// is lost.
    private func requests(_ live: LiveStatus) -> [MetricCard] {
        let outcomes = L10n.format(
            "live.caption.outcomes", live.completed, live.cancelled, live.failed
        )
        let failures = L10n.format(
            "live.caption.failure_kinds", live.capacityFailures, live.metalFailures
        )
        let waits = L10n.format(
            "live.caption.wait_kinds", live.waitingMemory, live.waitingConcurrency, live.suspended
        )
        return [
            MetricCard(title: L10n.string("live.submitted"), value: "\(live.submitted)",
                       caption: outcomes,
                       systemImage: "arrow.up.circle.fill", tint: SplashPalette.blue),
            MetricCard(title: L10n.string("live.failed"),
                       value: "\(live.failed)",
                       caption: failures,
                       systemImage: "xmark.octagon.fill", tint: SplashPalette.rose),
            MetricCard(title: L10n.string("live.pending"), value: "\(live.pending) / \(live.pendingLimit)",
                       caption: L10n.string("live.caption.queue"),
                       systemImage: "hourglass", tint: SplashPalette.violet,
                       progress: live.pendingLimit > 0
                           ? Double(live.pending) / Double(live.pendingLimit) : nil),
            MetricCard(title: L10n.string("live.waiting"), value: "\(live.waiting)",
                       caption: waits,
                       systemImage: "clock.fill", tint: SplashPalette.amber),
        ]
    }

    /// Memory in four blocks: what the engine holds against the budget it is
    /// held to, that budget, the Metal address space whose weight pages the
    /// host may reclaim, and the SSD cache. The budget names the panel's own
    /// `--max-memory` when it sets one, and says so, so a card never hides
    /// whether the limit shown is the configured one or the automatic plan's;
    /// the held card keeps the peak in its bar's note, which the figure and
    /// the budget alone cannot show.
    private func memory(_ live: LiveStatus) -> [MetricCard] {
        let limit = live.configuredMemoryLimitBytes ?? live.governorLimitBytes
        let held = share(live.physicalBytes, of: limit)
        let peak = live.peakPhysicalBytes ?? live.peakBytes
        let limitCaption = live.configuredMemoryLimitBytes != nil
            ? L10n.string("live.caption.limit_configured")
            : L10n.string("live.caption.limit_automatic")
        return [
            MetricCard(title: L10n.string("live.physical_memory"),
                       value: bytes(live.physicalBytes),
                       caption: L10n.string("live.physical_memory_caption"),
                       systemImage: "cpu", tint: SplashPalette.mint,
                       progress: held.value,
                       progressCaption: heldNote(held.caption, peak)),
            MetricCard(title: L10n.string("live.memory_limit"),
                       value: bytes(limit),
                       caption: limitCaption,
                       systemImage: "memorychip", tint: SplashPalette.blue),
            MetricCard(title: L10n.string("live.current_memory"), value: bytes(live.currentBytes),
                       caption: L10n.string("live.current_memory_caption"),
                       systemImage: "square.stack.3d.up.fill", tint: SplashPalette.violet),
            MetricCard(title: L10n.string("live.weights"),
                       value: weightsState(live),
                       caption: weightsCaption(live),
                       systemImage: live.weightsRestoring ? "arrow.down.circle.fill"
                           : (live.weightsReleased ? "moon.zzz.fill" : "lock.fill"),
                       tint: live.weightsRestoring ? SplashPalette.blue
                           : (live.weightsReleased ? SplashPalette.amber : SplashPalette.mint)),
            MetricCard(title: L10n.string("live.disk_cache"),
                       value: disk(live.diskUsedBytes, live.diskCapacityBytes),
                       caption: diskCacheCaption(live),
                       systemImage: "internaldrive.fill", tint: SplashPalette.cyan,
                       progress: share(live.diskUsedBytes, of: live.diskCapacityBytes).value),
        ]
    }

    /// The disk cache's own line: what the panel's switch does to it, since a
    /// tier that is off reports no capacity and the figures alone would read
    /// as a cache that is empty rather than one that is not used.
    private func diskCacheCaption(_ live: LiveStatus) -> String {
        if !model.global.diskCache { return L10n.string("live.caption.disk_cache_off") }
        let offload = idleOffloadCaption(live)
        return offload.isEmpty ? L10n.string("live.caption.disk_cache_on") : offload
    }

    /// The weights' idle release as one card, beside the memory they hold:
    /// resident, released (the next request pays to take them back), or
    /// restoring now. The caption names the timer and how many restores the
    /// engine has made, so a released value does not look like a fault.
    private func weightsState(_ live: LiveStatus) -> String {
        guard live.weightsKnown else { return "—" }
        if live.weightsRestoring { return L10n.string("live.weights.restoring") }
        if live.weightsReleased { return L10n.string("live.weights.released") }
        return L10n.string("live.weights.resident")
    }

    private func weightsCaption(_ live: LiveStatus) -> String {
        guard live.weightsKnown else { return "" }
        guard let idle = live.idleReleaseSeconds else {
            return L10n.string("live.weights.caption_off")
        }
        let release = L10n.format("live.weights.caption", Int(idle.rounded()))
        guard live.weightsRestores > 0 else { return release }
        return L10n.format("live.weights.caption_restores", release, live.weightsRestores)
    }

    private func caches(_ live: LiveStatus) -> [MetricCard] {
        return [
            MetricCard(title: L10n.string("live.hit_rate"), value: percent(live.cacheHitRate),
                       systemImage: "target", tint: SplashPalette.mint,
                       progress: live.cacheHitRate),
            MetricCard(title: L10n.string("live.kv_disk_hit"), value: percent(live.kvDiskHitRate),
                       caption: diskReuseCaption(live),
                       systemImage: "arrow.down.circle.fill", tint: SplashPalette.cyan,
                       progress: live.kvDiskHitRate),
            MetricCard(title: L10n.string("live.state_hit"), value: number(live.stateDiskHits),
                       systemImage: "arrow.triangle.2.circlepath", tint: SplashPalette.blue),
        ]
    }

    /// "96 of 128 tokens": the counts behind the disk reuse rate, so a rate
    /// read alone does not hide how much was reused at all.
    private func diskReuseCaption(_ live: LiveStatus) -> String {
        guard let reused = live.reusedTokens, reused > 0 else { return "" }
        return L10n.format("live.caption.share", number(live.kvDiskHitTokens), number(reused))
    }

    /// How the engine's answers feel, from the native latency histograms:
    /// time to first token and the gap between tokens, each with the tail the
    /// median hides.
    private func latency(_ live: LiveStatus) -> [MetricCard] {
        [
            MetricCard(
                title: L10n.string("live.ttft"),
                value: milliseconds(live.ttftP50Milliseconds),
                unit: "ms",
                caption: latencyCaption(p95: live.ttftP95Milliseconds, samples: live.ttftSamples),
                systemImage: "timer",
                tint: SplashPalette.violet
            ),
            MetricCard(
                title: L10n.string("live.itl"),
                value: milliseconds(live.itlP50Milliseconds),
                unit: "ms",
                caption: latencyCaption(p95: live.itlP95Milliseconds, samples: live.itlSamples),
                systemImage: "waveform.path.ecg",
                tint: SplashPalette.cyan
            ),
        ]
    }

    private func milliseconds(_ value: Double?) -> String {
        guard let value else { return "—" }
        return value < 10 ? String(format: "%.1f", value) : String(format: "%.0f", value)
    }

    private func latencyCaption(p95: Double?, samples: Int) -> String {
        guard samples > 0, let p95 else {
            return L10n.string("live.caption.latency_empty")
        }
        return L10n.format("live.caption.latency", milliseconds(p95), samples)
    }

    /// What the Mac is doing about the heat, as the thermal page reads it in
    /// full: the fans as one number, the hottest of the four temperature
    /// groups, and the share of the wall time prefill may keep the GPU busy.
    private func thermal(_ live: LiveStatus) -> [MetricCard] {
        let fans = live.fanSpeeds
        let hottest = TemperatureGroup.hottest(live.temperatures)
        return [
            MetricCard(title: L10n.string("live.fans"),
                       value: fans.isEmpty ? "—" : "\(fans.max() ?? 0)",
                       caption: fans.isEmpty
                           ? L10n.string("thermal.no_fans")
                           : L10n.format("thermal.fan.count", fans.count),
                       systemImage: "fan.fill", tint: SplashPalette.cyan),
            MetricCard(title: L10n.string("live.hottest"),
                       value: hottest.map { String(format: "%.1f °C", $0.celsius) } ?? "—",
                       caption: hottest?.group.title ?? L10n.string("thermal.unavailable"),
                       systemImage: "thermometer.medium", tint: hottest.map { heatTint($0.celsius) } ?? .gray),
            MetricCard(title: L10n.string("live.duty"),
                       value: live.dutyPercent.map { String(format: "%.0f%%", $0) } ?? "—",
                       caption: live.thermalScale.map {
                           L10n.format("thermal.scale.caption", $0)
                       } ?? "",
                       systemImage: "metronome.fill", tint: SplashPalette.amber,
                       progress: live.dutyPercent.map { $0 / 100 }),
        ]
    }

    /// The bar's note under the physical memory card: the share of the budget
    /// and the peak it has reached, either of which can be missing without
    /// taking the other with it.
    private func heldNote(_ share: String?, _ peak: UInt64?) -> String? {
        let notes = [share, peak.map { L10n.format("live.caption.peak_of", bytes($0)) }]
            .compactMap { $0 }
        return notes.isEmpty ? nil : notes.joined(separator: " · ")
    }

    /// A used-of-total pair as a fraction and a caption, either of which is nil
    /// when the total is unknown so nothing renders a misleading bar.
    private func share(
        _ used: UInt64?, of total: UInt64?
    ) -> (value: Double?, caption: String?) {
        guard let used, let total, total > 0 else { return (nil, nil) }
        let fraction = min(1, Double(used) / Double(total))
        return (fraction, L10n.format("live.caption.share", percent(fraction), bytes(total)))
    }

    private func tokens(_ value: Int?) -> String {
        number(value)
    }

    /// A count with grouping separators, or a dash when the status left it out.
    private func number(_ value: Int?) -> String {
        value.map { $0.formatted() } ?? "—"
    }

    private func bytes(_ value: UInt64?) -> String {
        guard let value else { return "—" }
        return ByteCountFormatter.string(fromByteCount: Int64(value), countStyle: .memory)
    }

    private func rate(_ value: Double?) -> String {
        guard let value else { return "—" }
        return String(format: "%.0f", value)
    }

    private func percent(_ value: Double?) -> String {
        guard let value else { return "—" }
        return String(format: "%.1f%%", value * 100)
    }

    private func disk(_ used: UInt64?, _ capacity: UInt64?) -> String {
        switch (used, capacity) {
        case (nil, nil): return "—"
        case (let used?, nil): return bytes(used)
        default:
            let percentValue = capacity.map { 100.0 * Double(used!) / Double($0) } ?? 0
            return String(format: "%@ · %.0f%%", bytes(used), percentValue)
        }
    }

    private func idleOffloadCaption(_ live: LiveStatus) -> String {
        guard live.idleOffloadPasses > 0 || live.idleOffloadBytes != nil else { return "" }
        let offloadBytes = live.idleOffloadBytes.map {
            ByteCountFormatter.string(fromByteCount: Int64($0), countStyle: .memory)
        } ?? "—"
        return L10n.format("live.caption.idle_offload", live.idleOffloadPasses, offloadBytes)
    }
}

/// The colour each stage wears wherever it appears: amber while requests wait,
/// cyan while prompts are read, mint while tokens stream, and a quiet slate at
/// rest.
func stageColor(_ stage: RequestStage) -> Color {
    switch stage {
    case .idle: return SplashPalette.muted
    case .queued: return SplashPalette.amber
    case .prefilling: return SplashPalette.cyan
    case .decoding: return SplashPalette.mint
    }
}

/// What the engine is doing now: the stage in words, the last reading of each
/// rate, and the pipeline that led there.
private struct LifecycleCard: View {
    let live: LiveStatus
    let stage: RequestStage
    let updated: Date?

    private var color: Color { stageColor(stage) }

    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            HStack(spacing: 10) {
                SectionLabel(
                    title: L10n.string("live.lifecycle"),
                    systemImage: "point.3.connected.trianglepath.dotted",
                    tint: color
                )
                Spacer(minLength: 0)
                if updated != nil {
                    Text(L10n.string("live.lifecycle.caption"))
                        .font(.system(size: 9.5))
                        .foregroundStyle(SplashPalette.faint)
                }
            }

            HStack(alignment: .center, spacing: 16) {
                ZStack {
                    Circle()
                        .fill(color.opacity(0.13))
                        .frame(width: 54, height: 54)
                    Circle()
                        .strokeBorder(color.opacity(0.35), lineWidth: 1)
                        .frame(width: 54, height: 54)
                    Image(systemName: stage.systemImage)
                        .font(.system(size: 20, weight: .bold))
                        .foregroundStyle(color)
                        .shadow(color: color.opacity(0.7), radius: 8)
                }

                VStack(alignment: .leading, spacing: 3) {
                    Text(stage.title.uppercased())
                        .font(.system(size: 19, weight: .bold, design: .rounded))
                        .tracking(0.8)
                        .foregroundStyle(color)
                        .contentTransition(.interpolate)
                    Text(caption)
                        .font(.system(size: 11.5))
                        .foregroundStyle(SplashPalette.muted)
                        .lineLimit(1)
                }

                Spacer(minLength: 10)

                HStack(spacing: 22) {
                    smallReading(
                        L10n.string("live.decode"),
                        rate(live.decodeTokensPerSecond),
                        SplashPalette.mint
                    )
                    smallReading(
                        L10n.string("live.prefill"),
                        rate(live.prefillTokensPerSecond),
                        SplashPalette.cyan
                    )
                    smallReading(
                        L10n.string("live.pending"),
                        "\(live.pending)",
                        SplashPalette.amber
                    )
                }
            }

            pipeline
        }
        .padding(18)
        .glassPanel(tint: color, raised: true)
    }

    private var caption: String {
        switch stage {
        case .idle:
            return L10n.string("stage.idle.caption")
        case .queued:
            return L10n.format("stage.queued.caption", max(live.pending, 1))
        case .prefilling:
            return L10n.string("stage.prefilling.caption")
        case .decoding:
            return L10n.format("stage.decoding.caption", rate(live.decodeTokensPerSecond))
        }
    }

    private func smallReading(_ title: String, _ value: String, _ tint: Color) -> some View {
        VStack(alignment: .trailing, spacing: 1) {
            Text(title.uppercased())
                .font(.system(size: 8.5, weight: .bold))
                .tracking(0.8)
                .foregroundStyle(SplashPalette.faint)
            Text(value)
                .font(.system(size: 18, weight: .bold, design: .rounded))
                .monospacedDigit()
                .foregroundStyle(tint)
                .contentTransition(.numericText())
        }
    }

    /// The pipeline, in the order a request moves through it: queue, prompt
    /// ingestion, generation, rest.
    private var pipeline: some View {
        let order: [RequestStage] = [.queued, .prefilling, .decoding, .idle]
        let activeIndex = order.firstIndex(of: stage) ?? order.count - 1
        return HStack(spacing: 8) {
            ForEach(Array(order.enumerated()), id: \.element) { index, item in
                pipelineNode(
                    item,
                    active: index == activeIndex,
                    passed: index < activeIndex
                )
                if index < order.count - 1 {
                    Capsule()
                        .fill(
                            index < activeIndex
                                ? AnyShapeStyle(
                                    LinearGradient(
                                        colors: [stageColor(item).opacity(0.7), stageColor(order[index + 1]).opacity(0.7)],
                                        startPoint: .leading,
                                        endPoint: .trailing
                                    )
                                )
                                : AnyShapeStyle(SplashPalette.hairline)
                        )
                        .frame(height: 2)
                        .frame(maxWidth: .infinity)
                }
            }
        }
        .padding(.top, 2)
    }

    private func pipelineNode(
        _ item: RequestStage,
        active: Bool,
        passed: Bool
    ) -> some View {
        let tint = stageColor(item)
        return HStack(spacing: 7) {
            ZStack {
                Circle()
                    .fill(active ? tint.opacity(0.18) : SplashPalette.surface.opacity(0.5))
                    .frame(width: 25, height: 25)
                Circle()
                    .strokeBorder(
                        active ? tint.opacity(0.7) : SplashPalette.border.opacity(0.5),
                        lineWidth: active ? 1.2 : 0.8
                    )
                    .frame(width: 25, height: 25)
                Image(systemName: item.systemImage)
                    .font(.system(size: 9.5, weight: .bold))
                    .foregroundStyle(active ? tint : (passed ? tint.opacity(0.7) : SplashPalette.faint))
            }
            Text(item.title)
                .font(.system(size: 10.5, weight: active ? .bold : .medium))
                .foregroundStyle(active ? tint : SplashPalette.muted)
                .lineLimit(1)
                .fixedSize(horizontal: true, vertical: false)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(
            active ? tint.opacity(0.09) : .clear,
            in: Capsule()
        )
    }

    private func rate(_ value: Double?) -> String {
        guard let value else { return "—" }
        return String(format: "%.0f", value)
    }
}

/// The Metal memory the engine holds, as a ring against its budget, with the
/// minute behind it and the peak it has reached.
private struct MemoryPanel: View {
    let live: LiveStatus
    let history: [Double]

    var body: some View {
        let limit = live.configuredMemoryLimitBytes ?? live.governorLimitBytes
        let fraction = fraction(live.physicalBytes, of: limit)
        let tint = fraction > 0.85 ? SplashPalette.rose : SplashPalette.blue

        return HStack(spacing: 18) {
            RingGauge(
                fraction: fraction,
                tint: tint,
                lineWidth: 11,
                label: fraction > 0 ? String(format: "%.0f%%", fraction * 100) : "—",
                detail: L10n.string("live.memory_of_budget")
            )
            .frame(width: 122, height: 122)

            VStack(alignment: .leading, spacing: 6) {
                SectionLabel(
                    title: L10n.string("live.physical_memory"),
                    systemImage: "memorychip.fill",
                    tint: tint
                )
                HStack(alignment: .firstTextBaseline, spacing: 5) {
                    Text(bytes(live.physicalBytes))
                        .font(.system(size: 23, weight: .bold, design: .rounded))
                        .monospacedDigit()
                        .contentTransition(.numericText())
                    Text(verbatim: "/ \(bytes(limit))")
                        .font(.system(size: 10.5, weight: .medium))
                        .foregroundStyle(SplashPalette.faint)
                        .lineLimit(1)
                }
                if let peak = live.peakPhysicalBytes ?? live.peakBytes {
                    Text(L10n.format("live.caption.peak_of", bytes(peak)))
                        .font(.system(size: 9.5))
                        .foregroundStyle(SplashPalette.faint)
                        .lineLimit(1)
                }
                Sparkline(values: history, tint: tint, upperBound: 1)
                    .frame(height: 24)
                HStack(spacing: 6) {
                    Pill(
                        text: L10n.string(
                            live.configuredMemoryLimitBytes != nil
                                ? "live.memory.configured"
                                : "live.memory.auto"
                        ),
                        tint: SplashPalette.blue
                    )
                    if let current = live.currentBytes {
                        Pill(
                            text: L10n.format("live.caption.allocated", bytes(current)),
                            tint: SplashPalette.violet
                        )
                    }
                    if live.weightsKnown {
                        Pill(text: weightsPill.text, tint: weightsPill.tint)
                    }
                }
            }
            Spacer(minLength: 0)
        }
        .padding(17)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)
        .glassPanel(tint: tint, raised: true)
    }

    /// The weights' idle state, so the overview shows a released or
    /// restoring model without opening the memory tab.
    private var weightsPill: (text: String, tint: Color) {
        if live.weightsRestoring {
            return (L10n.string("live.weights.restoring"), SplashPalette.blue)
        }
        if live.weightsReleased {
            return (L10n.string("live.weights.released"), SplashPalette.amber)
        }
        return (L10n.string("live.weights.resident"), SplashPalette.mint)
    }

    private func fraction(_ used: UInt64?, of total: UInt64?) -> Double {
        guard let used, let total, total > 0 else { return 0 }
        return min(1, Double(used) / Double(total))
    }

    private func bytes(_ value: UInt64?) -> String {
        guard let value else { return "—" }
        return ByteCountFormatter.string(fromByteCount: Int64(value), countStyle: .memory)
    }
}

/// Who is talking to the engine: one chip per local process with an
/// established socket on the inference port, with the app's own icon when
/// macOS has one. An empty board says where to point an app instead.
private struct ClientsPanel: View {
    let clients: [ConnectedClient]
    let endpoint: String
    let onRefresh: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                SectionLabel(
                    title: L10n.string("clients.title"),
                    systemImage: "app.connected.to.app.below.fill",
                    tint: SplashPalette.blue
                )
                Spacer(minLength: 6)
                if !clients.isEmpty {
                    Pill(
                        text: clients.count == 1
                            ? L10n.string("clients.one")
                            : L10n.format("clients.count", clients.count),
                        tint: SplashPalette.blue
                    )
                }
                Button(action: onRefresh) {
                    Image(systemName: "arrow.clockwise")
                        .font(.system(size: 9.5, weight: .bold))
                        .foregroundStyle(SplashPalette.muted)
                        .frame(width: 22, height: 20)
                        .background(SplashPalette.surface.opacity(0.45), in: RoundedRectangle(cornerRadius: 6, style: .continuous))
                        .overlay(
                            RoundedRectangle(cornerRadius: 6, style: .continuous)
                                .strokeBorder(SplashPalette.border.opacity(0.4), lineWidth: 0.6)
                        )
                }
                .buttonStyle(.plain)
                .help(L10n.string("clients.refresh"))
            }

            if clients.isEmpty {
                HStack(spacing: 9) {
                    Image(systemName: "network.slash")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(SplashPalette.faint)
                    VStack(alignment: .leading, spacing: 1) {
                        Text(L10n.string("clients.empty"))
                            .font(.system(size: 10.5, weight: .medium))
                            .foregroundStyle(SplashPalette.muted)
                        Text(L10n.format("clients.hint", endpoint))
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundStyle(SplashPalette.faint)
                            .lineLimit(1)
                    }
                    Spacer(minLength: 0)
                }
                .padding(.vertical, 4)
            } else {
                ScrollView(.horizontal) {
                    HStack(spacing: 8) {
                        ForEach(clients) { client in
                            chip(client)
                        }
                    }
                }
                .scrollIndicators(.hidden)
            }
        }
        .padding(15)
        .glassPanel(tint: SplashPalette.blue)
    }

    private func chip(_ client: ConnectedClient) -> some View {
        let tint = tint(client.kind)
        return HStack(spacing: 8) {
            icon(client)
            VStack(alignment: .leading, spacing: 1) {
                Text(client.name)
                    .font(.system(size: 10.5, weight: .semibold))
                    .foregroundStyle(.primary)
                    .lineLimit(1)
                Text(
                    client.connections == 1
                        ? L10n.format("clients.detail.one", Int(client.pid))
                        : L10n.format("clients.detail", Int(client.pid), client.connections)
                )
                    .font(.system(size: 8.5, design: .monospaced))
                    .foregroundStyle(SplashPalette.faint)
                    .lineLimit(1)
            }
        }
        .padding(.horizontal, 9)
        .padding(.vertical, 7)
        .background(SplashPalette.surface.opacity(0.40), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(tint.opacity(0.26), lineWidth: 0.7)
        )
    }

    @ViewBuilder
    private func icon(_ client: ConnectedClient) -> some View {
        if let app = NSRunningApplication(processIdentifier: client.pid),
           let image = app.icon {
            Image(nsImage: image)
                .resizable()
                .frame(width: 19, height: 19)
        } else {
            ZStack {
                Circle()
                    .fill(tint(client.kind).opacity(0.14))
                    .frame(width: 23, height: 23)
                Image(systemName: symbol(client.kind))
                    .font(.system(size: 9.5, weight: .bold))
                    .foregroundStyle(tint(client.kind))
            }
        }
    }

    private func symbol(_ kind: ConnectedClient.Kind) -> String {
        switch kind {
        case .agent: return "sparkles"
        case .code: return "chevron.left.forwardslash.chevron.right"
        case .chat: return "bubble.left.and.bubble.right.fill"
        case .terminal: return "terminal.fill"
        case .other: return "app.connected.to.app.below.fill"
        }
    }

    private func tint(_ kind: ConnectedClient.Kind) -> Color {
        switch kind {
        case .agent: return SplashPalette.violet
        case .code: return SplashPalette.blue
        case .chat: return SplashPalette.cyan
        case .terminal: return SplashPalette.amber
        case .other: return SplashPalette.muted
        }
    }
}

/// The last few request outcomes as a horizontal timeline: a live log of what
/// the engine finished while the board watched, not only what it is doing now.
private struct ActivityPanel: View {
    let events: [RequestEvent]
    let completed: Int
    let uptime: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 11) {
            HStack(spacing: 8) {
                SectionLabel(
                    title: L10n.string("activity.title"),
                    systemImage: "list.bullet.rectangle",
                    tint: SplashPalette.violet
                )
                Spacer(minLength: 6)
                if let uptime {
                    Pill(text: L10n.format("sidebar.uptime", uptime), tint: SplashPalette.cyan)
                }
                Pill(text: L10n.format("activity.done", completed), tint: SplashPalette.mint)
            }

            if events.isEmpty {
                HStack(spacing: 8) {
                    Image(systemName: "hourglass")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(SplashPalette.faint)
                    Text(L10n.string("activity.empty"))
                        .font(.system(size: 10.5))
                        .foregroundStyle(SplashPalette.faint)
                    Spacer(minLength: 0)
                }
                .padding(.vertical, 6)
            } else {
                ScrollView(.horizontal) {
                    HStack(spacing: 8) {
                        ForEach(Array(events.suffix(6).reversed())) { event in
                            chip(event)
                        }
                    }
                }
                .scrollIndicators(.hidden)
            }
        }
        .padding(15)
        .glassPanel(tint: SplashPalette.violet)
    }

    private func chip(_ event: RequestEvent) -> some View {
        let tint = tint(event.kind)
        return HStack(spacing: 8) {
            ZStack {
                Circle()
                    .fill(tint.opacity(0.14))
                    .frame(width: 23, height: 23)
                Image(systemName: icon(event.kind))
                    .font(.system(size: 9.5, weight: .bold))
                    .foregroundStyle(tint)
            }
            VStack(alignment: .leading, spacing: 1) {
                Text(title(event.kind))
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(.primary)
                    .lineLimit(1)
                Text(detail(event))
                    .font(.system(size: 8.5))
                    .foregroundStyle(SplashPalette.faint)
                    .lineLimit(1)
            }
            Spacer(minLength: 5)
            Text(event.date, style: .time)
                .font(.system(size: 8, design: .monospaced))
                .foregroundStyle(SplashPalette.faint)
        }
        .padding(.horizontal, 9)
        .padding(.vertical, 7)
        .frame(minWidth: 158, alignment: .leading)
        .background(SplashPalette.surface.opacity(0.40), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(tint.opacity(0.26), lineWidth: 0.7)
        )
    }

    private func title(_ kind: RequestEvent.Kind) -> String {
        switch kind {
        case .completed: return L10n.string("activity.completed")
        case .failed: return L10n.string("activity.failed")
        case .cancelled: return L10n.string("activity.cancelled")
        }
    }

    private func icon(_ kind: RequestEvent.Kind) -> String {
        switch kind {
        case .completed: return "checkmark"
        case .failed: return "xmark"
        case .cancelled: return "minus"
        }
    }

    private func tint(_ kind: RequestEvent.Kind) -> Color {
        switch kind {
        case .completed: return SplashPalette.mint
        case .failed: return SplashPalette.rose
        case .cancelled: return SplashPalette.amber
        }
    }

    private func detail(_ event: RequestEvent) -> String {
        switch event.kind {
        case .completed:
            var parts: [String] = []
            if event.count > 1 {
                parts.append(L10n.format("activity.multi", event.count))
            }
            if let rate = event.tokensPerSecond, rate > 0 {
                parts.append(L10n.format("activity.rate", String(format: "%.0f", rate)))
            }
            return parts.isEmpty ? L10n.string("activity.completed") : parts.joined(separator: " · ")
        case .failed:
            return L10n.string("activity.failed_detail")
        case .cancelled:
            return L10n.string("activity.cancelled_detail")
        }
    }
}

/// The two rates over the last minute on one shared scale, so the input and
/// output of a batch are read against each other rather than each against its
/// own peak. Hovering the plot pins a sample: its time, its two rates and a
/// crosshair over the point.
private struct RateChartPanel: View {
    let decode: [Double]
    let prefill: [Double]

    @State private var hover: Hover?

    private struct Hover: Equatable {
        let index: Int
    }

    private var count: Int { max(decode.count, prefill.count) }
    private var peak: Double { max(decode.max() ?? 0, prefill.max() ?? 0, 1) }

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack {
                SectionLabel(
                    title: L10n.string("live.last_minute"),
                    systemImage: "waveform.path.ecg",
                    tint: SplashPalette.mint
                )
                Spacer(minLength: 6)
                legend(
                    L10n.string("live.decode"),
                    SplashPalette.mint,
                    hover == nil ? decode.last : sample(decode, at: hover?.index)
                )
                legend(
                    L10n.string("live.prefill"),
                    SplashPalette.cyan,
                    hover == nil ? prefill.last : sample(prefill, at: hover?.index)
                )
            }

            GeometryReader { geo in
                ZStack(alignment: .topLeading) {
                    RateChart(decode: decode, prefill: prefill)
                    if let hover {
                        crosshair(hover, size: geo.size)
                    }
                }
                .contentShape(Rectangle())
                .onContinuousHover { phase in
                    switch phase {
                    case .active(let point):
                        update(point, width: geo.size.width)
                    case .ended:
                        hover = nil
                    }
                }
            }

            HStack {
                Text(L10n.string("chart.60s"))
                Spacer()
                Text(L10n.string("chart.now"))
            }
            .font(.system(size: 8, design: .monospaced))
            .foregroundStyle(SplashPalette.faint)
        }
        .padding(17)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)
        .glassPanel(tint: SplashPalette.mint, raised: true)
    }

    private func legend(_ title: String, _ tint: Color, _ value: Double?) -> some View {
        HStack(spacing: 4) {
            Circle()
                .fill(tint)
                .frame(width: 5, height: 5)
                .shadow(color: tint.opacity(0.8), radius: 3)
            Text(title)
                .font(.system(size: 9))
                .foregroundStyle(SplashPalette.faint)
                .lineLimit(1)
            if let value {
                Text(String(format: "%.0f", value))
                    .font(.system(size: 9, weight: .bold, design: .rounded))
                    .monospacedDigit()
                    .foregroundStyle(tint)
                    .contentTransition(.numericText())
            }
        }
    }

    private func update(_ point: CGPoint, width: CGFloat) {
        guard count > 1, width > 1 else { return }
        let step = width / CGFloat(count - 1)
        let index = min(max(Int((point.x / step).rounded()), 0), count - 1)
        hover = Hover(index: index)
    }

    /// A series can be shorter than the sample count while it warms up; its
    /// newest value is the newest sample, so align from the right.
    private func sample(_ values: [Double], at index: Int?) -> Double? {
        guard let index, count > 0 else { return nil }
        let position = index - (count - values.count)
        guard values.indices.contains(position) else { return nil }
        return values[position]
    }

    private func crosshair(_ hover: Hover, size: CGSize) -> some View {
        let decodeValue = sample(decode, at: hover.index)
        let prefillValue = sample(prefill, at: hover.index)
        let x = count > 1 ? size.width * CGFloat(hover.index) / CGFloat(count - 1) : 0
        let tooltipWidth: CGFloat = 126
        let tooltipX = min(max(x - tooltipWidth / 2, 0), max(0, size.width - tooltipWidth))
        let seconds = max(0, count - 1 - hover.index)

        return ZStack(alignment: .topLeading) {
            Rectangle()
                .fill(SplashPalette.muted.opacity(0.30))
                .frame(width: 1, height: size.height)
                .offset(x: x)

            if let prefillValue {
                dot(at: x, y: y(prefillValue, in: size.height), tint: SplashPalette.cyan)
            }
            if let decodeValue {
                dot(at: x, y: y(decodeValue, in: size.height), tint: SplashPalette.mint)
            }

            VStack(alignment: .leading, spacing: 3) {
                Text(seconds == 0 ? L10n.string("chart.now") : L10n.format("chart.ago", seconds))
                    .font(.system(size: 8, weight: .semibold, design: .monospaced))
                    .foregroundStyle(SplashPalette.faint)
                if let decodeValue {
                    tooltipRow(SplashPalette.mint, L10n.string("live.decode"), decodeValue)
                }
                if let prefillValue {
                    tooltipRow(SplashPalette.cyan, L10n.string("live.prefill"), prefillValue)
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
            .frame(width: tooltipWidth, alignment: .leading)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 9, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 9, style: .continuous)
                    .strokeBorder(SplashPalette.border.opacity(0.5), lineWidth: 0.7)
            )
            .shadow(color: .black.opacity(0.18), radius: 8, y: 3)
            .offset(x: tooltipX, y: 0)
        }
        .allowsHitTesting(false)
    }

    private func dot(at x: CGFloat, y: CGFloat, tint: Color) -> some View {
        Circle()
            .fill(tint)
            .frame(width: 6, height: 6)
            .shadow(color: tint.opacity(0.9), radius: 4)
            .offset(x: x - 3, y: y - 3)
    }

    private func y(_ value: Double, in height: CGFloat) -> CGFloat {
        height - CGFloat(min(value, peak) / peak) * (height - 4) - 2
    }

    private func tooltipRow(_ tint: Color, _ title: String, _ value: Double) -> some View {
        HStack(spacing: 5) {
            Circle().fill(tint).frame(width: 4, height: 4)
            Text(title)
                .font(.system(size: 8))
                .foregroundStyle(SplashPalette.faint)
            Spacer(minLength: 3)
            Text(String(format: "%.0f", value))
                .font(.system(size: 8.5, weight: .bold, design: .rounded))
                .monospacedDigit()
                .foregroundStyle(tint)
        }
    }
}

private struct RateChart: View {
    let decode: [Double]
    let prefill: [Double]

    var body: some View {
        Canvas { context, size in
            guard size.width > 2, size.height > 2 else { return }
            let peak = max(decode.max() ?? 0, prefill.max() ?? 0, 1)

            var grid = Path()
            for step in 1...3 {
                let y = size.height * CGFloat(step) / 4
                grid.move(to: CGPoint(x: 0, y: y))
                grid.addLine(to: CGPoint(x: size.width, y: y))
            }
            context.stroke(
                grid,
                with: .color(SplashPalette.border.opacity(0.35)),
                style: StrokeStyle(lineWidth: 0.6, dash: [3, 5])
            )

            func path(_ values: [Double]) -> Path {
                var path = Path()
                guard values.count > 1 else { return path }
                let step = size.width / CGFloat(values.count - 1)
                for (index, value) in values.enumerated() {
                    let y = size.height - CGFloat(min(value, peak) / peak) * (size.height - 4) - 2
                    let point = CGPoint(x: CGFloat(index) * step, y: y)
                    if index == 0 { path.move(to: point) } else { path.addLine(to: point) }
                }
                return path
            }

            let decodePath = path(decode)
            let prefillPath = path(prefill)

            if decode.count > 1 {
                var area = decodePath
                area.addLine(to: CGPoint(x: size.width, y: size.height))
                area.addLine(to: CGPoint(x: 0, y: size.height))
                area.closeSubpath()
                context.fill(
                    area,
                    with: .linearGradient(
                        Gradient(colors: [
                            SplashPalette.mint.opacity(0.22),
                            SplashPalette.mint.opacity(0),
                        ]),
                        startPoint: CGPoint(x: 0, y: 0),
                        endPoint: CGPoint(x: 0, y: size.height)
                    )
                )
            }

            context.stroke(
                prefillPath,
                with: .color(SplashPalette.cyan.opacity(0.95)),
                style: StrokeStyle(lineWidth: 1.5, lineCap: .round, lineJoin: .round)
            )
            context.stroke(
                decodePath,
                with: .color(SplashPalette.mint),
                style: StrokeStyle(lineWidth: 1.9, lineCap: .round, lineJoin: .round)
            )
        }
        .animation(.easeOut(duration: 0.3), value: decode.last)
    }
}

/// One tab of the live board: its icon, its name, and the queue's count while
/// requests wait, with the hover and selection chrome the rest of the app uses.
private struct LiveTabButton: View {
    let title: String
    let systemImage: String
    let tint: Color
    let selected: Bool
    var badge: Int?
    let action: () -> Void

    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 7) {
                Image(systemName: systemImage)
                    .font(.system(size: 11, weight: .semibold))
                Text(title)
                    .font(.system(size: 11.5, weight: selected ? .semibold : .medium))
                if let badge {
                    Text("\(badge)")
                        .font(.system(size: 9, weight: .bold, design: .rounded))
                        .monospacedDigit()
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1.5)
                        .background(
                            selected ? Color.white.opacity(0.22) : tint.opacity(0.16),
                            in: Capsule()
                        )
                        .foregroundStyle(selected ? .white : tint)
                }
                if selected {
                    Circle()
                        .fill(tint)
                        .frame(width: 4, height: 4)
                }
            }
            .foregroundStyle(selected ? .white : SplashPalette.muted)
            .padding(.horizontal, 13)
            .padding(.vertical, 8)
            .background(
                selected
                    ? AnyShapeStyle(
                        LinearGradient(
                            colors: [tint, tint.opacity(0.72)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    : AnyShapeStyle(hovering ? SplashPalette.surface.opacity(0.45) : Color.clear),
                in: Capsule()
            )
            .overlay(
                Capsule().strokeBorder(
                    selected ? tint.opacity(0.5) : SplashPalette.border.opacity(0.45),
                    lineWidth: 0.8
                )
            )
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .animation(.easeOut(duration: 0.12), value: hovering)
        .accessibilityLabel(title)
        .accessibilityAddTraits(selected ? .isSelected : [])
    }
}
