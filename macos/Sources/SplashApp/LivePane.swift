import SwiftUI

/// A live view of the engine's `/status`: throughput, request counts, Metal
/// memory, caches and admission, refreshed once a second while it serves.
struct LivePane: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        ScrollView {
            let live = model.live
            VStack(alignment: .leading, spacing: 18) {
                PaneHeader(title: L10n.string("view.live"), subtitle: L10n.string("live.subtitle"))
                if !live.updated {
                    PaneEmpty(
                        title: L10n.string("live.empty.title"),
                        message: L10n.string("live.empty.description"),
                        systemImage: "chart.line.uptrend.xyaxis",
                        actionTitle: model.canStart ? L10n.string("start") : nil,
                        action: model.canStart ? { model.start() } : nil
                    )
                    .frame(minHeight: 300)
                } else {
                    VStack(alignment: .leading, spacing: 20) {
                        throughput(live)
                        MetricSection(
                            title: L10n.string("live.serving"),
                            systemImage: "bolt.fill",
                            cards: serving(live)
                        )
                        MetricSection(
                            title: L10n.string("live.requests"),
                            systemImage: "tray.full.fill",
                            cards: requests(live)
                        )
                        MetricSection(
                            title: L10n.string("live.memory"),
                            systemImage: "memorychip.fill",
                            cards: memory(live)
                        )
                        MetricSection(
                            title: L10n.string("live.caches"),
                            systemImage: "externaldrive.fill",
                            cards: caches(live)
                        )
                        MetricSection(
                            title: L10n.string("live.thermal"),
                            systemImage: "fan.fill",
                            cards: thermal(live)
                        )
                    }
                }
            }
            .padding(Theme.pagePadding)
            .frame(maxWidth: .infinity, alignment: .topLeading)
        }
        .background(.background)
    }

    /// The three numbers that decide whether a request feels fast, given the
    /// full width so they are readable at a glance from across the desk. Each
    /// hero carries the fastest the engine has been seen to run in its note:
    /// the figure above is the lifetime average, which a long conversation
    /// pulls down, so the peak is what says how fast this Mac really is.
    private func throughput(_ live: LiveStatus) -> some View {
        HStack(spacing: 12) {
            HeroCard(
                title: L10n.string("live.decode"),
                value: rate(live.decodeTokensPerSecond),
                unit: L10n.string("live.caption.tokens_per_second"),
                systemImage: "speedometer",
                tint: .green,
                note: peakNote("live.caption.peak_output", live.peakDecodeTokensPerSecond)
            )
            HeroCard(
                title: L10n.string("live.prefill"),
                value: rate(live.prefillTokensPerSecond),
                unit: L10n.string("live.caption.tokens_per_second"),
                systemImage: "arrow.down.to.line",
                tint: .teal,
                note: peakNote("live.caption.peak_input", live.peakPrefillTokensPerSecond)
            )
            HeroCard(
                title: L10n.string("live.draft_acceptance"),
                value: percent(live.draftAcceptanceRate),
                unit: L10n.string("live.caption.draft_acceptance"),
                systemImage: "checkmark.seal",
                tint: .mint,
                progress: live.draftAcceptanceRate,
                note: peakNote(
                    "live.caption.peak_acceptance", live.peakDraftAcceptanceRate,
                    percent: true
                )
            )
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
                systemImage: "cube", tint: .indigo
            ),
            MetricCard(
                title: L10n.string("live.state"), value: state, caption: restarts,
                systemImage: "power", tint: ready ? .green : .orange
            ),
            MetricCard(
                title: L10n.string("live.context"), value: tokens(live.contextTokens),
                caption: contextCaption,
                systemImage: "text.alignleft", tint: .indigo
            ),
        ]
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
                       systemImage: "arrow.up.circle", tint: .blue),
            MetricCard(title: L10n.string("live.failed"),
                       value: "\(live.failed)",
                       caption: failures,
                       systemImage: "xmark.octagon", tint: .red),
            MetricCard(title: L10n.string("live.pending"), value: "\(live.pending) / \(live.pendingLimit)",
                       caption: L10n.string("live.caption.queue"),
                       systemImage: "hourglass", tint: .purple,
                       progress: live.pendingLimit > 0
                           ? Double(live.pending) / Double(live.pendingLimit) : nil),
            MetricCard(title: L10n.string("live.waiting"), value: "\(live.waiting)",
                       caption: waits,
                       systemImage: "clock", tint: .orange),
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
                       systemImage: "cpu", tint: .green,
                       progress: held.value,
                       progressCaption: heldNote(held.caption, peak)),
            MetricCard(title: L10n.string("live.memory_limit"),
                       value: bytes(limit),
                       caption: limitCaption,
                       systemImage: "memorychip", tint: .blue),
            MetricCard(title: L10n.string("live.current_memory"), value: bytes(live.currentBytes),
                       caption: L10n.string("live.current_memory_caption"),
                       systemImage: "square.stack.3d.up", tint: .indigo),
            MetricCard(title: L10n.string("live.disk_cache"),
                       value: disk(live.diskUsedBytes, live.diskCapacityBytes),
                       caption: diskCacheCaption(live),
                       systemImage: "internaldrive", tint: .teal,
                       progress: share(live.diskUsedBytes, of: live.diskCapacityBytes).value),
        ]
    }

    /// The disk cache's own line: what the panel's switch does to it, since a
    /// tier that is off reports no capacity and the figures alone would read
    /// as a cache that is empty rather than one that is not used.
    private func diskCacheCaption(_ live: LiveStatus) -> String {
        if !model.diskCache { return L10n.string("live.caption.disk_cache_off") }
        let offload = idleOffloadCaption(live)
        return offload.isEmpty ? L10n.string("live.caption.disk_cache_on") : offload
    }

    private func caches(_ live: LiveStatus) -> [MetricCard] {
        return [
            MetricCard(title: L10n.string("live.hit_rate"), value: percent(live.cacheHitRate),
                       systemImage: "target", tint: .green,
                       progress: live.cacheHitRate),
            MetricCard(title: L10n.string("live.kv_disk_hit"), value: percent(live.kvDiskHitRate),
                       caption: diskReuseCaption(live),
                       systemImage: "arrow.down.circle", tint: .teal,
                       progress: live.kvDiskHitRate),
            MetricCard(title: L10n.string("live.state_hit"), value: number(live.stateDiskHits),
                       systemImage: "arrow.triangle.2.circlepath", tint: .blue),
        ]
    }

    /// "96 of 128 tokens": the counts behind the disk reuse rate, so a rate
    /// read alone does not hide how much was reused at all.
    private func diskReuseCaption(_ live: LiveStatus) -> String {
        guard let reused = live.reusedTokens, reused > 0 else { return "" }
        return L10n.format("live.caption.share", number(live.kvDiskHitTokens), number(reused))
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
                       systemImage: "fan.fill", tint: .cyan),
            MetricCard(title: L10n.string("live.hottest"),
                       value: hottest.map { String(format: "%.1f °C", $0.celsius) } ?? "—",
                       caption: hottest?.group.title ?? L10n.string("thermal.unavailable"),
                       systemImage: "thermometer", tint: hottest.map { heatTint($0.celsius) } ?? .gray),
            MetricCard(title: L10n.string("live.duty"),
                       value: live.dutyPercent.map { String(format: "%.0f%%", $0) } ?? "—",
                       caption: live.thermalScale.map {
                           L10n.format("thermal.scale.caption", $0)
                       } ?? "",
                       systemImage: "metronome.fill", tint: .teal,
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
