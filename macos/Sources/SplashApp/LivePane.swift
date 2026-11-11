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
                            .transition(.opacity)
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
        // No stage or tab animation on the pane itself: the stage changes
        // every second or two while the engine serves, and a spring on the
        // whole board would re-layout it for a badge colour. The tab switch
        // crossfades through the transition above alone.
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
                    updated: model.statusUpdatedAt != nil
                )
                .equatable()
                heroes(live)
                HStack(spacing: 12) {
                    MemoryPanel(live: live, history: model.memoryHistory)
                        .equatable()
                    RateChartPanel(
                        decode: model.decodeHistory,
                        prefill: model.prefillHistory
                    )
                    .equatable()
                }
                .frame(height: 212)
                ClientsPanel(
                    clients: model.connectedClients,
                    endpoint: model.apiBaseURL
                ) {
                    model.refreshConnectedClients()
                }
                .equatable()
            }
        case .requests:
            VStack(alignment: .leading, spacing: 20) {
                ActivityPanel(
                    events: model.recentEvents,
                    completed: live.completed,
                    uptime: { model.uptimeText }
                )
                .equatable()
                MetricSection(
                    title: L10n.string("live.serving"),
                    systemImage: "power",
                    cards: serving(live),
                    tint: SplashPalette.blue
                )
                .equatable()
                MetricSection(
                    title: L10n.string("live.requests"),
                    systemImage: "tray.full.fill",
                    cards: requests(live),
                    tint: SplashPalette.violet
                )
                .equatable()
                MetricSection(
                    title: L10n.string("live.latency"),
                    systemImage: "timer",
                    cards: latency(live),
                    tint: SplashPalette.violet
                )
                .equatable()
            }
        case .memory:
            VStack(alignment: .leading, spacing: 20) {
                HStack(spacing: 12) {
                    MemoryPanel(live: live, history: model.memoryHistory)
                        .equatable()
                    RateChartPanel(
                        decode: model.decodeHistory,
                        prefill: model.prefillHistory
                    )
                    .equatable()
                }
                .frame(height: 212)
                MetricSection(
                    title: L10n.string("live.memory"),
                    systemImage: "memorychip.fill",
                    cards: memory(live),
                    tint: SplashPalette.mint
                )
                .equatable()
                MetricSection(
                    title: L10n.string("live.caches"),
                    systemImage: "externaldrive.fill",
                    cards: caches(live),
                    tint: SplashPalette.cyan
                )
                .equatable()
            }
        case .thermal:
            VStack(alignment: .leading, spacing: 20) {
                MetricSection(
                    title: L10n.string("live.thermal"),
                    systemImage: "fan.fill",
                    cards: thermal(live),
                    tint: SplashPalette.amber
                )
                .equatable()
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
            .equatable()
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
            .equatable()
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
            .equatable()
        }
    }

    /// The title and the one live badge: what the engine is doing, and when the
    /// reading behind it was taken. The badge is its own view so the per-second
    /// timestamp costs one small re-render, not the board under it.
    private func header(_ live: LiveStatus) -> some View {
        PaneHeader(
            title: L10n.string("view.live"),
            subtitle: L10n.string("live.subtitle"),
            eyebrow: L10n.string("live.eyebrow")
        ) {
            if live.updated {
                LiveStageBadge(stage: model.stage, pollAt: { model.statusUpdatedAt })
            }
        }
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
                // A flat tint fill rather than a gradient: the selected tab
                // changes a few times a minute, and a gradient fill would
                // blend an extra layer for a two-state button.
                selected
                    ? AnyShapeStyle(tint.opacity(0.85))
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
        .accessibilityLabel(title)
        .accessibilityAddTraits(selected ? .isSelected : [])
    }
}
