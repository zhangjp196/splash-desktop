import AppKit
import SwiftUI

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

/// The header's one live badge: the stage in words and the moment the reading
/// behind it was taken. It is the only part of the board that moves every
/// second even on a quiet engine, so it keeps a clock of its own — a
/// one-second TimelineView that re-reads the poll time through `pollAt` —
/// instead of the model publishing a fresh date every second and waking the
/// whole board for it. The clock stays a plain formatted string rather than
/// `Text(_, style: .time)`, which keeps a ticking timer of its own.
struct LiveStageBadge: View {
    let stage: RequestStage
    /// Read fresh on the badge's own clock each second.
    let pollAt: () -> Date?

    private static let clock: DateFormatter = {
        let formatter = DateFormatter()
        formatter.timeStyle = .medium
        formatter.dateStyle = .none
        return formatter
    }()

    var body: some View {
        HStack(spacing: 9) {
            StatusBeacon(color: color, active: true)
            VStack(alignment: .leading, spacing: 1) {
                Text(stage.title.uppercased())
                    .font(.system(size: 10, weight: .bold))
                    .tracking(1.1)
                    .foregroundStyle(color)
                TimelineView(.periodic(from: .now, by: 1)) { _ in
                    if let updated = pollAt() {
                        HStack(spacing: 3) {
                            Text(L10n.string("live.updated"))
                            Text(verbatim: Self.clock.string(from: updated))
                        }
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundStyle(SplashPalette.faint)
                    }
                }
            }
        }
        .padding(.horizontal, 13)
        .padding(.vertical, 8)
        .glassPanel(radius: 12, tint: color)
    }

    private var color: Color { stageColor(stage) }
}

/// What the engine is doing now: the stage in words, the last reading of each
/// rate, and the pipeline that led there. Equatable over its plain inputs, so
/// a poll that changed nothing skips the whole card.
struct LifecycleCard: View, Equatable {
    let live: LiveStatus
    let stage: RequestStage
    let updated: Bool

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
                if updated {
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
                }

                VStack(alignment: .leading, spacing: 3) {
                    Text(stage.title.uppercased())
                        .font(.system(size: 19, weight: .bold, design: .rounded))
                        .tracking(0.8)
                        .foregroundStyle(color)
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
        }
    }

    /// The pipeline, in the order a request moves through it: queue, prompt
    /// ingestion, generation, rest. The connectors are flat hairlines — a
    /// gradient per connector would be four more offscreen-blended fills on
    /// the one card that redraws every second.
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
                                ? stageColor(item).opacity(0.45)
                                : SplashPalette.hairline
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
/// minute behind it and the peak it has reached. Equatable over its inputs, so
/// an unchanged reading skips the ring and the pills.
struct MemoryPanel: View, Equatable {
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
/// Equatable over the board's real data, so the scan that finds the same
/// clients again skips the whole panel; the refresh closure is left out of
/// the comparison on purpose, since it is always this panel's own button
/// calling the same refresh and carries nothing to compare.
struct ClientsPanel: View, Equatable {
    let clients: [ConnectedClient]
    let endpoint: String
    let onRefresh: () -> Void

    static func == (lhs: ClientsPanel, rhs: ClientsPanel) -> Bool {
        lhs.clients == rhs.clients && lhs.endpoint == rhs.endpoint
    }

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

    /// Icons by process identifier, remembered across renders. Reading an
    /// icon goes through `NSRunningApplication` and AppKit each time, and a
    /// board of clients would repeat that round trip on every pass. A miss is
    /// remembered too, so a pid whose process has exited answers nil without
    /// a fresh query — but only for a minute, because the system reuses pids
    /// and an old nil must not hide a new process's icon for longer than
    /// that. The cache is cleared outright once it grows past a few dozen
    /// entries; a board this small refills it in one pass.
    @MainActor private static var iconCache: [Int32: (icon: NSImage?, stamped: Date)] = [:]

    /// The app icon for a pid, through the cache.
    @MainActor private static func cachedIcon(for pid: Int32) -> NSImage? {
        let now = Date()
        if let cached = iconCache[pid],
           cached.icon != nil || now.timeIntervalSince(cached.stamped) < 60 {
            return cached.icon
        }
        let icon = NSRunningApplication(processIdentifier: pid)?.icon
        if iconCache.count > 64 { iconCache.removeAll() }
        iconCache[pid] = (icon, now)
        return icon
    }

    @ViewBuilder
    private func icon(_ client: ConnectedClient) -> some View {
        if let image = Self.cachedIcon(for: client.pid) {
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
/// Equatable over the feed itself: the uptime pill beside the title is the one
/// part that moves every second, so it reads the model on a clock of its own
/// (see `UptimePill`) and is left out of the comparison — an equal feed then
/// skips the whole panel while the pill keeps ticking.
struct ActivityPanel: View, Equatable {
    let events: [RequestEvent]
    let completed: Int
    /// Read fresh on the pill's own clock each second, so the panel holds no
    /// per-second value that would make it unequal and redraw.
    let uptime: () -> String?

    static func == (lhs: ActivityPanel, rhs: ActivityPanel) -> Bool {
        lhs.events == rhs.events && lhs.completed == rhs.completed
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 11) {
            HStack(spacing: 8) {
                SectionLabel(
                    title: L10n.string("activity.title"),
                    systemImage: "list.bullet.rectangle",
                    tint: SplashPalette.violet
                )
                Spacer(minLength: 6)
                UptimePill(uptime: uptime)
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

/// The uptime pill on its own clock. It is the one part of the panel that
/// moves every second even on a quiet engine, and the model no longer
/// publishes every second for it (an idle engine must not wake the whole
/// interface), so the pill re-reads the running time on a one-second
/// TimelineView instead. The closure keeps the reading live across ticks —
/// including the moment the server starts or stops, which the next tick
/// picks up without the panel around it redrawing.
private struct UptimePill: View {
    let uptime: () -> String?

    var body: some View {
        TimelineView(.periodic(from: .now, by: 1)) { _ in
            if let uptime = uptime() {
                Pill(text: L10n.format("sidebar.uptime", uptime), tint: SplashPalette.cyan)
            }
        }
    }
}

/// The two rates over the last minute on one shared scale, so the input and
/// output of a batch are read against each other rather than each against its
/// own peak. Hovering the plot pins a sample: its time, its two rates and a
/// crosshair over the point. Equatable over its series, so a redraw happens
/// only when a sample actually moved.
struct RateChartPanel: View, Equatable {
    let decode: [Double]
    let prefill: [Double]

    @State private var hover: Hover?

    private struct Hover: Equatable {
        let index: Int
    }

    /// Equality over the series alone: the hover pin is the pointer's own
    /// state and must not keep a redraw from being skipped.
    static func == (lhs: RateChartPanel, rhs: RateChartPanel) -> Bool {
        lhs.decode == rhs.decode && lhs.prefill == rhs.prefill
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
            Text(title)
                .font(.system(size: 9))
                .foregroundStyle(SplashPalette.faint)
                .lineLimit(1)
            if let value {
                Text(String(format: "%.0f", value))
                    .font(.system(size: 9, weight: .bold, design: .rounded))
                    .monospacedDigit()
                    .foregroundStyle(tint)
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
            .background(SplashPalette.surface.opacity(0.92), in: RoundedRectangle(cornerRadius: 9, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 9, style: .continuous)
                    .strokeBorder(SplashPalette.border.opacity(0.5), lineWidth: 0.7)
            )
            .shadow(color: .black.opacity(0.18), radius: 6, y: 3)
            .offset(x: tooltipX, y: 0)
        }
        .allowsHitTesting(false)
    }

    private func dot(at x: CGFloat, y: CGFloat, tint: Color) -> some View {
        Circle()
            .fill(tint)
            .frame(width: 6, height: 6)
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
        // No value animation: the plot is redrawn from the series each sample,
        // and animating a Canvas blends two full redraws every second.
    }
}
