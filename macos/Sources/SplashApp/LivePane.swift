import SwiftUI

/// A live view of the engine's `/status`: throughput, request counts, Metal
/// memory, caches and admission, refreshed once a second while it serves.
struct LivePane: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        ScrollView {
            if !model.live.updated {
                ContentUnavailableView(
                    L10n.string("live.empty.title"),
                    systemImage: "chart.line.uptrend.xyaxis",
                    description: Text(verbatim: L10n.string("live.empty.description"))
                )
                .padding(.top, 80)
            } else {
                let live = model.live
                VStack(alignment: .leading, spacing: 22) {
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
                }
                .padding(18)
            }
        }
        .background(.background)
    }

    private func serving(_ live: LiveStatus) -> [MetricCard] {
        let ready = live.ready
        let state = ready
            ? L10n.string("live.ready")
            : (model.phase == .starting ? L10n.string("live.starting") : L10n.string("live.not_ready"))
        let restarts = live.restarts > 0 ? L10n.format("live.restarts", live.restarts) : ""
        return [
            MetricCard(
                title: L10n.string("live.state"), value: state, caption: restarts,
                systemImage: "power", tint: ready ? .green : .orange
            ),
            MetricCard(
                title: L10n.string("live.context"), value: tokens(live.contextTokens),
                systemImage: "text.alignleft", tint: .indigo
            ),
            MetricCard(
                title: L10n.string("live.decode"), value: rate(live.decodeTokensPerSecond),
                caption: L10n.string("live.caption.tokens_per_second"),
                systemImage: "speedometer", tint: .green
            ),
            MetricCard(
                title: L10n.string("live.prefill"), value: rate(live.prefillTokensPerSecond),
                caption: L10n.string("live.caption.tokens_per_second"),
                systemImage: "arrow.down.to.line", tint: .teal
            ),
            MetricCard(
                title: L10n.string("live.draft_acceptance"), value: percent(live.draftAcceptanceRate),
                systemImage: "checkmark.seal", tint: .mint
            ),
        ]
    }

    private func requests(_ live: LiveStatus) -> [MetricCard] {
        return [
            MetricCard(title: L10n.string("live.submitted"), value: "\(live.submitted)",
                       systemImage: "arrow.up.circle", tint: .blue),
            MetricCard(title: L10n.string("live.completed"), value: "\(live.completed)",
                       systemImage: "checkmark.circle", tint: .green),
            MetricCard(title: L10n.string("live.failed"), value: "\(live.failed)",
                       systemImage: "xmark.octagon", tint: .red),
            MetricCard(title: L10n.string("live.cancelled"), value: "\(live.cancelled)",
                       systemImage: "slash.circle", tint: .orange),
            MetricCard(title: L10n.string("live.pending"), value: "\(live.pending) / \(live.pendingLimit)",
                       systemImage: "hourglass", tint: .purple),
            MetricCard(title: L10n.string("live.waiting"), value: "\(live.waiting)",
                       systemImage: "clock", tint: .orange),
            MetricCard(title: L10n.string("live.waiting_memory"), value: "\(live.waitingMemory)",
                       systemImage: "memorychip", tint: .pink),
            MetricCard(title: L10n.string("live.waiting_concurrency"), value: "\(live.waitingConcurrency)",
                       systemImage: "person.2", tint: .brown),
            MetricCard(title: L10n.string("live.suspended"), value: "\(live.suspended)",
                       systemImage: "pause.circle", tint: .gray),
            MetricCard(title: L10n.string("live.capacity_failures"), value: "\(live.capacityFailures)",
                       systemImage: "exclamationmark.triangle", tint: .red),
            MetricCard(title: L10n.string("live.metal_failures"), value: "\(live.metalFailures)",
                       systemImage: "exclamationmark.octagon", tint: .red),
        ]
    }

    private func memory(_ live: LiveStatus) -> [MetricCard] {
        return [
            MetricCard(title: L10n.string("live.current_memory"), value: bytes(live.currentBytes),
                       systemImage: "memorychip", tint: .blue),
            MetricCard(title: L10n.string("live.peak_memory"), value: bytes(live.peakBytes),
                       systemImage: "chart.bar", tint: .indigo),
            MetricCard(title: L10n.string("live.disk_cache"),
                       value: disk(live.diskUsedBytes, live.diskCapacityBytes),
                       systemImage: "internaldrive", tint: .teal),
        ]
    }

    private func caches(_ live: LiveStatus) -> [MetricCard] {
        return [
            MetricCard(title: L10n.string("live.hit_rate"), value: percent(live.cacheHitRate),
                       systemImage: "target", tint: .green),
            MetricCard(title: L10n.string("live.kv_disk_hit"), value: tokens(live.kvDiskHitTokens),
                       systemImage: "arrow.down.circle", tint: .teal),
            MetricCard(title: L10n.string("live.state_hit"), value: tokens(live.stateHitTokens),
                       systemImage: "arrow.triangle.2.circlepath", tint: .blue),
        ]
    }

    private func tokens(_ value: Int?) -> String {
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
}

private struct MetricSection: View {
    let title: String
    let systemImage: String
    let cards: [MetricCard]

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            Label(title, systemImage: systemImage)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.secondary)
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 190), spacing: 10)], spacing: 10) {
                ForEach(cards) { card in
                    card
                }
            }
        }
    }
}

private struct MetricCard: View, Identifiable {
    let id = UUID()
    let title: String
    let value: String
    var caption = ""
    var systemImage = "circle"
    var tint: Color = .accentColor

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: systemImage)
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(tint)
                .frame(width: 28, height: 28)
                .background(tint.opacity(0.14), in: RoundedRectangle(cornerRadius: 8))
            VStack(alignment: .leading, spacing: 3) {
                Text(title).font(.caption).foregroundStyle(.secondary)
                Text(value)
                    .font(.title3.weight(.semibold).monospacedDigit())
                    .lineLimit(1)
                    .minimumScaleFactor(0.6)
                if !caption.isEmpty {
                    Text(caption).font(.caption2).foregroundStyle(.tertiary)
                }
            }
            Spacer(minLength: 0)
        }
        .padding(10)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .strokeBorder(.separator.opacity(0.5))
        )
    }
}