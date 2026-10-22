import SwiftUI

/// The heat-and-noise ladder, in the server's own names
/// (server/serve_options.py THERMAL_POLICIES): the five rungs the app offers,
/// loudest first, each the share of wall time prefill may keep the GPU busy.
/// `auto` sets nothing at all — the engine paces nothing and macOS's own
/// thermal management decides the fans. Prefill is the one command that pins
/// the GPU for seconds at a time, so it is what heats a Mac and what its fans
/// answer; decoding keeps its speed at every rung, and a lower one also backs
/// off further as the Mac's thermal state rises.
enum ThermalRung: String, CaseIterable, Identifiable {
    case auto, performance, balanced, moderate, quiet

    var id: String { rawValue }

    /// The rung's duty, which is what the server passes as --prefill-duty and
    /// what /v1/thermal changes on a running engine.
    var duty: Double {
        switch self {
        case .auto: return 100
        case .performance: return 90
        case .balanced: return 75
        case .moderate: return 50
        case .quiet: return 30
        }
    }

    var title: String { L10n.string("thermal." + rawValue) }

    /// How much longer a prompt takes at this rung, as the multiple a person
    /// reads: prefill runs the given share of the wall time.
    var cost: Double { 100 / duty }

    /// The rung a duty the engine holds is on, or nil for a duty between the
    /// rungs, as a server started outside the app with --prefill-duty leaves.
    init?(duty: Double) {
        guard let match = Self.allCases.first(where: { abs($0.duty - duty) < 0.5 })
        else { return nil }
        self = match
    }
}

/// The thermal page: the heat the Mac reports, read-only, and the one control
/// that trades prompt speed for quiet. The control belongs to the app rather
/// than to a model configuration, and a change reaches a running engine at
/// once through /v1/thermal, so nothing needs a restart.
struct ThermalPane: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                ladder
                if model.live.updated {
                    MetricSection(
                        title: L10n.string("thermal.fans"),
                        systemImage: "fan.fill",
                        cards: fanCards
                    )
                    MetricSection(
                        title: L10n.string("thermal.sensors"),
                        systemImage: "thermometer.medium",
                        cards: temperatureCards
                    )
                    MetricSection(
                        title: L10n.string("thermal.pacing"),
                        systemImage: "metronome.fill",
                        cards: pacingCards
                    )
                } else {
                    ContentUnavailableView(
                        L10n.string("thermal.empty.title"),
                        systemImage: "fan",
                        description: Text(verbatim: L10n.string("thermal.empty.description"))
                    )
                    .padding(.top, 40)
                }
            }
            .padding(18)
        }
        .background(.background)
    }

    /// The five rungs, with the one in effect marked. While a server runs, what
    /// is marked is what the engine holds, which is the truth a change is read
    /// back from; while none does, it is the rung the next start will use.
    private var ladder: some View {
        let running = model.live.updated
        let marked = running ? model.engineRung : model.thermalRung
        return VStack(alignment: .leading, spacing: 13) {
            HStack(spacing: 8) {
                Label(L10n.string("thermal.ladder.title"), systemImage: "dial.medium.fill")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.secondary)
                Spacer(minLength: 0)
                if running, let duty = model.engineDutyPercent {
                    Text(L10n.format("thermal.duty.held", Int(duty.rounded())))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            HStack(spacing: 10) {
                ForEach(ThermalRung.allCases) { rung in
                    RungButton(rung: rung, selected: marked == rung) {
                        model.setThermalRung(rung)
                    }
                }
            }
            Text(costLine())
                .font(.callout)
                .foregroundStyle(.secondary)
            HStack(spacing: 6) {
                Image(systemName: running ? "bolt.fill" : "moon.zzz.fill")
                    .font(.caption)
                    .foregroundStyle(running ? .green : .secondary)
                Text(running ? L10n.string("thermal.applies") : L10n.string("thermal.next_start"))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if let error = model.thermalError {
                Label(error, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
        .overlay(
            RoundedRectangle(cornerRadius: 14).strokeBorder(.separator.opacity(0.5))
        )
    }

    /// The price of the rung in effect, which while a server runs is the one
    /// the engine holds: a server started elsewhere, or a change made through
    /// the CLI, need not be the rung the app would start.
    private func costLine() -> String {
        let duty = model.live.updated ? model.engineDutyPercent : model.thermalRung.duty
        guard let duty, duty < 100 else { return L10n.string("thermal.cost.full") }
        return L10n.format("thermal.cost.paced", 100 / duty)
    }

    /// One block per part: CPU, GPU, IO and the disk, each the hottest of the
    /// SMC keys that belong to it. A Mac that reports none of a group's keys
    /// keeps the block with no reading rather than hiding it, so the four are
    /// always read the same way.
    private var temperatureCards: [MetricCard] {
        let readings = model.live.temperatures
        let hottest = TemperatureGroup.hottest(readings)?.group
        return TemperatureGroup.allCases.map { group in
            let celsius = TemperatureGroup.peak(readings, in: group)
            return MetricCard(
                title: group.title,
                value: celsius.map { String(format: "%.1f °C", $0) }
                    ?? L10n.string("thermal.unavailable"),
                caption: group == hottest ? L10n.string("thermal.sensor.hottest") : "",
                systemImage: icon(group),
                tint: celsius.map(heatTint) ?? .gray
            )
        }
    }

    private func icon(_ group: TemperatureGroup) -> String {
        switch group {
        case .cpu: return "cpu"
        case .gpu: return "square.stack.3d.up.fill"
        case .io: return "arrow.left.arrow.right"
        case .disk: return "internaldrive"
        }
    }

    /// One card per fan, read-only: the speed against what the fan can do, so
    /// the bar says how hard it is already working.
    private var fanCards: [MetricCard] {
        let live = model.live
        guard live.fansAvailable, !live.fanSpeeds.isEmpty else {
            return [
                MetricCard(title: L10n.string("thermal.fans"),
                           value: L10n.string("thermal.unavailable"),
                           caption: L10n.string("thermal.no_fans"),
                           systemImage: "fan", tint: .gray)
            ]
        }
        return live.fanSpeeds.enumerated().map { index, rpm in
            let maximum = live.fanMaximums.indices.contains(index)
                ? live.fanMaximums[index] : nil
            let minimum = live.fanMinimums.indices.contains(index)
                ? live.fanMinimums[index] : nil
            return MetricCard(
                title: L10n.format("thermal.fan", index + 1),
                value: "\(rpm)",
                caption: fanCaption(minimum: minimum, maximum: maximum),
                systemImage: "fan.fill",
                tint: rpm == 0 ? .secondary : .cyan,
                progress: maximum.flatMap { $0 > 0 ? Double(rpm) / Double($0) : nil }
            )
        }
    }

    private func fanCaption(minimum: Int?, maximum: Int?) -> String {
        guard let minimum, let maximum else { return L10n.string("thermal.caption.rpm") }
        return L10n.format("thermal.fan.caption", minimum, maximum)
    }

    /// What the race against the heat costs while it serves: the duty the
    /// engine holds, what the Mac's thermal state leaves of it, and the waits
    /// the pacing has cost so far.
    private var pacingCards: [MetricCard] {
        let live = model.live
        let scale = live.thermalScale
        return [
            MetricCard(title: L10n.string("thermal.duty"),
                       value: live.dutyPercent.map { String(format: "%.0f%%", $0) } ?? "—",
                       caption: L10n.string("thermal.duty.caption"),
                       systemImage: "metronome.fill", tint: .teal,
                       progress: live.dutyPercent.map { $0 / 100 }),
            MetricCard(title: L10n.string("thermal.scale"),
                       value: scale.map { String(format: "×%.2f", $0) } ?? "—",
                       caption: L10n.string("thermal.scale.caption"),
                       systemImage: "thermometer.snowflake",
                       tint: (scale ?? 1) >= 1 ? .green : .orange,
                       progress: scale),
            MetricCard(title: L10n.string("thermal.waits"),
                       value: "\(live.pacedWaits ?? 0)",
                       caption: waitsCaption,
                       systemImage: "hourglass", tint: .orange),
        ]
    }

    private var waitsCaption: String {
        guard let waited = model.live.pacedWaitMilliseconds, waited > 0 else {
            return L10n.string("thermal.waits.none")
        }
        return L10n.format("thermal.waits.caption", seconds(waited / 1000))
    }

    private func seconds(_ value: Double) -> String {
        value < 60 ? String(format: "%.1fs", value)
                   : String(format: "%.0fm", value / 60)
    }
}

/// Green while the Mac is cool, then orange, then red as it approaches the
/// ceiling the runtime takes readings to.
func heatTint(_ celsius: Double) -> Color {
    celsius < 70 ? .green : (celsius < 90 ? .orange : .red)
}

/// One rung of the ladder: its name and the share of the wall time prefill may
/// run at it, marked when it is the rung in effect.
private struct RungButton: View {
    let rung: ThermalRung
    let selected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 3) {
                Text(rung.title)
                    .font(.callout.weight(selected ? .semibold : .regular))
                    .lineLimit(1)
                    .minimumScaleFactor(0.65)
                Text("\(Int(rung.duty))%")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(selected ? .primary : .secondary)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 9)
            .background(
                selected ? Color.accentColor.opacity(0.16) : Color.secondary.opacity(0.08),
                in: RoundedRectangle(cornerRadius: 10)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 10)
                    .strokeBorder(selected ? Color.accentColor.opacity(0.7) : .clear)
            )
        }
        .buttonStyle(.plain)
        .help(L10n.format("thermal.rung.help", Int(rung.duty)))
    }
}
