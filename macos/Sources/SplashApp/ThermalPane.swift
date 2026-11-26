import SwiftUI

/// The three thermal choices. Both smart modes adapt their prefill duty from
/// live fan, temperature and macOS pressure readings; system default leaves
/// pacing to macOS.
enum ThermalRung: String, CaseIterable, Identifiable {
    case smartOne, smartTwo, auto

    var id: String { rawValue }
    var isSmart: Bool { self != .auto }

    var title: String {
        switch self {
        case .smartOne: return L10n.string("thermal.smart.one")
        case .smartTwo: return L10n.string("thermal.smart.two")
        case .auto: return L10n.string("thermal.auto")
        }
    }

    var subtitle: String {
        switch self {
        case .smartOne: return L10n.string("thermal.smart.one.caption")
        case .smartTwo: return L10n.string("thermal.smart.two.caption")
        case .auto: return L10n.string("thermal.auto.caption")
        }
    }

    var fanShare: Double {
        switch self {
        case .smartOne: return 0.45
        case .smartTwo: return 0.70
        case .auto: return 1
        }
    }

    var ceiling: Double {
        switch self {
        case .smartOne: return 72
        case .smartTwo: return 92
        case .auto: return 100
        }
    }

    var floor: Double { 15 }
    var duty: Double? { self == .auto ? 100 : nil }
}

/// The thermal page: the heat the Mac reports, read-only, and the one control
/// that trades prompt speed for quiet. The control belongs to the app rather
/// than to a model configuration, and a change reaches a running engine at
/// once through /v1/thermal, so nothing needs a restart.
///
/// The page splits by how often its parts move: the ladder is the user's own
/// choice and changes when they act, while the readings follow /status every
/// second. The readings are their own Equatable view over the status alone,
/// so a poll that moved no sensor leaves the ladder and the header alone.
struct ThermalPane: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                PaneHeader(
                    title: L10n.string("view.thermal")
                )
                ladder
                if model.live.updated {
                    ThermalReadings(live: model.live)
                        .equatable()
                } else {
                    PaneEmpty(
                        title: L10n.string("thermal.empty.title"),
                        systemImage: "fan",
                        actionTitle: model.canStart ? L10n.string("start") : nil,
                        action: model.canStart ? { model.start() } : nil
                    )
                    .frame(minHeight: 260)
                }
            }
            .padding(Theme.pagePadding)
            .frame(maxWidth: .infinity, alignment: .topLeading)
        }
        .background(Color.clear)
    }

    /// The modes, with the one in effect marked. While a server runs, what
    /// is marked is what the engine holds, which is the truth a change is read
    /// back from; while none does, it is the mode the next start will use.
    /// Either smart choice computes its own duty from the current readings.
    private var ladder: some View {
        let running = model.live.updated
        // `smart` holds no rung of the server's ladder: the duty the engine
        // carries is one the app computed, so it can land between the rungs
        // and match none of them. What is marked is therefore the app's own
        // choice while it is the one pacing the engine.
        let marked = running
            ? (model.thermalRung.isSmart ? model.thermalRung : model.engineRung)
            : model.thermalRung
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
            if model.thermalRung.isSmart {
                smartStatusRow
            }
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
        .glassPanel(tint: SplashPalette.amber, raised: true)
    }

    private var smartStatusRow: some View {
        let pressure = model.live.thermalState
        let reason = smartReason
        return HStack(spacing: 8) {
            Image(systemName: pressureIcon(pressure))
                .foregroundStyle(thermalPressureTint(pressure))
            Text(reason)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(.primary)
                .lineLimit(1)
                .minimumScaleFactor(0.85)
            Spacer(minLength: 4)
            Text(L10n.format("thermal.smart.target", Int(model.smartDuty().rounded())))
                .font(.system(size: 12, weight: .semibold, design: .rounded).monospacedDigit())
                .foregroundStyle(SplashPalette.blue)
        }
        .padding(.top, 2)
    }

    private var smartReason: String {
        let live = model.live
        if live.thermalState == .critical { return L10n.string("thermal.reason.critical") }
        if live.thermalState == .serious { return L10n.string("thermal.reason.pressure") }
        if let hottest = TemperatureGroup.hottest(live.temperatures)?.celsius, hottest >= 75 {
            return L10n.string("thermal.reason.temperature")
        }
        if let share = fanUtilization(live), share >= model.thermalRung.fanShare {
            return L10n.string("thermal.reason.fans")
        }
        if live.thermalState == .fair { return L10n.string("thermal.reason.pressure") }
        if live.temperatures.isEmpty && (!live.fansAvailable || live.fanSpeeds.isEmpty) {
            return L10n.string("thermal.reason.unknown")
        }
        return L10n.string("thermal.reason.cool")
    }

    private func pressureIcon(_ pressure: ThermalPressure?) -> String {
        switch pressure {
        case .nominal: return "checkmark.circle.fill"
        case .fair: return "thermometer.medium"
        case .serious: return "thermometer.high"
        case .critical: return "exclamationmark.triangle.fill"
        case nil: return "waveform.path"
        }
    }

}

/// The page's read-only half: the heat the Mac reports and what the pacing has
/// cost, drawn from one `/status` reading. It is Equatable over that reading
/// alone, so a poll that moved no sensor, fan or counter skips both sections
/// instead of redrawing them beside an untouched ladder.
private struct ThermalReadings: View, Equatable {
    let live: LiveStatus

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            MetricSection(
                title: L10n.string("thermal.heat"),
                systemImage: "thermometer.medium",
                cards: heatCards
            )
            .equatable()
            MetricSection(
                title: L10n.string("thermal.pacing"),
                systemImage: "metronome.fill",
                cards: pacingCards
            )
            .equatable()
        }
    }

    /// The Mac's heat in one row: the fans as a single block — the fastest
    /// against what it can do, the rest of the pack in its caption — then one
    /// block per part (CPU, GPU, IO and the disk), each the hottest of the SMC
    /// keys that belong to it. A Mac that reports none of a group's keys keeps
    /// the block with no reading rather than hiding it, so the row is always
    /// read the same way.
    private var heatCards: [MetricCard] {
        [fanCard] + temperatureCards
    }

    /// Every fan folded into one block: a MacBook's two and a Mac Studio's two
    /// answer the same question, and the row is read by comparing the parts of
    /// the Mac, not one fan against another.
    private var fanCard: MetricCard {
        guard live.fansAvailable, !live.fanSpeeds.isEmpty else {
            return MetricCard(title: L10n.string("thermal.fans"),
                              value: L10n.string("thermal.unavailable"),
                              caption: L10n.string("thermal.no_fans"),
                              systemImage: "fan", tint: .gray)
        }
        let fastest = live.fanSpeeds.max() ?? 0
        return MetricCard(
            title: L10n.string("thermal.fans"),
            value: "\(fastest)",
            caption: fanCaption(),
            systemImage: "fan.fill",
            tint: fastest == 0 ? .secondary : .cyan,
            progress: fanUtilization(live)
        )
    }

    /// "2 fans, 1342–3197 RPM", or the count alone when the SMC gave no range.
    private func fanCaption() -> String {
        let speeds = live.fanSpeeds
        guard let low = speeds.min(), let high = speeds.max(), speeds.count > 1 else {
            return speeds.count > 1
                ? L10n.format("thermal.fan.count", speeds.count)
                : L10n.string("thermal.caption.rpm")
        }
        return L10n.format("thermal.fan.range", speeds.count, low, high)
    }

    private var temperatureCards: [MetricCard] {
        let readings = live.temperatures
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

    /// What the race against the heat costs while it serves, in two blocks:
    /// the duty the engine holds with the backoff the Mac's thermal state
    /// leaves of it, and the waits that pacing has cost so far.
    private var pacingCards: [MetricCard] {
        let scale = live.thermalScale
        let backoff = scale.map { L10n.format("thermal.scale.caption", $0) }
        let effectiveDuty = live.dutyPercent.flatMap { duty in
            live.thermalScale.map { duty * $0 }
        }
        return [
            MetricCard(title: L10n.string("thermal.duty"),
                       value: live.dutyPercent.map { String(format: "%.0f%%", $0) } ?? "—",
                       caption: L10n.string("thermal.duty.caption"),
                       systemImage: "metronome.fill", tint: .teal,
                       progress: live.dutyPercent.map { $0 / 100 }),
            MetricCard(title: L10n.string("thermal.effective_duty"),
                       value: effectiveDuty.map { String(format: "%.0f%%", $0) } ?? "—",
                       caption: backoff ?? L10n.string("thermal.duty.caption"),
                       systemImage: "waveform.path", tint: .cyan,
                       progress: effectiveDuty.map { $0 / 100 }),
            MetricCard(title: L10n.string("thermal.pressure"),
                       value: live.thermalState?.title ?? L10n.string("thermal.unavailable"),
                       caption: L10n.string("thermal.pressure.caption"),
                       systemImage: "thermometer.medium",
                       tint: thermalPressureTint(live.thermalState)),
            MetricCard(title: L10n.string("thermal.waits"),
                       value: "\(live.pacedWaits ?? 0)",
                       caption: waitsCaption,
                       systemImage: "hourglass", tint: .orange),
        ]
    }

    private var waitsCaption: String {
        guard let waited = live.pacedWaitMilliseconds, waited > 0 else {
            return L10n.string("thermal.waits.none")
        }
        return L10n.format("thermal.waits.caption", seconds(waited / 1000))
    }

    private func seconds(_ value: Double) -> String {
        value < 60 ? String(format: "%.1fs", value)
                   : String(format: "%.0fm", value / 60)
    }
}

private func thermalPressureTint(_ pressure: ThermalPressure?) -> Color {
    switch pressure {
    case .nominal: return .green
    case .fair: return .yellow
    case .serious: return .orange
    case .critical: return .red
    case nil: return .gray
    }
}

/// Green while the Mac is cool, then orange, then red as it approaches the
/// ceiling the runtime takes readings to.
func heatTint(_ celsius: Double) -> Color {
    celsius < 70 ? .green : (celsius < 90 ? .orange : .red)
}

/// One of the three available thermal modes.
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
                Text(rung.subtitle)
                    .font(.caption)
                    .foregroundStyle(selected ? .primary : .secondary)
                    .lineLimit(1)
                    .minimumScaleFactor(0.75)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 9)
            .background(
                // A flat tint wash rather than a gradient: the ladder is a
                // three-state choice, and a gradient fill would blend an
                // extra layer for a state that changes a few times a day.
                selected
                    ? AnyShapeStyle(SplashPalette.blue.opacity(0.20))
                    : AnyShapeStyle(SplashPalette.surface.opacity(0.38)),
                in: RoundedRectangle(cornerRadius: 10, style: .continuous)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(
                        selected ? SplashPalette.cyan.opacity(0.6) : SplashPalette.border.opacity(0.35),
                        lineWidth: selected ? 1 : 0.7
                    )
            )
        }
        .buttonStyle(.plain)
    }


}
