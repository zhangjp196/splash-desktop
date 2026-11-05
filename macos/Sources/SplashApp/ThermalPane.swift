import SwiftUI

/// The heat-and-noise control, in the server's own names
/// (server/serve_options.py THERMAL_POLICIES): the three modes the app
/// offers, loudest first. `auto` sets nothing at all — the engine paces
/// nothing and macOS's own thermal management decides the fans. `smart` is
/// not a fixed mode: the app reads the fans and temperatures from /status
/// each second and pushes a computed duty to the engine through
/// /v1/thermal, so the Mac stays as loud as it needs to be and no louder.
/// `fixed` runs at one of the ladder's rungs, each the share of wall time
/// prefill may keep the GPU busy. Prefill is the one command that pins the
/// GPU for seconds at a time, so it is what heats a Mac and what its fans
/// answer; decoding keeps its speed in every mode, and a lower duty also
/// backs off further as the Mac's thermal state rises.
enum ThermalRung: String, CaseIterable, Identifiable {
    case auto, smart, fixed

    var id: String { rawValue }

    /// The mode's fixed duty, which is what the server passes as
    /// --prefill-duty and what /v1/thermal changes on a running engine.
    /// `auto` is the only mode with a duty of its own: `smart` holds none,
    /// the app computes one from the fans and temperatures, and `fixed`
    /// carries one in its level (FixedLevel).
    var duty: Double? {
        switch self {
        case .auto: return 100
        case .smart: return nil
        case .fixed: return nil
        }
    }

    var title: String { L10n.string("thermal." + rawValue) }

    /// How much longer a prompt takes at this mode, as the multiple a person
    /// reads: prefill runs the given share of the wall time.
    var cost: Double? { duty.map { 100 / $0 } }
}

/// The levels of the fixed mode, in the server's own names: the rungs of
/// the ladder the server passes as --thermal-policy, each the share of
/// wall time prefill may keep the GPU busy. Unlike the smart levels, each is
/// a duty itself: held steady while the fixed mode is chosen, and what
/// /v1/thermal changes on a running engine.
enum FixedLevel: String, CaseIterable, Identifiable {
    case balanced, moderate, quiet

    var id: String { rawValue }

    var title: String { L10n.string("thermal." + rawValue) }

    /// The share of wall time prefill may keep the GPU busy at this level,
    /// what the server passes as --prefill-duty and what /v1/thermal changes
    /// on a running engine.
    var duty: Double {
        switch self {
        case .balanced: return 68
        case .moderate: return 48
        case .quiet: return 33
        }
    }

    /// The level a duty the engine holds is on, or nil for a duty between the
    /// levels, as a server started outside the app with --prefill-duty leaves.
    init?(duty: Double) {
        guard let match = Self.allCases.first(where: { abs($0.duty - duty) < 0.5 })
        else { return nil }
        self = match
    }
}

/// How hard the smart mode works. Unlike the fixed rungs, none of the three is
/// a duty: each is the fan the Mac may hear before prefill gives way, and the
/// duty follows the fans and the sensors from second to second (AppModel
/// smartDuty). High lets the fans run up and keeps the GPU busy; low holds them
/// near where they idle and trades prompt speed for quiet.
enum SmartLevel: String, CaseIterable, Identifiable {
    case high, medium, low

    var id: String { rawValue }

    var title: String { L10n.string("thermal.smart." + rawValue) }

    /// The share of a fan's own range this level lets the Mac reach before
    /// prefill starts to give way: below it the fans are the Mac's quiet answer
    /// and prefill runs at the ceiling, above it the duty falls off toward the
    /// floor as the fans approach their top.
    var fanShare: Double {
        switch self {
        case .high: return 0.85
        case .medium: return 0.60
        case .low: return 0.40
        }
    }

    /// The most prefill may keep the GPU busy while the fans are still below
    /// their level's share.
    var ceiling: Double {
        switch self {
        case .high: return 100
        case .medium: return 75
        case .low: return 55
        }
    }

    /// The least the smart mode backs off to, so a hot Mac still makes
    /// progress instead of stopping.
    var floor: Double { 12 }
}

/// The thermal page: the heat the Mac reports, read-only, and the one control
/// that trades prompt speed for quiet. The control belongs to the app rather
/// than to a model configuration, and a change reaches a running engine at
/// once through /v1/thermal, so nothing needs a restart.
struct ThermalPane: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                PaneHeader(
                    title: L10n.string("view.thermal"),
                    subtitle: L10n.string("thermal.subtitle"),
                    eyebrow: L10n.string("thermal.eyebrow")
                )
                ladder
                if model.live.updated {
                    MetricSection(
                        title: L10n.string("thermal.heat"),
                        systemImage: "thermometer.medium",
                        cards: heatCards
                    )
                    MetricSection(
                        title: L10n.string("thermal.pacing"),
                        systemImage: "metronome.fill",
                        cards: pacingCards
                    )
                } else {
                    PaneEmpty(
                        title: L10n.string("thermal.empty.title"),
                        message: L10n.string("thermal.empty.description"),
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
    /// `smart` shows its level picker, since its duty is computed, not fixed,
    /// and `fixed` shows one too, since its duty is the level's.
    private var ladder: some View {
        let running = model.live.updated
        // `smart` holds no rung of the server's ladder: the duty the engine
        // carries is one the app computed, so it can land between the rungs
        // and match none of them. What is marked is therefore the app's own
        // choice while it is the one pacing the engine.
        let marked = running
            ? (model.thermalRung == .smart ? ThermalRung.smart : model.engineRung)
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
                    RungButton(rung: rung, duty: effectiveDuty(for: rung), selected: marked == rung) {
                        model.setThermalRung(rung)
                    }
                }
            }
            if model.thermalRung == .smart {
                smartLevelRow
            } else if model.thermalRung == .fixed {
                fixedLevelRow
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
        .glassPanel(tint: SplashPalette.amber, raised: true)
    }

    /// The duty a mode carries in effect, for its button's label: the mode's
    /// own for `auto`, the chosen level's for `fixed`, and none for `smart`,
    /// whose duty is computed from the fans and temperatures.
    private func effectiveDuty(for rung: ThermalRung) -> Double? {
        switch rung {
        case .auto: return rung.duty
        case .smart: return nil
        case .fixed: return model.fixedLevel.duty
        }
    }

    /// The smart mode's levels, with the one in effect marked and what it
    /// decides, in the same words the rungs use for their duty: the fan band
    /// this level allows before prefill gives way.
    private var smartLevelRow: some View {
        HStack(spacing: 10) {
            Text(L10n.string("thermal.smart.level"))
                .font(.caption)
                .foregroundStyle(.secondary)
            Picker("", selection: Binding(
                get: { model.smartLevel },
                set: { model.setSmartLevel($0) }
            )) {
                ForEach(SmartLevel.allCases) { level in
                    Text(level.title).tag(level)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .frame(maxWidth: 240)
            Spacer(minLength: 0)
            Text(L10n.format("thermal.smart.band", model.smartLevel.fanShare * 100))
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
        }
    }

    /// The fixed mode's levels, in the same shape the smart mode's picker has:
    /// the level in effect, and what it decides, in the same words the rungs
    /// use for their duty: the share of the wall time prefill may run at it.
    private var fixedLevelRow: some View {
        HStack(spacing: 10) {
            Text(L10n.string("thermal.fixed.level"))
                .font(.caption)
                .foregroundStyle(.secondary)
            Picker("", selection: Binding(
                get: { model.fixedLevel },
                set: { model.setFixedLevel($0) }
            )) {
                ForEach(FixedLevel.allCases) { level in
                    Text(level.title).tag(level)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .frame(maxWidth: 240)
            Spacer(minLength: 0)
            Text(L10n.format("thermal.fixed.duty", model.fixedLevel.duty))
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
        }
    }

    /// The price of the mode in effect, which while a server runs is the one
    /// the engine holds: a server started elsewhere, or a change made through
    /// the CLI, need not be the mode the app would start. `smart` has no fixed
    /// price: it is whatever the fans and temperatures ask for.
    private func costLine() -> String {
        if model.thermalRung == .smart {
            return L10n.string("thermal.cost.smart")
        }
        let duty = model.live.updated
            ? model.engineDutyPercent
            : (model.thermalRung == .fixed ? model.fixedLevel.duty : model.thermalRung.duty)
        guard let duty, duty < 100 else { return L10n.string("thermal.cost.full") }
        return L10n.format("thermal.cost.paced", 100 / duty)
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
        let live = model.live
        guard live.fansAvailable, !live.fanSpeeds.isEmpty else {
            return MetricCard(title: L10n.string("thermal.fans"),
                              value: L10n.string("thermal.unavailable"),
                              caption: L10n.string("thermal.no_fans"),
                              systemImage: "fan", tint: .gray)
        }
        let fastest = live.fanSpeeds.max() ?? 0
        let maximum = live.fanMaximums.max()
        return MetricCard(
            title: L10n.string("thermal.fans"),
            value: "\(fastest)",
            caption: fanCaption(),
            systemImage: "fan.fill",
            tint: fastest == 0 ? .secondary : .cyan,
            progress: maximum.flatMap { $0 > 0 ? Double(fastest) / Double($0) : nil }
        )
    }

    /// "2 fans, 1342–3197 RPM", or the count alone when the SMC gave no range.
    private func fanCaption() -> String {
        let speeds = model.live.fanSpeeds
        guard let low = speeds.min(), let high = speeds.max(), speeds.count > 1 else {
            return speeds.count > 1
                ? L10n.format("thermal.fan.count", speeds.count)
                : L10n.string("thermal.caption.rpm")
        }
        return L10n.format("thermal.fan.range", speeds.count, low, high)
    }

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

    /// What the race against the heat costs while it serves, in two blocks:
    /// the duty the engine holds with the backoff the Mac's thermal state
    /// leaves of it, and the waits that pacing has cost so far.
    private var pacingCards: [MetricCard] {
        let live = model.live
        let scale = live.thermalScale
        let backoff = scale.map { L10n.format("thermal.scale.caption", $0) }
        return [
            MetricCard(title: L10n.string("thermal.duty"),
                       value: live.dutyPercent.map { String(format: "%.0f%%", $0) } ?? "—",
                       caption: backoff ?? L10n.string("thermal.duty.caption"),
                       systemImage: "metronome.fill", tint: .teal,
                       progress: live.dutyPercent.map { $0 / 100 }),
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

/// One mode of the control: its name and the share of the wall time prefill
/// may run at it, marked when it is the mode in effect. `smart` shows no
/// fixed share: its duty is computed from the fans and temperatures; `fixed`
/// shows its level's share.
private struct RungButton: View {
    let rung: ThermalRung
    /// The duty the mode carries in effect: the mode's own for `auto`, the
    /// chosen level's for `fixed`, and none for `smart`.
    let duty: Double?
    let selected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 3) {
                Text(rung.title)
                    .font(.callout.weight(selected ? .semibold : .regular))
                    .lineLimit(1)
                    .minimumScaleFactor(0.65)
                Text(dutyLabel)
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(selected ? .primary : .secondary)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 9)
            .background(
                selected
                    ? AnyShapeStyle(
                        LinearGradient(
                            colors: [
                                SplashPalette.blue.opacity(0.24),
                                SplashPalette.cyan.opacity(0.09),
                            ],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
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
        .help(helpText)
    }

    private var dutyLabel: String {
        guard let duty else { return L10n.string("thermal.smart.duty") }
        return "\(Int(duty))%"
    }

    private var helpText: String {
        guard let duty else { return L10n.string("thermal.rung.help.smart") }
        return L10n.format("thermal.rung.help", Int(duty))
    }
}
