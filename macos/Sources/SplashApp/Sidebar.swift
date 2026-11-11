import SwiftUI

/// The sidebar: the brand and what it serves, the panes, and the server's
/// state and controls pinned beneath them.
struct Sidebar: View {
    @EnvironmentObject private var model: AppModel
    @Binding var selection: RootView.Section
    @Binding var showPalette: Bool

    var body: some View {
        VStack(spacing: 0) {
            brand
                .padding(.horizontal, 14)
                .padding(.top, 14)
                .padding(.bottom, 8)

            ScrollView {
                VStack(alignment: .leading, spacing: 2) {
                    row(.chat)
                    groupLabel("sidebar.serve")
                    row(.models)
                    groupLabel("sidebar.monitor")
                    row(.live)
                    row(.thermal)
                    row(.log)
                    groupLabel("sidebar.app")
                    row(.settings)
                    groupLabel("sidebar.about")
                    row(.about)
                }
                .padding(.horizontal, 10)
                .padding(.top, 4)
            }
            .scrollIndicators(.hidden)

            ServerControls()
        }
        .background {
            // A flat translucent wash, not a Material: the sidebar sits over
            // the aurora for the whole app's life, and a backdrop blur under
            // it would re-sample the scene every frame for a shade the panes
            // already carry.
            ZStack {
                Rectangle().fill(SplashPalette.surface.opacity(0.55))
                LinearGradient(
                    colors: [
                        SplashPalette.blue.opacity(0.10),
                        SplashPalette.canvas.opacity(0.22),
                        SplashPalette.cyan.opacity(0.05),
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
            }
        }
        .overlay(alignment: .trailing) {
            Rectangle()
                .fill(SplashPalette.border.opacity(0.35))
                .frame(width: 1)
        }
    }

    private var brand: some View {
        HStack(spacing: 10) {
            BrandMark(size: 34)
            VStack(alignment: .leading, spacing: 1) {
                Text(verbatim: "Splash")
                    .font(.system(size: 17, weight: .bold, design: .rounded))
                    .tracking(-0.3)
                Text(verbatim: model.runningConfig?.displayName
                     ?? model.activeConfig?.displayName
                     ?? model.live.modelID
                     ?? L10n.string("models.none"))
                    .font(.system(size: 9.5, weight: .medium))
                    .foregroundStyle(SplashPalette.faint)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            Spacer(minLength: 0)
            Button {
                showPalette = true
            } label: {
                Image(systemName: "command")
                    .font(.system(size: 10.5, weight: .semibold))
                    .foregroundStyle(SplashPalette.muted)
                    .frame(width: 25, height: 23)
                    .background(SplashPalette.surface.opacity(0.5), in: RoundedRectangle(cornerRadius: 7, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: 7, style: .continuous)
                            .strokeBorder(SplashPalette.border.opacity(0.5), lineWidth: 0.7)
                    )
            }
            .buttonStyle(.plain)
            .help(L10n.string("palette.shortcut"))
        }
    }

    @State private var hovered: RootView.Section?

    private func row(_ section: RootView.Section) -> some View {
        let active = selection == section
        return Button {
            selection = section
        } label: {
            HStack(spacing: 10) {
                Image(systemName: section.systemImage)
                    .font(.system(size: 11.5, weight: .semibold))
                    .frame(width: 18)
                    .foregroundStyle(active ? section.tint : SplashPalette.muted)
                Text(section.title)
                    .font(.system(size: 12.5, weight: active ? .semibold : .regular))
                    .foregroundStyle(active ? Color.primary : SplashPalette.muted)
                Spacer(minLength: 0)
                if section == .live, model.phase == .ready {
                    StatusBeacon(color: stageColor(model.stage), active: true)
                        .scaleEffect(0.85)
                } else if active {
                    Circle()
                        .fill(section.tint)
                        .frame(width: 4, height: 4)
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .background {
                if active {
                    RoundedRectangle(cornerRadius: 9, style: .continuous)
                        .fill(
                            LinearGradient(
                                colors: [section.tint.opacity(0.20), section.tint.opacity(0.05)],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 9, style: .continuous)
                                .strokeBorder(section.tint.opacity(0.25), lineWidth: 0.8)
                        )
                } else if hovered == section {
                    RoundedRectangle(cornerRadius: 9, style: .continuous)
                        .fill(SplashPalette.sheen.opacity(0.05))
                }
            }
            .overlay(alignment: .leading) {
                if active {
                    Capsule()
                        .fill(section.tint)
                        .frame(width: 2.5, height: 14)
                        .offset(x: 2)
                }
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .keyboardShortcut(section.shortcut, modifiers: .command)
        .onHover { inside in
            if inside { hovered = section }
            else if hovered == section { hovered = nil }
        }
        .animation(.easeOut(duration: 0.14), value: active)
        .animation(.easeOut(duration: 0.14), value: hovered)
    }

    private func groupLabel(_ key: String) -> some View {
        Text(L10n.string(key).uppercased())
            .font(.system(size: 9, weight: .bold))
            .tracking(1.4)
            .foregroundStyle(SplashPalette.faint.opacity(0.8))
            .padding(.horizontal, 10)
            .padding(.top, 14)
            .padding(.bottom, 4)
    }
}

/// The running state and the primary actions, pinned under the sidebar.
private struct ServerControls: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                StatusBeacon(color: indicator, active: model.isRunning)
                VStack(alignment: .leading, spacing: 1) {
                    Text(model.statusText)
                        .font(.system(size: 10.5, weight: .semibold))
                        .lineLimit(1)
                        .truncationMode(.tail)
                    HStack(spacing: 5) {
                        // The uptime ticks on a clock of its own: the model
                        // no longer publishes every second (an idle engine
                        // must not wake the whole interface), so this line
                        // re-reads the running time on a one-second
                        // TimelineView instead.
                        TimelineView(.periodic(from: .now, by: 1)) { _ in
                            if let uptime = model.uptimeText {
                                Text(L10n.format("sidebar.uptime", uptime))
                            } else {
                                Text(model.phase == .idle ? L10n.string("sidebar.offline") : L10n.string("sidebar.working"))
                            }
                        }
                        if let tokens = model.contextTokens, model.phase == .ready {
                            Text(verbatim: "·")
                            Text(L10n.format("sidebar.context", tokens.formatted()))
                        }
                    }
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(SplashPalette.faint)
                    .lineLimit(1)
                }
                Spacer(minLength: 4)
            }

            if model.isRunning {
                HStack(spacing: 0) {
                    rateCell(L10n.string("live.decode"), model.live.decodeTokensPerSecond, SplashPalette.mint)
                    Rectangle()
                        .fill(SplashPalette.hairline)
                        .frame(width: 1, height: 22)
                    rateCell(L10n.string("live.prefill"), model.live.prefillTokensPerSecond, SplashPalette.cyan)
                }
                .padding(.vertical, 4)
                .background(SplashPalette.surface.opacity(0.42), in: RoundedRectangle(cornerRadius: 9, style: .continuous))

                if model.phase == .ready {
                    Sparkline(values: model.decodeHistory, tint: SplashPalette.mint)
                        .frame(height: 14)
                        .padding(.horizontal, 1)
                        .opacity(0.9)
                }
            }

            if model.isRunning {
                Button(role: .destructive) {
                    model.stop()
                } label: {
                    powerLabel(
                        L10n.string("stop"),
                        systemImage: "stop.fill",
                        colors: [SplashPalette.rose, Color(red: 0.9, green: 0.25, blue: 0.32)]
                    )
                }
                .buttonStyle(.plain)
                .disabled(model.phase == .stopping)

                HStack(spacing: 6) {
                    smallButton("safari.fill", title: L10n.string("open.in.browser")) {
                        model.openInBrowser()
                    }
                    .disabled(model.phase != .ready)
                    smallButton("doc.on.doc.fill", title: L10n.string("menu.copy_api")) {
                        model.copyToClipboard(model.apiBaseURL)
                    }
                }
            } else {
                Button {
                    model.start()
                } label: {
                    powerLabel(
                        L10n.string("start"),
                        systemImage: "play.fill",
                        colors: [SplashPalette.blue, SplashPalette.cyan]
                    )
                }
                .buttonStyle(.plain)
                .disabled(!model.canStart)
                .keyboardShortcut(.return, modifiers: .command)
            }
        }
        .padding(12)
        .glassPanel(radius: 16, tint: indicator.opacity(0.55), raised: true)
        .padding(.horizontal, 12)
        .padding(.bottom, 12)
        .animation(.easeOut(duration: 0.2), value: model.phase)
    }

    private func rateCell(_ title: String, _ value: Double?, _ tint: Color) -> some View {
        VStack(spacing: 1) {
            Text(title.uppercased())
                .font(.system(size: 7.5, weight: .bold))
                .tracking(0.5)
                .foregroundStyle(SplashPalette.faint)
                .lineLimit(1)
            Text(value.map { String(format: "%.0f", $0) } ?? "—")
                .font(.system(size: 12, weight: .bold, design: .rounded))
                .monospacedDigit()
                .foregroundStyle(tint)
                .lineLimit(1)
        }
        .frame(maxWidth: .infinity)
    }

    private func powerLabel(_ title: String, systemImage: String, colors: [Color]) -> some View {
        HStack(spacing: 7) {
            Image(systemName: systemImage)
                .font(.system(size: 10, weight: .bold))
            Text(title)
                .font(.system(size: 12, weight: .semibold))
            Spacer(minLength: 4)
            Image(systemName: "arrow.right")
                .font(.system(size: 9, weight: .bold))
                .opacity(0.65)
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 12)
        .padding(.vertical, 9)
        .frame(maxWidth: .infinity)
        .background(
            LinearGradient(colors: colors, startPoint: .topLeading, endPoint: .bottomTrailing),
            in: RoundedRectangle(cornerRadius: 10, style: .continuous)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(.white.opacity(0.2), lineWidth: 0.8)
        )
        .contentShape(Rectangle())
    }

    private func smallButton(_ systemImage: String, title: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Label(title, systemImage: systemImage)
                .labelStyle(.iconOnly)
                .font(.system(size: 10.5, weight: .semibold))
                .foregroundStyle(SplashPalette.muted)
                .frame(maxWidth: .infinity)
                .frame(height: 25)
                .background(SplashPalette.surface.opacity(0.45), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .strokeBorder(SplashPalette.border.opacity(0.45), lineWidth: 0.7)
                )
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(title)
    }

    private var indicator: Color {
        switch model.phase {
        case .ready: return SplashPalette.mint
        case .starting, .stopping: return SplashPalette.amber
        case .failed: return SplashPalette.rose
        case .idle: return SplashPalette.muted
        }
    }
}
