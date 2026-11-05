import AppKit
import SwiftUI

/// The window: a sidebar that switches the pane, the pane itself, and the
/// conversation beside it, with a divider that drags freely. The chat is part
/// of the window rather than a browser tab, so a request and the numbers it
/// moves are a glance apart.
struct RootView: View {
    @EnvironmentObject private var model: AppModel
    @AppStorage("sidebar.selection") private var selectionRaw = Section.models.rawValue
    @AppStorage("appearance") private var appearanceRaw = AppAppearance.dark.rawValue
    @State private var showPalette = false
    @State private var toast: String?
    @State private var toastToken = 0

    private var section: Section { Section(rawValue: selectionRaw) ?? .models }
    private var selection: Binding<Section> {
        Binding(
            get: { Section(rawValue: selectionRaw) ?? .models },
            set: { selectionRaw = $0.rawValue }
        )
    }
    private var appearance: AppAppearance {
        AppAppearance(rawValue: appearanceRaw) ?? .dark
    }

    enum Section: String, CaseIterable, Identifiable {
        case chat, models, live, thermal, log, about

        var id: String { rawValue }

        var title: String {
            switch self {
            case .chat: return L10n.string("view.chat")
            case .models: return L10n.string("view.models")
            case .live: return L10n.string("view.live")
            case .thermal: return L10n.string("view.thermal")
            case .log: return L10n.string("view.log")
            case .about: return L10n.string("view.about")
            }
        }

        var systemImage: String {
            switch self {
            case .chat: return "bubble.left.and.bubble.right.fill"
            case .models: return "square.stack.3d.up.fill"
            case .live: return "waveform.path.ecg"
            case .thermal: return "fan.fill"
            case .log: return "terminal.fill"
            case .about: return "info.circle.fill"
            }
        }

        var tint: Color {
            switch self {
            case .chat: return SplashPalette.cyan
            case .models: return SplashPalette.blue
            case .live: return SplashPalette.mint
            case .thermal: return SplashPalette.amber
            case .log: return SplashPalette.violet
            case .about: return SplashPalette.muted
            }
        }

        /// ⌘1 through ⌘6, in the sidebar's order, for moving without the mouse.
        var shortcut: KeyEquivalent {
            switch self {
            case .chat: return "1"
            case .models: return "2"
            case .live: return "3"
            case .thermal: return "4"
            case .log: return "5"
            case .about: return "6"
            }
        }
    }

    var body: some View {
        // A fixed sidebar rather than a split view: navigation is the app's
        // frame, and a window with it collapsed has no way back.
        ZStack {
            AuroraBackground(
                tint: section == .live && model.phase == .ready
                    ? stageColor(model.stage)
                    : section.tint
            )

            HStack(spacing: 0) {
                Sidebar(selection: selection, showPalette: $showPalette)
                    .frame(width: Theme.sidebarWidth)

                Rectangle()
                    .fill(SplashPalette.border.opacity(0.35))
                    .frame(width: 1)

                ZStack {
                    pane
                    // The conversation stays mounted whichever tab is open, so a
                    // reply that is streaming is still streaming on return.
                    ChatPane()
                        .opacity(section == .chat ? 1 : 0)
                        .allowsHitTesting(section == .chat)
                        .accessibilityHidden(section != .chat)
                        .zIndex(section == .chat ? 1 : 0)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .toolbar { toolbar }
                .toolbarBackground(SplashPalette.canvas, for: .windowToolbar)
                .environment(\.openChat) { selectionRaw = Section.chat.rawValue }
                .animation(.easeOut(duration: 0.22), value: selectionRaw)
                .onChange(of: section) { _, section in
                    if section != .chat { NSApp.keyWindow?.makeFirstResponder(nil) }
                }
            }

            if showPalette {
                CommandPalette(actions: paletteActions, isPresented: $showPalette)
                    .transition(.opacity.combined(with: .scale(scale: 0.98, anchor: .top)))
                    .zIndex(20)
            }

            if let toast {
                VStack {
                    Spacer()
                    ToastBanner(text: toast)
                        .padding(.bottom, 24)
                }
                .allowsHitTesting(false)
                .transition(.move(edge: .bottom).combined(with: .opacity))
                .zIndex(30)
            }
        }
        .preferredColorScheme(appearance.colorScheme)
        .onAppear { applyAppearance() }
        .onChange(of: appearanceRaw) { _, _ in applyAppearance() }
        .onReceive(NotificationCenter.default.publisher(for: .showCommandPalette)) { _ in
            showPalette = true
        }
        .onReceive(NotificationCenter.default.publisher(for: .splashToast)) { note in
            showToast(note.object as? String)
        }
        .animation(.easeOut(duration: 0.16), value: showPalette)
        .frame(minWidth: 1080, idealWidth: 1360, minHeight: 720, idealHeight: 900)
    }

    /// Give the whole app the window's chosen appearance, not only the SwiftUI
    /// content: the title bar, the menus and the embedded chat follow too.
    private func applyAppearance() {
        switch appearance {
        case .system:
            NSApp.appearance = nil
        case .dark:
            NSApp.appearance = NSAppearance(named: .darkAqua)
        case .light:
            NSApp.appearance = NSAppearance(named: .aqua)
        }
    }

    private func showToast(_ text: String?) {
        guard let text, !text.isEmpty else { return }
        toastToken += 1
        let token = toastToken
        withAnimation(.spring(response: 0.36, dampingFraction: 0.84)) {
            toast = text
        }
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(2.2))
            guard token == toastToken else { return }
            withAnimation(.easeOut(duration: 0.2)) {
                toast = nil
            }
        }
    }

    @ViewBuilder private var pane: some View {
        Group {
            switch section {
            case .chat: Color.clear
            case .models: ModelsPane()
            case .live: LivePane()
            case .thermal: ThermalPane()
            case .log: LogPane()
            case .about: AboutPane()
            }
        }
        .id(section)
        .transition(.opacity.combined(with: .offset(y: 6)))
    }

    /// The detail's toolbar: the search palette, the appearance, the language,
    /// and the way to a browser tab.
    @ToolbarContentBuilder private var toolbar: some ToolbarContent {
        ToolbarItemGroup(placement: .primaryAction) {
            Button {
                showPalette = true
            } label: {
                Image(systemName: "command")
            }
            .help(L10n.string("palette.shortcut"))

            Menu {
                ForEach(AppAppearance.allCases) { option in
                    Button {
                        appearanceRaw = option.rawValue
                    } label: {
                        if option == appearance {
                            Label(option.title, systemImage: "checkmark")
                        } else {
                            Label(option.title, systemImage: option.systemImage)
                        }
                    }
                }
            } label: {
                Image(systemName: appearance.systemImage)
            }
            .help(L10n.string("appearance.title"))

            Menu {
                ForEach(AppModel.InterfaceLanguage.allCases) { language in
                    Button {
                        model.language = language
                        model.applyLanguage()
                    } label: {
                        if model.language == language {
                            Label(languageLabel(language), systemImage: "checkmark")
                        } else {
                            Text(languageLabel(language))
                        }
                    }
                }
            } label: {
                Image(systemName: "globe")
            }
            .help(L10n.string("settings.language"))

            Button {
                model.openInBrowser()
            } label: {
                Image(systemName: "safari")
            }
            .disabled(model.phase != .ready)
            .help(L10n.string("open.in.browser"))
        }
    }

    private func languageLabel(_ language: AppModel.InterfaceLanguage) -> String {
        switch language {
        case .followSystem: return L10n.string("language.follow")
        case .chinese: return "中文"
        case .english: return "English"
        }
    }

    /// Everything the palette can run from one place, so a new section or
    /// server action is one entry there too.
    private var paletteActions: [PaletteAction] {
        var actions = Section.allCases.map { item in
            PaletteAction(
                id: "goto." + item.rawValue,
                title: item.title,
                subtitle: L10n.string("palette.goto"),
                systemImage: item.systemImage,
                tint: item.tint
            ) {
                selectionRaw = item.rawValue
            }
        }
        if model.isRunning {
            actions.append(
                PaletteAction(
                    id: "server.stop",
                    title: L10n.string("menu.stop"),
                    subtitle: L10n.string("palette.server"),
                    systemImage: "stop.fill",
                    tint: SplashPalette.rose
                ) { model.stop() }
            )
            actions.append(
                PaletteAction(
                    id: "server.copy",
                    title: L10n.string("menu.copy_api"),
                    subtitle: model.apiBaseURL,
                    systemImage: "doc.on.doc.fill",
                    tint: SplashPalette.blue
                ) { model.copyToClipboard(model.apiBaseURL) }
            )
            actions.append(
                PaletteAction(
                    id: "server.open",
                    title: L10n.string("menu.open"),
                    subtitle: model.apiBaseURL,
                    systemImage: "safari.fill",
                    tint: SplashPalette.cyan
                ) { model.openInBrowser() }
            )
        } else {
            actions.append(
                PaletteAction(
                    id: "server.start",
                    title: L10n.string("menu.start"),
                    subtitle: L10n.string("palette.server"),
                    systemImage: "play.fill",
                    tint: SplashPalette.mint
                ) { model.start() }
            )
        }
        actions.append(
            PaletteAction(
                id: "appearance",
                title: L10n.string("appearance.cycle"),
                subtitle: L10n.string("palette.appearance"),
                systemImage: appearance.systemImage,
                tint: SplashPalette.violet
            ) {
                let all = AppAppearance.allCases
                let next = all[(all.firstIndex(of: appearance).map { $0 + 1 } ?? 0) % all.count]
                appearanceRaw = next.rawValue
            }
        )
        return actions
    }
}

/// The sidebar: the brand and what it serves, the panes, and the server's
/// state and controls pinned beneath them.
private struct Sidebar: View {
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
            ZStack {
                Rectangle().fill(.ultraThinMaterial)
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
                    .shadow(color: active ? section.tint.opacity(0.7) : .clear, radius: 5)
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
                        .shadow(color: section.tint, radius: 4)
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
                        .shadow(color: section.tint, radius: 5)
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

/// The brand mark: the same gradient drop the page shows in its title bar.
struct BrandMark: View {
    var size: CGFloat = 24

    var body: some View {
        Image(systemName: "drop.fill")
            .font(.system(size: size * 0.46, weight: .bold))
            .foregroundStyle(.white)
            .frame(width: size, height: size)
            .background(
                LinearGradient(
                    colors: [
                        SplashPalette.blue,
                        SplashPalette.cyan,
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                ),
                in: RoundedRectangle(cornerRadius: size * 0.29, style: .continuous)
            )
            .overlay(
                RoundedRectangle(cornerRadius: size * 0.29, style: .continuous)
                    .strokeBorder(.white.opacity(0.25), lineWidth: 0.8)
            )
            .shadow(color: SplashPalette.blue.opacity(0.45), radius: 8, y: 4)
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
                        if let uptime = model.uptimeText {
                            Text(L10n.format("sidebar.uptime", uptime))
                        } else {
                            Text(model.phase == .idle ? L10n.string("sidebar.offline") : L10n.string("sidebar.working"))
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
                .contentTransition(.numericText())
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
        .shadow(color: colors.first?.opacity(0.45) ?? .clear, radius: 12, y: 5)
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

/// The server's log, in the window rather than a terminal: monospaced, live,
/// and scrolled to the end as it grows. The console keeps its own dark glass
/// whatever the window appearance, so long output is easy on the eyes.
private struct LogPane: View {
    @EnvironmentObject private var model: AppModel
    @State private var hovering = false

    var body: some View {
        VStack(spacing: 0) {
            PaneHeader(title: L10n.string("view.log"), subtitle: logSubtitle, eyebrow: L10n.string("log.eyebrow")) {
                HStack(spacing: 8) {
                    if model.isRunning {
                        HStack(spacing: 5) {
                            StatusBeacon(color: model.phase == .ready ? SplashPalette.mint : SplashPalette.amber, active: true)
                                .scaleEffect(0.75)
                            Text(L10n.string("log.live").uppercased())
                                .font(.system(size: 9, weight: .bold))
                                .tracking(1)
                                .foregroundStyle(SplashPalette.mint)
                        }
                        .padding(.horizontal, 9)
                        .padding(.vertical, 5)
                        .background(SplashPalette.mint.opacity(0.10), in: Capsule())
                    }
                    Button {
                        model.clearLog()
                    } label: {
                        Label(L10n.string("log.clear"), systemImage: "trash")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .buttonStyle(.borderless)
                    .disabled(model.log.isEmpty)
                }
            }
            .padding(.horizontal, Theme.pagePadding)
            .padding(.top, Theme.pagePadding)
            .padding(.bottom, 16)

            console
                .padding(.horizontal, Theme.pagePadding)
                .padding(.bottom, Theme.pagePadding)
        }
        .background(Color.clear)
    }

    private var console: some View {
        VStack(spacing: 0) {
            HStack(spacing: 6) {
                ForEach([Color(red: 1, green: 0.37, blue: 0.34), Color(red: 1, green: 0.74, blue: 0.18), Color(red: 0.16, green: 0.79, blue: 0.25)], id: \.self) { color in
                    Circle().fill(color.opacity(0.85)).frame(width: 7, height: 7)
                }
                Spacer()
                Text(verbatim: "splash — live output")
                    .font(.system(size: 9, weight: .medium, design: .monospaced))
                    .foregroundStyle(Color.white.opacity(0.32))
                Spacer()
                Text(verbatim: "\(model.log.count.formatted()) chars")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(Color.white.opacity(0.25))
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 9)
            .background(Color.white.opacity(0.025))

            Rectangle()
                .fill(Color.white.opacity(0.06))
                .frame(height: 1)

            ScrollViewReader { proxy in
                ScrollView {
                    Text(model.log.isEmpty ? L10n.string("log.empty") : model.log)
                        .font(.system(size: 11.5, design: .monospaced))
                        .foregroundStyle(model.log.isEmpty ? Color.white.opacity(0.35) : Color(red: 0.78, green: 0.94, blue: 0.88))
                        .textSelection(.enabled)
                        .lineSpacing(3)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 14)
                        .id("end")
                }
                .onChange(of: model.log) {
                    proxy.scrollTo("end", anchor: .bottom)
                }
            }
        }
        .background(
            RoundedRectangle(cornerRadius: Theme.cardRadius, style: .continuous)
                .fill(
                    LinearGradient(
                        colors: [
                            Color(red: 0.035, green: 0.052, blue: 0.075),
                            Color(red: 0.025, green: 0.035, blue: 0.055),
                        ],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
        )
        .overlay(
            RoundedRectangle(cornerRadius: Theme.cardRadius, style: .continuous)
                .strokeBorder(
                    LinearGradient(
                        colors: [Color.white.opacity(0.16), Color.white.opacity(0.03)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    ),
                    lineWidth: 0.8
                )
        )
        .clipShape(RoundedRectangle(cornerRadius: Theme.cardRadius, style: .continuous))
        .shadow(color: .black.opacity(0.35), radius: 24, y: 12)
        .scaleEffect(hovering ? 1.002 : 1)
        .animation(.easeOut(duration: 0.2), value: hovering)
        .onHover { hovering = $0 }
    }

    private var logSubtitle: String? {
        guard model.isRunning else { return nil }
        return model.log.isEmpty ? nil : L10n.string("log.live")
    }
}

/// A short confirmation that floats over the window's bottom edge: quiet,
/// self-dismissing, and out of the way of the content.
private struct ToastBanner: View {
    let text: String

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 12, weight: .bold))
                .foregroundStyle(SplashPalette.mint)
            Text(text)
                .font(.system(size: 11.5, weight: .semibold))
                .foregroundStyle(.primary)
        }
        .padding(.horizontal, 15)
        .padding(.vertical, 9)
        .background(.ultraThinMaterial, in: Capsule())
        .overlay(
            Capsule().strokeBorder(SplashPalette.border.opacity(0.55), lineWidth: 0.8)
        )
        .shadow(color: .black.opacity(0.22), radius: 18, y: 8)
    }
}
