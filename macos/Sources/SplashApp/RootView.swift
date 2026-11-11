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
    @State private var showTerminal = false
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
        case chat, models, live, thermal, log, settings, about

        var id: String { rawValue }

        var title: String {
            switch self {
            case .chat: return L10n.string("view.chat")
            case .models: return L10n.string("view.models")
            case .live: return L10n.string("view.live")
            case .thermal: return L10n.string("view.thermal")
            case .log: return L10n.string("view.log")
            case .settings: return L10n.string("view.settings")
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
            case .settings: return "gearshape.fill"
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
            case .settings: return SplashPalette.cyan
            case .about: return SplashPalette.muted
            }
        }

        /// ⌘1 through ⌘7, in the sidebar's order, for moving without the mouse.
        var shortcut: KeyEquivalent {
            switch self {
            case .chat: return "1"
            case .models: return "2"
            case .live: return "3"
            case .thermal: return "4"
            case .log: return "5"
            case .settings: return "6"
            case .about: return "7"
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
                .sheet(isPresented: $showTerminal) { TerminalDrawer() }
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
        .frame(minWidth: 810, idealWidth: 990, minHeight: 540, idealHeight: 660)
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
            case .settings: GlobalSettingsPane()
            case .about: AboutPane()
            }
        }
        .id(section)
        .transition(.opacity)
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

            Button {
                showTerminal = true
            } label: {
                Image(systemName: "terminal")
            }
            .help(L10n.string("terminal.title"))
            .keyboardShortcut("j", modifiers: .command)

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
                            Label(language.title, systemImage: "checkmark")
                        } else {
                            Text(language.title)
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
        .background(SplashPalette.surface.opacity(0.85), in: Capsule())
        .overlay(
            Capsule().strokeBorder(SplashPalette.border.opacity(0.55), lineWidth: 0.8)
        )
        .shadow(color: .black.opacity(0.22), radius: 12, y: 6)
    }
}
