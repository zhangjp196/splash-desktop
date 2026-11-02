import AppKit
import SwiftUI

/// The window: a sidebar that switches the pane, the pane itself, and the
/// conversation beside it, with a divider that drags freely. The chat is part
/// of the window rather than a browser tab, so a request and the numbers it
/// moves are a glance apart.
struct RootView: View {
    @EnvironmentObject private var model: AppModel
    @AppStorage("sidebar.selection") private var selectionRaw = Section.models.rawValue

    private var section: Section { Section(rawValue: selectionRaw) ?? .models }
    private var selection: Binding<Section> {
        Binding(
            get: { Section(rawValue: selectionRaw) ?? .models },
            set: { selectionRaw = $0.rawValue }
        )
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
            case .chat: return "bubble.left.and.bubble.right"
            case .models: return "square.stack.3d.up"
            case .live: return "chart.line.uptrend.xyaxis"
            case .thermal: return "fan.fill"
            case .log: return "terminal"
            case .about: return "info.circle"
            }
        }

    }

    var body: some View {
        // A fixed sidebar rather than a split view: navigation is the app's
        // frame, and a window with it collapsed has no way back.
        HStack(spacing: 0) {
            Sidebar(selection: selection)
                .frame(width: 224)
                .background(Color(nsColor: .controlBackgroundColor))

            Rectangle()
                .fill(Color(nsColor: .separatorColor).opacity(0.5))
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
            .toolbar { ToolbarItems() }
            .environment(\.openChat) { selectionRaw = Section.chat.rawValue }
            .onChange(of: section) { _, section in
                if section != .chat { NSApp.keyWindow?.makeFirstResponder(nil) }
            }
        }
        .frame(minWidth: 1080, idealWidth: 1320, minHeight: 720, idealHeight: 880)
    }

    @ViewBuilder private var pane: some View {
        switch section {
        case .chat: Color.clear
        case .models: ModelsPane()
        case .live: LivePane()
        case .thermal: ThermalPane()
        case .log: LogPane()
        case .about: AboutPane()
        }
    }
}

/// The sidebar: the brand and what it serves, the panes, and the server's
/// state and controls pinned beneath them.
private struct Sidebar: View {
    @EnvironmentObject private var model: AppModel
    @Binding var selection: RootView.Section

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 9) {
                BrandMark()
                VStack(alignment: .leading, spacing: 1) {
                    Text(verbatim: "Splash Desktop")
                        .font(.system(size: 13.5, weight: .semibold))
                        .lineLimit(1)
                    Text(verbatim: model.activeConfig?.displayName ?? model.live.modelID ?? L10n.string("models.none"))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 12)
            .padding(.top, 8)
            .padding(.bottom, 6)

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
                .padding(.horizontal, 8)
                .padding(.top, 2)
            }
            .scrollIndicators(.hidden)

            ServerControls()
        }
    }

    @State private var hovered: RootView.Section?

    private func row(_ section: RootView.Section) -> some View {
        let active = selection == section
        return Button {
            selection = section
        } label: {
            HStack(spacing: 9) {
                Image(systemName: section.systemImage)
                    .font(.system(size: 11.5, weight: .medium))
                    .frame(width: 17)
                    .foregroundStyle(active ? Color.accentColor : Color.secondary)
                Text(section.title)
                    .font(.system(size: 12.5, weight: active ? .semibold : .regular))
                    .foregroundStyle(active ? Color.primary : Color.primary.opacity(0.82))
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .background(
                active
                    ? Color.accentColor.opacity(0.14)
                    : (hovered == section ? Color.primary.opacity(0.05) : Color.clear),
                in: RoundedRectangle(cornerRadius: 7)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { inside in
            if inside { hovered = section }
            else if hovered == section { hovered = nil }
        }
    }

    private func groupLabel(_ key: String) -> some View {
        Text(L10n.string(key))
            .font(.system(size: 10.5, weight: .semibold))
            .foregroundStyle(.tertiary)
            .padding(.horizontal, 8)
            .padding(.top, 12)
            .padding(.bottom, 3)
    }
}

/// The brand mark: the same gradient drop the page shows in its title bar.
struct BrandMark: View {
    var body: some View {
        Image(systemName: "drop.fill")
            .font(.system(size: 11, weight: .bold))
            .foregroundStyle(.white)
            .frame(width: 24, height: 24)
            .background(
                LinearGradient(
                    colors: [
                        Color(red: 0.19, green: 0.42, blue: 0.97),
                        Color(red: 0.09, green: 0.78, blue: 0.76),
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                ),
                in: RoundedRectangle(cornerRadius: 7)
            )
            .shadow(color: Color(red: 0.19, green: 0.42, blue: 0.97).opacity(0.35), radius: 4, y: 2)
    }
}

/// The running state and the primary actions, pinned under the sidebar.
private struct ServerControls: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack(spacing: 7) {
                Circle()
                    .fill(indicator)
                    .frame(width: 7, height: 7)
                    .shadow(color: indicator.opacity(0.55), radius: 3)
                Text(model.statusText)
                    .font(.caption.weight(.medium))
                    .lineLimit(1)
                    .truncationMode(.tail)
                Spacer(minLength: 4)
                if let tokens = model.contextTokens, model.phase == .ready {
                    Text(tokens.formatted())
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.secondary)
                        .help(L10n.format("context.tokens", tokens.formatted()))
                }
            }
            .animation(.default, value: model.phase)

            if model.isRunning {
                Button(role: .destructive) {
                    model.stop()
                } label: {
                    Label(L10n.string("stop"), systemImage: "stop.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(.red)
                .disabled(model.phase == .stopping)

                HStack(spacing: 6) {
                    smallButton("safari", title: L10n.string("open.in.browser")) {
                        model.openInBrowser()
                    }
                    .disabled(model.phase != .ready)
                    smallButton("doc.on.doc", title: L10n.string("menu.copy_api")) {
                        model.copyToClipboard(model.apiBaseURL)
                    }
                }
                .frame(maxWidth: .infinity)
            } else {
                Button {
                    model.start()
                } label: {
                    Label(L10n.string("start"), systemImage: "play.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(!model.canStart)
                .keyboardShortcut(.return, modifiers: .command)
            }
        }
        .padding(10)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: Theme.cardRadius))
        .overlay(
            RoundedRectangle(cornerRadius: Theme.cardRadius)
                .strokeBorder(.separator.opacity(0.45))
        )
        .padding(10)
    }

    private func smallButton(_ systemImage: String, title: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Label(title, systemImage: systemImage)
                .labelStyle(.iconOnly)
                .font(.system(size: 11, weight: .medium))
                .frame(width: 52, height: 22)
                .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 6))
                .contentShape(Rectangle())
        }
        .buttonStyle(.borderless)
        .help(title)
    }

    private var indicator: Color {
        switch model.phase {
        case .ready: return .green
        case .starting, .stopping: return .orange
        case .failed: return .red
        case .idle: return .secondary
        }
    }
}

/// The detail's toolbar: the interface language, and the way to a browser tab.
private struct ToolbarItems: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
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
    }

    private func languageLabel(_ language: AppModel.InterfaceLanguage) -> String {
        switch language {
        case .followSystem: return L10n.string("language.follow")
        case .chinese: return "中文"
        case .english: return "English"
        }
    }
}

/// The server's log, in the window rather than a terminal: monospaced, live,
/// and scrolled to the end as it grows.
private struct LogPane: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(spacing: 0) {
            PaneHeader(title: L10n.string("view.log"), subtitle: logSubtitle) {
                Button {
                    model.clearLog()
                } label: {
                    Label(L10n.string("log.clear"), systemImage: "trash")
                }
                .buttonStyle(.borderless)
                .disabled(model.log.isEmpty)
            }
            .padding(.horizontal, Theme.pagePadding)
            .padding(.vertical, 14)

            Divider()

            ScrollViewReader { proxy in
                ScrollView {
                    Text(model.log.isEmpty ? L10n.string("log.empty") : model.log)
                        .font(.system(size: 11.5, design: .monospaced))
                        .foregroundStyle(model.log.isEmpty ? .secondary : .primary)
                        .textSelection(.enabled)
                        .lineSpacing(2)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 18)
                        .padding(.vertical, 14)
                        .id("end")
                }
                .background(Color(nsColor: .textBackgroundColor))
                .onChange(of: model.log) {
                    proxy.scrollTo("end", anchor: .bottom)
                }
            }
        }
        .background(.background)
    }

    private var logSubtitle: String? {
        guard model.isRunning else { return nil }
        return model.log.isEmpty ? nil : L10n.string("log.live")
    }
}
