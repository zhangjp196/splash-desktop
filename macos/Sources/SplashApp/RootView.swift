import SwiftUI

/// The window: a header that always shows the state and the actions, and the
/// views the app has. Live leads because the running numbers are what the
/// window is opened for, then control, then the log. The chat is the server's
/// own web page, opened in the browser.
struct RootView: View {
    @EnvironmentObject private var model: AppModel
    @State private var tab = Tab.live

    enum Tab: Hashable, CaseIterable {
        case live, control, log

        var title: String {
            switch self {
            case .live: return L10n.string("view.live")
            case .control: return L10n.string("view.control")
            case .log: return L10n.string("view.log")
            }
        }

        var systemImage: String {
            switch self {
            case .live: return "chart.line.uptrend.xyaxis"
            case .control: return "slider.horizontal.3"
            case .log: return "terminal"
            }
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            Header()
            Divider()
            TabBar(selection: $tab)
            Divider()
            switch tab {
            case .live: LivePane()
            case .control: ControlPanelView()
            case .log: LogPane()
            }
        }
        .background(.background)
    }
}

/// The navigation band under the header: the three views, full width, so the
/// window's shape is obvious and the header keeps only state and actions.
private struct TabBar: View {
    @Binding var selection: RootView.Tab

    var body: some View {
        Picker("", selection: $selection) {
            ForEach(RootView.Tab.allCases, id: \.self) { tab in
                Label(tab.title, systemImage: tab.systemImage).tag(tab)
            }
        }
        .labelStyle(.titleAndIcon)
        .pickerStyle(.segmented)
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.bar)
    }
}

/// The brand row: a gradient mark, the running state as a pill, and the
/// primary actions, on a bar of the window's own material. It carries no
/// navigation, so the tab band below it is the only place the window changes
/// what it shows.
private struct Header: View {
    @EnvironmentObject private var model: AppModel

    private let brandGradient = LinearGradient(
        colors: [
            Color(red: 0.19, green: 0.42, blue: 0.97),
            Color(red: 0.09, green: 0.78, blue: 0.76),
        ],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "drop.fill")
                .font(.system(size: 15, weight: .bold))
                .foregroundStyle(.white)
                .frame(width: 30, height: 30)
                .background(brandGradient, in: RoundedRectangle(cornerRadius: 8))
            Text("Splash").font(.title3.weight(.semibold))

            statusPill
            Spacer()

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
            .menuStyle(.borderlessButton)
            .frame(width: 26)
            .help(L10n.string("settings.language"))

            Button {
                model.openInBrowser()
            } label: {
                Label(L10n.string("open.chat"), systemImage: "bubble.left.and.bubble.right.fill")
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(model.phase != .ready)
            .help(L10n.string("open.in.browser"))

            if model.isRunning {
                Button(L10n.string("stop"), role: .destructive) { model.stop() }
                    .controlSize(.large)
                    .disabled(model.phase == .stopping)
            } else {
                Button(L10n.string("start"), action: model.start)
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .keyboardShortcut(.return, modifiers: .command)
                    .disabled(!model.canStart)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(.bar)
    }

    private var statusPill: some View {
        HStack(spacing: 7) {
            Circle()
                .fill(indicator)
                .frame(width: 8, height: 8)
                .shadow(color: indicator.opacity(0.6), radius: 3)
            Text(model.statusText).font(.callout)
            if let tokens = model.contextTokens, model.phase == .ready {
                Text("·").foregroundStyle(.tertiary)
                Text(L10n.format("context.tokens", tokens.formatted()))
                    .font(.callout).foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 11)
        .padding(.vertical, 5)
        .background(.quaternary.opacity(0.4), in: Capsule())
        .animation(.default, value: model.phase)
    }

    private var indicator: Color {
        switch model.phase {
        case .ready: return .green
        case .starting, .stopping: return .orange
        case .failed: return .red
        case .idle: return .secondary
        }
    }

    private func languageLabel(_ language: AppModel.InterfaceLanguage) -> String {
        switch language {
        case .followSystem: return L10n.string("language.follow")
        case .chinese: return "中文"
        case .english: return "English"
        }
    }
}

private struct LogPane: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 8) {
                Label(L10n.string("view.log"), systemImage: "terminal")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.secondary)
                Spacer()
                Button(L10n.string("log.clear")) { model.clearLog() }
                    .buttonStyle(.link)
                    .disabled(model.log.isEmpty)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            Divider()
            ScrollViewReader { proxy in
                ScrollView {
                    Text(model.log.isEmpty ? L10n.string("log.empty") : model.log)
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(model.log.isEmpty ? .secondary : .primary)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(14)
                        .id("end")
                }
                .background(.quaternary.opacity(0.25))
                .onChange(of: model.log) {
                    proxy.scrollTo("end", anchor: .bottom)
                }
            }
        }
    }
}