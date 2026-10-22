import AppKit
import SwiftUI

/// The window: a sidebar that switches the pane, a detail that shows it, and
/// the server's state and controls pinned under the sidebar. The models pane
/// leads, because choosing what to serve is what the window is opened for;
/// the live numbers, the settings and the log follow.
struct RootView: View {
    @EnvironmentObject private var model: AppModel
    @State private var section: Section = .models

    enum Section: String, CaseIterable, Identifiable {
        case models, live, thermal, log, about

        var id: String { rawValue }

        var title: String {
            switch self {
            case .models: return L10n.string("view.models")
            case .live: return L10n.string("view.live")
            case .thermal: return L10n.string("view.thermal")
            case .log: return L10n.string("view.log")
            case .about: return L10n.string("view.about")
            }
        }

        var systemImage: String {
            switch self {
            case .models: return "square.stack.3d.up"
            case .live: return "chart.line.uptrend.xyaxis"
            case .thermal: return "fan.fill"
            case .log: return "terminal"
            case .about: return "info.circle"
            }
        }

        /// The panes as the sidebar's groups show them: what is served, then
        /// how it is running.
        static let serving: [Section] = [.models]
        static let monitoring: [Section] = [.live, .thermal, .log]
        static let aboutSection: [Section] = [.about]
    }

    var body: some View {
        NavigationSplitView {
            Sidebar(selection: $section)
                .navigationSplitViewColumnWidth(min: 196, ideal: 214, max: 280)
        } detail: {
            Group {
                switch section {
                case .models: ModelsPane()
                case .live: LivePane()
                case .thermal: ThermalPane()
                case .log: LogPane()
                case .about: AboutPane()
                }
            }
            .navigationTitle(section.title)
            .toolbar { ToolbarItems() }
        }
        .background(.background)
    }
}

/// The sidebar: the brand, the panes, and the server's state and controls, so
/// the app can be driven without leaving the pane you are in.
private struct Sidebar: View {
    @EnvironmentObject private var model: AppModel
    @Binding var selection: RootView.Section

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 8) {
                BrandMark()
                Text("Splash Desktop").font(.headline)
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 14)
            .padding(.top, 10)
            .padding(.bottom, 6)

            List(selection: $selection) {
                Section {
                    ForEach(RootView.Section.serving) { section in
                        Label(section.title, systemImage: section.systemImage).tag(section)
                    }
                } header: {
                    Text(L10n.string("sidebar.serve"))
                }
                Section {
                    ForEach(RootView.Section.monitoring) { section in
                        Label(section.title, systemImage: section.systemImage).tag(section)
                    }
                } header: {
                    Text(L10n.string("sidebar.monitor"))
                }
                Section {
                    ForEach(RootView.Section.aboutSection) { section in
                        Label(section.title, systemImage: section.systemImage).tag(section)
                    }
                } header: {
                    Text(L10n.string("sidebar.about"))
                }
            }
            .listStyle(.sidebar)
        }
        .safeAreaInset(edge: .bottom, spacing: 0) {
            ServerControls()
        }
    }
}

/// The brand mark, shared by the sidebar.
struct BrandMark: View {
    var body: some View {
        Image(systemName: "drop.fill")
            .font(.system(size: 12, weight: .bold))
            .foregroundStyle(.white)
            .frame(width: 22, height: 22)
            .background(
                LinearGradient(
                    colors: [
                        Color(red: 0.19, green: 0.42, blue: 0.97),
                        Color(red: 0.09, green: 0.78, blue: 0.76),
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                ),
                in: RoundedRectangle(cornerRadius: 6)
            )
    }
}

/// The running state and the primary actions, pinned under the sidebar.
private struct ServerControls: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            StatusPill()
            if model.isRunning {
                HStack(spacing: 8) {
                    Button(role: .destructive) {
                        model.stop()
                    } label: {
                        Label(L10n.string("stop"), systemImage: "stop.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .disabled(model.phase == .stopping)
                    Button {
                        model.openInBrowser()
                    } label: {
                        Label(L10n.string("open.chat"), systemImage: "bubble.left.and.bubble.right.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .disabled(model.phase != .ready)
                }
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
        .padding(12)
        .background(.bar)
    }
}

/// A capsule with the run state and the served context.
struct StatusPill: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        HStack(spacing: 7) {
            Circle()
                .fill(indicator)
                .frame(width: 8, height: 8)
                .shadow(color: indicator.opacity(0.6), radius: 3)
            Text(model.statusText).font(.callout).lineLimit(1)
            Spacer(minLength: 0)
            if let tokens = model.contextTokens, model.phase == .ready {
                Text(L10n.format("context.tokens", tokens.formatted()))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 11)
        .padding(.vertical, 6)
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
}

/// The detail's toolbar: the interface language and the way to the chat.
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

        Button {
            model.openInBrowser()
        } label: {
            Label(L10n.string("open.chat"), systemImage: "bubble.left.and.bubble.right.fill")
        }
        .disabled(model.phase != .ready)
        .help(L10n.string("open.in.browser"))
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
