import SwiftUI

/// The control panel: settings before serving, a live status panel and the
/// log once it has. Conversations live in the server's own web page, opened
/// in the browser.
struct RootView: View {
    @EnvironmentObject private var model: AppModel
    @State private var tab = Tab.control

    enum Tab: Hashable { case control, live, log }

    var body: some View {
        VStack(spacing: 0) {
            Header()
            Divider()
            switch tab {
            case .control: ControlPanelView()
            case .live: LivePane()
            case .log: LogPane()
            }
        }
        .background(.background)
        .toolbar {
            ToolbarItem(placement: .principal) {
                Picker("", selection: $tab) {
                    Label(L10n.string("view.control"), systemImage: "slider.horizontal.3")
                        .tag(Tab.control)
                    Label(L10n.string("view.live"), systemImage: "chart.line.uptrend.xyaxis")
                        .tag(Tab.live)
                    Label(L10n.string("view.log"), systemImage: "terminal")
                        .tag(Tab.log)
                }
                .labelStyle(.titleAndIcon)
                .pickerStyle(.segmented)
                .frame(width: 360)
            }
        }
    }
}

/// The brand row: a gradient mark, the running state as a pill, and the
/// primary actions, on the window's toolbar material.
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