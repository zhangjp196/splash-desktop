import AppKit
import SwiftUI

/// The menu bar controls: a small live dashboard, start and stop without
/// opening the window, and a way back to the window after it is closed.
struct MenuBarView: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        VStack(alignment: .leading, spacing: 11) {
            HStack(spacing: 8) {
                BrandMark(size: 24)
                VStack(alignment: .leading, spacing: 1) {
                    Text(verbatim: "Splash")
                        .font(.system(size: 13, weight: .bold, design: .rounded))
                    Text(model.statusText)
                        .font(.system(size: 9))
                        .foregroundStyle(SplashPalette.faint)
                        .lineLimit(1)
                }
                Spacer(minLength: 4)
                StatusBeacon(color: indicator, active: model.isRunning)
                Text(model.stage.title.uppercased())
                    .font(.system(size: 8, weight: .bold))
                    .tracking(0.7)
                    .foregroundStyle(stageColor(model.stage))
            }

            if model.phase == .ready {
                HStack(spacing: 8) {
                    metric(
                        L10n.string("live.decode"),
                        model.live.decodeTokensPerSecond.map { String(format: "%.0f", $0) } ?? "—",
                        L10n.string("live.caption.tokens_per_second"),
                        SplashPalette.mint
                    )
                    metric(
                        L10n.string("live.prefill"),
                        model.live.prefillTokensPerSecond.map { String(format: "%.0f", $0) } ?? "—",
                        L10n.string("live.caption.tokens_per_second"),
                        SplashPalette.cyan
                    )
                    metric(
                        L10n.string("live.pending"),
                        "\(model.live.pending)",
                        L10n.string("live.caption.queue_short"),
                        SplashPalette.amber
                    )
                }

                Sparkline(values: model.decodeHistory, tint: SplashPalette.mint)
                    .frame(height: 26)

                HStack(spacing: 5) {
                    Circle()
                        .fill(memoryTint)
                        .frame(width: 5, height: 5)
                    Text(L10n.format(
                        "menu.memory",
                        bytes(model.live.physicalBytes),
                        bytes(model.live.configuredMemoryLimitBytes ?? model.live.governorLimitBytes)
                    ))
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(SplashPalette.faint)
                    Spacer(minLength: 0)
                    if let uptime = model.uptimeText {
                        Text(L10n.format("sidebar.uptime", uptime))
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundStyle(SplashPalette.faint)
                    }
                }
            }

            Divider()

            Button {
                show(.live)
            } label: {
                menuRow(L10n.string("menu.dashboard"), "waveform.path.ecg", SplashPalette.mint)
            }
            .buttonStyle(.plain)

            if model.isRunning {
                Button {
                    model.openInBrowser()
                } label: {
                    menuRow(L10n.string("menu.open"), "safari.fill", SplashPalette.cyan)
                }
                .buttonStyle(.plain)
                Button {
                    model.copyToClipboard(model.apiBaseURL)
                } label: {
                    menuRow(L10n.string("menu.copy_api"), "doc.on.doc.fill", SplashPalette.blue)
                }
                .buttonStyle(.plain)
                Button {
                    model.stop()
                } label: {
                    menuRow(L10n.string("menu.stop"), "stop.fill", SplashPalette.rose)
                }
                .buttonStyle(.plain)
            } else {
                Button {
                    model.start()
                } label: {
                    menuRow(L10n.string("menu.start"), "play.fill", SplashPalette.mint)
                }
                .buttonStyle(.plain)
                .disabled(!model.canStart)
            }

            Divider()

            HStack {
                Button(L10n.string("menu.show")) {
                    show(nil)
                }
                .buttonStyle(.link)
                Spacer()
                Button(L10n.string("menu.quit")) {
                    NSApplication.shared.terminate(nil)
                }
                .buttonStyle(.link)
                .keyboardShortcut("q")
            }
            .font(.system(size: 11))
        }
        .padding(13)
        .frame(width: 302)
    }

    private func show(_ section: RootView.Section?) {
        if let section {
            UserDefaults.standard.set(section.rawValue, forKey: "sidebar.selection")
        }
        NSApp.activate(ignoringOtherApps: true)
        openWindow(id: "main")
    }

    private func metric(_ title: String, _ value: String, _ unit: String, _ tint: Color) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(title.uppercased())
                .font(.system(size: 7.5, weight: .bold))
                .tracking(0.6)
                .foregroundStyle(SplashPalette.faint)
                .lineLimit(1)
            HStack(alignment: .firstTextBaseline, spacing: 3) {
                Text(value)
                    .font(.system(size: 17, weight: .bold, design: .rounded))
                    .monospacedDigit()
                    .foregroundStyle(tint)
                    .contentTransition(.numericText())
                Text(unit)
                    .font(.system(size: 7.5))
                    .foregroundStyle(SplashPalette.faint)
                    .lineLimit(1)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(8)
        .background(tint.opacity(0.08), in: RoundedRectangle(cornerRadius: 9, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 9, style: .continuous)
                .strokeBorder(tint.opacity(0.18), lineWidth: 0.7)
        )
    }

    private func menuRow(_ title: String, _ icon: String, _ tint: Color) -> some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .font(.system(size: 10, weight: .bold))
                .foregroundStyle(tint)
                .frame(width: 22, height: 20)
                .background(tint.opacity(0.12), in: RoundedRectangle(cornerRadius: 6, style: .continuous))
            Text(title)
                .font(.system(size: 11.5, weight: .medium))
                .foregroundStyle(.primary)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 4)
        .contentShape(Rectangle())
    }

    private var indicator: Color {
        switch model.phase {
        case .ready: return SplashPalette.mint
        case .starting, .stopping: return SplashPalette.amber
        case .failed: return SplashPalette.rose
        case .idle: return SplashPalette.muted
        }
    }

    private var memoryTint: Color {
        guard let used = model.live.physicalBytes,
              let limit = model.live.configuredMemoryLimitBytes ?? model.live.governorLimitBytes,
              limit > 0 else { return SplashPalette.muted }
        let share = Double(used) / Double(limit)
        return share > 0.85 ? SplashPalette.rose : SplashPalette.blue
    }

    private func bytes(_ value: UInt64?) -> String {
        guard let value else { return "—" }
        return ByteCountFormatter.string(fromByteCount: Int64(value), countStyle: .memory)
    }
}

/// Keeps the app, and a running server, alive when its window closes; stops
/// the tracked server when the app quits; and hints once that closing the
/// window leaves the server running in the menu bar. A window, the menu bar
/// item and this delegate all share AppModel.shared.
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var quitting = false
    private static let hintShown = "splash.background-hint-shown"

    func applicationDidFinishLaunching(_ notification: Notification) {
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(windowWillClose(_:)),
            name: NSWindow.willCloseNotification,
            object: nil
        )
    }

    /// The menu bar item keeps the app alive after its last window closes.
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
    }

    /// Clicking the dock icon after the window closed brings it back.
    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        true
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        quitting = true
        return .terminateNow
    }

    func applicationWillTerminate(_ notification: Notification) {
        if Thread.isMainThread {
            AppModel.shared.stopOnQuit()
        } else {
            DispatchQueue.main.sync {
                AppModel.shared.stopOnQuit()
            }
        }
    }

    /// The window's own close, not the menu bar panel's: only a window that
    /// can become main is the app's, and only a running server is worth a hint.
    @MainActor
    @objc private func windowWillClose(_ note: Notification) {
        guard let window = note.object as? NSWindow, window.canBecomeMain else { return }
        guard !quitting, AppModel.shared.isRunning else { return }
        guard !UserDefaults.standard.bool(forKey: Self.hintShown) else { return }
        UserDefaults.standard.set(true, forKey: Self.hintShown)
        DispatchQueue.main.async {
            let alert = NSAlert()
            alert.alertStyle = .informational
            alert.messageText = L10n.string("background.alert.title")
            alert.informativeText = L10n.string("background.alert.message")
            alert.addButton(withTitle: L10n.string("background.alert.ok"))
            alert.runModal()
        }
    }
}

struct SplashApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var model = AppModel.shared

    var body: some Scene {
        WindowGroup("Splash Desktop", id: "main") {
            RootView()
                .environmentObject(model)
                .frame(minWidth: 1080, minHeight: 720)
        }
        .defaultSize(width: 1320, height: 880)
        .windowResizability(.contentMinSize)
        .commands {
            CommandGroup(replacing: .newItem) {}
            // The sidebar is the app's navigation and stays put; the standard
            // Hide/Show Sidebar commands would leave an empty window.
            CommandGroup(replacing: .sidebar) {}
            CommandGroup(after: .sidebar) {
                Divider()
                Button(L10n.string("palette.shortcut")) {
                    NotificationCenter.default.post(name: .showCommandPalette, object: nil)
                }
                .keyboardShortcut("k", modifiers: .command)
            }
        }

        MenuBarExtra {
            MenuBarView().environmentObject(model)
        } label: {
            Image(systemName: model.menuBarSymbol)
        }
    }
}
