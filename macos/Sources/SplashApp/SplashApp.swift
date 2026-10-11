import AppKit
import SwiftUI

/// The menu bar controls: start and stop without opening the window, and a way
/// back to the window after it is closed.
struct MenuBarView: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        Text(verbatim: model.statusText)
        if let tokens = model.contextTokens, model.phase == .ready {
            Text(verbatim: L10n.format("context.tokens", tokens.formatted()))
        }
        Divider()
        Button(L10n.string("menu.show")) {
            NSApp.activate(ignoringOtherApps: true)
            openWindow(id: "main")
        }
        if model.isRunning {
            Button(L10n.string("menu.stop")) { model.stop() }
            Button(L10n.string("menu.open")) { model.openInBrowser() }
            Button(L10n.string("menu.copy_api")) { model.copyToClipboard(model.apiBaseURL) }
        } else {
            Button(L10n.string("menu.start")) { model.start() }
                .disabled(!model.canStart)
        }
        Divider()
        Button(L10n.string("menu.quit")) { NSApplication.shared.terminate(nil) }
            .keyboardShortcut("q")
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
        WindowGroup("Splash", id: "main") {
            RootView()
                .environmentObject(model)
                .frame(minWidth: 860, minHeight: 620)
        }
        .commands {
            CommandGroup(replacing: .newItem) {}
        }

        MenuBarExtra {
            MenuBarView().environmentObject(model)
        } label: {
            Image(systemName: model.menuBarSymbol)
        }
    }
}
