import AppKit
import SwiftUI

/// The menu bar controls: start and stop without opening the window.
struct MenuBarView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        Text(verbatim: model.statusText)
        if let tokens = model.contextTokens, model.phase == .ready {
            Text(verbatim: L10n.format("context.tokens", tokens.formatted()))
        }
        Divider()
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

/// Stops the tracked server when the app quits: a window, the menu bar item
/// and this delegate all share AppModel.shared.
final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationWillTerminate(_ notification: Notification) {
        if Thread.isMainThread {
            AppModel.shared.stopOnQuit()
        } else {
            DispatchQueue.main.sync {
                AppModel.shared.stopOnQuit()
            }
        }
    }
}

struct SplashApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var model = AppModel.shared

    var body: some Scene {
        WindowGroup("Splash") {
            RootView()
                .environmentObject(model)
                .frame(minWidth: 900, minHeight: 660)
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
