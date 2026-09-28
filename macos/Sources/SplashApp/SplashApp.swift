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
        } else {
            Button(L10n.string("menu.start")) { model.start() }
                .disabled(!model.canStart)
        }
        Divider()
        Button(L10n.string("menu.quit")) { NSApplication.shared.terminate(nil) }
            .keyboardShortcut("q")
    }
}

struct SplashApp: App {
    @StateObject private var model = AppModel()

    var body: some Scene {
        WindowGroup("Splash") {
            RootView()
                .environmentObject(model)
                .frame(minWidth: 880, minHeight: 620)
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
