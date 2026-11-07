import SwiftUI
import SwiftTerm

/// The terminal drawer: a login shell in the window, for the commands a
/// session needs beside the server it drives (⌘J). SwiftTerm owns the
/// pseudo-terminal and draws its output; the shell ends with the view.
struct TerminalDrawer: View {
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Label(L10n.string("terminal.title"), systemImage: "terminal")
                    .font(.system(size: 12, weight: .semibold))
                Spacer()
                Button(L10n.string("terminal.close")) { dismiss() }
                    .buttonStyle(.plain)
                    .keyboardShortcut(.cancelAction)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            Rectangle()
                .fill(SplashPalette.border.opacity(0.35))
                .frame(height: 1)
            TerminalSession()
        }
        .frame(minWidth: 720, minHeight: 420)
    }
}

private struct TerminalSession: NSViewRepresentable {
    func makeNSView(context: Context) -> LocalProcessTerminalView {
        let terminal = LocalProcessTerminalView(frame: .zero)
        terminal.processDelegate = context.coordinator
        let shell = ProcessInfo.processInfo.environment["SHELL"] ?? "/bin/zsh"
        terminal.startProcess(executable: shell, args: ["-l"], environment: nil)
        return terminal
    }

    func updateNSView(_ terminal: LocalProcessTerminalView, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator: NSObject, LocalProcessTerminalViewDelegate {
        func processTerminated(source: TerminalView, exitCode: Int32?) {
            // The shell ended: keep the last output visible.
        }

        func sizeChanged(source: LocalProcessTerminalView, newCols: Int, newRows: Int) {}

        func setTerminalTitle(source: LocalProcessTerminalView, title: String) {}

        func hostCurrentDirectoryUpdate(source: TerminalView, directory: String?) {}
    }
}
