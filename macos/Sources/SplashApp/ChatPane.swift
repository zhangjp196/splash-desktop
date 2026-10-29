import SwiftUI
import WebKit

/// One saved conversation as the page reports it: the id its history menu
/// opens and the title it shows.
struct ChatSummary: Identifiable, Equatable {
    let id: String
    let title: String
}

/// The state the embedded page posts back, and the calls the app makes into
/// it. The page owns the conversations and the stream; this only mirrors what
/// the native header shows and asks for an action by id.
@MainActor
final class ChatBridge: ObservableObject {
    @Published private(set) var recents: [ChatSummary] = []
    @Published private(set) var title = ""
    @Published private(set) var activeID: String?
    @Published private(set) var running = false
    @Published private(set) var canUndo = false

    weak var webView: WKWebView?

    func apply(_ payload: [String: Any]) {
        guard payload["type"] as? String == "state" else { return }
        title = payload["title"] as? String ?? ""
        activeID = payload["active"] as? String
        running = payload["running"] as? Bool ?? false
        canUndo = payload["canUndo"] as? Bool ?? false
        let rows = payload["recents"] as? [[String: Any]] ?? []
        recents = rows.compactMap { row in
            guard let id = row["id"] as? String, let title = row["title"] as? String
            else { return nil }
            return ChatSummary(id: id, title: title)
        }
    }

    func open(_ id: String) { run("window.splash && window.splash.open(\(Self.quoted(id)))") }
    func newChat() { run("window.splash && window.splash.newChat()") }
    func clearHistory() { run("window.splash && window.splash.clearHistory()") }
    func undoClear() { run("window.splash && window.splash.undoClear()") }

    private func run(_ script: String) {
        webView?.evaluateJavaScript(script, completionHandler: nil)
    }

    /// A Swift string as a JavaScript string literal, quotes and all.
    private static func quoted(_ value: String) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: [value]),
              let encoded = String(data: data, encoding: .utf8)
        else { return "\"\"" }
        return String(encoded.dropFirst().dropLast())
    }
}

/// The conversation, as its own tab: the server's own chat page in embed mode,
/// under one native band that holds the history, a new chat, and clearing it.
struct ChatPane: View {
    @EnvironmentObject private var model: AppModel
    @StateObject private var bridge = ChatBridge()
    @State private var reloadToken = 0
    @State private var undoVisible = false

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 6) {
                history
                if bridge.running {
                    ProgressView()
                        .controlSize(.mini)
                        .padding(.leading, 2)
                }
                Spacer(minLength: 8)
                iconButton("square.and.pencil", help: L10n.string("chat.new")) {
                    bridge.newChat()
                }
                .disabled(model.phase != .ready || bridge.running)
                iconButton("trash", help: L10n.string("chat.clear")) {
                    bridge.clearHistory()
                }
                .disabled(bridge.recents.isEmpty || bridge.running)
                iconButton("safari", help: L10n.string("open.in.browser")) {
                    model.openInBrowser()
                }
                .disabled(model.phase != .ready)
            }
            .padding(.horizontal, 11)
            .padding(.vertical, 7)
            .background(.bar)

            if undoVisible {
                undoBar
                    .transition(.move(edge: .top).combined(with: .opacity))
            }

            Divider()

            content
        }
        .background(.background)
        .animation(.easeOut(duration: 0.16), value: undoVisible)
        .onChange(of: model.phase) { _, phase in
            if phase == .ready { reloadToken += 1 }
        }
        .onChange(of: bridge.canUndo) { _, canUndo in
            undoVisible = canUndo
        }
        .task(id: bridge.canUndo) {
            guard bridge.canUndo else { return }
            try? await Task.sleep(for: .seconds(10))
            guard !Task.isCancelled else { return }
            undoVisible = false
        }
    }

    /// The saved conversations, newest first, and the same clear action the
    /// header's trash button offers.
    private var history: some View {
        Menu {
            if bridge.recents.isEmpty {
                Text(L10n.string("chat.no_history"))
            } else {
                ForEach(bridge.recents) { chat in
                    Button(chat.title) { bridge.open(chat.id) }
                        .disabled(bridge.running)
                }
                Divider()
                Button(L10n.string("chat.clear"), role: .destructive) {
                    bridge.clearHistory()
                }
                .disabled(bridge.running)
            }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "clock.arrow.circlepath")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.secondary)
                Text(bridge.title.isEmpty ? L10n.string("chat.new") : bridge.title)
                    .font(.system(size: 12.5, weight: .semibold))
                    .lineLimit(1)
                Image(systemName: "chevron.down")
                    .font(.system(size: 8, weight: .bold))
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 9)
            .padding(.vertical, 4)
            .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 7))
            .contentShape(Rectangle())
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .disabled(model.phase != .ready)
        .help(L10n.string("chat.history"))
    }

    private var undoBar: some View {
        HStack(spacing: 7) {
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(.green)
            Text(L10n.string("chat.cleared"))
                .lineLimit(1)
            Spacer(minLength: 8)
            Button(L10n.string("chat.undo")) {
                bridge.undoClear()
                undoVisible = false
            }
            .buttonStyle(.borderless)
            .font(.caption.weight(.semibold))
        }
        .font(.caption)
        .padding(.horizontal, Theme.pagePadding)
        .padding(.vertical, 7)
        .background(Color.green.opacity(0.09))
    }

    @ViewBuilder private var content: some View {
        if model.phase == .ready, let url = model.chatEmbedURL {
            WebChat(url: url, reloadToken: reloadToken, bridge: bridge)
        } else if model.phase == .starting || model.phase == .stopping {
            VStack(spacing: 10) {
                ProgressView()
                Text(model.statusText)
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            PaneEmpty(
                title: L10n.string("chat.empty.title"),
                message: isFailed ? model.statusText : L10n.string("chat.empty.message"),
                systemImage: "bubble.left.and.bubble.right",
                actionTitle: model.canStart ? L10n.string("start") : nil,
                action: model.canStart ? { model.start() } : nil
            )
        }
    }

    private var isFailed: Bool {
        if case .failed = model.phase { return true }
        return false
    }

    private func iconButton(_ systemImage: String, help: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.system(size: 12, weight: .medium))
                .frame(width: 24, height: 22)
                .contentShape(Rectangle())
        }
        .buttonStyle(.borderless)
        .help(help)
    }
}

/// A `WKWebView` kept out of SwiftUI's layout changes: it loads once per port
/// and reloads only when the app asks it to.
private struct WebChat: NSViewRepresentable {
    let url: URL
    let reloadToken: Int
    let bridge: ChatBridge

    func makeCoordinator() -> Coordinator { Coordinator(bridge: bridge) }

    func makeNSView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .default()
        configuration.userContentController.add(context.coordinator, name: "splash")
        let view = WKWebView(frame: .zero, configuration: configuration)
        view.navigationDelegate = context.coordinator
        view.underPageBackgroundColor = .textBackgroundColor
        view.allowsMagnification = false
        view.allowsBackForwardNavigationGestures = false
        view.setValue(false, forKey: "drawsBackground")
        bridge.webView = view
        context.coordinator.loadedURL = url
        context.coordinator.loadedToken = reloadToken
        view.load(URLRequest(url: url))
        return view
    }

    func updateNSView(_ view: WKWebView, context: Context) {
        bridge.webView = view
        guard context.coordinator.loadedURL != url
                || context.coordinator.loadedToken != reloadToken else { return }
        context.coordinator.loadedURL = url
        context.coordinator.loadedToken = reloadToken
        view.load(URLRequest(url: url))
    }

    static func dismantleNSView(_ view: WKWebView, coordinator: Coordinator) {
        view.configuration.userContentController.removeScriptMessageHandler(forName: "splash")
        view.navigationDelegate = nil
    }

    @MainActor
    final class Coordinator: NSObject, WKNavigationDelegate, WKScriptMessageHandler {
        let bridge: ChatBridge
        var loadedURL: URL?
        var loadedToken = 0
        /// The server may still be opening its port when the first request
        /// goes out; two quiet retries turn a blank pane into the page.
        var retries = 0

        init(bridge: ChatBridge) {
            self.bridge = bridge
        }

        func userContentController(
            _ controller: WKUserContentController,
            didReceive message: WKScriptMessage
        ) {
            guard let payload = message.body as? [String: Any] else { return }
            bridge.apply(payload)
        }

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            retries = 0
        }

        func webView(
            _ webView: WKWebView,
            didFailProvisionalNavigation navigation: WKNavigation!,
            withError error: Error
        ) {
            retry(webView)
        }

        func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
            retry(webView)
        }

        private func retry(_ webView: WKWebView) {
            guard retries < 2, let url = loadedURL else { return }
            retries += 1
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.7) { [weak webView] in
                webView?.load(URLRequest(url: url))
            }
        }

        /// The chat is the page; every other link belongs to the browser.
        func webView(
            _ webView: WKWebView,
            decidePolicyFor action: WKNavigationAction,
            decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
        ) {
            guard let url = action.request.url, let host = url.host()?.lowercased() else {
                decisionHandler(.allow)
                return
            }
            if host == "127.0.0.1" || host == "localhost" {
                decisionHandler(.allow)
            } else {
                NSWorkspace.shared.open(url)
                decisionHandler(.cancel)
            }
        }
    }
}
