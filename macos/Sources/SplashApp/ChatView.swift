import AppKit
import SwiftUI

/// The built-in chat panel: messages are streamed straight from the server's
/// OpenAI-compatible `/v1/chat/completions`, so the window no longer opens a
/// browser page. The model and its API address are shown up top, with copy
/// actions for the surface other tools would need to talk to the server.
struct ChatView: View {
    @EnvironmentObject private var model: AppModel
    @State private var copiedLabel: String?
    @FocusState private var inputFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            toolbar
            Divider()
            if !model.isRunning {
                ContentUnavailableView(
                    L10n.string("chat.empty.title"),
                    systemImage: "bubble.left.and.bubble.right",
                    description: Text(verbatim: L10n.string("chat.empty.description"))
                )
                .padding(.top, 80)
            } else {
                messageList
                inputBar
            }
        }
        .background(.background)
    }

    private var toolbar: some View {
        HStack(spacing: 8) {
            Label(model.servedModelID.isEmpty ? "—" : model.servedModelID, systemImage: "cube")
                .font(.subheadline.weight(.semibold))
                .lineLimit(1)
                .truncationMode(.middle)
            Text(verbatim: model.apiBaseURL)
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer()
            if let copiedLabel {
                Label(copiedLabel, systemImage: "checkmark.circle.fill")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.green)
                    .transition(.opacity)
            }
            Button {
                copy(L10n.string("copy.api"), model.apiBaseURL)
            } label: {
                Label(L10n.string("copy.api"), systemImage: "doc.on.doc")
            }
            .buttonStyle(.bordered)
            .help(L10n.string("copy.api"))

            Menu {
                Button {
                    copy(L10n.string("copy.base"), model.serviceBaseURL)
                } label: {
                    Label(L10n.string("copy.base"), systemImage: "network")
                }
                Button {
                    copy(L10n.string("copy.api"), model.apiBaseURL)
                } label: {
                    Label(L10n.string("copy.api"), systemImage: "doc.on.doc")
                }
                Button {
                    copy(L10n.string("copy.key"), model.apiKey)
                } label: {
                    Label(L10n.string("copy.key"), systemImage: "key")
                }
                .disabled(model.apiKey.isEmpty)
                Button {
                    copy(L10n.string("copy.chat_url"), model.chatURLString)
                } label: {
                    Label(L10n.string("copy.chat_url"), systemImage: "link")
                }
                Divider()
                Button {
                    model.clearChat()
                } label: {
                    Label(L10n.string("chat.clear"), systemImage: "trash")
                }
                .disabled(model.chatMessages.isEmpty)
            } label: {
                Image(systemName: "ellipsis.circle")
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(.bar)
    }

    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 10) {
                    ForEach(model.chatMessages) { message in
                        MessageBubble(
                            message: message,
                            streaming: model.chatSending
                                && message.id == model.chatMessages.last?.id
                        )
                        .id(message.id)
                    }
                    if model.chatMessages.isEmpty {
                        Text(verbatim: L10n.string("chat.hint"))
                            .font(.callout)
                            .foregroundStyle(.secondary)
                            .padding(.top, 60)
                    }
                }
                .padding(14)
            }
            .background(.background)
            .onChange(of: model.chatMessages.count) { scrollToTail(proxy) }
            .onChange(of: model.chatMessages.last?.content) { scrollToTail(proxy) }
        }
    }

    /// Keep the newest bubble in view as it streams in; the id anchor gives
    /// SwiftUI a stable place to land on every content change.
    private func scrollToTail(_ proxy: ScrollViewProxy) {
        guard let last = model.chatMessages.last else { return }
        withAnimation(.easeOut(duration: 0.12)) {
            proxy.scrollTo(last.id, anchor: .bottom)
        }
    }

    private var inputBar: some View {
        HStack(spacing: 8) {
            TextField(L10n.string("chat.placeholder"), text: $model.chatInput)
                .textFieldStyle(.plain)
                .focused($inputFocused)
                .onSubmit { model.sendChat() }
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 10))
            if model.chatSending {
                Button {
                    model.cancelChat()
                } label: {
                    Label(L10n.string("chat.stop"), systemImage: "stop.fill")
                }
                .buttonStyle(.borderedProminent)
                .tint(.red)
            } else {
                Button {
                    model.sendChat()
                } label: {
                    Label(L10n.string("chat.send"), systemImage: "paperplane.fill")
                }
                .buttonStyle(.borderedProminent)
                .disabled(model.chatInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(12)
        .background(.bar)
    }

    private func copy(_ label: String, _ value: String) {
        guard !value.isEmpty else { return }
        model.copyToClipboard(value)
        withAnimation(.easeOut(duration: 0.15)) { copiedLabel = label }
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 1_400_000_000)
            withAnimation(.easeIn(duration: 0.15)) {
                if copiedLabel == label { copiedLabel = nil }
            }
        }
    }
}

/// One message as a bubble: the user's on the right in the accent color, the
/// model's on the left on a material. A streaming bubble carries a cursor so
/// it reads as in progress.
private struct MessageBubble: View {
    @EnvironmentObject private var model: AppModel
    let message: ChatMessage
    var streaming = false

    var body: some View {
        HStack(spacing: 12) {
            if message.role == .user { Spacer(minLength: 60) }
            Text(streaming && !message.content.isEmpty ? message.content + "▍" : message.content)
                .textSelection(.enabled)
                .font(.callout)
                .lineSpacing(2)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(
                    message.role == .user
                        ? Color.accentColor
                        : Color(nsColor: .underPageBackgroundColor),
                    in: RoundedRectangle(cornerRadius: 12)
                )
                .foregroundStyle(message.role == .user ? Color.white : Color.primary)
                .contextMenu {
                    Button {
                        model.copyToClipboard(message.content)
                    } label: {
                        Label(L10n.string("copy.message"), systemImage: "doc.on.doc")
                    }
                    if message.role == .assistant {
                        Button {
                            model.chatInput = message.content
                        } label: {
                            Label(L10n.string("chat.reuse"), systemImage: "arrow.uturn.down")
                        }
                    }
                }
            if message.role == .assistant { Spacer(minLength: 60) }
        }
        .frame(maxWidth: .infinity, alignment: message.role == .user ? .trailing : .leading)
    }
}