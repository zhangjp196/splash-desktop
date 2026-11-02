import SwiftUI

/// The models pane: the model this Mac is serving, and the settings the next
/// start runs with. There is nothing to choose among — the pane shows what is
/// running, or that nothing is, above the settings themselves.
struct ModelsPane: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.openChat) private var openChat
    /// The settings the form edits. A field writes here first, so a keystroke
    /// is never redistributed through every stored setting and back into the
    /// field that is being typed in; the edit reaches the model, and the
    /// debounced store, from the change below.
    @State private var draft = ServerSettings.default

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            PaneHeader(title: L10n.string("view.models"), subtitle: L10n.string("models.subtitle")) {
                if model.isRunning {
                    Button(role: .destructive) {
                        model.stop()
                    } label: {
                        Label(L10n.string("stop"), systemImage: "stop.fill")
                    }
                    .disabled(model.phase == .stopping)
                } else {
                    Button {
                        model.start()
                    } label: {
                        Label(L10n.string("start"), systemImage: "play.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!model.canStart)
                }
            }
            runningCard
            settingsCard
            if model.phase == .ready {
                apiRow
            }
        }
        .padding(Theme.pagePadding)
        .frame(maxWidth: Theme.contentWidth, alignment: .leading)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(.background)
        .onAppear { draft = model.settings }
        .onChange(of: draft) { _, edited in
            model.update(edited)
        }
        // The settings can change without the form: Restore Defaults, or a
        // configuration the app attached to. The draft follows, unless it is
        // already what the model holds.
        .onChange(of: model.settingsFingerprint) { _, _ in
            if draft != model.settings { draft = model.settings }
        }
    }

    /// What is being served right now: the one screen where a glance is enough
    /// before opening the chat. Nothing is listed when nothing runs.
    private var runningCard: some View {
        HStack(spacing: 18) {
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 7) {
                    Circle()
                        .fill(model.runningModelName == nil ? Color.secondary : .green)
                        .frame(width: 7, height: 7)
                        .shadow(
                            color: (model.runningModelName == nil ? Color.secondary : .green).opacity(0.55),
                            radius: 3
                        )
                    Text(verbatim: model.runningModelName ?? L10n.string("models.none"))
                        .font(.headline)
                        .lineLimit(1)
                }
                Text(model.statusText)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            Spacer(minLength: 12)
            if model.runningModelName != nil {
                MiniStat(
                    title: L10n.string("live.decode"),
                    value: rate(model.live.decodeTokensPerSecond),
                    unit: L10n.string("live.caption.tokens_per_second")
                )
                MiniStat(
                    title: L10n.string("live.prefill"),
                    value: rate(model.live.prefillTokensPerSecond),
                    unit: L10n.string("live.caption.tokens_per_second")
                )
                MiniStat(
                    title: L10n.string("live.physical_memory"),
                    value: bytes(model.live.physicalBytes)
                )
                Button {
                    openChat()
                } label: {
                    Label(L10n.string("open.chat"), systemImage: "bubble.left.and.bubble.right")
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .padding(.horizontal, 15)
        .padding(.vertical, 13)
        .background {
            RoundedRectangle(cornerRadius: 14)
                .fill(
                    LinearGradient(
                        colors: [Color.accentColor.opacity(0.07), Color.teal.opacity(0.05)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
        }
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .strokeBorder(Color.accentColor.opacity(0.16))
        )
    }

    /// The settings the next start carries, edited in place: one set, because
    /// one Mac serves one model at a time.
    private var settingsCard: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 8) {
                Label(L10n.string("models.settings"), systemImage: "slider.horizontal.3")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.secondary)
                Spacer(minLength: 8)
                if model.isRunning {
                    Text(L10n.string("models.settings.next_start"))
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 9)

            Divider()

            SettingsTabs(settings: $draft)
        }
        .frame(maxHeight: .infinity, alignment: .top)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: Theme.cardRadius))
        .overlay(
            RoundedRectangle(cornerRadius: Theme.cardRadius)
                .strokeBorder(.separator.opacity(0.5))
        )
    }

    private var apiRow: some View {
        HStack(spacing: 8) {
            Image(systemName: "link")
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(.tertiary)
            Text(verbatim: model.apiBaseURL)
                .font(.system(size: 11.5, design: .monospaced))
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
            Spacer(minLength: 8)
            Button {
                model.copyToClipboard(model.apiBaseURL)
            } label: {
                Image(systemName: "doc.on.doc")
                    .font(.system(size: 10.5))
            }
            .buttonStyle(.borderless)
            .help(L10n.string("menu.copy_api"))
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.quaternary.opacity(0.14), in: RoundedRectangle(cornerRadius: 9))
    }

    private func rate(_ value: Double?) -> String {
        value.map { String(format: "%.0f", $0) } ?? "—"
    }

    private func bytes(_ value: UInt64?) -> String {
        guard let value else { return "—" }
        return ByteCountFormatter.string(fromByteCount: Int64(value), countStyle: .memory)
    }
}

/// One number in the running card: a quiet label and a figure big enough to
/// read across the desk.
private struct MiniStat: View {
    let title: String
    let value: String
    var unit: String = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(1)
            HStack(alignment: .firstTextBaseline, spacing: 3) {
                Text(value)
                    .font(.system(size: 16, weight: .semibold).monospacedDigit())
                if !unit.isEmpty {
                    Text(unit)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            .lineLimit(1)
        }
    }
}
