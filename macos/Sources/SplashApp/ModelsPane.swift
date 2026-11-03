import SwiftUI

/// The models pane: the configurations the app can serve, added and edited
/// here.
struct ModelsPane: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.openChat) private var openChat
    @State private var editing: EditingConfig?
    @State private var pendingRemoval: ModelConfig?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                PaneHeader(title: L10n.string("view.models"), subtitle: subtitle) {
                    Button {
                        editing = EditingConfig(config: model.newConfig(), isNew: true)
                    } label: {
                        Label(L10n.string("models.add_config"), systemImage: "plus")
                    }
                    .buttonStyle(.borderedProminent)
                }
                if model.phase == .ready {
                    runningCard
                }
                if model.configs.isEmpty {
                    EmptyHint(L10n.string("models.empty.configs"))
                } else {
                    LazyVStack(spacing: Theme.cardSpacing) {
                        ForEach(model.configs) { config in
                            ConfigRow(
                                config: config,
                                onEdit: { editing = EditingConfig(config: config, isNew: false) },
                                onRemove: { pendingRemoval = config }
                            )
                        }
                    }
                }
                if model.phase == .ready {
                    apiRow
                }
            }
            .padding(Theme.pagePadding)
            .frame(maxWidth: Theme.contentWidth, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .topLeading)
        }
        .background(.background)
        .sheet(item: $editing) { session in
            ConfigEditor(
                config: session.config,
                isNew: session.isNew,
                onSave: { edited in
                    model.save(edited)
                    editing = nil
                },
                onCancel: { editing = nil }
            )
            .frame(minWidth: 560, minHeight: 640)
        }
        .alert(
            L10n.string("models.remove.title"),
            isPresented: Binding(
                get: { pendingRemoval != nil },
                set: { if !$0 { pendingRemoval = nil } }
            ),
            presenting: pendingRemoval
        ) { config in
            Button(L10n.string("config.cancel"), role: .cancel) { pendingRemoval = nil }
            Button(L10n.string("models.remove"), role: .destructive) { model.removeConfig(config) }
        } message: { config in
            Text(verbatim: L10n.format("models.remove.message", config.displayName))
        }
    }

    /// What is being served right now, above the configurations: the one
    /// screen where a glance is enough before opening the chat.
    private var runningCard: some View {
        HStack(spacing: 18) {
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 7) {
                    Circle()
                        .fill(.green)
                        .frame(width: 7, height: 7)
                        .shadow(color: .green.opacity(0.55), radius: 3)
                    Text(verbatim: model.activeConfig?.displayName ?? model.live.modelID ?? L10n.string("models.none"))
                        .font(.headline)
                        .lineLimit(1)
                }
                Text(model.statusText)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            Spacer(minLength: 12)
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

    private var subtitle: String {
        model.configs.isEmpty
            ? L10n.string("models.empty.subtitle")
            : L10n.format("models.count", model.configs.count)
    }
}

/// A saved configuration, as a card: what it serves, how the server runs it,
/// and the ways to use, edit and remove it.
private struct ConfigRow: View {
    @EnvironmentObject private var model: AppModel
    let config: ModelConfig
    var onEdit: () -> Void
    var onRemove: () -> Void

    @State private var hovering = false

    private var isActive: Bool { model.activeConfigID == config.id }

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: config.settings.modelMode == .local ? "folder.fill" : "shippingbox.fill")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(isActive ? .white : Color.accentColor)
                .frame(width: 30, height: 30)
                .background(
                    isActive
                        ? AnyShapeStyle(Color.accentColor)
                        : AnyShapeStyle(Color.accentColor.opacity(0.12)),
                    in: RoundedRectangle(cornerRadius: 8)
                )

            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    Text(config.displayName)
                        .font(.body.weight(.medium))
                        .lineLimit(1)
                    if isActive {
                        Pill(text: L10n.string("models.active"), tint: .green)
                    }
                }
                HStack(spacing: 5) {
                    ForEach(chips, id: \.self) { chip in
                        Pill(text: chip)
                    }
                }
                .lineLimit(1)
                .fixedSize(horizontal: true, vertical: false)
                .clipped()
            }

            Spacer(minLength: 10)

            if !isActive {
                Button {
                    model.apply(config)
                } label: {
                    Image(systemName: "checkmark.circle")
                        .font(.system(size: 13))
                }
                .buttonStyle(.borderless)
                .help(L10n.string("models.use"))
            }

            Button {
                model.apply(config)
                model.start()
            } label: {
                Label(L10n.string("models.start"), systemImage: "play.fill")
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.regular)
            .disabled(model.isRunning || model.isBusy)

            Menu {
                Button(L10n.string("models.use")) { model.apply(config) }
                    .disabled(isActive)
                Button(L10n.string("models.edit")) { onEdit() }
                Divider()
                Button(L10n.string("models.remove"), role: .destructive) { onRemove() }
            } label: {
                Image(systemName: "ellipsis")
                    .font(.system(size: 12, weight: .semibold))
                    .frame(width: 24, height: 22)
                    .contentShape(Rectangle())
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .fixedSize()
        }
        .padding(.horizontal, 13)
        .padding(.vertical, 10)
        .background {
            RoundedRectangle(cornerRadius: Theme.cardRadius)
                .fill(isActive ? Color.green.opacity(0.08) : Color.primary.opacity(hovering ? 0.045 : 0.025))
        }
        .overlay {
            RoundedRectangle(cornerRadius: Theme.cardRadius)
                .strokeBorder(
                    isActive ? Color.green.opacity(0.45) : Color(nsColor: .separatorColor).opacity(0.6)
                )
        }
        .contentShape(Rectangle())
        .onTapGesture { model.apply(config) }
        .onHover { hovering = $0 }
        .animation(.easeOut(duration: 0.12), value: hovering)
        .contextMenu {
            Button(L10n.string("models.edit")) { onEdit() }
            Button(L10n.string("models.remove"), role: .destructive) { onRemove() }
        }
    }

    /// One short capsule per characteristic, so the row reads at a glance
    /// rather than as a run-on subtitle.
    private var chips: [String] {
        var parts: [String] = []
        switch config.settings.modelMode {
        case .upstream:
            parts.append(L10n.string("mode.upstream.short"))
            if !config.settings.modelID.isEmpty { parts.append(short(config.settings.modelID, 30)) }
        case .local:
            parts.append(L10n.string("mode.local.short"))
            if !config.settings.modelDirectory.isEmpty {
                parts.append(short((config.settings.modelDirectory as NSString).lastPathComponent, 26))
            }
        }
        parts.append(config.settings.kvFormat.uppercased())
        parts.append(L10n.format("models.concurrency", config.settings.maxConcurrency))
        if !config.settings.neuralEngine { parts.append(L10n.string("models.ane_off")) }
        if config.settings.languageOnly { parts.append(L10n.string("models.text_only")) }
        if !config.settings.diskCache { parts.append(L10n.string("models.disk_cache_off")) }
        return Array(parts.prefix(4))
    }

    private func short(_ value: String, _ limit: Int) -> String {
        value.count > limit ? String(value.prefix(limit)) + "…" : value
    }
}

/// The configuration editor: a name and the whole settings form, in a sheet.
/// Each instance owns an independent draft, so editing one configuration
/// never bleeds into another.
private struct ConfigEditor: View {
    let isNew: Bool
    var onSave: (ModelConfig) -> Void
    var onCancel: () -> Void

    @State private var draft: ModelConfig

    init(config: ModelConfig, isNew: Bool, onSave: @escaping (ModelConfig) -> Void, onCancel: @escaping () -> Void) {
        self.isNew = isNew
        self.onSave = onSave
        self.onCancel = onCancel
        _draft = State(initialValue: config)
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                BrandMark()
                Text(L10n.string(isNew ? "config.add.title" : "config.edit.title"))
                    .font(.headline)
                Spacer()
                Button(L10n.string("config.cancel"), action: onCancel)
                Button(L10n.string("config.save")) { onSave(draft) }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.defaultAction)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .background(.bar)
            Divider()
            HStack(spacing: 8) {
                Text(L10n.string("config.name")).foregroundStyle(.secondary)
                TextField("", text: $draft.name, prompt: Text(verbatim: draft.displayName))
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 340)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            Divider()
            SettingsTabs(settings: $draft.settings)
        }
        .frame(minWidth: 660, minHeight: 600)
    }
}

/// An independent editing session for one configuration.
private struct EditingConfig: Identifiable {
    let id = UUID()
    var config: ModelConfig
    let isNew: Bool
}

/// A placeholder for the empty configuration list: dashed, quiet, and the
/// same shape as the cards that replace it.
private struct EmptyHint: View {
    let text: String

    init(_ text: String) { self.text = text }

    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: "square.stack.3d.up")
                .font(.system(size: 22, weight: .light))
                .foregroundStyle(.tertiary)
            Text(text)
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 30)
        .background(.quaternary.opacity(0.12), in: RoundedRectangle(cornerRadius: Theme.cardRadius))
        .overlay(
            RoundedRectangle(cornerRadius: Theme.cardRadius)
                .strokeBorder(style: StrokeStyle(lineWidth: 1, dash: [5, 4]))
                .foregroundStyle(.separator)
        )
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
