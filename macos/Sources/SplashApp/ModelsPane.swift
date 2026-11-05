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
                PaneHeader(
                    title: L10n.string("view.models"),
                    subtitle: subtitle,
                    eyebrow: L10n.string("models.eyebrow")
                ) {
                    Button {
                        editing = EditingConfig(config: model.newConfig(), isNew: true)
                    } label: {
                        Label(L10n.string("models.add_config"), systemImage: "plus")
                            .font(.system(size: 12, weight: .semibold))
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(SplashPalette.blue)
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
        .background(Color.clear)
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
                    StatusBeacon(color: SplashPalette.mint, active: true)
                    Text(verbatim: model.runningConfig?.displayName
                         ?? model.live.modelID
                         ?? model.activeConfig?.displayName
                         ?? L10n.string("models.none"))
                        .font(.system(size: 15, weight: .semibold, design: .rounded))
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
        .glassPanel(tint: SplashPalette.mint, raised: true)
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
        .padding(.vertical, 9)
        .background(SplashPalette.surface.opacity(0.45), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(SplashPalette.border.opacity(0.45), lineWidth: 0.7)
        )
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

/// A saved configuration, as a card: what it serves and how the server runs
/// it. Start is the only way to make one the configuration the server runs —
/// it applies the configuration and starts the server — and the menu holds
/// edit and remove.
private struct ConfigRow: View {
    @EnvironmentObject private var model: AppModel
    let config: ModelConfig
    var onEdit: () -> Void
    var onRemove: () -> Void

    @State private var hovering = false

    /// Green marks the configuration the running server was started with, and
    /// nothing else: selecting or editing another configuration is not use.
    private var isServing: Bool {
        model.isRunning && model.runningConfigID == config.id
    }

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: config.settings.modelMode == .local ? "folder.fill" : "shippingbox.fill")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(isServing ? .white : Color.accentColor)
                .frame(width: 30, height: 30)
                .background(
                    isServing
                        ? AnyShapeStyle(Color.accentColor)
                        : AnyShapeStyle(Color.accentColor.opacity(0.12)),
                    in: RoundedRectangle(cornerRadius: 8)
                )

            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    Text(config.displayName)
                        .font(.body.weight(.medium))
                        .lineLimit(1)
                    if isServing {
                        Pill(text: L10n.string("models.active"), tint: SplashPalette.mint)
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

            Button {
                model.apply(config)
                model.start()
            } label: {
                Label(
                    isServing ? L10n.string("models.active") : L10n.string("models.start"),
                    systemImage: isServing ? "checkmark.circle.fill" : "play.fill"
                )
            }
            .buttonStyle(.borderedProminent)
            .tint(isServing ? SplashPalette.mint : SplashPalette.blue)
            .controlSize(.regular)
            .disabled(model.isRunning || model.isBusy)

            Menu {
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
            RoundedRectangle(cornerRadius: Theme.cardRadius, style: .continuous)
                .fill(
                    isServing
                        ? AnyShapeStyle(
                            LinearGradient(
                                colors: [SplashPalette.mint.opacity(0.14), SplashPalette.cyan.opacity(0.04)],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        : AnyShapeStyle(SplashPalette.surface.opacity(hovering ? 0.58 : 0.40))
                )
        }
        .overlay {
            RoundedRectangle(cornerRadius: Theme.cardRadius, style: .continuous)
                .strokeBorder(
                    isServing
                        ? SplashPalette.mint.opacity(0.45)
                        : SplashPalette.border.opacity(0.5),
                    lineWidth: isServing ? 1 : 0.8
                )
        }
        .contentShape(Rectangle())
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
