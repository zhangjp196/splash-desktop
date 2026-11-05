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

/// The configuration editor: a name, the whole settings form, and a summary of
/// what saving will start, in one sheet. Each instance owns an independent
/// draft, so editing one configuration never bleeds into another.
private struct ConfigEditor: View {
    let isNew: Bool
    var onSave: (ModelConfig) -> Void
    var onCancel: () -> Void

    /// The configuration the sheet opened on, kept to tell an edited section
    /// from an untouched one.
    private let original: ModelConfig
    @State private var draft: ModelConfig
    @FocusState private var nameFocused: Bool

    init(config: ModelConfig, isNew: Bool, onSave: @escaping (ModelConfig) -> Void, onCancel: @escaping () -> Void) {
        self.isNew = isNew
        self.onSave = onSave
        self.onCancel = onCancel
        self.original = config
        _draft = State(initialValue: config)
    }

    /// A configuration with nothing to serve would save and then fail to
    /// start, so the editor asks for the model before it lets go.
    private var canSave: Bool {
        switch draft.settings.modelMode {
        case .upstream:
            return !draft.settings.modelID.trimmingCharacters(in: .whitespaces).isEmpty
        case .local:
            return !draft.settings.modelDirectory.trimmingCharacters(in: .whitespaces).isEmpty
        }
    }

    private var hasChanges: Bool { draft != original }

    private var summary: String {
        let mode = draft.settings.modelMode == .local
            ? L10n.string("mode.local.short")
            : L10n.string("mode.upstream.short")
        return "\(mode) · \(draft.settings.kvFormat.uppercased())"
    }

    var body: some View {
        ZStack {
            AuroraBackground(tint: SplashPalette.blue)

            VStack(spacing: 0) {
                header
                Rectangle()
                    .fill(SplashPalette.border.opacity(0.4))
                    .frame(height: 0.7)
                SettingsEditor(settings: $draft.settings, baseline: original.settings)
                footer
            }
        }
        .frame(width: 900, height: 680)
        .onExitCommand(perform: onCancel)
        .onAppear {
            guard isNew else { return }
            DispatchQueue.main.async { nameFocused = true }
        }
    }

    private var header: some View {
        HStack(spacing: 14) {
            BrandMark(size: 32)

            VStack(alignment: .leading, spacing: 3) {
                Text(L10n.string(isNew ? "config.eyebrow.new" : "config.eyebrow.edit"))
                    .font(.system(size: 9.5, weight: .bold))
                    .tracking(1.5)
                    .foregroundStyle(SplashPalette.cyan)
                TextField(
                    "",
                    text: $draft.name,
                    prompt: Text(L10n.string("config.name.placeholder"))
                )
                .textFieldStyle(.plain)
                .font(.system(size: 21, weight: .bold, design: .rounded))
                .tracking(-0.3)
                .focused($nameFocused)
                .onSubmit(save)
                .overlay(alignment: .bottom) {
                    Rectangle()
                        .fill(nameFocused
                            ? SplashPalette.blue.opacity(0.7)
                            : SplashPalette.border.opacity(0.45))
                        .frame(height: 0.8)
                        .offset(y: 3)
                }
                .frame(maxWidth: 420)
            }

            Spacer(minLength: 12)

            if hasChanges {
                Pill(text: L10n.string("config.changed"), tint: SplashPalette.amber)
                    .transition(.opacity)
            }

            Button(action: onCancel) {
                Text(L10n.string("config.cancel"))
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.primary)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 7)
                    .background(
                        SplashPalette.surface.opacity(0.55),
                        in: Capsule()
                    )
                    .overlay(
                        Capsule().strokeBorder(
                            SplashPalette.border.opacity(0.6),
                            lineWidth: 0.8
                        )
                    )
            }
            .buttonStyle(.plain)
            .keyboardShortcut(.cancelAction)

            Button(action: save) {
                Text(L10n.string("config.save"))
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 18)
                    .padding(.vertical, 7)
                    .background(SplashPalette.accentFill, in: Capsule())
                    .shadow(color: SplashPalette.blue.opacity(0.35), radius: 8, y: 3)
                    .opacity(canSave ? 1 : 0.4)
            }
            .buttonStyle(.plain)
            .keyboardShortcut(.defaultAction)
            .disabled(!canSave)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
    }

    private var footer: some View {
        HStack(spacing: 10) {
            Image(systemName: draft.settings.modelMode == .local ? "folder.fill" : "shippingbox.fill")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(SplashPalette.cyan)
                .frame(width: 26, height: 26)
                .background(
                    SplashPalette.cyan.opacity(0.12),
                    in: RoundedRectangle(cornerRadius: 7, style: .continuous)
                )
            VStack(alignment: .leading, spacing: 1) {
                Text(verbatim: draft.displayName)
                    .font(.system(size: 11.5, weight: .semibold))
                    .lineLimit(1)
                Text(verbatim: summary)
                    .font(.system(size: 10.5))
                    .foregroundStyle(SplashPalette.muted)
                    .lineLimit(1)
            }

            Spacer(minLength: 12)

            if !canSave {
                Label(L10n.string("config.needs_model"), systemImage: "exclamationmark.triangle.fill")
                    .font(.system(size: 10.5))
                    .foregroundStyle(SplashPalette.amber)
            } else if draft.settings.lanAccess && draft.settings.apiKey.isEmpty {
                Label(L10n.string("help.lan_access.no_key"), systemImage: "exclamationmark.triangle.fill")
                    .font(.system(size: 10.5))
                    .foregroundStyle(SplashPalette.amber)
                    .lineLimit(1)
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 10)
        .background(SplashPalette.surface.opacity(0.32))
        .overlay(alignment: .top) {
            Rectangle()
                .fill(SplashPalette.border.opacity(0.4))
                .frame(height: 0.7)
        }
    }

    private func save() {
        guard canSave else { return }
        var saved = draft
        saved.name = saved.name.trimmingCharacters(in: .whitespacesAndNewlines)
        onSave(saved)
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
