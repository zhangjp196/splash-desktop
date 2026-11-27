import SwiftUI

/// The models pane: the configurations the app can serve, added and edited
/// here, listed by the kind of weights they load.
struct ModelsPane: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.openChat) private var openChat
    @State private var editing: EditingConfig?
    @State private var pendingRemoval: ModelConfig?
    @State private var downloading = false
    @State private var kind = ConfigKind.mlx
    @StateObject private var downloader = Downloader()

    /// The configurations the pane lists under the chosen kind. An online
    /// target has no local weights, so it is filed under neither and the pane
    /// shows it under its own heading.
    private var shown: [ModelConfig] {
        model.configs.filter { ConfigKind(of: $0) == kind }
    }

    private var counts: [ConfigKind: Int] {
        model.configs.reduce(into: [:]) { totals, config in
            totals[ConfigKind(of: config), default: 0] += 1
        }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                PaneHeader(
                    title: L10n.string("view.models")
                ) {
                    Button {
                        downloader.reset()
                        downloading = true
                    } label: {
                        Label(L10n.string("download.title"), systemImage: "magnifyingglass.circle")
                            .font(.system(size: 12, weight: .semibold))
                    }
                    .buttonStyle(.bordered)
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
                    EmptyView()
                } else {
                    kindPicker
                    if shown.isEmpty {
                        emptyKind
                    } else {
                        VStack(spacing: 6) {
                            ForEach(shown) { config in
                                ConfigRow(
                                    config: config,
                                    onEdit: { editing = EditingConfig(config: config, isNew: false) },
                                    onRemove: { pendingRemoval = config }
                                )
                            }
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
        .sheet(isPresented: $downloading) {
            DownloadSheet(
                downloader: downloader,
                onClose: { downloading = false }
            )
        }
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
        }
    }

    /// GGUF or MLX, with each kind's count, and no third tab: a configuration
    /// loads one of the two.
    private var kindPicker: some View {
        Picker("", selection: $kind) {
            ForEach(ConfigKind.allCases) { option in
                Text(verbatim: counts[option].map { "\(option.title) \($0)" } ?? option.title)
                    .tag(option)
            }
        }
        .pickerStyle(.segmented)
        .labelsHidden()
    }

    /// The chosen kind with nothing filed under it: the tab is not empty, the
    /// pane has simply not added one yet.
    private var emptyKind: some View {
        HStack(spacing: 8) {
            Image(systemName: kind.icon)
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(SplashPalette.muted)
            Text(L10n.format("models.kind.empty", kind.title))
                .font(.system(size: 12))
                .foregroundStyle(.secondary)
            Spacer()
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 16)
        .background(SplashPalette.surface.opacity(0.34),
                    in: RoundedRectangle(cornerRadius: 10, style: .continuous))
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

}

/// The kind of weights a configuration loads, which the pane lists them by.
enum ConfigKind: String, CaseIterable, Identifiable {
    // MLX first: the pane opens on the kind most Macs serve fastest.
    case mlx
    case gguf

    init(of config: ModelConfig) {
        // An online target has no local weights; it is listed under the kind
        // its saved settings last had, so it never disappears from the pane.
        self = config.settings.localKind == "mlx" ? .mlx : .gguf
    }

    var id: String { rawValue }

    /// The name the model settings use for it.
    var title: String {
        L10n.string(rawValue == "gguf" ? "models.kind.gguf" : "models.kind.mlx")
    }

    var icon: String {
        rawValue == "mlx" ? "cube.fill" : "doc.fill"
    }
}

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

    private var canStart: Bool {
        !model.isRunning && !model.isBusy && !config.missingFiles
    }

    var body: some View {
        HStack(spacing: 11) {
            kindBadge
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(config.displayName)
                        .font(.system(size: 12.5, weight: .semibold))
                        .foregroundStyle(isServing ? SplashPalette.mint : Color.primary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    if isServing {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.system(size: 11.5, weight: .semibold))
                            .foregroundStyle(SplashPalette.mint)
                    }
                }
                chipsLine
            }
            Spacer(minLength: 10)
            actions
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 9)
        .background(fill, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(
                    isServing ? SplashPalette.mint.opacity(0.5)
                              : SplashPalette.border.opacity(hovering ? 0.6 : 0.3),
                    lineWidth: isServing ? 1.1 : 0.8
                )
        )
        .contentShape(Rectangle())
        .onHover { hovering = $0 }
        .onTapGesture { if canStart { start() } }
        .contextMenu {
            Button(L10n.string("models.start")) { start() }.disabled(!canStart)
            Button(L10n.string("models.edit")) { onEdit() }
            Divider()
            Button(L10n.string("models.remove"), role: .destructive) { onRemove() }
        }
    }

    /// The row's leading square: which kind of weights it loads.
    private var kindBadge: some View {
        Image(systemName: icon)
            .font(.system(size: 14, weight: .semibold))
            .foregroundStyle(isServing ? SplashPalette.mint : SplashPalette.cyan)
            .frame(width: 30, height: 30)
            .background(
                (isServing ? SplashPalette.mint : SplashPalette.cyan).opacity(0.12),
                in: RoundedRectangle(cornerRadius: 8, style: .continuous)
            )
    }

    /// The characteristics, one capsule each, so a row reads across rather
    /// than as a run-on subtitle.
    private var chipsLine: some View {
        HStack(spacing: 4) {
            if config.missingFiles {
                Pill(text: L10n.string("models.files_missing"), tint: SplashPalette.rose)
            } else {
                ForEach(Array(chips.prefix(4).enumerated()), id: \.offset) { _, chip in
                    Pill(text: chip)
                }
            }
        }
        .lineLimit(1)
    }

    /// Start, edit and remove, shown on the row the pointer is over.
    private var actions: some View {
        HStack(spacing: 6) {
            if hovering || isServing {
                Button(action: start) {
                    Label(L10n.string("models.start"), systemImage: "play.fill")
                        .font(.system(size: 10.5, weight: .semibold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 9)
                        .padding(.vertical, 4)
                        .background(
                            canStart ? SplashPalette.blue : SplashPalette.muted.opacity(0.6),
                            in: Capsule()
                        )
                }
                .buttonStyle(.plain)
                .disabled(!canStart)
            }
            if hovering {
                Button(action: onEdit) {
                    Image(systemName: "slider.horizontal.3")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(SplashPalette.muted)
                        .frame(width: 24, height: 22)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help(L10n.string("models.edit"))
            }
        }
    }

    private var fill: AnyShapeStyle {
        if isServing {
            // A flat tint wash rather than a gradient: the serving mark is a
            // two-state fill, and a gradient would blend an extra layer for
            // it.
            return AnyShapeStyle(SplashPalette.mint.opacity(0.13))
        }
        return AnyShapeStyle(SplashPalette.surface.opacity(hovering ? 0.6 : 0.4))
    }

    private var icon: String {
        switch config.settings.modelMode {
        case .upstream: return "shippingbox.fill"
        case .local: return config.settings.localKind == "mlx" ? "cube.fill" : "doc.fill"
        }
    }

    private func start() {
        model.apply(config)
        model.start()
    }

    /// One short capsule per characteristic, so the tile reads at a glance
    /// rather than as a run-on subtitle.
    private var chips: [String] {
        var parts: [String] = []
        switch config.settings.modelMode {
        case .upstream:
            if !config.settings.modelID.isEmpty { parts.append(short(config.settings.modelID, 22)) }
        case .local:
            if !config.settings.modelDirectory.isEmpty {
                parts.append(short((config.settings.modelDirectory as NSString).lastPathComponent, 22))
            }
        }
        parts.append(config.settings.kvFormat.uppercased())
        parts.append(L10n.string("speculation.\(config.settings.speculation)"))
        parts.append(L10n.format("config.review.context", config.settings.maxContext))
        if !config.settings.maxMemory.isEmpty { parts.append(config.settings.maxMemory) }
        return parts
    }

    private func short(_ value: String, _ limit: Int) -> String {
        value.count > limit ? String(value.prefix(limit)) + "…" : value
    }
}

/// The configuration editor: a name and the model settings form, in one sheet. Each instance owns an independent
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
        // An unnamed configuration opens on the name its model gives it, so
        // the header never reads as an untitled blank: the value is the
        // default, and it is the user's to change or clear.
        var opened = config
        if opened.name.trimmingCharacters(in: .whitespaces).isEmpty,
           let derived = opened.derivedName {
            opened.name = derived
        }
        self.original = opened
        _draft = State(initialValue: opened)
    }

    /// A configuration with nothing to serve would save and then fail to
    /// start, so the editor asks for the model before it lets go.
    private var canSave: Bool {
        !draft.settings.modelDirectory.trimmingCharacters(in: .whitespaces).isEmpty
    }

    private var hasChanges: Bool { draft != original }

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
        .frame(width: 840, height: 620)
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
                    // With the field cleared, the header still names the
                    // configuration the way the models pane will.
                    prompt: Text(draft.derivedName ?? L10n.string("config.name.placeholder"))
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
            }

            Spacer(minLength: 12)

            if !canSave {
                Label(L10n.string("config.needs_model"), systemImage: "exclamationmark.triangle.fill")
                    .font(.system(size: 10.5))
                    .foregroundStyle(SplashPalette.amber)
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
        ServerSettings.sanitized(&saved.settings)
        // The panel's model is a local one: an edit of a configuration the
        // older online mode saved moves it here, where the downloader puts
        // what it fetches. A draft is a folder now, not a repository ID.
        saved.settings.modelMode = .local
        saved.settings.draftMode =
            saved.settings.draftDirectory.trimmingCharacters(in: .whitespaces).isEmpty
                ? "none" : "folder"
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
