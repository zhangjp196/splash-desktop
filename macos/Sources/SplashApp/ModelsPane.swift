import SwiftUI

/// The models pane: the configurations the app can serve, added and edited
/// here.
struct ModelsPane: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.openChat) private var openChat
    @State private var editing: EditingConfig?
    @State private var pendingRemoval: ModelConfig?
    @State private var downloading = false
    @StateObject private var downloader = Downloader()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                PaneHeader(
                    title: L10n.string("view.models"),
                    subtitle: subtitle,
                    eyebrow: L10n.string("models.eyebrow")
                ) {
                    Button {
                        downloader.reset()
                        downloading = true
                    } label: {
                        Label(L10n.string("download.title"), systemImage: "arrow.down.circle")
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
                    EmptyHint(L10n.string("models.empty.configs"))
                } else {
                    LazyVGrid(
                        columns: [GridItem(.adaptive(minimum: 164, maximum: 196),
                                           spacing: 16, alignment: .top)],
                        alignment: .leading,
                        spacing: 20
                    ) {
                        ForEach(model.configs) { config in
                            ConfigTile(
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
        .sheet(isPresented: $downloading) {
            DownloadSheet(
                downloader: downloader,
                onClose: { downloading = false }
            )
            .environmentObject(model)
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

/// A saved configuration, as a rectangle: the icon and the kind inside, the
/// name and the state outside it, so a row of several reads across the desk.
/// Clicking the tile starts it — applying the configuration and starting the
/// server is the one way to make it the configuration the server runs — and
/// the menu holds edit and remove.
private struct ConfigTile: View {
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
        VStack(spacing: 7) {
            block
            label
        }
        .frame(width: 172)
        .contentShape(Rectangle())
        .onHover { hovering = $0 }
        .animation(.easeOut(duration: 0.12), value: hovering)
        .onTapGesture { if canStart { start() } }
        .contextMenu {
            Button(L10n.string("models.start")) { start() }.disabled(!canStart)
            Button(L10n.string("models.edit")) { onEdit() }
            Divider()
            Button(L10n.string("models.remove"), role: .destructive) { onRemove() }
        }
    }

    /// The rectangle itself: an icon, the kind, the serving mark, and — on a
    /// tile that can start — the play affordance the click means.
    private var block: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(fill)
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(
                    isServing ? SplashPalette.mint.opacity(0.55) : SplashPalette.border.opacity(hovering ? 0.75 : 0.5),
                    lineWidth: isServing ? 1.2 : 0.8
                )

            VStack(spacing: 8) {
                Image(systemName: icon)
                    .font(.system(size: 25, weight: .semibold))
                    .foregroundStyle(isServing ? SplashPalette.mint : SplashPalette.cyan)
                Pill(text: kindTag, tint: kindTint)
            }

            if canStart && hovering {
                VStack {
                    Spacer(minLength: 0)
                    Label(L10n.string("models.start"), systemImage: "play.fill")
                        .font(.system(size: 10.5, weight: .semibold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 4)
                        .background(SplashPalette.blue, in: Capsule())
                }
                .padding(.bottom, 8)
                .transition(.opacity)
            }

            VStack {
                HStack {
                    Spacer(minLength: 0)
                    if isServing {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(SplashPalette.mint)
                    } else if hovering {
                        Button(action: onEdit) {
                            Image(systemName: "slider.horizontal.3")
                                .font(.system(size: 10.5, weight: .semibold))
                                .foregroundStyle(SplashPalette.muted)
                                .frame(width: 20, height: 18)
                                .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .help(L10n.string("models.edit"))
                    }
                }
                Spacer(minLength: 0)
            }
            .padding(8)
        }
        .frame(height: 104)
    }

    /// Outside the rectangle: the name, and what the tile is serving with.
    private var label: some View {
        VStack(spacing: 3) {
            Text(config.displayName)
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(isServing ? SplashPalette.mint : Color.primary)
                .lineLimit(1)
                .truncationMode(.middle)
            HStack(spacing: 4) {
                if config.missingFiles {
                    Pill(text: L10n.string("models.files_missing"), tint: SplashPalette.rose)
                } else {
                    ForEach(chips.prefix(2), id: \.self) { chip in
                        Pill(text: chip)
                    }
                }
            }
            .lineLimit(1)
        }
        .frame(maxWidth: .infinity)
        .help(config.settings.modelMode == .local
              ? config.settings.modelDirectory
              : config.settings.modelID)
    }

    private var fill: AnyShapeStyle {
        if isServing {
            return AnyShapeStyle(
                LinearGradient(
                    colors: [SplashPalette.mint.opacity(0.16), SplashPalette.cyan.opacity(0.05)],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
            )
        }
        return AnyShapeStyle(SplashPalette.surface.opacity(hovering ? 0.62 : 0.42))
    }

    private var icon: String {
        switch config.settings.modelMode {
        case .upstream: return "shippingbox.fill"
        case .local: return config.settings.localKind == "mlx" ? "cube.fill" : "doc.fill"
        }
    }

    private var kindTag: String {
        switch config.settings.modelMode {
        case .upstream: return L10n.string("models.kind.hub")
        case .local: return L10n.string(config.settings.localKind == "mlx" ? "models.kind.mlx" : "models.kind.gguf")
        }
    }

    private var kindTint: Color {
        config.settings.modelMode == .upstream ? SplashPalette.violet : SplashPalette.cyan
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
        parts.append(L10n.format("models.concurrency", config.settings.maxConcurrency))
        return parts
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
