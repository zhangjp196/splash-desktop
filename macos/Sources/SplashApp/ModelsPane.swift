import SwiftUI

/// The models pane: the configurations the app can serve, added and edited
/// here.
struct ModelsPane: View {
    @EnvironmentObject private var model: AppModel
    @State private var editing: EditingConfig?
    @State private var pendingRemoval: ModelConfig?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                configsSection
            }
            .padding(20)
            .frame(maxWidth: 880, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .topLeading)
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

    private var configsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                SectionHeader(L10n.string("models.configs"), systemImage: "slider.horizontal.3")
                Spacer()
                Button {
                    editing = EditingConfig(config: model.newConfig(), isNew: true)
                } label: {
                    Label(L10n.string("models.add_config"), systemImage: "plus")
                }
            }
            if model.configs.isEmpty {
                EmptyHint(L10n.string("models.empty.configs"))
            } else {
                VStack(spacing: 10) {
                    ForEach(model.configs) { config in
                        ConfigRow(
                            config: config,
                            onEdit: { editing = EditingConfig(config: config, isNew: false) },
                            onRemove: { pendingRemoval = config }
                        )
                    }
                }
            }
        }
    }
}

/// A saved configuration, as a card: what it serves, how the server runs it,
/// and the ways to use, edit and remove it.
private struct ConfigRow: View {
    @EnvironmentObject private var model: AppModel
    let config: ModelConfig
    var onEdit: () -> Void
    var onRemove: () -> Void

    private var isActive: Bool { model.activeConfigID == config.id }

    var body: some View {
        HStack(spacing: 12) {
            HStack(spacing: 12) {
                Image(systemName: config.settings.modelMode == .local ? "folder" : "shippingbox")
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(Color.accentColor)
                    .frame(width: 32, height: 32)
                    .background(Color.accentColor.opacity(0.14), in: RoundedRectangle(cornerRadius: 9))
                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 6) {
                        Text(config.displayName).font(.body.weight(.medium)).lineLimit(1)
                        if isActive { ActiveBadge() }
                    }
                    HStack(spacing: 5) {
                        ForEach(chips, id: \.self) { chip in
                            Chip(text: chip)
                        }
                    }
                    .lineLimit(1)
                }
            }
            .contentShape(Rectangle())
            .onTapGesture { model.apply(config) }
            .help(L10n.string("models.use"))
            Spacer(minLength: 8)
            Button {
                model.apply(config)
                model.start()
            } label: {
                Label(L10n.string("models.start"), systemImage: "play.fill")
            }
            .buttonStyle(.borderedProminent)
            .disabled(model.isRunning || model.isBusy)
            Button(action: onEdit) {
                Image(systemName: "pencil")
            }
            .buttonStyle(.borderless)
            .help(L10n.string("models.edit"))
            Menu {
                Button(L10n.string("models.use")) { model.apply(config) }
                    .disabled(isActive)
                Button(L10n.string("models.edit")) { onEdit() }
                Divider()
                Button(L10n.string("models.remove"), role: .destructive) { onRemove() }
            } label: {
                Image(systemName: "ellipsis.circle")
            }
            .menuStyle(.borderlessButton)
            .frame(width: 24)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
        .background {
            RoundedRectangle(cornerRadius: 12).fill(.quaternary.opacity(0.25))
            if isActive {
                RoundedRectangle(cornerRadius: 12).fill(Color.green.opacity(0.10))
            }
        }
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .strokeBorder(isActive ? Color.green.opacity(0.5) : Color.secondary.opacity(0.25))
        )
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
            parts.append(L10n.string("mode.upstream"))
            if !config.settings.modelID.isEmpty { parts.append(config.settings.modelID) }
        case .local:
            parts.append(L10n.string("mode.local"))
            if !config.settings.modelDirectory.isEmpty {
                parts.append((config.settings.modelDirectory as NSString).lastPathComponent)
            }
            if !config.settings.draftDirectory.isEmpty {
                parts.append("Draft: \((config.settings.draftDirectory as NSString).lastPathComponent)")
            }
            if !config.settings.mmproj.isEmpty {
                parts.append("Vision: \((config.settings.mmproj as NSString).lastPathComponent)")
            }
        }
        parts.append(config.settings.kvFormat.uppercased())
        if !config.settings.neuralEngine { parts.append(L10n.string("models.ane_off")) }
        if config.settings.languageOnly { parts.append(L10n.string("models.text_only")) }
        if !config.settings.diskCache { parts.append(L10n.string("models.disk_cache_off")) }
        parts.append(L10n.format("models.concurrency", config.settings.maxConcurrency))
        return parts
    }
}

/// One characteristic of a configuration, as a small capsule.
private struct Chip: View {
    let text: String

    var body: some View {
        Text(text)
            .font(.caption2)
            .foregroundStyle(.secondary)
            .lineLimit(1)
            .truncationMode(.middle)
            .padding(.horizontal, 6)
            .padding(.vertical, 1)
            .background(.quaternary.opacity(0.5), in: Capsule())
    }
}

/// A small capsule on the configuration in use.
private struct ActiveBadge: View {
    var body: some View {
        Text(L10n.string("models.active"))
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 6)
            .padding(.vertical, 1)
            .background(Color.green.opacity(0.18), in: Capsule())
            .foregroundStyle(.green)
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

/// A titled band above a card.
private struct SectionHeader: View {
    let title: String
    let systemImage: String

    init(_ title: String, systemImage: String) {
        self.title = title
        self.systemImage = systemImage
    }

    var body: some View {
        Label(title, systemImage: systemImage).font(.headline)
    }
}

/// A dashed placeholder for an empty section.
private struct EmptyHint: View {
    let text: String

    init(_ text: String) { self.text = text }

    var body: some View {
        Text(text)
            .font(.callout)
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, alignment: .center)
            .padding(.vertical, 22)
            .background(.quaternary.opacity(0.18), in: RoundedRectangle(cornerRadius: 10))
            .overlay(
                RoundedRectangle(cornerRadius: 10)
                    .strokeBorder(style: StrokeStyle(lineWidth: 1, dash: [5, 4]))
                    .foregroundStyle(.separator)
            )
    }
}
