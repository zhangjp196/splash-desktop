import AppKit
import SwiftUI

/// The models pane: the configurations the app can serve, added and edited
/// here, and the folders it manages, so a local model is one pick away.
struct ModelsPane: View {
    @EnvironmentObject private var model: AppModel
    @State private var draft = ModelConfig(settings: ServerSettings.default)
    @State private var editing = false
    @State private var isNew = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                configsSection
                directoriesSection
            }
            .padding(20)
            .frame(maxWidth: 880, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .topLeading)
        }
        .sheet(isPresented: $editing) {
            ConfigEditor(
                config: $draft,
                isNew: isNew,
                onSave: {
                    model.save(draft)
                    editing = false
                },
                onCancel: { editing = false }
            )
            .frame(minWidth: 560, minHeight: 640)
        }
    }

    private var configsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                SectionHeader(L10n.string("models.configs"), systemImage: "slider.horizontal.3")
                Spacer()
                Button {
                    draft = model.newConfig()
                    isNew = true
                    editing = true
                } label: {
                    Label(L10n.string("models.add_config"), systemImage: "plus")
                }
            }
            if model.configs.isEmpty {
                EmptyHint(L10n.string("models.empty.configs"))
            } else {
                Card {
                    ForEach(model.configs) { config in
                        ConfigRow(config: config) {
                            draft = config
                            isNew = false
                            editing = true
                        }
                        if config.id != model.configs.last?.id { Divider() }
                    }
                }
            }
        }
    }

    private var directoriesSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                SectionHeader(L10n.string("models.directories"), systemImage: "folder")
                Spacer()
                Button {
                    addDirectories()
                } label: {
                    Label(L10n.string("models.add_directory"), systemImage: "plus")
                }
            }
            if model.modelDirectories.isEmpty {
                EmptyHint(L10n.string("models.empty.dirs"))
            } else {
                Card {
                    ForEach(model.modelDirectories) { directory in
                        DirectoryRow(directory: directory)
                        if directory.id != model.modelDirectories.last?.id { Divider() }
                    }
                }
            }
        }
    }

    private func addDirectories() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = true
        panel.prompt = L10n.string("models.add_directory")
        if panel.runModal() == .OK {
            for url in panel.urls { model.addModelDirectory(url) }
        }
    }
}

/// A saved configuration, with the ways to use it and edit it.
private struct ConfigRow: View {
    @EnvironmentObject private var model: AppModel
    let config: ModelConfig
    var onEdit: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            HStack(spacing: 12) {
                Image(systemName: config.settings.modelMode == .local ? "folder" : "shippingbox")
                    .foregroundStyle(.secondary)
                    .frame(width: 22)
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text(config.displayName).font(.body.weight(.medium)).lineLimit(1)
                        if model.activeConfigID == config.id { ActiveBadge() }
                    }
                    Text(subtitle).font(.caption).foregroundStyle(.secondary)
                        .lineLimit(1).truncationMode(.middle)
                }
            }
            .contentShape(Rectangle())
            .onTapGesture { model.apply(config) }
            .help(L10n.string("models.use"))
            Spacer()
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
                    .disabled(model.activeConfigID == config.id)
                Button(L10n.string("models.edit")) { onEdit() }
                Divider()
                Button(L10n.string("models.remove"), role: .destructive) { model.removeConfig(config) }
            } label: {
                Image(systemName: "ellipsis.circle")
            }
            .menuStyle(.borderlessButton)
            .frame(width: 24)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .contextMenu {
            Button(L10n.string("models.edit")) { onEdit() }
            Button(L10n.string("models.remove"), role: .destructive) { model.removeConfig(config) }
        }
    }

    private var subtitle: String {
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
        }
        parts.append(config.settings.kvFormat.uppercased())
        if !config.settings.neuralEngine { parts.append(L10n.string("models.ane_off")) }
        if config.settings.languageOnly { parts.append(L10n.string("models.text_only")) }
        return parts.joined(separator: " · ")
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
private struct ConfigEditor: View {
    @Binding var config: ModelConfig
    let isNew: Bool
    var onSave: () -> Void
    var onCancel: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                BrandMark()
                Text(L10n.string(isNew ? "config.add.title" : "config.edit.title"))
                    .font(.headline)
                Spacer()
                Button(L10n.string("config.cancel"), action: onCancel)
                Button(L10n.string("config.save"), action: onSave)
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.defaultAction)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .background(.bar)
            Divider()
            HStack(spacing: 8) {
                Text(L10n.string("config.name")).foregroundStyle(.secondary)
                TextField("", text: $config.name, prompt: Text(verbatim: config.displayName))
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 340)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            Divider()
            SettingsTabs(settings: $config.settings)
        }
        .frame(minWidth: 660, minHeight: 600)
    }
}

/// A managed model directory, with what it holds and the ways to use it.
private struct DirectoryRow: View {
    @EnvironmentObject private var model: AppModel
    let directory: ModelDirectory

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: directory.kindValue == .gguf ? "shippingbox" : "cube.fill")
                .foregroundStyle(.secondary)
                .frame(width: 22)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(directory.name).font(.body.weight(.medium)).lineLimit(1)
                    KindBadge(kind: directory.kindValue)
                    if directory.hasVision {
                        Image(systemName: "eye").font(.caption2).foregroundStyle(.secondary)
                            .help(L10n.string("models.vision"))
                    }
                }
                Text(directory.path).font(.caption).foregroundStyle(.secondary)
                    .lineLimit(1).truncationMode(.middle)
            }
            Spacer()
            Text(directory.sizeBytes.map(formatBytes) ?? "…")
                .font(.caption).foregroundStyle(.secondary).monospacedDigit()
            Menu {
                Button(L10n.string("models.set_target")) { model.useAsTarget(directory) }
                Button(L10n.string("models.set_draft")) { model.useAsDraft(directory) }
                Divider()
                Button(L10n.string("models.reveal")) { model.reveal(directory.path) }
                Button(L10n.string("models.remove"), role: .destructive) {
                    model.removeModelDirectory(directory)
                }
            } label: {
                Image(systemName: "ellipsis.circle")
            }
            .menuStyle(.borderlessButton)
            .frame(width: 24)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
    }
}

/// A small capsule naming a directory's format.
private struct KindBadge: View {
    let kind: ModelKind

    var body: some View {
        Text(kind.label)
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 6)
            .padding(.vertical, 1)
            .background(color.opacity(0.16), in: Capsule())
            .foregroundStyle(color)
    }

    private var color: Color {
        switch kind {
        case .mlx: return .blue
        case .gguf: return .purple
        case .splash: return .orange
        case .unknown: return .secondary
        }
    }
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

/// A rounded card the rows sit in.
private struct Card<Content: View>: View {
    @ViewBuilder var content: Content

    var body: some View {
        VStack(spacing: 0) { content }
            .background(.quaternary.opacity(0.28), in: RoundedRectangle(cornerRadius: 10))
            .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(.separator.opacity(0.6)))
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
