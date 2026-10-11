import AppKit
import SwiftUI

/// The models pane: the configurations the app has served, for one-click
/// switching, and the directories it manages, so a local model is one pick
/// away.
struct ModelsPane: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                recentSection
                directoriesSection
            }
            .padding(20)
            .frame(maxWidth: 860, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .topLeading)
        }
    }

    private var recentSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            SectionHeader(L10n.string("models.recent"), systemImage: "clock.arrow.circlepath")
            if model.recentCombos.isEmpty {
                EmptyHint(L10n.string("models.empty.recent"))
            } else {
                Card {
                    ForEach(model.recentCombos) { combo in
                        ComboRow(combo: combo)
                        if combo.id != model.recentCombos.last?.id { Divider() }
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

/// A remembered configuration, with the ways to use it again.
private struct ComboRow: View {
    @EnvironmentObject private var model: AppModel
    let combo: ModelCombo

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: combo.modeValue == .local ? "folder" : "shippingbox")
                .foregroundStyle(.secondary)
                .frame(width: 22)
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.body.weight(.medium)).lineLimit(1)
                Text(subtitle).font(.caption).foregroundStyle(.secondary)
                    .lineLimit(1).truncationMode(.middle)
            }
            Spacer()
            Button(L10n.string("models.use")) { model.apply(combo) }
                .disabled(model.isRunning)
            Button {
                model.apply(combo)
                model.start()
            } label: {
                Label(L10n.string("models.use_start"), systemImage: "play.fill")
            }
            .buttonStyle(.borderedProminent)
            .disabled(model.isRunning || model.isBusy)
            Menu {
                Button(L10n.string("models.remove"), role: .destructive) { model.removeCombo(combo) }
            } label: {
                Image(systemName: "ellipsis.circle")
            }
            .menuStyle(.borderlessButton)
            .frame(width: 24)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
    }

    private var title: String {
        switch combo.modeValue {
        case .local: return (combo.modelDirectory as NSString).lastPathComponent
        case .upstream: return combo.modelID
        }
    }

    private var subtitle: String {
        var parts = [modeLabel]
        if combo.modeValue == .local, !combo.draftDirectory.isEmpty {
            parts.append("Draft: \((combo.draftDirectory as NSString).lastPathComponent)")
        }
        parts.append(combo.kvFormat.uppercased())
        if combo.languageOnly { parts.append(L10n.string("models.text_only")) }
        return parts.joined(separator: " · ")
    }

    private var modeLabel: String {
        switch combo.modeValue {
        case .upstream: return L10n.string("mode.upstream")
        case .local: return L10n.string("mode.local")
        }
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
        Label(title, systemImage: systemImage)
            .font(.headline)
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
