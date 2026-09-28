import AppKit
import SwiftUI

/// The model library: a sidebar of the saved entries (SQLite) and a detail
/// editor for the selected one. Adding saves one entry at a time; Start runs
/// whatever the detail currently shows.
struct ModelsPane: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        HSplitView {
            ModelSidebar()
            ModelDetailForm()
        }
    }
}

private struct ModelSidebar: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Label(L10n.string("models.title"), systemImage: "building.2.crop.circle.fill")
                    .font(.subheadline.weight(.semibold))
                Spacer()
                Button {
                    model.newModel()
                } label: {
                    Label(L10n.string("models.add"), systemImage: "plus")
                }
                .labelStyle(.titleAndIcon)
                .disabled(model.isRunning)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            Divider()
            if model.library.isEmpty {
                empty
            } else {
                List(selection: $model.selectedModelID) {
                    ForEach(model.library) { entry in
                        ModelRow(entry: entry).tag(entry.id)
                    }
                }
                .listStyle(.sidebar)
                .onChange(of: model.selectedModelID) { _, newValue in
                    model.applySelection(newValue)
                }
            }
        }
        .frame(minWidth: 230, idealWidth: 260)
    }

    private var empty: some View {
        Button {
            model.newModel()
        } label: {
            VStack(spacing: 10) {
                Image(systemName: "cube.box")
                    .font(.system(size: 32))
                    .foregroundStyle(.tertiary)
                Text(L10n.string("models.empty"))
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 18)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .padding()
    }
}

private struct ModelRow: View {
    let entry: StoredModel

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(entry.name).lineLimit(1)
            Text(subtitle)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
        }
        .padding(.vertical, 2)
    }

    private var subtitle: String {
        switch entry.mode {
        case "splash":
            return entry.modelID.isEmpty ? L10n.string("mode.splash") : entry.modelID
        case "upstream":
            return entry.modelID.isEmpty ? L10n.string("mode.upstream") : entry.modelID
        default:
            let directory = entry.modelDirectory.isEmpty
                ? L10n.string("mode.local")
                : entry.modelDirectory
            return entry.draftDirectory.isEmpty
                ? directory
                : "\(directory) · \(L10n.string("side.draft"))"
        }
    }
}

private struct ModelDetailForm: View {
    @EnvironmentObject private var model: AppModel
    @State private var confirmDelete = false
    @FocusState private var nameFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            if model.selectedModelID == nil {
                HStack(spacing: 8) {
                    Image(systemName: "plus.circle.fill").foregroundStyle(.blue)
                    Text(verbatim: L10n.string("models.unsaved"))
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.blue.opacity(0.06))
            }
            Form {
                Section {
                    TextField(L10n.string("models.name"), text: $model.modelName)
                        .textFieldStyle(.roundedBorder)
                        .focused($nameFocused)
                        .disabled(model.isRunning)
                } header: {
                    Label(L10n.string("models.name"), systemImage: "tag")
                }

            Section {
                Picker(L10n.string("model.picker"), selection: $model.modelMode) {
                    ForEach(AppModel.ModelMode.allCases) { mode in
                        Text(verbatim: label(mode)).tag(mode)
                    }
                }
                .pickerStyle(.menu)
                .disabled(model.isRunning)

                switch model.modelMode {
                case .splash:
                    ModelField(
                        modelID: $model.modelID,
                        catalog: model.catalogIDs.filter { $0.lowercased().contains("splash") },
                        disabled: model.isRunning
                    )
                case .upstream:
                    ModelField(
                        modelID: $model.modelID,
                        catalog: model.catalogIDs.filter { !$0.lowercased().contains("splash") },
                        disabled: model.isRunning
                    )
                case .local:
                    DirectoryField(
                        title: L10n.string("field.target"),
                        path: $model.modelDirectory,
                        prompt: L10n.string("field.target.prompt")
                    )
                    .disabled(model.isRunning)
                    DirectoryField(
                        title: L10n.string("field.draft"),
                        path: $model.draftDirectory,
                        prompt: L10n.string("field.draft.prompt")
                    )
                    .disabled(model.isRunning)
                }
            } header: {
                Label(L10n.string("section.model"), systemImage: "cube.box")
            }

            Section {
                TextField(L10n.string("field.port"), value: $model.port, format: .number)
                    .frame(width: 120)
                    .disabled(model.isRunning)
                Toggle(L10n.string("field.language_only"), isOn: $model.languageOnly)
                    .disabled(model.isRunning || model.modelMode == .splash)
                Picker(L10n.string("field.kv"), selection: $model.kvFormat) {
                    Text(verbatim: L10n.string("kv.int8")).tag("int8")
                    Text(verbatim: L10n.string("kv.bf16")).tag("bf16")
                }
                .pickerStyle(.segmented)
                .frame(width: 220)
                .disabled(model.isRunning)
            } header: {
                Label(L10n.string("section.server"), systemImage: "network")
            }

            Section {
                TextField(L10n.string("field.max_memory"), text: $model.maxMemory)
                    .disabled(model.isRunning)
                TextField(L10n.string("field.max_context"), text: $model.maxContext)
                    .disabled(model.isRunning)
                TextField(L10n.string("field.max_cache_disk"), text: $model.maxCacheDisk)
                    .disabled(model.isRunning)
            } header: {
                Label(L10n.string("section.limits"), systemImage: "slider.horizontal.3")
            }

            Section {
                SecureField(L10n.string("field.api_key"), text: $model.apiKey)
                    .disabled(model.isRunning)
                TextField(L10n.string("field.served_names"), text: $model.servedNames)
                    .disabled(model.isRunning)
                TextField(L10n.string("field.max_request_size"), text: $model.maxRequestSize)
                    .disabled(model.isRunning)
                Picker(L10n.string("field.reasoning_effort"), selection: $model.reasoningEffort) {
                    Text(verbatim: L10n.string("reasoning.default")).tag("")
                    ForEach(["none", "minimal", "low", "medium", "high", "xhigh", "max"], id: \.self) { effort in
                        Text(verbatim: effort).tag(effort)
                    }
                }
                .pickerStyle(.menu)
                .disabled(model.isRunning)
            } header: {
                Label(L10n.string("section.advanced"), systemImage: "gearshape.2")
            }

            Section {
                HStack {
                    Button(L10n.string("start"), action: model.start)
                        .buttonStyle(.borderedProminent)
                        .disabled(!model.canStart)
                    Spacer()
                    Button(L10n.string("save")) { model.saveModel() }
                        .disabled(model.isRunning)
                    Button(L10n.string("delete"), role: .destructive) { confirmDelete = true }
                        .disabled(model.selectedModelID == nil || model.isRunning)
                }
            }
        }
        .formStyle(.grouped)
        .scrollContentBackground(.hidden)
        }
        .alert(L10n.string("delete.title"), isPresented: $confirmDelete) {
            Button(L10n.string("delete.confirm"), role: .destructive) {
                model.deleteSelectedModel()
            }
            Button(L10n.string("cancel"), role: .cancel) {}
        } message: {
            Text(verbatim: L10n.string("delete.message"))
        }
        .alert(
            L10n.string("error.library.title"),
            isPresented: saveErrorBinding,
            presenting: model.modelStoreError
        ) { _ in
            Button(L10n.string("ok"), role: .cancel) {}
        } message: { error in
            Text(verbatim: error)
        }
        .onChange(of: model.selectedModelID) { _, newValue in
            if newValue == nil { nameFocused = true }
        }
    }

    private var saveErrorBinding: Binding<Bool> {
        Binding(
            get: { model.modelStoreError != nil },
            set: { if !$0 { model.modelStoreError = nil } }
        )
    }

    private func label(_ mode: AppModel.ModelMode) -> String {
        switch mode {
        case .splash: return L10n.string("mode.splash")
        case .upstream: return L10n.string("mode.upstream")
        case .local: return L10n.string("mode.local")
        }
    }
}

/// A model ID field with a menu of matching catalog examples, kept in sync:
/// choosing an example replaces the current text.
private struct ModelField: View {
    @Binding var modelID: String
    let catalog: [String]
    let disabled: Bool

    var body: some View {
        HStack {
            TextField("owner/repo[:variant]", text: $modelID)
                .textFieldStyle(.roundedBorder)
                .disabled(disabled)
            if !catalog.isEmpty {
                Menu(L10n.string("examples")) {
                    ForEach(catalog, id: \.self) { identifier in
                        Button(identifier) { modelID = identifier }
                    }
                }
                .fixedSize()
                .disabled(disabled)
            }
        }
    }
}

/// A read-only path with a Choose button; the panel asks for a directory.
private struct DirectoryField: View {
    let title: String
    @Binding var path: String
    let prompt: String

    var body: some View {
        LabeledContent(title) {
            HStack {
                Text(path.isEmpty ? prompt : path)
                    .foregroundStyle(path.isEmpty ? .secondary : .primary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .frame(maxWidth: .infinity, alignment: .leading)
                Button(L10n.string("choose"), action: choose)
            }
        }
    }

    private func choose() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = L10n.string("choose")
        if !path.isEmpty {
            panel.directoryURL = URL(fileURLWithPath: path, isDirectory: true)
        }
        if panel.runModal() == .OK, let url = panel.url {
            path = url.path
        }
    }
}