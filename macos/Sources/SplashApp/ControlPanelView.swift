import AppKit
import SwiftUI

/// Model source and server settings, grouped by what they decide: which model,
/// what the server exposes, how much memory it may use, what it does while
/// idle, and what a request may carry. Every field feeds the settings value
/// SQLite keeps, so this form is where the next launch comes from.
struct ControlPanelView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        Form {
            Section {
                Picker(L10n.string("model.picker"), selection: $model.modelMode) {
                    ForEach(AppModel.ModelMode.allCases) { mode in
                        Text(verbatim: label(mode)).tag(mode)
                    }
                }
                .pickerStyle(.menu)
                .disabled(model.isRunning)
                .onChange(of: model.modelMode) { edited() }

                switch model.modelMode {
                case .splash:
                    ModelField(
                        modelID: $model.modelID,
                        catalog: model.catalogIDs.filter { $0.lowercased().contains("splash") },
                        disabled: model.isRunning,
                        onEdit: edited
                    )
                case .upstream:
                    ModelField(
                        modelID: $model.modelID,
                        catalog: model.catalogIDs.filter { !$0.lowercased().contains("splash") },
                        disabled: model.isRunning,
                        onEdit: edited
                    )
                case .local:
                    DirectoryField(
                        title: L10n.string("field.target"),
                        path: $model.modelDirectory,
                        prompt: L10n.string("field.target.prompt"),
                        disabled: model.isRunning,
                        onEdit: edited
                    )
                    DirectoryField(
                        title: L10n.string("field.draft"),
                        path: $model.draftDirectory,
                        prompt: L10n.string("field.draft.prompt"),
                        disabled: model.isRunning,
                        onEdit: edited
                    )
                }
            } header: {
                Label(L10n.string("section.model"), systemImage: "cube.box")
            }

            Section {
                TextField(L10n.string("field.port"), value: $model.port, format: .number)
                    .frame(width: 120)
                    .disabled(model.isRunning)
                    .onChange(of: model.port) { edited() }
                Toggle(L10n.string("field.language_only"), isOn: $model.languageOnly)
                    .disabled(model.isRunning || model.modelMode == .splash)
                    .onChange(of: model.languageOnly) { edited() }
                Picker(L10n.string("field.kv"), selection: $model.kvFormat) {
                    Text(verbatim: L10n.string("kv.int8")).tag("int8")
                    Text(verbatim: L10n.string("kv.bf16")).tag("bf16")
                }
                .pickerStyle(.segmented)
                .frame(width: 220)
                .disabled(model.isRunning)
                .onChange(of: model.kvFormat) { edited() }
            } header: {
                Label(L10n.string("section.server"), systemImage: "network")
            }

            Section {
                TextField(L10n.string("field.max_memory"), text: $model.maxMemory)
                    .disabled(model.isRunning)
                    .onChange(of: model.maxMemory) { edited() }
                TextField(L10n.string("field.max_context"), text: $model.maxContext)
                    .disabled(model.isRunning)
                    .onChange(of: model.maxContext) { edited() }
                TextField(L10n.string("field.max_cache_disk"), text: $model.maxCacheDisk)
                    .disabled(model.isRunning)
                    .onChange(of: model.maxCacheDisk) { edited() }
            } header: {
                Label(L10n.string("section.memory"), systemImage: "memorychip")
            }

            Section {
                TextField(L10n.string("field.idle_offload"), text: $model.idleOffloadSeconds)
                    .disabled(model.isRunning)
                    .onChange(of: model.idleOffloadSeconds) { edited() }
                TextField(L10n.string("field.residency"), text: $model.residencySeconds)
                    .disabled(model.isRunning)
                    .onChange(of: model.residencySeconds) { edited() }
            } header: {
                Label(L10n.string("section.idle"), systemImage: "leaf")
            } footer: {
                Text(verbatim: L10n.string("footer.idle"))
            }

            Section {
                SecureField(L10n.string("field.api_key"), text: $model.apiKey)
                    .disabled(model.isRunning)
                    .onChange(of: model.apiKey) { edited() }
                TextField(L10n.string("field.served_names"), text: $model.servedNames)
                    .disabled(model.isRunning)
                    .onChange(of: model.servedNames) { edited() }
                TextField(L10n.string("field.max_request_size"), text: $model.maxRequestSize)
                    .disabled(model.isRunning)
                    .onChange(of: model.maxRequestSize) { edited() }
                Picker(L10n.string("field.reasoning_effort"), selection: $model.reasoningEffort) {
                    Text(verbatim: L10n.string("reasoning.default")).tag("")
                    ForEach(ServerSettings.reasoningEfforts, id: \.self) { effort in
                        Text(verbatim: effort).tag(effort)
                    }
                }
                .pickerStyle(.menu)
                .disabled(model.isRunning)
                .onChange(of: model.reasoningEffort) { edited() }
            } header: {
                Label(L10n.string("section.request"), systemImage: "text.bubble")
            }

            Section {
                HStack {
                    Label(L10n.string("control.saved"), systemImage: "externaldrive.fill")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button {
                        model.restoreDefaultSettings()
                    } label: {
                        Label(L10n.string("control.defaults"), systemImage: "arrow.counterclockwise")
                    }
                    .disabled(model.isRunning)
                }
            }
        }
        .formStyle(.grouped)
        .scrollContentBackground(.hidden)
    }

    private func label(_ mode: AppModel.ModelMode) -> String {
        switch mode {
        case .splash: return L10n.string("mode.splash")
        case .upstream: return L10n.string("mode.upstream")
        case .local: return L10n.string("mode.local")
        }
    }

    /// Every field routes its edits here, so a keystroke anywhere in the form
    /// schedules one debounced write of the whole settings value.
    private func edited() {
        model.settingsEdited()
    }
}

/// A model ID field with a menu of matching catalog examples, kept in sync:
/// choosing an example replaces the current text.
private struct ModelField: View {
    @Binding var modelID: String
    let catalog: [String]
    let disabled: Bool
    var onEdit: () -> Void = {}

    var body: some View {
        HStack {
            TextField("owner/repo[:variant]", text: $modelID)
                .textFieldStyle(.roundedBorder)
                .disabled(disabled)
                .onChange(of: modelID) { onEdit() }
            if !catalog.isEmpty {
                Menu(L10n.string("examples")) {
                    ForEach(catalog, id: \.self) { identifier in
                        Button(identifier) {
                            modelID = identifier
                            onEdit()
                        }
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
    var disabled = false
    var onEdit: () -> Void = {}

    var body: some View {
        LabeledContent(title) {
            HStack {
                Text(path.isEmpty ? prompt : path)
                    .foregroundStyle(path.isEmpty ? .secondary : .primary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .frame(maxWidth: .infinity, alignment: .leading)
                Button(L10n.string("choose")) {
                    choose()
                    onEdit()
                }
                .disabled(disabled)
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