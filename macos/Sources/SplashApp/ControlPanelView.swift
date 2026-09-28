import AppKit
import SwiftUI

/// Model source and server settings. A Splash package carries its own DFlash2
/// draft and vision, an upstream MLX/GGUF model's draft is selected by the
/// installer, and a local directory names one unless it is a Splash package.
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
                TextField(L10n.string("field.idle_offload"), text: $model.idleOffloadSeconds)
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
                        .controlSize(.large)
                        .disabled(!model.canStart)
                    Spacer()
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