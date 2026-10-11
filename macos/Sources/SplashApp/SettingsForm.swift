import AppKit
import SwiftUI

/// The configuration's settings, split across tabs like a preferences
/// window: what model, how the server serves it, memory, idle behaviour and
/// what a request may carry. Bound to a configuration the models pane edits.
struct SettingsTabs: View {
    @Binding var settings: ServerSettings
    var disabled = false

    var body: some View {
        TabView {
            tab { modelSection }
                .tabItem { Label(L10n.string("section.model"), systemImage: "cube.box") }
            tab { serverSection }
                .tabItem { Label(L10n.string("section.server"), systemImage: "network") }
            tab { memorySection }
                .tabItem { Label(L10n.string("section.memory"), systemImage: "memorychip") }
            tab { idleSection }
                .tabItem { Label(L10n.string("section.idle"), systemImage: "leaf") }
            tab { requestsSection }
                .tabItem { Label(L10n.string("section.request"), systemImage: "text.bubble") }
        }
    }

    private func tab<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        Form { content() }
            .formStyle(.grouped)
            .scrollContentBackground(.hidden)
    }

    // MARK: Sections

    @ViewBuilder private var modelSection: some View {
        Section {
            Picker(L10n.string("model.picker"), selection: $settings.modelMode) {
                ForEach(AppModel.ModelMode.allCases) { mode in
                    Text(verbatim: label(mode)).tag(mode)
                }
            }
            .pickerStyle(.menu)
            .disabled(disabled)

            switch settings.modelMode {
            case .upstream:
                ModelField(modelID: $settings.modelID, catalog: Runtime.catalog, disabled: disabled)
            case .local:
                DirectoryField(
                    title: L10n.string("field.target"),
                    path: $settings.modelDirectory,
                    prompt: L10n.string("field.target.prompt"),
                    disabled: disabled
                )
                DirectoryField(
                    title: L10n.string("field.draft"),
                    path: $settings.draftDirectory,
                    prompt: L10n.string("field.draft.prompt"),
                    disabled: disabled
                )
            }
        } footer: {
            Text(verbatim: L10n.string("footer.model"))
        }
    }

    @ViewBuilder private var serverSection: some View {
        Section {
            TextField(L10n.string("field.port"), value: $settings.port, format: .number)
                .disabled(disabled)
            Picker(L10n.string("field.kv"), selection: $settings.kvFormat) {
                Text(verbatim: L10n.string("kv.int8")).tag("int8")
                Text(verbatim: L10n.string("kv.bf16")).tag("bf16")
            }
            .pickerStyle(.segmented)
            .disabled(disabled)
            Toggle(L10n.string("field.language_only"), isOn: $settings.languageOnly)
                .disabled(disabled)
            Toggle(L10n.string("field.neural_engine"), isOn: $settings.neuralEngine)
                .disabled(disabled)
        } footer: {
            Text(verbatim: L10n.string("help.neural_engine"))
        }
    }

    @ViewBuilder private var memorySection: some View {
        Section {
            TextField(L10n.string("field.max_memory"), text: $settings.maxMemory)
                .disabled(disabled)
            TextField(L10n.string("field.max_context"), text: $settings.maxContext)
                .disabled(disabled)
            TextField(L10n.string("field.max_cache_disk"), text: $settings.maxCacheDisk)
                .disabled(disabled)
        }
    }

    @ViewBuilder private var idleSection: some View {
        Section {
            TextField(L10n.string("field.idle_offload"), text: $settings.idleOffloadSeconds)
                .disabled(disabled)
            TextField(L10n.string("field.residency"), text: $settings.residencySeconds)
                .disabled(disabled)
        } footer: {
            Text(verbatim: L10n.string("footer.idle"))
        }
    }

    @ViewBuilder private var requestsSection: some View {
        Section {
            SecureField(L10n.string("field.api_key"), text: $settings.apiKey)
                .disabled(disabled)
            TextField(L10n.string("field.served_names"), text: $settings.servedNames)
                .disabled(disabled)
            TextField(L10n.string("field.max_request_size"), text: $settings.maxRequestSize)
                .disabled(disabled)
            Picker(L10n.string("field.reasoning_effort"), selection: $settings.reasoningEffort) {
                Text(verbatim: L10n.string("reasoning.default")).tag("")
                ForEach(ServerSettings.reasoningEfforts, id: \.self) { effort in
                    Text(verbatim: effort).tag(effort)
                }
            }
            .pickerStyle(.menu)
            .disabled(disabled)
        }
    }

    private func label(_ mode: AppModel.ModelMode) -> String {
        switch mode {
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
        LabeledContent("Model") {
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
}

/// A read-only path with a Choose button; the panel asks for a directory.
private struct DirectoryField: View {
    let title: String
    @Binding var path: String
    let prompt: String
    var disabled = false

    var body: some View {
        LabeledContent(title) {
            HStack {
                Text(path.isEmpty ? prompt : path)
                    .foregroundStyle(path.isEmpty ? .secondary : .primary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .frame(maxWidth: .infinity, alignment: .leading)
                Button(L10n.string("choose")) { choose() }
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
