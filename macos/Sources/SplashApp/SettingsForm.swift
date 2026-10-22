import AppKit
import SwiftUI

/// The configuration's settings, grouped in a rail the editor switches
/// between: what model, how the server serves it, admission, memory, idle
/// behaviour and what a request may carry. Bound to a configuration the
/// models pane edits.
struct SettingsTabs: View {
    @Binding var settings: ServerSettings
    var disabled = false
    @State private var section: Pane? = .model

    /// The form's groups. The admission limit sits under Scheduling with the
    /// rest of how requests are taken, rather than hidden among the server's
    /// transport options.
    private enum Pane: String, CaseIterable, Identifiable {
        case model, server, schedule, memory, cache, request

        var id: String { rawValue }

        var title: String { L10n.string("section.\(rawValue)") }

        var systemImage: String {
            switch self {
            case .model: return "cube.box"
            case .server: return "network"
            case .schedule: return "speedometer"
            case .memory: return "memorychip"
            case .cache: return "internaldrive"
            case .request: return "text.bubble"
            }
        }
    }

    var body: some View {
        HStack(spacing: 0) {
            List(selection: $section) {
                ForEach(Pane.allCases) { item in
                    Label(item.title, systemImage: item.systemImage).tag(item)
                }
            }
            .listStyle(.sidebar)
            .frame(width: 176)

            Divider()

            Form { current }
                .formStyle(.grouped)
                .scrollContentBackground(.hidden)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
    }

    @ViewBuilder private var current: some View {
        switch section ?? .model {
        case .model: modelSection
        case .server: serverSection
        case .schedule: scheduleSection
        case .memory: memorySection
        case .cache: cacheSection
        case .request: requestsSection
        }
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
                    label: L10n.string("field.target"),
                    prompt: L10n.string("field.target.prompt"),
                    path: $settings.modelDirectory,
                    disabled: disabled
                )
                DirectoryField(
                    label: L10n.string("field.draft"),
                    prompt: L10n.string("field.draft.prompt"),
                    path: $settings.draftDirectory,
                    disabled: disabled
                )
            }
        }
    }

    @ViewBuilder private var serverSection: some View {
        Section {
            TextField(L10n.string("field.port"), value: $settings.port, format: .number)
                .disabled(disabled)
            Toggle(L10n.string("field.lan_access"), isOn: $settings.lanAccess)
                .disabled(disabled)
            TextField(L10n.string("field.allowed_hosts"), text: $settings.allowedHosts)
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
            if settings.lanAccess && settings.apiKey.isEmpty {
                Label(
                    L10n.string("help.lan_access.no_key"),
                    systemImage: "exclamationmark.triangle.fill"
                )
                .foregroundStyle(.orange)
            }
        }
    }

    @ViewBuilder private var scheduleSection: some View {
        Section {
            TextField(L10n.string("field.max_concurrency"), value: $settings.maxConcurrency, format: .number)
                .disabled(disabled)
            Picker(L10n.string("field.verify_rows"), selection: $settings.verifyRows) {
                Text(verbatim: L10n.string("verify.auto")).tag("auto")
                Text(verbatim: L10n.string("verify.all")).tag("all")
            }
            .pickerStyle(.segmented)
            .disabled(disabled)
        }
    }

    @ViewBuilder private var memorySection: some View {
        Section {
            TextField(L10n.string("field.max_memory"), text: $settings.maxMemory)
                .disabled(disabled)
            TextField(L10n.string("field.max_context"), text: $settings.maxContext)
                .disabled(disabled)
            TextField(L10n.string("field.expert_memory"), text: $settings.expertMemory)
                .disabled(disabled)
            TextField(L10n.string("field.ple_memory"), text: $settings.pleMemory)
                .disabled(disabled)
            TextField(L10n.string("field.sparse_kv_memory"), text: $settings.sparseKvMemory)
                .disabled(disabled)
        }
    }

    @ViewBuilder private var cacheSection: some View {
        Section {
            TextField(L10n.string("field.residency"), text: $settings.residencySeconds)
                .disabled(disabled)
            Toggle(L10n.string("field.disk_cache"), isOn: $settings.diskCache)
                .disabled(disabled)
            TextField(L10n.string("field.cache_quota"), text: $settings.maxCacheDisk)
                .disabled(disabled || !settings.diskCache)
            TextField(L10n.string("field.cache_writes"), text: $settings.maxCacheDiskWrites)
                .disabled(disabled || !settings.diskCache)
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
                if !modelID.isEmpty {
                    Button {
                        modelID = ""
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.borderless)
                    .help(L10n.string("field.clear"))
                    .disabled(disabled)
                }
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

/// A directory path field with a browse button (NSOpenPanel) and a clear button.
private struct DirectoryField: View {
    let label: String
    let prompt: String
    @Binding var path: String
    let disabled: Bool

    var body: some View {
        HStack(spacing: 6) {
            TextField(label, text: $path, prompt: Text(prompt))
                .textFieldStyle(.roundedBorder)
                .disabled(disabled)
            Button(L10n.string("field.browse")) {
                selectFolder()
            }
            .disabled(disabled)
            if !path.isEmpty {
                Button {
                    path = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.borderless)
                .help(L10n.string("field.clear"))
                .disabled(disabled)
            }
        }
    }

    private func selectFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = L10n.string("field.browse")
        if panel.runModal() == .OK, let url = panel.url {
            path = url.path
        }
    }
}
