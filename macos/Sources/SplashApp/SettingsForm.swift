import SwiftUI

/// The configuration's settings, grouped in a rail the editor switches
/// between: how the server serves, admission, memory, idle behaviour and what
/// a request may carry. Bound to the settings the models pane edits.
struct SettingsTabs: View {
    @Binding var settings: ServerSettings
    var disabled = false
    @State private var section: Pane? = .server

    /// The form's groups. The admission limit sits under Scheduling with the
    /// rest of how requests are taken, rather than hidden among the server's
    /// transport options.
    private enum Pane: String, CaseIterable, Identifiable {
        case server, schedule, memory, cache, request

        var id: String { rawValue }

        var title: String { L10n.string("section.\(rawValue)") }

        var systemImage: String {
            switch self {
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
        switch section ?? .server {
        case .server: serverSection
        case .schedule: scheduleSection
        case .memory: memorySection
        case .cache: cacheSection
        case .request: requestsSection
        }
    }

    // MARK: Sections

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
                Text(verbatim: L10n.string("kv.int4")).tag("int4")
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
}
