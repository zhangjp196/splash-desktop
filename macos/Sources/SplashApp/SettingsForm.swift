import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// The groups the configuration editor's rail switches between. Each one owns
/// a short description, so the rail explains the form instead of only naming
/// it.
enum SettingsSection: String, CaseIterable, Identifiable {
    case model, server, schedule, memory, cache, request

    var id: String { rawValue }

    var title: String { L10n.string("section.\(rawValue)") }
    var summary: String { L10n.string("section.\(rawValue).help") }

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

    var tint: Color {
        switch self {
        case .model: return SplashPalette.blue
        case .server: return SplashPalette.cyan
        case .schedule: return SplashPalette.amber
        case .memory: return SplashPalette.mint
        case .cache: return SplashPalette.violet
        case .request: return SplashPalette.rose
        }
    }
}

/// The configuration form: a rail of sections beside the selected section's
/// cards. It edits a draft the caller owns, so nothing is applied until the
/// caller saves it. `baseline` is the configuration the draft started from;
/// the rail marks the sections that differ from it.
struct SettingsEditor: View {
    @Binding var settings: ServerSettings
    let baseline: ServerSettings
    @State private var section: SettingsSection

    init(settings: Binding<ServerSettings>, baseline: ServerSettings, section: SettingsSection = .model) {
        self._settings = settings
        self.baseline = baseline
        self._section = State(initialValue: section)
    }

    /// What the picked model can use, so the form shows only those options.
    private var profile: ModelProfile {
        ModelProfile.of(modelID: settings.modelID, directory: settings.modelDirectory)
    }

    var body: some View {
        HStack(spacing: 0) {
            rail
            Rectangle()
                .fill(SplashPalette.border.opacity(0.35))
                .frame(width: 1)
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    sectionHeader
                    page
                    Spacer(minLength: 14)
                    reviewCard
                }
                .padding(.horizontal, 22)
                .padding(.vertical, 20)
                .frame(maxWidth: .infinity, minHeight: 540, alignment: .topLeading)
            }
        }
    }

    // MARK: Rail

    private var rail: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(L10n.string("config.sections").uppercased())
                .font(.system(size: 9.5, weight: .bold))
                .tracking(1.4)
                .foregroundStyle(SplashPalette.faint)
                .padding(.horizontal, 9)
                .padding(.bottom, 7)

            ForEach(SettingsSection.allCases) { item in
                RailItem(
                    section: item,
                    selected: section == item,
                    changed: changed(item)
                ) {
                    withAnimation(.easeOut(duration: 0.14)) { section = item }
                }
            }

            Spacer(minLength: 10)

            Text(L10n.string("config.hint"))
                .font(.system(size: 10))
                .foregroundStyle(SplashPalette.faint)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.horizontal, 9)
        }
        .padding(12)
        .frame(width: 226, alignment: .top)
    }

    /// Whether this section holds a value that differs from the configuration
    /// the editor opened on. A dot in the rail, rather than a page-wide
    /// "unsaved" flag, tells the user where the edit actually is.
    private func changed(_ item: SettingsSection) -> Bool {
        let s = settings
        let b = baseline
        switch item {
        case .model:
            return s.modelMode != b.modelMode || s.modelID != b.modelID
                || s.modelDirectory != b.modelDirectory
                || s.draftDirectory != b.draftDirectory || s.mmproj != b.mmproj
        case .server:
            return s.port != b.port || s.lanAccess != b.lanAccess
                || s.allowedHosts != b.allowedHosts || s.allowedOrigins != b.allowedOrigins
                || s.kvFormat != b.kvFormat || s.stateFormat != b.stateFormat
                || s.neuralEngine != b.neuralEngine || s.languageOnly != b.languageOnly
        case .schedule:
            return s.maxConcurrency != b.maxConcurrency || s.verifyRows != b.verifyRows
                || s.decodeShare != b.decodeShare
                || s.responsivePrefill != b.responsivePrefill
                || s.prefillSliceMilliseconds != b.prefillSliceMilliseconds
        case .memory:
            return s.maxMemory != b.maxMemory || s.maxContext != b.maxContext
                || s.expertMemory != b.expertMemory
                || s.expertMemoryGrow != b.expertMemoryGrow
                || s.pleMemory != b.pleMemory
                || s.sparseKvMemory != b.sparseKvMemory
        case .cache:
            return s.unloadIdleWeights != b.unloadIdleWeights
                || s.residencySeconds != b.residencySeconds || s.diskCache != b.diskCache
                || s.cacheIdleSpill != b.cacheIdleSpill || s.cacheIdleEvict != b.cacheIdleEvict
                || s.residencyKeepAlive != b.residencyKeepAlive || s.cacheDir != b.cacheDir
                || s.maxCacheDisk != b.maxCacheDisk
                || s.maxCacheDiskWrites != b.maxCacheDiskWrites
                || s.cacheEvictionPreset != b.cacheEvictionPreset
                || s.cacheEvictionPolicy != b.cacheEvictionPolicy
                || s.cacheEvictionProtect != b.cacheEvictionProtect
        case .request:
            return s.apiKey != b.apiKey || s.servedNames != b.servedNames
                || s.announceServedName != b.announceServedName
                || s.maxRequestSize != b.maxRequestSize
                || s.requestTimeout != b.requestTimeout
                || s.maxImagePixels != b.maxImagePixels
                || s.allowIdleSleep != b.allowIdleSleep
                || s.thinkingTokenBudget != b.thinkingTokenBudget
                || s.thinkingEndText != b.thinkingEndText
                || s.reasoningEffort != b.reasoningEffort
        }
    }

    // MARK: Page

    private var sectionHeader: some View {
        HStack(spacing: 12) {
            Image(systemName: section.systemImage)
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(.white)
                .frame(width: 34, height: 34)
                .background(
                    LinearGradient(
                        colors: [section.tint, section.tint.opacity(0.68)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    ),
                    in: RoundedRectangle(cornerRadius: 9, style: .continuous)
                )
                .shadow(color: section.tint.opacity(0.35), radius: 7, y: 3)
            VStack(alignment: .leading, spacing: 2) {
                Text(section.title)
                    .font(.system(size: 19, weight: .bold, design: .rounded))
                    .tracking(-0.2)
                Text(section.summary)
                    .font(.system(size: 11.5))
                    .foregroundStyle(SplashPalette.muted)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
        }
    }

    @ViewBuilder private var page: some View {
        switch section {
        case .model:
            modelPage
        case .server:
            VStack(alignment: .leading, spacing: 14) {
                endpointCard
                endpointNote
                runtimeCard
            }
        case .schedule:
            scheduleCard
        case .memory:
            VStack(alignment: .leading, spacing: 14) {
                budgetCard
                if profile.experts || profile.ple || profile.sparseKv {
                    streamedCard
                }
            }
        case .cache:
            VStack(alignment: .leading, spacing: 14) {
                idleCard
                diskCard
            }
        case .request:
            VStack(alignment: .leading, spacing: 14) {
                accessCard
                payloadCard
            }
        }
    }

    // MARK: Model

    @ViewBuilder private var modelPage: some View {
        // Local models only: an online model is fetched by the downloader
        // beside the configurations, and then chosen here as a local target.
        // A configuration saved by the older online mode keeps its repository
        // ID and still starts, and editing it here moves it to local.
        if settings.modelMode == .upstream {
            EditorNote(
                text: L10n.string("help.legacy_upstream"),
                tone: SplashPalette.amber,
                systemImage: "exclamationmark.triangle"
            )
        }

        EditorCard(
            title: L10n.string("config.group.files"),
            systemImage: "folder",
            tint: SplashPalette.cyan
        ) {
            // A GGUF target is a file (the first of several: its shards lie
            // beside it), an MLX target a model directory.
            SettingRow(
                title: L10n.string("field.local_kind"),
                help: L10n.string("help.local_kind"),
                controlWidth: 300
            ) {
                Picker("", selection: $settings.localKind) {
                    Text(L10n.string("models.kind.gguf")).tag("gguf")
                    Text(L10n.string("models.kind.mlx")).tag("mlx")
                }
                .pickerStyle(.segmented)
                .labelsHidden()
            }
            rowDivider
            SettingRow(
                title: L10n.string("field.target"),
                help: L10n.string("field.target.prompt"),
                controlWidth: 300
            ) {
                DirectoryPicker(
                    path: $settings.modelDirectory,
                    prompt: L10n.string("field.target.prompt")
                )
            }
            if settings.localKind == "gguf" && profile.vision {
                rowDivider
                SettingRow(
                    title: L10n.string("field.mmproj"),
                    help: L10n.string("field.mmproj.prompt"),
                    controlWidth: 300
                ) {
                    DirectoryPicker(
                        path: $settings.mmproj,
                        prompt: L10n.string("field.mmproj.prompt"),
                        files: true
                    )
                }
                .disabled(settings.languageOnly)
                .opacity(settings.languageOnly ? 0.55 : 1)
            }
        }
        if settings.localKind == "gguf" && profile.vision && settings.languageOnly {
            EditorNote(
                text: L10n.string("help.mmproj.text_only"),
                tone: SplashPalette.muted,
                systemImage: "info.circle"
            )
        }

        // DFlash2 is its own choice: a local draft folder, like any other
        // model the panel serves — a Hub one is fetched by the downloader and
        // then chosen here.
        EditorCard(
            title: L10n.string("config.group.draft"),
            systemImage: "bolt.badge.clock",
            tint: SplashPalette.violet
        ) {
            SettingRow(
                title: L10n.string("field.draft"),
                help: L10n.string("help.draft_folder"),
                controlWidth: 300
            ) {
                DirectoryPicker(
                    path: $settings.draftDirectory,
                    prompt: L10n.string("field.draft.prompt")
                )
            }
        }
    }

    // MARK: Endpoint and access

    private var endpointCard: some View {
        EditorCard(
            title: L10n.string("config.group.endpoint"),
            systemImage: "network",
            tint: SplashPalette.cyan
        ) {
            SettingColumns {
                FieldColumn(title: L10n.string("field.port"), help: L10n.string("help.port")) {
                    EditorNumberField(value: $settings.port, range: 1...65535, width: 88)
                }
                FieldColumn(title: L10n.string("field.lan_access"), help: L10n.string("help.lan_access")) {
                    Toggle("", isOn: $settings.lanAccess)
                        .labelsHidden()
                        .toggleStyle(.switch)
                        .tint(SplashPalette.cyan)
                }
            }
            rowDivider
            SettingColumns {
                FieldColumn(title: L10n.string("field.allowed_hosts"), help: L10n.string("help.allowed_hosts")) {
                    EditorField(text: $settings.allowedHosts, prompt: "studio.local, proxy.example")
                }
                FieldColumn(title: L10n.string("field.allowed_origins"), help: L10n.string("help.allowed_origins")) {
                    EditorField(text: $settings.allowedOrigins, prompt: "tauri://localhost")
                }
            }
        }
    }

    private var endpointNote: some View {
        Group {
            if settings.lanAccess && settings.apiKey.isEmpty {
                EditorNote(text: L10n.string("help.lan_access.no_key"))
            }
        }
    }

    private var accessCard: some View {
        EditorCard(
            title: L10n.string("config.group.access"),
            systemImage: "key",
            tint: SplashPalette.rose
        ) {
            SettingRow(
                title: L10n.string("field.api_key"),
                help: L10n.string("help.api_key"),
                controlWidth: 300
            ) {
                SecretField(text: $settings.apiKey)
            }
            rowDivider
            SettingColumns {
                FieldColumn(title: L10n.string("field.served_names"), help: L10n.string("help.served_names")) {
                    EditorField(text: $settings.servedNames, prompt: "alias-a, alias-b")
                }
                FieldColumn(title: L10n.string("field.announce_served_name"), help: L10n.string("help.announce_served_name")) {
                    Toggle("", isOn: $settings.announceServedName)
                        .labelsHidden()
                        .toggleStyle(.switch)
                        .tint(SplashPalette.rose)
                        .disabled(settings.servedNames.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            }
        }
    }

    // MARK: Runtime

    private var runtimeCard: some View {
        EditorCard(
            title: L10n.string("config.group.runtime"),
            systemImage: "cpu",
            tint: SplashPalette.mint
        ) {
            SettingColumns {
                FieldColumn(title: L10n.string("field.kv"), help: L10n.string("help.kv")) {
                    Picker("", selection: $settings.kvFormat) {
                        Text(verbatim: L10n.string("kv.int8")).tag("int8")
                        Text(verbatim: L10n.string("kv.int4")).tag("int4")
                        Text(verbatim: L10n.string("kv.k8v4")).tag("k8v4")
                        Text(verbatim: L10n.string("kv.bf16")).tag("bf16")
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                }
                FieldColumn(title: L10n.string("field.state_format"), help: L10n.string("help.state_format")) {
                    Picker("", selection: $settings.stateFormat) {
                        Text(verbatim: L10n.string("state.fp32")).tag("fp32")
                        Text(verbatim: L10n.string("state.bf16")).tag("bf16")
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                }
            }
            rowDivider
            SettingColumns {
                FieldColumn(title: L10n.string("field.neural_engine"), help: L10n.string("help.neural_engine")) {
                    Toggle("", isOn: $settings.neuralEngine)
                        .labelsHidden()
                        .toggleStyle(.switch)
                        .tint(SplashPalette.mint)
                }
                if profile.vision {
                    FieldColumn(title: L10n.string("field.language_only"), help: L10n.string("help.language_only")) {
                        Toggle("", isOn: $settings.languageOnly)
                            .labelsHidden()
                            .toggleStyle(.switch)
                            .tint(SplashPalette.mint)
                    }
                }
            }
        }
    }

    // MARK: Memory and scheduling

    private var budgetCard: some View {
        EditorCard(
            title: L10n.string("config.group.budget"),
            systemImage: "memorychip",
            tint: SplashPalette.mint
        ) {
            SettingColumns {
                FieldColumn(title: L10n.string("field.max_memory"), help: L10n.string("help.max_memory")) {
                    SizeField(text: $settings.maxMemory, prompt: "28G")
                }
                FieldColumn(title: L10n.string("field.max_context"), help: L10n.string("help.max_context")) {
                    EditorField(text: $settings.maxContext, prompt: "204800", monospaced: true)
                }
            }
        }
    }

    private var streamedCard: some View {
        EditorCard(
            title: L10n.string("config.group.streamed"),
            systemImage: "square.stack.3d.down.right",
            tint: SplashPalette.violet
        ) {
            SettingColumns {
                if profile.experts {
                    FieldColumn(title: L10n.string("field.expert_memory"), help: L10n.string("help.expert_memory")) {
                        SizeField(text: $settings.expertMemory, prompt: L10n.string("size.auto"))
                    }
                }
                if profile.ple {
                    FieldColumn(title: L10n.string("field.ple_memory"), help: L10n.string("help.ple_memory")) {
                        SizeField(text: $settings.pleMemory, prompt: L10n.string("size.auto"))
                    }
                }
                if profile.sparseKv {
                    FieldColumn(title: L10n.string("field.sparse_kv_memory"), help: L10n.string("help.sparse_kv_memory")) {
                        SizeField(text: $settings.sparseKvMemory, prompt: L10n.string("size.auto"))
                    }
                }
            }
            if profile.experts {
                rowDivider
                SettingRow(
                    title: L10n.string("field.expert_memory_grow"),
                    help: L10n.string("help.expert_memory_grow"),
                    controlWidth: 300
                ) {
                    Toggle("", isOn: $settings.expertMemoryGrow)
                        .labelsHidden()
                        .toggleStyle(.switch)
                        .tint(SplashPalette.violet)
                }
            }
        }
    }

    private var scheduleCard: some View {
        EditorCard(
            title: L10n.string("config.group.pacing"),
            systemImage: "speedometer",
            tint: SplashPalette.amber
        ) {
            SettingColumns {
                FieldColumn(title: L10n.string("field.max_concurrency"), help: L10n.string("help.max_concurrency")) {
                    HStack(spacing: 7) {
                        EditorNumberField(value: $settings.maxConcurrency, range: 1...4096, width: 72)
                        Stepper("", value: $settings.maxConcurrency, in: 1...4096)
                            .labelsHidden()
                    }
                }
                FieldColumn(title: L10n.string("field.verify_rows"), help: L10n.string("help.verify_rows")) {
                    Picker("", selection: $settings.verifyRows) {
                        Text(verbatim: L10n.string("verify.auto")).tag("auto")
                        Text(verbatim: L10n.string("verify.all")).tag("all")
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                }
                FieldColumn(title: L10n.string("field.decode_share"), help: L10n.string("help.decode_share")) {
                    EditorField(text: $settings.decodeShare, prompt: "0.5", monospaced: true)
                }
            }
            rowDivider
            SettingColumns {
                FieldColumn(
                    title: L10n.string("field.responsive_prefill"),
                    help: L10n.string("help.responsive_prefill")
                ) {
                    Toggle("", isOn: $settings.responsivePrefill)
                        .labelsHidden()
                        .toggleStyle(.switch)
                        .tint(SplashPalette.amber)
                }
                FieldColumn(
                    title: L10n.string("field.prefill_slice"),
                    help: L10n.string("help.prefill_slice")
                ) {
                    EditorField(
                        text: $settings.prefillSliceMilliseconds,
                        prompt: "1000",
                        monospaced: true,
                        suffix: L10n.string("unit.milliseconds")
                    )
                    .disabled(!settings.responsivePrefill)
                    .opacity(settings.responsivePrefill ? 1 : 0.45)
                }
            }
        }
    }

    private var payloadCard: some View {
        EditorCard(
            title: L10n.string("config.group.payload"),
            systemImage: "doc.text",
            tint: SplashPalette.cyan
        ) {
            SettingColumns {
                FieldColumn(title: L10n.string("field.max_request_size"), help: L10n.string("help.max_request_size")) {
                    SizeField(text: $settings.maxRequestSize, prompt: "128M")
                }
                FieldColumn(title: L10n.string("field.request_timeout"), help: L10n.string("help.request_timeout")) {
                    EditorField(text: $settings.requestTimeout, prompt: "30m", monospaced: true)
                }
                if profile.vision {
                    FieldColumn(title: L10n.string("field.max_image_pixels"), help: L10n.string("help.max_image_pixels")) {
                        EditorField(text: $settings.maxImagePixels, prompt: "4194304", monospaced: true)
                    }
                }
            }
            rowDivider
            SettingColumns {
                FieldColumn(title: L10n.string("field.allow_idle_sleep"), help: L10n.string("help.allow_idle_sleep")) {
                    Toggle("", isOn: $settings.allowIdleSleep)
                        .labelsHidden()
                        .toggleStyle(.switch)
                        .tint(SplashPalette.cyan)
                }
                if profile.thinking {
                    FieldColumn(title: L10n.string("field.reasoning_effort"), help: L10n.string("help.reasoning_effort")) {
                        Picker("", selection: $settings.reasoningEffort) {
                            Text(verbatim: L10n.string("reasoning.default")).tag("")
                            ForEach(ServerSettings.reasoningEfforts, id: \.self) { effort in
                                Text(verbatim: effort).tag(effort)
                            }
                        }
                        .pickerStyle(.menu)
                        .labelsHidden()
                    }
                }
            }
            if profile.thinking {
                rowDivider
                SettingColumns {
                    FieldColumn(title: L10n.string("field.thinking_token_budget"), help: L10n.string("help.thinking_token_budget")) {
                        EditorField(text: $settings.thinkingTokenBudget, prompt: "4096", monospaced: true)
                    }
                    FieldColumn(title: L10n.string("field.thinking_end_text"), help: L10n.string("help.thinking_end_text")) {
                        EditorField(text: $settings.thinkingEndText, prompt: L10n.string("thinking_end.prompt"))
                    }
                }
            }
        }
    }

    // MARK: Cache

    private var idleCard: some View {
        EditorCard(
            title: L10n.string("config.group.idle"),
            systemImage: "moon.zzz",
            tint: SplashPalette.violet
        ) {
            SettingRow(
                title: L10n.string("field.idle_unload"),
                help: L10n.string("help.idle_unload"),
                controlWidth: 300
            ) {
                Toggle("", isOn: $settings.unloadIdleWeights)
                    .labelsHidden()
                    .toggleStyle(.switch)
                    .tint(SplashPalette.violet)
            }
            rowDivider
            SettingColumns {
                FieldColumn(title: L10n.string("field.residency"), help: L10n.string("help.residency")) {
                    EditorField(
                        text: $settings.residencySeconds,
                        prompt: "600",
                        monospaced: true,
                        suffix: L10n.string("unit.seconds")
                    )
                    .disabled(!settings.unloadIdleWeights)
                    .opacity(settings.unloadIdleWeights ? 1 : 0.45)
                }
                FieldColumn(title: L10n.string("field.cache_idle_spill"), help: L10n.string("help.cache_idle_spill")) {
                    EditorField(text: $settings.cacheIdleSpill, prompt: "5m", monospaced: true)
                }
                FieldColumn(title: L10n.string("field.residency_keepalive"), help: L10n.string("help.residency_keepalive")) {
                    EditorField(text: $settings.residencyKeepAlive, prompt: "follows idle release", monospaced: true)
                }
            }
            rowDivider
            SettingRow(
                title: L10n.string("field.cache_idle_evict"),
                help: L10n.string("help.cache_idle_evict"),
                controlWidth: 300
            ) {
                Toggle("", isOn: $settings.cacheIdleEvict)
                    .labelsHidden()
                    .toggleStyle(.switch)
                    .tint(SplashPalette.violet)
            }
        }
    }

    private var diskCard: some View {
        EditorCard(
            title: L10n.string("config.group.disk"),
            systemImage: "internaldrive",
            tint: SplashPalette.violet
        ) {
            SettingColumns {
                FieldColumn(title: L10n.string("field.disk_cache"), help: L10n.string("help.disk_cache")) {
                    Toggle("", isOn: $settings.diskCache)
                        .labelsHidden()
                        .toggleStyle(.switch)
                        .tint(SplashPalette.violet)
                }
                FieldColumn(title: L10n.string("field.cache_quota"), help: L10n.string("help.cache_quota")) {
                    SizeField(text: $settings.maxCacheDisk, prompt: "16G")
                }
                FieldColumn(title: L10n.string("field.cache_writes"), help: L10n.string("help.cache_writes")) {
                    SizeField(text: $settings.maxCacheDiskWrites, prompt: "32G")
                }
            }
            rowDivider
            SettingRow(
                title: L10n.string("field.cache_eviction"),
                help: L10n.string("help.cache_eviction")
            ) {
                Picker("", selection: $settings.cacheEvictionPreset) {
                    Text(verbatim: L10n.string("eviction.light")).tag("light")
                    Text(verbatim: L10n.string("eviction.balanced")).tag("balanced")
                    Text(verbatim: L10n.string("eviction.strong")).tag("strong")
                    Text(verbatim: L10n.string("eviction.custom")).tag("custom")
                }
                .pickerStyle(.menu)
                .labelsHidden()
            }
            if settings.cacheEvictionPreset == "custom" {
                rowDivider
                SettingColumns {
                    FieldColumn(title: L10n.string("field.cache_policy")) {
                        Picker("", selection: $settings.cacheEvictionPolicy) {
                            Text(verbatim: L10n.string("eviction.policy.lru")).tag("lru")
                            Text(verbatim: L10n.string("eviction.policy.fifo")).tag("fifo")
                            Text(verbatim: L10n.string("eviction.policy.lfu")).tag("lfu")
                            Text(verbatim: L10n.string("eviction.policy.largest")).tag("largest")
                        }
                        .pickerStyle(.menu)
                        .labelsHidden()
                    }
                    FieldColumn(title: L10n.string("field.cache_protect")) {
                        Picker("", selection: $settings.cacheEvictionProtect) {
                            ForEach(0...8, id: \.self) { count in
                                Text(verbatim: count == 0
                                    ? L10n.string("eviction.protect.off")
                                    : String(count))
                                    .tag(count)
                            }
                        }
                        .pickerStyle(.menu)
                        .labelsHidden()
                    }
                }
            }
            if settings.diskCache {
                rowDivider
                SettingRow(
                    title: L10n.string("field.cache_dir"),
                    help: L10n.string("help.cache_dir"),
                    controlWidth: 300
                ) {
                    EditorField(text: $settings.cacheDir, prompt: L10n.string("cache_dir.prompt"))
                }
            }
        }
    }

    /// The four numbers a start would use, kept at the foot of every page so
    /// the form has an answer without leaving the section being edited.
    private var reviewCard: some View {
        EditorCard(
            title: L10n.string("config.review"),
            systemImage: "play.circle",
            tint: SplashPalette.mint
        ) {
            HStack(spacing: 0) {
                ReviewStat(title: L10n.string("live.model"), value: modelLabel)
                reviewDivider
                ReviewStat(
                    title: L10n.string("field.port"),
                    value: String(settings.port),
                    caption: settings.lanAccess
                        ? L10n.string("config.review.lan")
                        : L10n.string("config.review.local")
                )
                reviewDivider
                ReviewStat(
                    title: L10n.string("config.group.budget"),
                    value: settings.maxMemory,
                    caption: L10n.format("config.review.context", settings.maxContext)
                )
                reviewDivider
                ReviewStat(
                    title: L10n.string("field.max_concurrency"),
                    value: String(settings.maxConcurrency),
                    caption: settings.kvFormat.uppercased()
                )
            }
            .padding(.top, 7)
        }
    }

    private var modelLabel: String {
        switch settings.modelMode {
        case .upstream:
            return settings.modelID.isEmpty
                ? L10n.string("models.none")
                : settings.modelID
        case .local:
            return settings.modelDirectory.isEmpty
                ? L10n.string("models.none")
                : (settings.modelDirectory as NSString).lastPathComponent
        }
    }

    private var reviewDivider: some View {
        Rectangle()
            .fill(SplashPalette.hairline.opacity(0.55))
            .frame(width: 0.6, height: 30)
    }

    private var rowDivider: some View {
        Rectangle()
            .fill(SplashPalette.hairline.opacity(0.55))
            .frame(height: 0.6)
    }
}

/// One figure in the "when you start" strip.
private struct ReviewStat: View {
    let title: String
    let value: String
    var caption: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.system(size: 9.5, weight: .semibold))
                .foregroundStyle(SplashPalette.faint)
                .lineLimit(1)
            Text(value.isEmpty ? "—" : value)
                .font(.system(size: 13, weight: .semibold, design: .rounded))
                .lineLimit(1)
                .truncationMode(.middle)
            if let caption, !caption.isEmpty {
                Text(caption)
                    .font(.system(size: 9.5))
                    .foregroundStyle(SplashPalette.muted)
                    .lineLimit(1)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 10)
    }
}

// MARK: - Rail

/// One section in the editor's rail: an icon, a name, a one-line explanation,
/// and a dot when the section differs from the configuration it opened on.
private struct RailItem: View {
    let section: SettingsSection
    let selected: Bool
    let changed: Bool
    let action: () -> Void

    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                Image(systemName: section.systemImage)
                    .font(.system(size: 11.5, weight: .semibold))
                    .foregroundStyle(selected ? .white : section.tint)
                    .frame(width: 26, height: 26)
                    .background(
                        selected
                            ? AnyShapeStyle(
                                LinearGradient(
                                    colors: [section.tint, section.tint.opacity(0.72)],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                )
                            )
                            : AnyShapeStyle(section.tint.opacity(0.13)),
                        in: RoundedRectangle(cornerRadius: 7, style: .continuous)
                    )
                VStack(alignment: .leading, spacing: 2) {
                    Text(section.title)
                        .font(.system(size: 12, weight: selected ? .semibold : .medium))
                        .foregroundStyle(.primary)
                    Text(section.summary)
                        .font(.system(size: 9.5))
                        .foregroundStyle(SplashPalette.muted)
                        .lineLimit(2)
                        .multilineTextAlignment(.leading)
                }
                Spacer(minLength: 0)
                if changed {
                    Circle()
                        .fill(section.tint)
                        .frame(width: 5, height: 5)
                        .shadow(color: section.tint.opacity(0.7), radius: 3)
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 7)
            .background(
                selected
                    ? AnyShapeStyle(section.tint.opacity(0.14))
                    : AnyShapeStyle(hovering ? SplashPalette.surface.opacity(0.5) : Color.clear),
                in: RoundedRectangle(cornerRadius: 10, style: .continuous)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(
                        selected ? section.tint.opacity(0.38) : Color.clear,
                        lineWidth: 0.8
                    )
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .animation(.easeOut(duration: 0.12), value: hovering)
    }
}

// MARK: - Cards and rows

/// A titled card in the form: a small section label above a stack of rows.
private struct EditorCard<Content: View>: View {
    let title: String
    let systemImage: String
    var tint: Color = SplashPalette.cyan
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            SectionLabel(title: title, systemImage: systemImage, tint: tint)
                .padding(.bottom, 3)
            content
        }
        .padding(14)
        .glassPanel(radius: 14)
    }
}

/// A row of two or three controls, each labelled above its value: the form
/// uses these where a single setting's own row would waste the width.
private struct SettingColumns<Content: View>: View {
    @ViewBuilder var content: Content

    var body: some View {
        HStack(alignment: .top, spacing: 16) {
            content
        }
        .padding(.vertical, 9)
    }
}

/// One labelled control in a SettingColumns row. Its title and explanation
/// sit above the control, so several read as one band rather than a form.
private struct FieldColumn<Control: View>: View {
    let title: String
    var help: String?
    @ViewBuilder var control: Control

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(spacing: 4) {
                Text(title)
                    .font(.system(size: 10.5, weight: .semibold))
                    .foregroundStyle(SplashPalette.muted)
                    .lineLimit(1)
                if let help, !help.isEmpty {
                    Image(systemName: "info.circle")
                        .font(.system(size: 9))
                        .foregroundStyle(SplashPalette.faint)
                        .help(help)
                }
            }
            control
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// A labelled row: the name and its explanation on the left, the control on
/// the right at a shared width so the form scans as a column.
private struct SettingRow<Control: View>: View {
    let title: String
    var help: String?
    var controlWidth: CGFloat = 250
    @ViewBuilder var control: Control

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 16) {
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 4) {
                    Text(title)
                        .font(.system(size: 12.5, weight: .medium))
                    if let help, !help.isEmpty {
                        Image(systemName: "info.circle")
                            .font(.system(size: 10))
                            .foregroundStyle(SplashPalette.faint)
                            .help(help)
                    }
                }
                if let help, !help.isEmpty {
                    Text(help)
                        .font(.system(size: 10.5))
                        .foregroundStyle(SplashPalette.muted)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer(minLength: 12)
            control
                .frame(width: controlWidth, alignment: .trailing)
        }
        .padding(.vertical, 9)
    }
}

/// A short note under a page or inside a card: a warning, or a quiet hint.
private struct EditorNote: View {
    let text: String
    var tone: Color = SplashPalette.amber
    var systemImage: String = "exclamationmark.triangle.fill"

    var body: some View {
        HStack(alignment: .top, spacing: 7) {
            Image(systemName: systemImage)
                .font(.system(size: 10.5, weight: .semibold))
            Text(text)
                .font(.system(size: 11))
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
        .foregroundStyle(tone)
        .padding(.horizontal, 11)
        .padding(.vertical, 9)
        .background(tone.opacity(0.10), in: RoundedRectangle(cornerRadius: 9, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 9, style: .continuous)
                .strokeBorder(tone.opacity(0.24), lineWidth: 0.7)
        )
    }
}

/// The two mode choices, as cards rather than a menu: the difference between
/// downloading a model and serving a folder is worth a sentence.
/// The rounded field chrome every editor text field shares, including the
/// focus edge the plain text style otherwise hides.
private struct EditorFieldShell: ViewModifier {
    var focused = false

    func body(content: Content) -> some View {
        content
            .padding(.horizontal, 9)
            .padding(.vertical, 6)
            .background(
                SplashPalette.canvas.opacity(0.55),
                in: RoundedRectangle(cornerRadius: 8, style: .continuous)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .strokeBorder(
                        focused
                            ? SplashPalette.blue.opacity(0.75)
                            : SplashPalette.border.opacity(0.55),
                        lineWidth: focused ? 1.2 : 0.8
                    )
            )
            .shadow(
                color: focused ? SplashPalette.blue.opacity(0.18) : .clear,
                radius: 4
            )
    }
}

/// A monospaced single-line field the settings form and the downloader
/// sheet share.
struct EditorField: View {
    @Binding var text: String
    var prompt = ""
    var monospaced = false
    var suffix: String?
    @FocusState private var focused: Bool

    var body: some View {
        HStack(spacing: 6) {
            TextField("", text: $text, prompt: Text(prompt))
                .textFieldStyle(.plain)
                .font(.system(size: 12, design: monospaced ? .monospaced : .default))
                .focused($focused)
            if let suffix, !suffix.isEmpty {
                Text(suffix)
                    .font(.system(size: 10))
                    .foregroundStyle(SplashPalette.faint)
            }
            if !text.isEmpty {
                Button {
                    text = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 10))
                        .foregroundStyle(SplashPalette.faint)
                }
                .buttonStyle(.plain)
                .help(L10n.string("field.clear"))
            }
        }
        .modifier(EditorFieldShell(focused: focused))
    }
}

private struct EditorNumberField: View {
    @Binding var value: Int
    var range: ClosedRange<Int>?
    var width: CGFloat = 90
    @FocusState private var focused: Bool

    var body: some View {
        TextField("", value: $value, format: .number.grouping(.never))
            .textFieldStyle(.plain)
            .font(.system(size: 12, design: .monospaced))
            .multilineTextAlignment(.trailing)
            .focused($focused)
            .modifier(EditorFieldShell(focused: focused))
            .frame(width: width)
            .onSubmit {
                if let range {
                    value = min(max(value, range.lowerBound), range.upperBound)
                }
            }
    }
}

/// A size field (`16G`, `off`, or empty for automatic) with the unit hint
/// kept outside the text, so the value stays easy to copy.
private struct SizeField: View {
    @Binding var text: String
    var prompt: String

    var body: some View {
        EditorField(text: $text, prompt: prompt, monospaced: true)
    }
}

private struct SecretField: View {
    @Binding var text: String
    @State private var revealed = false
    @FocusState private var focused: Bool

    var body: some View {
        HStack(spacing: 6) {
            Group {
                if revealed {
                    TextField("", text: $text, prompt: Text("sk-…"))
                } else {
                    SecureField("", text: $text, prompt: Text("sk-…"))
                }
            }
            .textFieldStyle(.plain)
            .font(.system(size: 12, design: .monospaced))
            .focused($focused)
            Button {
                revealed.toggle()
            } label: {
                Image(systemName: revealed ? "eye.slash" : "eye")
                    .font(.system(size: 10.5))
                    .foregroundStyle(SplashPalette.faint)
            }
            .buttonStyle(.plain)
            .help(L10n.string(revealed ? "field.hide_key" : "field.show_key"))
        }
        .modifier(EditorFieldShell(focused: focused))
    }
}

/// A path field with a browse button, and a clear button inside the field.
private struct DirectoryPicker: View {
    @Binding var path: String
    let prompt: String
    /// Choose a GGUF file rather than a directory: only a vision projector
    /// does, since a target is a folder.
    var files = false

    var body: some View {
        HStack(spacing: 6) {
            EditorField(text: $path, prompt: prompt, monospaced: true)
            Button {
                select()
            } label: {
                Image(systemName: files ? "doc.badge.plus" : "folder")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(SplashPalette.cyan)
                    .frame(width: 28, height: 28)
                    .background(
                        SplashPalette.surface.opacity(0.72),
                        in: RoundedRectangle(cornerRadius: 7, style: .continuous)
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 7, style: .continuous)
                            .strokeBorder(SplashPalette.border.opacity(0.5), lineWidth: 0.7)
                    )
            }
            .buttonStyle(.plain)
            .help(L10n.string("field.browse"))
        }
    }

    private func select() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = files
        panel.canChooseDirectories = !files
        panel.allowsMultipleSelection = false
        if files {
            panel.allowedContentTypes = [UTType(filenameExtension: "gguf")].compactMap { $0 }
        }
        panel.prompt = L10n.string("field.browse")
        if panel.runModal() == .OK, let url = panel.urls.first {
            path = url.path
        }
    }
}
