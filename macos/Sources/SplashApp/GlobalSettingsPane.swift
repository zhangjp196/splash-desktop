import SwiftUI

/// The groups the app-wide settings page switches between. They describe the
/// server every configuration shares, not the model one configuration serves.
enum GlobalSection: String, CaseIterable, Identifiable, RailSection {
    case application, connection, access, storage

    var id: String { rawValue }

    var title: String { L10n.string("settings.section.\(rawValue)") }
    var summary: String { L10n.string("settings.section.\(rawValue).help") }

    var systemImage: String {
        switch self {
        case .application: return "macwindow"
        case .connection: return "network"
        case .access: return "key"
        case .storage: return "internaldrive"
        }
    }

    var tint: Color {
        switch self {
        case .application: return SplashPalette.blue
        case .connection: return SplashPalette.cyan
        case .access: return SplashPalette.rose
        case .storage: return SplashPalette.violet
        }
    }
}

/// The Settings tab: the endpoint, the API's access and request limits, the
/// window's own preferences and the cache — everything a launch needs that is
/// true of the app rather than of the model it serves. One value
/// (`model.global`) backs the page, so an edit is an edit of the next start's
/// command, and every configuration uses it.
struct GlobalSettingsPane: View {
    @EnvironmentObject private var model: AppModel
    @AppStorage("appearance") private var appearanceRaw = AppAppearance.dark.rawValue
    @State private var section: GlobalSection
    @State private var confirmReset = false

    init(section: GlobalSection = .application) {
        _section = State(initialValue: section)
    }

    /// What the model the app currently points at can use, so the image-pixel
    /// row appears only when it would reach the engine. Switching models in
    /// the Models tab changes this page's rows with it.
    private var profile: ModelProfile {
        ModelProfile.of(modelID: model.modelID, directory: model.modelDirectory)
    }

    private var appearance: AppAppearance {
        AppAppearance(rawValue: appearanceRaw) ?? .dark
    }

    /// The picker writes the choice and makes it take effect at once; the
    /// toolbar's own menu does the same, and both share `model.language`.
    private var language: Binding<AppModel.InterfaceLanguage> {
        Binding(
            get: { model.language },
            set: { language in
                model.language = language
                model.applyLanguage()
            }
        )
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            PaneHeader(
                title: L10n.string("view.settings"),
                subtitle: L10n.string("settings.subtitle"),
                eyebrow: L10n.string("settings.eyebrow")
            ) {
                if model.isRunning && section != .application {
                    Pill(text: L10n.string("settings.next_start"), tint: SplashPalette.amber)
                }
                if section != .application {
                    Button {
                        confirmReset = true
                    } label: {
                        Label(L10n.string("control.defaults"), systemImage: "arrow.counterclockwise")
                            .font(.system(size: 12, weight: .semibold))
                    }
                    .buttonStyle(.bordered)
                    .help(L10n.string("settings.reset.help"))
                }
            }

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
                        if section == .application {
                            applicationReviewCard
                        } else {
                            reviewCard
                        }
                    }
                    .padding(.horizontal, 22)
                    .padding(.vertical, 20)
                    .frame(maxWidth: .infinity, alignment: .topLeading)
                }
                .scrollIndicators(.hidden)
            }
            .background(SplashPalette.surface.opacity(0.18), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
        .padding(Theme.pagePadding)
        .frame(maxWidth: Theme.contentWidth, alignment: .leading)
        .frame(maxWidth: .infinity, alignment: .topLeading)
        .onChange(of: model.global) { _, _ in
            model.settingsEdited()
        }
        .confirmationDialog(
            L10n.string("settings.reset.title"),
            isPresented: $confirmReset,
            titleVisibility: .visible
        ) {
            Button(L10n.string("control.defaults"), role: .destructive) {
                model.restoreDefaultGlobalSettings()
            }
            Button(L10n.string("config.cancel"), role: .cancel) {}
        } message: {
            Text(L10n.string("settings.reset.message"))
        }
    }

    // MARK: Rail

    private var rail: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(L10n.string("settings.sections").uppercased())
                .font(.system(size: 9.5, weight: .bold))
                .tracking(1.4)
                .foregroundStyle(SplashPalette.faint)
                .padding(.horizontal, 9)
                .padding(.bottom, 7)

            ForEach(GlobalSection.allCases) { item in
                RailItem(
                    section: item,
                    selected: section == item
                ) {
                    withAnimation(.easeOut(duration: 0.14)) { section = item }
                }
            }

            Spacer(minLength: 10)

            Text(L10n.string("settings.hint"))
                .font(.system(size: 10))
                .foregroundStyle(SplashPalette.faint)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.horizontal, 9)
        }
        .padding(12)
        .frame(width: 226, alignment: .top)
    }

    // MARK: Page

    private var sectionHeader: some View {
        HStack(spacing: 12) {
            Image(systemName: section.systemImage)
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(.white)
                .frame(width: 34, height: 34)
                .background(
                    // A flat tint fill rather than a gradient: the header
                    // re-renders on every section switch, and a gradient
                    // would blend an extra layer for a two-state chip.
                    section.tint.opacity(0.85),
                    in: RoundedRectangle(cornerRadius: 9, style: .continuous)
                )
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
        case .application:
            applicationPage
        case .connection:
            VStack(alignment: .leading, spacing: 14) {
                endpointCard
                endpointNote
            }
        case .access:
            VStack(alignment: .leading, spacing: 14) {
                accessCard
                payloadCard
            }
        case .storage:
            VStack(alignment: .leading, spacing: 14) {
                idleCard
                diskCard
            }
        }
    }

    // MARK: Application

    private var applicationPage: some View {
        VStack(alignment: .leading, spacing: 14) {
            EditorCard(
                title: L10n.string("settings.group.application"),
                systemImage: "macwindow",
                tint: SplashPalette.blue
            ) {
                SettingRow(
                    title: L10n.string("appearance.title"),
                    help: L10n.string("settings.help.appearance"),
                    controlWidth: 300
                ) {
                    Picker("", selection: $appearanceRaw) {
                        ForEach(AppAppearance.allCases) { option in
                            Text(option.title).tag(option.rawValue)
                        }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                }
                rowDivider
                SettingRow(
                    title: L10n.string("settings.language"),
                    help: L10n.string("settings.help.language"),
                    controlWidth: 300
                ) {
                    Picker("", selection: language) {
                        ForEach(AppModel.InterfaceLanguage.allCases) { option in
                            Text(verbatim: option.title).tag(option)
                        }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                }
            }

            EditorCard(
                title: L10n.string("download.source.title"),
                systemImage: "arrow.down.circle",
                tint: SplashPalette.cyan
            ) {
                SettingRow(
                    title: L10n.string("download.base"),
                    help: L10n.string("download.base.help"),
                    controlWidth: 300
                ) {
                    DirectoryPicker(
                        path: $model.source.baseDirectory,
                        prompt: L10n.string("settings.source.base.prompt")
                    )
                }
                rowDivider
                SettingRow(
                    title: L10n.string("download.endpoint"),
                    help: L10n.string("download.endpoint.help"),
                    controlWidth: 300
                ) {
                    EditorField(
                        text: $model.source.hubEndpoint,
                        prompt: "https://huggingface.co",
                        monospaced: true
                    )
                }
                rowDivider
                Text(L10n.string("download.source.help"))
                    .font(.system(size: 10.5))
                    .foregroundStyle(SplashPalette.muted)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.vertical, 7)
            }
        }
        .onChange(of: model.source) { _, _ in model.saveSource() }
    }

    // MARK: Connection

    private var endpointCard: some View {
        EditorCard(
            title: L10n.string("config.group.endpoint"),
            systemImage: "network",
            tint: SplashPalette.cyan
        ) {
            SettingColumns {
                FieldColumn(title: L10n.string("field.port"), help: L10n.string("settings.help.port")) {
                    EditorNumberField(value: $model.global.port, range: 1...65535, width: 88)
                }
                FieldColumn(title: L10n.string("field.lan_access"), help: L10n.string("settings.help.lan_access")) {
                    Toggle("", isOn: $model.global.lanAccess)
                        .labelsHidden()
                        .toggleStyle(.switch)
                        .tint(SplashPalette.cyan)
                }
            }
            rowDivider
            SettingColumns {
                FieldColumn(title: L10n.string("field.allowed_hosts"), help: L10n.string("help.allowed_hosts")) {
                    EditorField(text: $model.global.allowedHosts, prompt: "studio.local, proxy.example")
                }
                FieldColumn(title: L10n.string("field.allowed_origins"), help: L10n.string("help.allowed_origins")) {
                    EditorField(text: $model.global.allowedOrigins, prompt: "tauri://localhost")
                }
            }
        }
    }

    private var endpointNote: some View {
        Group {
            if model.global.lanAccess && model.global.apiKey.isEmpty {
                EditorNote(text: L10n.string("help.lan_access.no_key"))
            }
        }
    }

    // MARK: Access

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
                SecretField(text: $model.global.apiKey)
            }
            rowDivider
            SettingColumns {
                FieldColumn(title: L10n.string("field.served_names"), help: L10n.string("help.served_names")) {
                    EditorField(text: $model.global.servedNames, prompt: "alias-a, alias-b")
                }
                FieldColumn(title: L10n.string("field.announce_served_name"), help: L10n.string("help.announce_served_name")) {
                    Toggle("", isOn: $model.global.announceServedName)
                        .labelsHidden()
                        .toggleStyle(.switch)
                        .tint(SplashPalette.rose)
                        .disabled(model.global.servedNames.trimmingCharacters(in: .whitespaces).isEmpty)
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
                    SizeField(text: $model.global.maxRequestSize, prompt: "128M")
                }
                FieldColumn(title: L10n.string("field.request_timeout"), help: L10n.string("help.request_timeout")) {
                    EditorField(text: $model.global.requestTimeout, prompt: "30m", monospaced: true)
                }
                if profile.vision {
                    FieldColumn(title: L10n.string("field.max_image_pixels"), help: L10n.string("help.max_image_pixels")) {
                        EditorField(text: $model.global.maxImagePixels, prompt: "4194304", monospaced: true)
                    }
                }
            }
            rowDivider
            SettingColumns {
                FieldColumn(title: L10n.string("field.allow_idle_sleep"), help: L10n.string("help.allow_idle_sleep")) {
                    Toggle("", isOn: $model.global.allowIdleSleep)
                        .labelsHidden()
                        .toggleStyle(.switch)
                        .tint(SplashPalette.cyan)
                }
            }
        }
    }

    // MARK: Storage

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
                Toggle("", isOn: $model.global.unloadIdleWeights)
                    .labelsHidden()
                    .toggleStyle(.switch)
                    .tint(SplashPalette.violet)
            }
            rowDivider
            SettingColumns {
                FieldColumn(title: L10n.string("field.residency"), help: L10n.string("help.residency")) {
                    EditorField(
                        text: $model.global.residencySeconds,
                        prompt: "600",
                        monospaced: true,
                        suffix: L10n.string("unit.seconds")
                    )
                    .disabled(!model.global.unloadIdleWeights)
                    .opacity(model.global.unloadIdleWeights ? 1 : 0.45)
                }
                FieldColumn(title: L10n.string("field.cache_idle_spill"), help: L10n.string("help.cache_idle_spill")) {
                    EditorField(text: $model.global.cacheIdleSpill, prompt: "5m", monospaced: true)
                }
                FieldColumn(title: L10n.string("field.residency_keepalive"), help: L10n.string("help.residency_keepalive")) {
                    EditorField(text: $model.global.residencyKeepAlive, prompt: L10n.string("residency.prompt"), monospaced: true)
                }
            }
            rowDivider
            SettingRow(
                title: L10n.string("field.cache_idle_evict"),
                help: L10n.string("help.cache_idle_evict"),
                controlWidth: 300
            ) {
                Toggle("", isOn: $model.global.cacheIdleEvict)
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
                    Toggle("", isOn: $model.global.diskCache)
                        .labelsHidden()
                        .toggleStyle(.switch)
                        .tint(SplashPalette.violet)
                }
                FieldColumn(title: L10n.string("field.cache_quota"), help: L10n.string("help.cache_quota")) {
                    SizeField(text: $model.global.maxCacheDisk, prompt: "16G")
                }
                FieldColumn(title: L10n.string("field.cache_writes"), help: L10n.string("help.cache_writes")) {
                    SizeField(text: $model.global.maxCacheDiskWrites, prompt: "32G")
                }
            }
            rowDivider
            SettingRow(
                title: L10n.string("field.cache_eviction"),
                help: L10n.string("help.cache_eviction")
            ) {
                Picker("", selection: $model.global.cacheEvictionPreset) {
                    Text(verbatim: L10n.string("eviction.light")).tag("light")
                    Text(verbatim: L10n.string("eviction.balanced")).tag("balanced")
                    Text(verbatim: L10n.string("eviction.strong")).tag("strong")
                    Text(verbatim: L10n.string("eviction.custom")).tag("custom")
                }
                .pickerStyle(.menu)
                .labelsHidden()
            }
            if model.global.cacheEvictionPreset == "custom" {
                rowDivider
                SettingColumns {
                    FieldColumn(title: L10n.string("field.cache_policy")) {
                        Picker("", selection: $model.global.cacheEvictionPolicy) {
                            Text(verbatim: L10n.string("eviction.policy.lru")).tag("lru")
                            Text(verbatim: L10n.string("eviction.policy.fifo")).tag("fifo")
                            Text(verbatim: L10n.string("eviction.policy.lfu")).tag("lfu")
                            Text(verbatim: L10n.string("eviction.policy.largest")).tag("largest")
                        }
                        .pickerStyle(.menu)
                        .labelsHidden()
                    }
                    FieldColumn(title: L10n.string("field.cache_protect")) {
                        Picker("", selection: $model.global.cacheEvictionProtect) {
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
            if model.global.diskCache {
                rowDivider
                SettingRow(
                    title: L10n.string("field.cache_dir"),
                    help: L10n.string("help.cache_dir"),
                    controlWidth: 300
                ) {
                    EditorField(text: $model.global.cacheDir, prompt: L10n.string("cache_dir.prompt"))
                }
            }
        }
    }

    // MARK: Review

    /// What the app itself will do: how it looks, what language it speaks and
    /// where a download lands. The server figures on the other sections do not
    /// belong here.
    private var applicationReviewCard: some View {
        EditorCard(
            title: L10n.string("settings.review.application"),
            systemImage: "macwindow",
            tint: SplashPalette.blue
        ) {
            HStack(spacing: 0) {
                ReviewStat(title: L10n.string("appearance.title"), value: appearance.title)
                reviewDivider
                ReviewStat(title: L10n.string("settings.language"), value: model.language.title)
                reviewDivider
                ReviewStat(
                    title: L10n.string("download.base"),
                    value: model.source.baseDirectory.isEmpty
                        ? L10n.string("download.base.default")
                        : model.source.effectiveBaseDescription
                )
                reviewDivider
                ReviewStat(
                    title: L10n.string("download.endpoint"),
                    value: model.source.hubEndpoint.isEmpty
                        ? "huggingface.co"
                        : model.source.hubEndpoint
                )
            }
            .padding(.top, 7)
        }
    }

    /// The command a start would build, at a glance: where it listens, how many
    /// requests it admits, and whether the SSD cache is on. The page has no
    /// model figures because none of these settings is one.
    private var reviewCard: some View {
        EditorCard(
            title: L10n.string("settings.review"),
            systemImage: "globe",
            tint: SplashPalette.cyan
        ) {
            HStack(spacing: 0) {
                ReviewStat(
                    title: L10n.string("field.port"),
                    value: String(model.global.port),
                    caption: model.global.lanAccess
                        ? L10n.string("config.review.lan")
                        : L10n.string("config.review.local")
                )
                reviewDivider
                ReviewStat(
                    title: L10n.string("field.api_key"),
                    value: model.global.apiKey.isEmpty
                        ? L10n.string("settings.review.off")
                        : L10n.string("settings.review.on")
                )
                reviewDivider
                ReviewStat(
                    title: L10n.string("field.disk_cache"),
                    value: model.global.diskCache
                        ? L10n.string("settings.review.on")
                        : L10n.string("settings.review.off"),
                    caption: model.global.diskCache ? model.global.maxCacheDisk : nil
                )
            }
            .padding(.top, 7)
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
