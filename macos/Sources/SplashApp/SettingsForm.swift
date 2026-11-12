import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// The groups the configuration editor's rail switches between. Each one owns
/// a short description, so the rail explains the form instead of only naming
/// it.
enum SettingsSection: String, CaseIterable, Identifiable, RailSection {
    case model, draft, memory, serving

    var id: String { rawValue }

    var title: String { L10n.string("section.\(rawValue)") }
    var summary: String { L10n.string("section.\(rawValue).help") }

    var systemImage: String {
        switch self {
        case .model: return "cube.box"
        case .draft: return "bolt.badge.clock"
        case .memory: return "memorychip"
        case .serving: return "speedometer"
        }
    }

    var tint: Color {
        switch self {
        case .model: return SplashPalette.blue
        case .draft: return SplashPalette.violet
        case .memory: return SplashPalette.mint
        case .serving: return SplashPalette.amber
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
                || s.modelDirectory != b.modelDirectory || s.localKind != b.localKind
                || s.mmproj != b.mmproj || s.languageOnly != b.languageOnly
        case .draft:
            return s.speculation != b.speculation || s.requireDraft != b.requireDraft
                || s.draftDirectory != b.draftDirectory || s.draftMode != b.draftMode
                || s.draftID != b.draftID || s.verifyRows != b.verifyRows
        case .memory:
            return s.maxMemory != b.maxMemory || s.maxContext != b.maxContext
                || s.expertMemory != b.expertMemory
                || s.pleMemory != b.pleMemory
                || s.sparseKvMemory != b.sparseKvMemory
        case .serving:
            return s.kvFormat != b.kvFormat || s.stateFormat != b.stateFormat
                || s.neuralEngine != b.neuralEngine
                || s.maxConcurrency != b.maxConcurrency
                || s.decodeShare != b.decodeShare
                || s.responsivePrefill != b.responsivePrefill
                || s.prefillSliceMilliseconds != b.prefillSliceMilliseconds
                || s.reasoningEffort != b.reasoningEffort
                || s.thinkingTokenBudget != b.thinkingTokenBudget
                || s.thinkingEndText != b.thinkingEndText
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
        case .model:
            modelPage
        case .draft:
            draftPage
        case .memory:
            VStack(alignment: .leading, spacing: 14) {
                budgetCard
                if profile.experts || profile.ple || profile.sparseKv {
                    streamedCard
                }
            }
        case .serving:
            VStack(alignment: .leading, spacing: 14) {
                runtimeCard
                paceCard
                if profile.thinking {
                    thinkingCard
                }
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
            if profile.vision {
                rowDivider
                SettingRow(
                    title: L10n.string("field.language_only"),
                    help: L10n.string("help.language_only"),
                    controlWidth: 300
                ) {
                    Toggle("", isOn: $settings.languageOnly)
                        .labelsHidden()
                        .toggleStyle(.switch)
                        .tint(SplashPalette.cyan)
                }
            }
        }
        if settings.localKind == "gguf" && profile.vision && settings.languageOnly {
            EditorNote(
                text: L10n.string("help.mmproj.text_only"),
                tone: SplashPalette.muted,
                systemImage: "info.circle"
            )
        }
    }

    // MARK: Draft

    /// How decode proposes tokens, and the draft it proposes them from. This
    /// is the one part of a configuration besides its model and memory: a
    /// draft is tied to the target it verifies against, so it cannot be an
    /// app-wide setting.
    @ViewBuilder private var draftPage: some View {
        EditorCard(
            title: L10n.string("config.group.draft"),
            systemImage: "bolt.badge.clock",
            tint: SplashPalette.violet
        ) {
            SettingRow(
                title: L10n.string("field.speculation"),
                help: L10n.string("help.speculation"),
                controlWidth: 300
            ) {
                Picker("", selection: $settings.speculation) {
                    Text(verbatim: L10n.string("speculation.auto")).tag("auto")
                    Text(verbatim: L10n.string("speculation.dflash2")).tag("dflash2")
                }
                .pickerStyle(.segmented)
                .labelsHidden()
            }
            rowDivider
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
            rowDivider
            SettingRow(
                title: L10n.string("field.require_draft"),
                help: L10n.string("help.require_draft"),
                controlWidth: 300
            ) {
                Toggle("", isOn: $settings.requireDraft)
                    .labelsHidden()
                    .toggleStyle(.switch)
                    .tint(SplashPalette.violet)
            }
            rowDivider
            SettingRow(
                title: L10n.string("field.verify_rows"),
                help: L10n.string("help.verify_rows"),
                controlWidth: 300
            ) {
                Picker("", selection: $settings.verifyRows) {
                    Text(verbatim: L10n.string("verify.auto")).tag("auto")
                    Text(verbatim: L10n.string("verify.all")).tag("all")
                }
                .pickerStyle(.segmented)
                .labelsHidden()
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
                    MemorySizeField(text: $settings.maxMemory, cap: .budget, prompt: "28G")
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
                        MemorySizeField(text: $settings.expertMemory, cap: .experts, prompt: L10n.string("size.auto"))
                    }
                }
                if profile.ple {
                    FieldColumn(title: L10n.string("field.ple_memory"), help: L10n.string("help.ple_memory")) {
                        MemorySizeField(text: $settings.pleMemory, cap: .smallStore, prompt: L10n.string("size.auto"))
                    }
                }
                if profile.sparseKv {
                    FieldColumn(title: L10n.string("field.sparse_kv_memory"), help: L10n.string("help.sparse_kv_memory")) {
                        MemorySizeField(text: $settings.sparseKvMemory, cap: .smallStore, prompt: L10n.string("size.auto"))
                    }
                }
            }
        }
    }

    // MARK: Serving

    /// How this model's engine runs: precision, how many requests it admits,
    /// how prefill is paced, and the thinking defaults a request may use.
    /// These are per configuration because a model's files and memory plan
    /// decide what it can afford.
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
                FieldColumn(title: L10n.string("field.neural_engine"), help: L10n.string("help.neural_engine")) {
                    Toggle("", isOn: $settings.neuralEngine)
                        .labelsHidden()
                        .toggleStyle(.switch)
                        .tint(SplashPalette.mint)
                }
            }
        }
    }

    private var paceCard: some View {
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

    private var thinkingCard: some View {
        EditorCard(
            title: L10n.string("config.group.thinking"),
            systemImage: "brain",
            tint: SplashPalette.blue
        ) {
            SettingColumns {
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
                FieldColumn(title: L10n.string("field.thinking_token_budget"), help: L10n.string("help.thinking_token_budget")) {
                    EditorField(text: $settings.thinkingTokenBudget, prompt: "4096", monospaced: true)
                }
                FieldColumn(title: L10n.string("field.thinking_end_text"), help: L10n.string("help.thinking_end_text")) {
                    EditorField(text: $settings.thinkingEndText, prompt: L10n.string("thinking_end.prompt"))
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
                    title: L10n.string("config.group.budget"),
                    value: settings.maxMemory,
                    caption: L10n.format("config.review.context", settings.maxContext)
                )
                reviewDivider
                ReviewStat(
                    title: L10n.string("field.kv"),
                    value: settings.kvFormat.uppercased(),
                    caption: settings.stateFormat.uppercased()
                )
                reviewDivider
                ReviewStat(
                    title: L10n.string("field.speculation"),
                    value: L10n.string("speculation.\(settings.speculation)"),
                    caption: draftLabel
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

    private var draftLabel: String {
        let draft = settings.draftDirectory.trimmingCharacters(in: .whitespaces)
        guard !draft.isEmpty else { return L10n.string("models.none") }
        return (draft as NSString).lastPathComponent
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
struct ReviewStat: View {
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

// MARK: - Cards and rows

/// A titled card in the form: a small section label above a stack of rows.
struct EditorCard<Content: View>: View {
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
struct SettingColumns<Content: View>: View {
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
struct FieldColumn<Control: View>: View {
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
struct SettingRow<Control: View>: View {
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
struct EditorNote: View {
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

struct EditorNumberField: View {
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
struct SizeField: View {
    @Binding var text: String
    var prompt: String

    var body: some View {
        EditorField(text: $text, prompt: prompt, monospaced: true)
    }
}

/// How much of the Mac's memory a memory setting may claim: the slider's and
/// a typed size's upper limit. The overall budget takes the most of the
/// machine (90%); a store that lives inside that budget takes less, and the
/// small streaming caches less again, so no single setting can ask the whole
/// machine for itself.
enum MemoryCap {
    /// `--max-memory`, the ceiling on the engine's Metal allocations.
    case budget
    /// `--expert-memory`, the routed-expert store inside the budget.
    case experts
    /// `--ple-memory` and `--sparse-kv-memory`, the small streaming caches.
    case smallStore

    /// The cap in gigabytes, on a tenth-of-a-gigabyte grain so a 48G Mac
    /// reads `43.2G` rather than `43.200000000000004`.
    var gigabytes: Double {
        let physical = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
        let fraction: Double
        switch self {
        case .budget: fraction = 0.9
        case .experts: fraction = 0.75
        case .smallStore: fraction = 0.25
        }
        return max(0.5, (physical * fraction * 10).rounded() / 10)
    }

    /// The cap as the engine reads a size, such as `43.2G`.
    var label: String { MemorySizeField.label(gigabytes) }
}

/// A memory size the user can drag or type: a slider up to the share of the
/// Mac's memory the setting may claim, beside a field that still takes the
/// whole syntax (`auto`, `off`, or a size such as `16G`). Dragging writes the
/// size it lands on, the far left returning the setting to automatic; a size
/// typed above the cap is clamped to it when the field is submitted.
struct MemorySizeField: View {
    @Binding var text: String
    let cap: MemoryCap
    var prompt = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                Slider(value: slider, in: 0...cap.gigabytes)
                    .accessibilityLabel(Text(prompt.isEmpty ? cap.label : prompt))
                    .accessibilityValue(Text(MemorySizeField.label(slider.wrappedValue)))
                EditorField(text: $text, prompt: prompt, monospaced: true)
                    .frame(width: 84)
                    .onSubmit(clampTyped)
            }
            Text(L10n.format("memory.cap", cap.label))
                .font(.system(size: 9))
                .foregroundStyle(SplashPalette.faint)
        }
    }

    /// The slider reads the size the field holds and writes the size it is
    /// dragged to, so typing and dragging stay one value. `auto`, `off` and
    /// an empty field read as the far left, where a drag back returns the
    /// setting to automatic.
    private var slider: Binding<Double> {
        Binding(
            get: { min(MemorySizeField.gigabytes(text) ?? 0, cap.gigabytes) },
            set: { newValue in
                // Half-gigabyte steps keep a drag deliberate; below the first
                // step the drag means "back to automatic" rather than a size.
                let rounded = (newValue * 2).rounded() / 2
                text = rounded < 0.5 ? "" : MemorySizeField.label(rounded)
            }
        )
    }

    /// A typed size above the cap is clamped to it; `auto`, `off` and an
    /// empty field are not sizes, so they stay as they were typed.
    private func clampTyped() {
        guard let value = MemorySizeField.gigabytes(text), value > cap.gigabytes else { return }
        text = cap.label
    }

    /// The size a value holds in gigabytes, for the slider: nil for empty,
    /// `auto` or `off`, which the slider does not set. Sizes are binary, as
    /// the engine reads them: `28G` is 28 GiB.
    static func gigabytes(_ raw: String) -> Double? {
        var value = raw.trimmingCharacters(in: .whitespaces)
        guard !value.isEmpty,
              value.caseInsensitiveCompare("auto") != .orderedSame,
              value.caseInsensitiveCompare("off") != .orderedSame
        else { return nil }
        var scale = 1.0
        if let last = value.last {
            switch last {
            case "K", "k": scale = 1_024
            case "M", "m": scale = 1_048_576
            case "G", "g": scale = 1_073_741_824
            case "T", "t": scale = 1_099_511_627_776
            default: break
            }
        }
        if scale > 1 { value.removeLast() }
        guard let amount = Double(value), amount.isFinite, amount > 0 else { return nil }
        return amount * scale / 1_073_741_824
    }

    /// A size as the engine reads one: whole gigabytes plain, a fraction to
    /// one decimal, so a 48G Mac's budget cap reads `43.2G`.
    static func label(_ gigabytes: Double) -> String {
        let rounded = (gigabytes * 10).rounded() / 10
        if rounded == rounded.rounded() { return "\(Int(rounded))G" }
        return String(format: "%.1fG", rounded)
    }
}

struct SecretField: View {
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
struct DirectoryPicker: View {
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
