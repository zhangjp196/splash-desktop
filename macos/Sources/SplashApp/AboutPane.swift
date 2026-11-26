import SwiftUI

/// The About pane: what the app is, the version it runs, and the two GitHub
/// projects behind it — the desktop app and the inference engine it drives.
struct AboutPane: View {
    @AppStorage("updates.automatic") private var automaticUpdates = true
    @StateObject private var updateChecker = UpdateChecker.shared

    private static let appURL = URL(string: "https://github.com/zhangjp196/splash-desktop")!
    private static let engineURL = URL(string: "https://github.com/incoai/splash")!

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                hero
                updateCard
                VStack(alignment: .leading, spacing: 10) {
                    SectionLabel(
                        title: L10n.string("about.links"),
                        systemImage: "link",
                        tint: SplashPalette.cyan
                    )
                    link(
                        L10n.string("about.current"),
                        detail: Self.appURL.absoluteString,
                        destination: Self.appURL,
                        tint: SplashPalette.blue
                    )
                    link(
                        L10n.string("about.upstream"),
                        detail: Self.engineURL.absoluteString,
                        destination: Self.engineURL,
                        tint: SplashPalette.mint
                    )
                }
            }
            .padding(Theme.pagePadding)
            .frame(maxWidth: 760, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .top)
        }
        .background(Color.clear)
        .onAppear { updateChecker.checkIfDue(enabled: automaticUpdates) }
        .onChange(of: automaticUpdates) { _, enabled in
            if enabled { updateChecker.checkIfDue(enabled: true) }
        }
    }

    private var updateCard: some View {
        EditorCard(
            title: L10n.string("update.title"),
            systemImage: "arrow.down.circle",
            tint: SplashPalette.mint
        ) {
            SettingRow(
                title: L10n.string("update.automatic"),
                help: L10n.string("update.automatic.help"),
                controlWidth: 300
            ) {
                Toggle("", isOn: $automaticUpdates)
                    .labelsHidden()
                    .toggleStyle(.switch)
                    .tint(SplashPalette.mint)
            }
            Rectangle()
                .fill(SplashPalette.hairline.opacity(0.55))
                .frame(height: 0.6)
            SettingRow(
                title: L10n.string("update.check"),
                help: updateStatus,
                controlWidth: 300
            ) {
                HStack(spacing: 10) {
                    Button {
                        updateChecker.checkManually()
                    } label: {
                        if updateChecker.isChecking {
                            ProgressView().controlSize(.small)
                        } else {
                            Text(L10n.string("update.check"))
                        }
                    }
                    .disabled(updateChecker.isChecking)
                    if updateChecker.statusKey == "update.status.available" {
                        Link(L10n.string("update.open_release"), destination: UpdateChecker.releaseURL)
                    }
                }
            }
        }
    }

    private var updateStatus: String {
        switch updateChecker.statusKey {
        case "update.status.available":
            return L10n.format(updateChecker.statusKey, updateChecker.latestVersion ?? "")
        case "update.status.current":
            let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? ""
            return L10n.format(updateChecker.statusKey, version)
        default:
            return L10n.string(updateChecker.statusKey)
        }
    }

    private var hero: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(spacing: 16) {
                ZStack {
                    Circle()
                        .fill(
                            RadialGradient(
                                colors: [
                                    SplashPalette.blue.opacity(0.4),
                                    SplashPalette.cyan.opacity(0.05),
                                ],
                                center: .center,
                                startRadius: 2,
                                endRadius: 54
                            )
                        )
                        .frame(width: 88, height: 88)
                        .blur(radius: 2)
                    BrandMark(size: 58)
                }
                VStack(alignment: .leading, spacing: 6) {
                    Text("Splash Desktop")
                        .font(.system(size: 28, weight: .bold, design: .rounded))
                        .tracking(-0.5)
                    HStack(spacing: 7) {
                        if let version {
                            Pill(text: version, tint: SplashPalette.cyan)
                        }
                    }
                }
                Spacer(minLength: 0)
            }

        }
        .padding(22)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassPanel(tint: SplashPalette.blue, raised: true)
    }

    private var version: String? {
        guard let value = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String,
              !value.isEmpty
        else { return nil }
        return L10n.format("about.version", value)
    }

    private func link(
        _ title: String,
        detail: String,
        destination: URL,
        tint: Color
    ) -> some View {
        Link(destination: destination) {
            HStack(spacing: 12) {
                Image(systemName: "chevron.left.forwardslash.chevron.right")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(tint)
                    .frame(width: 32, height: 32)
                    .background(
                        tint.opacity(0.13),
                        in: RoundedRectangle(cornerRadius: 9, style: .continuous)
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 9, style: .continuous)
                            .strokeBorder(tint.opacity(0.22), lineWidth: 0.7)
                    )
                VStack(alignment: .leading, spacing: 3) {
                    Text(title)
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(.primary)
                    Text(detail)
                        .font(.system(size: 10.5, design: .monospaced))
                        .foregroundStyle(SplashPalette.faint)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                Spacer(minLength: 8)
                Image(systemName: "arrow.up.right")
                    .font(.system(size: 10, weight: .bold))
                    .foregroundStyle(tint)
            }
            .padding(12)
            .glassPanel(radius: 13, tint: tint)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}
