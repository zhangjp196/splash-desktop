import SwiftUI

/// The About pane: what the app is, the version it runs, and the two GitHub
/// projects behind it — the desktop app and the inference engine it drives.
struct AboutPane: View {
    private static let appURL = URL(string: "https://github.com/zhangjp196/splash-desktop")!
    private static let engineURL = URL(string: "https://github.com/incoai/splash")!

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                hero
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
                footer
            }
            .padding(Theme.pagePadding)
            .frame(maxWidth: 760, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .top)
        }
        .background(Color.clear)
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
                        Pill(text: L10n.string("about.badge"), tint: SplashPalette.mint)
                    }
                }
                Spacer(minLength: 0)
            }

            Text(L10n.string("about.description"))
                .font(.system(size: 13))
                .foregroundStyle(SplashPalette.muted)
                .fixedSize(horizontal: false, vertical: true)

            HStack(spacing: 8) {
                feature("bolt.fill", L10n.string("about.feature.speed"), SplashPalette.amber)
                feature("memorychip.fill", L10n.string("about.feature.memory"), SplashPalette.mint)
                feature("lock.shield.fill", L10n.string("about.feature.local"), SplashPalette.violet)
            }
        }
        .padding(22)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassPanel(tint: SplashPalette.blue, raised: true)
    }

    private func feature(_ icon: String, _ text: String, _ tint: Color) -> some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
                .font(.system(size: 9, weight: .bold))
                .foregroundStyle(tint)
            Text(text)
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(SplashPalette.muted)
                .lineLimit(1)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(SplashPalette.surface.opacity(0.45), in: Capsule())
        .overlay(Capsule().strokeBorder(SplashPalette.border.opacity(0.4), lineWidth: 0.7))
    }

    private var footer: some View {
        HStack(spacing: 7) {
            Image(systemName: "heart.fill")
                .font(.system(size: 9))
                .foregroundStyle(SplashPalette.rose)
            Text(L10n.string("about.footer"))
                .font(.system(size: 10))
                .foregroundStyle(SplashPalette.faint)
        }
        .padding(.top, 4)
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
        .help(destination.absoluteString)
    }
}
