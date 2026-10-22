import SwiftUI

/// The About pane: what the app is, the version it runs, and the two GitHub
/// projects behind it — the desktop app and the inference engine it drives.
struct AboutPane: View {
    private static let appURL = URL(string: "https://github.com/zhangjp196/splash-desktop")!
    private static let engineURL = URL(string: "https://github.com/incoai/splash")!

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                header
                Text(L10n.string("about.description"))
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                VStack(alignment: .leading, spacing: 10) {
                    Label(L10n.string("about.links"), systemImage: "link")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.secondary)
                    link(
                        L10n.string("about.current"),
                        detail: Self.appURL.absoluteString,
                        destination: Self.appURL
                    )
                    link(
                        L10n.string("about.upstream"),
                        detail: Self.engineURL.absoluteString,
                        destination: Self.engineURL
                    )
                }
            }
            .padding(20)
            .frame(maxWidth: 720, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .topLeading)
        }
        .background(.background)
    }

    private var header: some View {
        HStack(spacing: 12) {
            BrandMark()
                .scaleEffect(1.6)
                .frame(width: 38, height: 38)
            VStack(alignment: .leading, spacing: 3) {
                Text("Splash Desktop")
                    .font(.title2.weight(.semibold))
                if let version {
                    Text(version)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private var version: String? {
        guard let value = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String,
              !value.isEmpty
        else { return nil }
        return L10n.format("about.version", value)
    }

    private func link(_ title: String, detail: String, destination: URL) -> some View {
        Link(destination: destination) {
            HStack(spacing: 12) {
                Image(systemName: "chevron.left.forwardslash.chevron.right")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(Color.accentColor)
                    .frame(width: 30, height: 30)
                    .background(
                        Color.accentColor.opacity(0.14),
                        in: RoundedRectangle(cornerRadius: 8)
                    )
                VStack(alignment: .leading, spacing: 3) {
                    Text(title)
                        .font(.body.weight(.medium))
                        .foregroundStyle(.primary)
                    Text(detail)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                Spacer(minLength: 8)
                Image(systemName: "arrow.up.right.square")
                    .foregroundStyle(.secondary)
            }
            .padding(12)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(.quaternary.opacity(0.25))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .strokeBorder(.separator.opacity(0.5))
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(destination.absoluteString)
    }
}
