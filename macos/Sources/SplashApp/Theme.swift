import SwiftUI

/// The window's shared metrics. Panes read these instead of choosing their own
/// paddings and radii, so one change here reaches every page.
enum Theme {
    static let pagePadding: CGFloat = 22
    static let sectionSpacing: CGFloat = 22
    static let cardSpacing: CGFloat = 10
    static let cardRadius: CGFloat = 12
    static let controlRadius: CGFloat = 8
    static let contentWidth: CGFloat = 900

    /// The smallest comfortable width of the control panes and the chat, and
    /// the share of the pane area the chat opens on. The window itself opens
    /// at 3:2; the divider decides this ratio.
    static let paneMinWidth: CGFloat = 420
    static let chatMinWidth: CGFloat = 360
    static let chatDefaultFraction: Double = 0.42
}

/// A pane's own title band: what the page is, one line of context, and its
/// actions. Every pane starts with one so the window reads the same way
/// whichever page is open.
struct PaneHeader<Trailing: View>: View {
    let title: String
    var subtitle: String?
    @ViewBuilder var trailing: () -> Trailing

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.title3.weight(.semibold))
                if let subtitle, !subtitle.isEmpty {
                    Text(subtitle)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
            }
            Spacer(minLength: 12)
            trailing()
        }
    }
}

extension PaneHeader where Trailing == EmptyView {
    init(title: String, subtitle: String? = nil) {
        self.init(title: title, subtitle: subtitle, trailing: { EmptyView() })
    }
}

/// An empty page, with the same shape everywhere: an icon, a sentence, and the
/// one action that gets the pane going.
struct PaneEmpty: View {
    let title: String
    let message: String
    var systemImage: String
    var actionTitle: String?
    var action: (() -> Void)?

    var body: some View {
        VStack(spacing: 10) {
            Image(systemName: systemImage)
                .font(.system(size: 30, weight: .light))
                .foregroundStyle(.tertiary)
            Text(title)
                .font(.headline)
            Text(message)
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 340)
            if let actionTitle, let action {
                Button(actionTitle, action: action)
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .padding(.top, 6)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(30)
    }
}

/// A small rounded label for one characteristic of a value.
struct Pill: View {
    let text: String
    var tint: Color?

    var body: some View {
        Text(text)
            .font(.caption2.weight(.medium))
            .foregroundStyle(tint ?? .secondary)
            .lineLimit(1)
            .truncationMode(.middle)
            .padding(.horizontal, 7)
            .padding(.vertical, 2)
            .background((tint ?? .secondary).opacity(tint == nil ? 0.10 : 0.14), in: Capsule())
    }
}


/// Lets a pane send the window to the conversation without knowing how the
/// sidebar keeps its selection.
private struct OpenChatKey: EnvironmentKey {
    static let defaultValue: () -> Void = {}
}

extension EnvironmentValues {
    var openChat: () -> Void {
        get { self[OpenChatKey.self] }
        set { self[OpenChatKey.self] = newValue }
    }
}
