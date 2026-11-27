import AppKit
import SwiftUI

/// The window's shared metrics. Panes read these instead of choosing their own
/// paddings and radii, so one change here reaches every page.
enum Theme {
    static let pagePadding: CGFloat = 26
    static let sectionSpacing: CGFloat = 24
    static let cardSpacing: CGFloat = 12
    static let cardRadius: CGFloat = 18
    static let controlRadius: CGFloat = 10
    static let contentWidth: CGFloat = 980
    static let sidebarWidth: CGFloat = 238

    /// The smallest comfortable width of the control panes and the chat, and
    /// the share of the pane area the chat opens on. The window itself opens
    /// at 3:2; the divider decides this ratio.
    static let paneMinWidth: CGFloat = 420
    static let chatMinWidth: CGFloat = 360
    static let chatDefaultFraction: Double = 0.42
}

/// A clear blue palette shared by every pane.
enum SplashPalette {
    static let blue = Color.adaptive(light: (0.145, 0.388, 0.922), dark: (0.38, 0.62, 1.0))
    static let cyan = Color.adaptive(light: (0.02, 0.58, 0.88), dark: (0.27, 0.76, 1.0))
    static let mint = Color.adaptive(light: (0.23, 0.51, 0.94), dark: (0.49, 0.72, 1.0))
    static let violet = Color.adaptive(light: (0.34, 0.39, 0.82), dark: (0.62, 0.66, 1.0))
    static let amber = Color.adaptive(light: (0.72, 0.42, 0.06), dark: (0.98, 0.72, 0.28))
    static let rose = Color.adaptive(light: (0.76, 0.20, 0.25), dark: (0.98, 0.42, 0.47))

    static let gradient = LinearGradient(colors: [blue, cyan, mint], startPoint: .topLeading, endPoint: .bottomTrailing)
    static let accentFill = LinearGradient(colors: [blue, cyan], startPoint: .topLeading, endPoint: .bottomTrailing)

    static let canvas = Color.adaptive(light: (0.953, 0.965, 0.984), dark: (0.047, 0.067, 0.11))
    static let surface = Color.adaptive(light: (1, 1, 1), dark: (0.085, 0.11, 0.16))
    static let surfaceRaised = Color.adaptive(light: (1, 1, 1), dark: (0.11, 0.145, 0.21))
    static let border = Color.adaptive(light: (0.79, 0.83, 0.90), dark: (0.24, 0.31, 0.43))
    static let hairline = Color.adaptive(light: (0.86, 0.89, 0.94), dark: (0.18, 0.24, 0.34))
    static let ink = Color.adaptive(light: (0.055, 0.075, 0.12), dark: (0.92, 0.95, 1.0))
    static let muted = Color.adaptive(light: (0.36, 0.41, 0.51), dark: (0.66, 0.72, 0.83))
    static let faint = Color.adaptive(light: (0.50, 0.55, 0.64), dark: (0.49, 0.57, 0.70))
    static let sheen = Color.adaptive(light: (0.68, 0.75, 0.88), dark: (0.86, 0.92, 1.0))

    // The navigation rail stays deep navy in both appearance modes.
    static let sidebar = Color(red: 0.055, green: 0.085, blue: 0.16)
    static let sidebarSurface = Color(red: 0.105, green: 0.22, blue: 0.46)
    static let sidebarText = Color(red: 0.94, green: 0.96, blue: 1.0)
    static let sidebarMuted = Color(red: 0.69, green: 0.76, blue: 0.89)
    static let sidebarMint = Color(red: 0.70, green: 0.84, blue: 1.0)
}

extension Color {
    /// A colour that follows the window's appearance through AppKit, so plain
    /// `Color` views animate with a theme switch like the system ones do.
    static func adaptive(light: (Double, Double, Double), dark: (Double, Double, Double)) -> Color {
        Color(nsColor: NSColor(name: nil) { appearance in
            let isDark = appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
            let value = isDark ? dark : light
            return NSColor(srgbRed: value.0, green: value.1, blue: value.2, alpha: 1)
        })
    }
}

/// The window's appearance, the app's own choice rather than only the system's.
enum AppAppearance: String, CaseIterable, Identifiable {
    case system, dark, light

    var id: String { rawValue }
    var title: String { L10n.string("appearance." + rawValue) }

    var systemImage: String {
        switch self {
        case .system: return "circle.lefthalf.filled"
        case .dark: return "moon.stars.fill"
        case .light: return "sun.max.fill"
        }
    }

    var colorScheme: ColorScheme? {
        switch self {
        case .system: return nil
        case .dark: return .dark
        case .light: return .light
        }
    }
}

/// A quiet, flat canvas behind the native panes.
struct AuroraBackground: View {
    var tint: Color = SplashPalette.blue

    var body: some View {
        SplashPalette.canvas
            .ignoresSafeArea()
            .allowsHitTesting(false)
    }
}

/// Shared solid card surface with a restrained edge.
struct GlassPanel: ViewModifier {
    var radius: CGFloat = Theme.cardRadius
    var tint: Color?
    var raised = false
    @Environment(\.colorScheme) private var scheme

    func body(content: Content) -> some View {
        content
            .background {
                RoundedRectangle(cornerRadius: radius, style: .continuous)
                    .fill(SplashPalette.surfaceRaised)
                    .overlay {
                        if let tint {
                            RoundedRectangle(cornerRadius: radius, style: .continuous)
                                .fill(tint.opacity(scheme == .dark ? 0.07 : 0.035))
                        }
                    }
            }
            .overlay {
                RoundedRectangle(cornerRadius: radius, style: .continuous)
                    .strokeBorder(SplashPalette.border.opacity(0.8), lineWidth: 0.8)
            }
            .shadow(color: .black.opacity(raised ? 0.07 : 0), radius: 10, x: 0, y: 4)
    }
}

extension View {
    func glassPanel(radius: CGFloat = Theme.cardRadius, tint: Color? = nil, raised: Bool = false) -> some View {
        modifier(GlassPanel(radius: radius, tint: tint, raised: raised))
    }
}

/// A pane's title and actions, shared across the window.
struct PaneHeader<Trailing: View>: View {
    let title: String
    @ViewBuilder var trailing: () -> Trailing

    var body: some View {
        HStack(alignment: .center, spacing: 14) {
            VStack(alignment: .leading, spacing: 5) {
                Text(title)
                    .font(.system(size: 34, weight: .bold, design: .rounded))
                    .tracking(-0.8)
            }
            Spacer(minLength: 12)
            trailing()
        }
    }
}

extension PaneHeader where Trailing == EmptyView {
    init(title: String) {
        self.init(title: title, trailing: { EmptyView() })
    }
}

/// The small uppercase label above a section of cards.
struct SectionLabel: View {
    let title: String
    let systemImage: String
    var tint: Color = SplashPalette.cyan

    var body: some View {
        HStack(spacing: 9) {
            RoundedRectangle(cornerRadius: 2, style: .continuous)
                .fill(tint)
                .frame(width: 3, height: 14)
            Image(systemName: systemImage)
                .font(.system(size: 10.5, weight: .bold))
                .foregroundStyle(tint)
            Text(title.uppercased())
                .font(.system(size: 10.5, weight: .bold))
                .tracking(1.2)
                .foregroundStyle(SplashPalette.muted)
                .lineLimit(1)
                .truncationMode(.tail)
                .minimumScaleFactor(0.8)
            Spacer(minLength: 0)
        }
    }
}

/// An empty page, with the same shape everywhere: an icon, a title, and an optional action.
struct PaneEmpty: View {
    let title: String
    var systemImage: String
    var actionTitle: String?
    var action: (() -> Void)?

    var body: some View {
        VStack(spacing: 14) {
            ZStack {
                Circle()
                    .fill(SplashPalette.mint.opacity(0.12))
                    .frame(width: 76, height: 76)
                Image(systemName: systemImage)
                    .font(.system(size: 29, weight: .light))
                    .foregroundStyle(SplashPalette.gradient)
            }
            Text(title)
                .font(.system(size: 18, weight: .semibold, design: .rounded))
            if let actionTitle, let action {
                Button(action: action) {
                    Label(actionTitle, systemImage: "play.fill")
                        .font(.system(size: 12.5, weight: .semibold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 18)
                        .padding(.vertical, 9)
                        .background(SplashPalette.accentFill, in: Capsule())
                }
                .buttonStyle(.plain)
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
            .font(.system(size: 10, weight: .semibold))
            .foregroundStyle(tint ?? SplashPalette.muted)
            .lineLimit(1)
            .truncationMode(.middle)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background((tint ?? SplashPalette.muted).opacity(0.13), in: Capsule())
            .overlay(Capsule().strokeBorder((tint ?? SplashPalette.muted).opacity(0.22), lineWidth: 0.7))
    }
}

/// A beacon for a live state.
struct StatusBeacon: View {
    var color: Color
    var active = true

    var body: some View {
        ZStack {
            if active {
                Circle().fill(color.opacity(0.22)).frame(width: 14, height: 14)
            }
            Circle().fill(color).frame(width: 7, height: 7)
        }
        .frame(width: 16, height: 16)
    }
}

/// Lets a pane send the window to the conversation without knowing how the sidebar keeps its selection.
private struct OpenChatKey: EnvironmentKey {
    static let defaultValue: () -> Void = {}
}

extension EnvironmentValues {
    var openChat: () -> Void {
        get { self[OpenChatKey.self] }
        set { self[OpenChatKey.self] = newValue }
    }
}
