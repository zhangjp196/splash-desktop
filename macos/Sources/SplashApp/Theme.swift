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

/// The palette: a deep, blue-black canvas lit by electric blue, cyan and mint.
/// Every colour adapts to the appearance, so the same views read well in the
/// dark dashboard and in a daylight window.
enum SplashPalette {
    static let blue = Color.adaptive(light: (0.13, 0.36, 0.90), dark: (0.28, 0.47, 0.98))
    static let cyan = Color.adaptive(light: (0.02, 0.52, 0.66), dark: (0.13, 0.78, 0.90))
    static let mint = Color.adaptive(light: (0.02, 0.55, 0.39), dark: (0.20, 0.84, 0.62))
    static let violet = Color.adaptive(light: (0.43, 0.25, 0.86), dark: (0.62, 0.48, 0.98))
    static let amber = Color.adaptive(light: (0.78, 0.44, 0.02), dark: (0.98, 0.68, 0.20))
    static let rose = Color.adaptive(light: (0.82, 0.15, 0.23), dark: (0.96, 0.38, 0.44))

    static let gradient = LinearGradient(
        colors: [blue, cyan, mint],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )

    static let accentFill = LinearGradient(
        colors: [blue, cyan],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )

    static let canvas = Color.adaptive(
        light: (0.955, 0.965, 0.99),
        dark: (0.035, 0.047, 0.078)
    )

    static let surface = Color.adaptive(
        light: (1.0, 1.0, 1.0),
        dark: (0.075, 0.09, 0.14)
    )

    static let surfaceRaised = Color.adaptive(
        light: (0.985, 0.99, 1.0),
        dark: (0.10, 0.12, 0.18)
    )

    static let border = Color.adaptive(
        light: (0.80, 0.83, 0.90),
        dark: (0.24, 0.28, 0.38)
    )

    static let hairline = Color.adaptive(
        light: (0.86, 0.88, 0.93),
        dark: (0.18, 0.22, 0.30)
    )

    static let muted = Color.adaptive(
        light: (0.42, 0.45, 0.53),
        dark: (0.58, 0.63, 0.72)
    )

    static let faint = Color.adaptive(
        light: (0.60, 0.63, 0.70),
        dark: (0.40, 0.45, 0.54)
    )

    /// The faint top light on a glass card: a white sheen in the dark, a cool
    /// tint in the light.
    static let sheen = Color.adaptive(
        light: (0.55, 0.62, 0.80),
        dark: (1.0, 1.0, 1.0)
    )
}

extension Color {
    /// A colour that follows the window's appearance through AppKit, so plain
    /// `Color` views animate with a theme switch like the system ones do.
    static func adaptive(
        light: (Double, Double, Double),
        dark: (Double, Double, Double)
    ) -> Color {
        Color(nsColor: NSColor(name: nil) { appearance in
            let isDark = appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
            let value = isDark ? dark : light
            return NSColor(
                srgbRed: value.0, green: value.1, blue: value.2, alpha: 1
            )
        })
    }
}

/// The window's appearance, the app's own choice rather than only the system's:
/// the dashboard is designed dark first, and a daylight window is there for
/// anyone who prefers it.
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

/// The window's backdrop: the canvas, three slow aurora blooms, and a fine
/// grid. It sits under every pane, so switching pages never changes the
/// atmosphere; only the content on top moves.
struct AuroraBackground: View {
    var tint: Color = SplashPalette.blue
    @Environment(\.colorScheme) private var scheme
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var drift = false

    var body: some View {
        ZStack {
            SplashPalette.canvas

            bloom(SplashPalette.blue, size: 620, x: -180, y: -300, opacity: 0.22)
                .offset(x: drift ? 60 : -30, y: drift ? 30 : -20)
            bloom(SplashPalette.cyan, size: 520, x: 320, y: 360, opacity: 0.14)
                .offset(x: drift ? -40 : 30, y: drift ? -30 : 20)
            bloom(SplashPalette.violet, size: 460, x: 520, y: -80, opacity: 0.12)
                .offset(x: drift ? -20 : 20, y: drift ? 40 : -10)
            bloom(tint, size: 500, x: 330, y: -300, opacity: 0.10)
                .offset(x: drift ? -30 : 15, y: drift ? 25 : -15)

            grid
                .opacity(scheme == .dark ? 0.05 : 0.035)
        }
        .animation(.easeInOut(duration: 1.2), value: tint)
        .ignoresSafeArea()
        .allowsHitTesting(false)
        .onAppear {
            guard !reduceMotion else { return }
            withAnimation(.easeInOut(duration: 16).repeatForever(autoreverses: true)) {
                drift = true
            }
        }
    }

    private func bloom(
        _ color: Color, size: CGFloat, x: CGFloat, y: CGFloat, opacity: Double
    ) -> some View {
        Circle()
            .fill(
                RadialGradient(
                    colors: [color.opacity(opacity), color.opacity(0)],
                    center: .center,
                    startRadius: 0,
                    endRadius: size / 2
                )
            )
            .frame(width: size, height: size)
            .blur(radius: 34)
            .offset(x: x, y: y)
    }

    private var grid: some View {
        Canvas { context, size in
            var path = Path()
            let step: CGFloat = 38
            var x: CGFloat = 0
            while x <= size.width {
                path.move(to: CGPoint(x: x, y: 0))
                path.addLine(to: CGPoint(x: x, y: size.height))
                x += step
            }
            var y: CGFloat = 0
            while y <= size.height {
                path.move(to: CGPoint(x: 0, y: y))
                path.addLine(to: CGPoint(x: size.width, y: y))
                y += step
            }
            context.stroke(path, with: .color(.white), lineWidth: 0.5)
        }
    }
}

/// The shared glass surface: a translucent card with a fine gradient edge, a
/// top sheen, and a soft shadow. Panes pass a tint when the card belongs to
/// one reading, so the board reads in colour rather than in boxes.
struct GlassPanel: ViewModifier {
    var radius: CGFloat = Theme.cardRadius
    var tint: Color?
    var raised = false
    @Environment(\.colorScheme) private var scheme

    private var isDark: Bool { scheme == .dark }

    func body(content: Content) -> some View {
        content
            .background {
                ZStack {
                    RoundedRectangle(cornerRadius: radius, style: .continuous)
                        .fill(.ultraThinMaterial)
                    RoundedRectangle(cornerRadius: radius, style: .continuous)
                        .fill(
                            SplashPalette.surface.opacity(
                                isDark ? (raised ? 0.72 : 0.52) : (raised ? 0.90 : 0.74)
                            )
                        )
                    if let tint {
                        RoundedRectangle(cornerRadius: radius, style: .continuous)
                            .fill(
                                LinearGradient(
                                    colors: [
                                        tint.opacity(isDark ? 0.14 : 0.06),
                                        tint.opacity(isDark ? 0.02 : 0.01),
                                    ],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                )
                            )
                    }
                    RoundedRectangle(cornerRadius: radius, style: .continuous)
                        .fill(
                            LinearGradient(
                                colors: [
                                    isDark
                                        ? SplashPalette.sheen.opacity(0.10)
                                        : Color.white.opacity(0.42),
                                    .clear,
                                ],
                                startPoint: .top,
                                endPoint: .center
                            )
                        )
                }
            }
            .overlay {
                RoundedRectangle(cornerRadius: radius, style: .continuous)
                    .strokeBorder(
                        LinearGradient(
                            colors: isDark
                                ? [
                                    SplashPalette.sheen.opacity(0.22),
                                    SplashPalette.border.opacity(0.35),
                                    SplashPalette.sheen.opacity(0.04),
                                ]
                                : [
                                    Color.white.opacity(0.95),
                                    SplashPalette.border.opacity(0.55),
                                    Color.white.opacity(0.55),
                                ],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        ),
                        lineWidth: 0.9
                    )
            }
            .shadow(
                color: .black.opacity(isDark ? (raised ? 0.28 : 0.18) : (raised ? 0.10 : 0.06)),
                radius: raised ? 22 : 14,
                x: 0,
                y: raised ? 12 : 7
            )
    }
}

extension View {
    func glassPanel(
        radius: CGFloat = Theme.cardRadius,
        tint: Color? = nil,
        raised: Bool = false
    ) -> some View {
        modifier(GlassPanel(radius: radius, tint: tint, raised: raised))
    }
}

/// A pane's own title band: what the page is, one line of context, and its
/// actions. Every pane starts with one so the window reads the same way
/// whichever page is open.
struct PaneHeader<Trailing: View>: View {
    let title: String
    var subtitle: String?
    var eyebrow: String?
    @ViewBuilder var trailing: () -> Trailing

    var body: some View {
        HStack(alignment: .center, spacing: 14) {
            VStack(alignment: .leading, spacing: 5) {
                if let eyebrow, !eyebrow.isEmpty {
                    Text(eyebrow.uppercased())
                        .font(.system(size: 10, weight: .bold))
                        .tracking(1.6)
                        .foregroundStyle(SplashPalette.cyan)
                }
                Text(title)
                    .font(.system(size: 27, weight: .bold, design: .rounded))
                    .tracking(-0.4)
                if let subtitle, !subtitle.isEmpty {
                    Text(subtitle)
                        .font(.system(size: 12.5))
                        .foregroundStyle(SplashPalette.muted)
                        .lineLimit(2)
                }
            }
            Spacer(minLength: 12)
            trailing()
        }
    }
}

extension PaneHeader where Trailing == EmptyView {
    init(title: String, subtitle: String? = nil, eyebrow: String? = nil) {
        self.init(
            title: title, subtitle: subtitle, eyebrow: eyebrow,
            trailing: { EmptyView() }
        )
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
                .shadow(color: tint.opacity(0.6), radius: 5)
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

/// An empty page, with the same shape everywhere: an icon, a sentence, and the
/// one action that gets the pane going.
struct PaneEmpty: View {
    let title: String
    let message: String
    var systemImage: String
    var actionTitle: String?
    var action: (() -> Void)?

    var body: some View {
        VStack(spacing: 14) {
            ZStack {
                Circle()
                    .fill(
                        LinearGradient(
                            colors: [
                                SplashPalette.blue.opacity(0.22),
                                SplashPalette.cyan.opacity(0.08),
                            ],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: 76, height: 76)
                    .blur(radius: 1)
                Image(systemName: systemImage)
                    .font(.system(size: 29, weight: .light))
                    .foregroundStyle(SplashPalette.gradient)
            }
            Text(title)
                .font(.system(size: 18, weight: .semibold, design: .rounded))
            Text(message)
                .font(.system(size: 12.5))
                .foregroundStyle(SplashPalette.muted)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 360)
            if let actionTitle, let action {
                Button(action: action) {
                    Label(actionTitle, systemImage: "play.fill")
                        .font(.system(size: 12.5, weight: .semibold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 18)
                        .padding(.vertical, 9)
                        .background(SplashPalette.accentFill, in: Capsule())
                        .shadow(color: SplashPalette.blue.opacity(0.4), radius: 12, y: 5)
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
            .overlay(
                Capsule().strokeBorder((tint ?? SplashPalette.muted).opacity(0.22), lineWidth: 0.7)
            )
    }
}

/// A beacon for a live state: the dot breathes while the server is up.
struct StatusBeacon: View {
    var color: Color
    var active = true
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var pulse = false

    var body: some View {
        ZStack {
            if active {
                Circle()
                    .fill(color.opacity(0.35))
                    .frame(width: 16, height: 16)
                    .scaleEffect(pulse ? 1.35 : 0.6)
                    .opacity(pulse ? 0 : 0.8)
            }
            Circle()
                .fill(color)
                .frame(width: 7, height: 7)
                .shadow(color: color.opacity(0.8), radius: 4)
        }
        .frame(width: 16, height: 16)
        .onAppear {
            guard active, !reduceMotion else { return }
            withAnimation(.easeOut(duration: 1.8).repeatForever(autoreverses: false)) {
                pulse = true
            }
        }
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
