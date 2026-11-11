import SwiftUI

/// The brand mark: the glossy droplet on the app icon's gradient tile. The
/// same shape as `server/favicon.svg`, so the Dock, the title bar and the chat
/// page wear one mark.
struct BrandMark: View {
    var size: CGFloat = 24

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: size * 0.29, style: .continuous)
                .fill(
                    LinearGradient(
                        colors: [
                            Color(red: 0.024, green: 0.086, blue: 0.192),
                            Color(red: 0.055, green: 0.271, blue: 0.682),
                            Color(red: 0.043, green: 0.604, blue: 0.847),
                            Color(red: 0.169, green: 0.851, blue: 0.784),
                        ],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
            RoundedRectangle(cornerRadius: size * 0.29, style: .continuous)
                .fill(
                    LinearGradient(
                        colors: [.white.opacity(0.20), .clear],
                        startPoint: .top,
                        endPoint: .center
                    )
                )
            DropletMark()
                .frame(width: size * 0.64, height: size * 0.70)
                .offset(y: size * 0.015)
        }
        .frame(width: size, height: size)
        .overlay(
            RoundedRectangle(cornerRadius: size * 0.29, style: .continuous)
                .strokeBorder(
                    LinearGradient(
                        colors: [.white.opacity(0.42), .white.opacity(0.04)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    ),
                    lineWidth: size * 0.012
                )
        )
        .shadow(color: SplashPalette.blue.opacity(0.45), radius: 8, y: 4)
    }
}

/// The droplet's path in the icon's 1024-unit box, with the glossy sheen and
/// rim light the icon generator draws in `AppIcon.svg`/`AppIcon.png`.
private struct DropletMark: View {
    var body: some View {
        GeometryReader { geometry in
            let width = geometry.size.width
            let height = geometry.size.height
            ZStack {
                DropletShape()
                    .fill(
                        LinearGradient(
                            colors: [
                                .white,
                                Color(red: 0.957, green: 0.996, blue: 1.0),
                                Color(red: 0.706, green: 0.894, blue: 0.949),
                            ],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .shadow(color: .black.opacity(0.28), radius: width * 0.08, y: height * 0.06)

                // The cool inner shadow that rounds the bottom of the drop.
                DropletShape()
                    .fill(
                        LinearGradient(
                            colors: [.clear, Color(red: 0.043, green: 0.361, blue: 0.659).opacity(0.28)],
                            startPoint: .center,
                            endPoint: .bottom
                        )
                    )

                // A broad sheen down the left, and the crisp specular near the
                // shoulder.
                Ellipse()
                    .fill(.white.opacity(0.62))
                    .frame(width: width * 0.30, height: height * 0.48)
                    .rotationEffect(.degrees(-24))
                    .offset(x: -width * 0.10, y: -height * 0.02)
                    .blur(radius: width * 0.055)
                    .mask(DropletShape())

                Ellipse()
                    .fill(.white.opacity(0.9))
                    .frame(width: width * 0.10, height: height * 0.20)
                    .rotationEffect(.degrees(-26))
                    .offset(x: -width * 0.06, y: -height * 0.14)
                    .blur(radius: width * 0.018)
                    .mask(DropletShape())

                DropletShape()
                    .stroke(.white.opacity(0.75), lineWidth: max(0.6, width * 0.012))
            }
        }
    }
}

/// The droplet itself, scaled from the icon's 1024-unit path.
private struct DropletShape: Shape {
    func path(in rect: CGRect) -> Path {
        let sx = rect.width / 1024
        let sy = rect.height / 1024
        func point(_ x: CGFloat, _ y: CGFloat) -> CGPoint {
            CGPoint(x: rect.minX + x * sx, y: rect.minY + y * sy)
        }
        var path = Path()
        path.move(to: point(512, 188))
        path.addCurve(
            to: point(704, 642),
            control1: point(512, 188),
            control2: point(704, 432)
        )
        path.addCurve(
            to: point(512, 868),
            control1: point(704, 783),
            control2: point(614, 868)
        )
        path.addCurve(
            to: point(320, 642),
            control1: point(410, 868),
            control2: point(320, 783)
        )
        path.addCurve(
            to: point(512, 188),
            control1: point(320, 432),
            control2: point(512, 188)
        )
        path.closeSubpath()
        return path
    }
}
