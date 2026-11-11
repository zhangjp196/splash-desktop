import SwiftUI

/// The tiles every pane is built from: a section of cards, a card of one
/// number, the bar under it, and the headline tile. They are shared so the
/// live board and the thermal page read the same way.

/// The tile rows are deliberately uniform: a metrics board is read by
/// comparing neighbours, so every rectangle in a row is the same size rather
/// than sized to its own text. The heights hold the tallest content a tile can
/// have (a value, a bar and one line of caption), and captions are kept to one
/// line so nothing grows past them.
enum TileMetrics {
    static let heroHeight: CGFloat = 168
    static let cardHeight: CGFloat = 112
    static let corner: CGFloat = 16
}

struct MetricSection: View, Equatable {
    let title: String
    let systemImage: String
    let cards: [MetricCard]
    var tint: Color = SplashPalette.cyan

    var body: some View {
        VStack(alignment: .leading, spacing: 11) {
            SectionLabel(title: title, systemImage: systemImage, tint: tint)
            LazyVGrid(
                columns: [GridItem(.adaptive(minimum: 200), spacing: 12)],
                spacing: 12
            ) {
                ForEach(cards) { card in
                    card
                }
            }
        }
    }
}

/// Equatable over its plain data, so a poll that left a card's figures alone
/// skips its body — on a board of dozens, most cards are unchanged most
/// seconds.
struct MetricCard: View, Identifiable, Equatable {
    /// The title is stable across polls; a fresh UUID would restart every
    /// animation on each refresh.
    var id: String { title + systemImage }
    let title: String
    let value: String
    var unit = ""
    var caption = ""
    var systemImage = "circle"
    var tint: Color = SplashPalette.blue
    /// A 0...1 share of a known total; nil leaves the card without a bar.
    var progress: Double?
    var progressCaption: String?
    /// The last minute of readings, drawn under the value when present.
    var history: [Double] = []
    /// A fixed top for the history chart, for a reading already on a 0–100
    /// scale; nil lets the card scale to its own peak.
    var chartMax: Double?

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack(spacing: 8) {
                Image(systemName: systemImage)
                    .font(.system(size: 11, weight: .bold))
                    .foregroundStyle(tint)
                    .frame(width: 25, height: 25)
                    .background(tint.opacity(0.14), in: RoundedRectangle(cornerRadius: 7, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: 7, style: .continuous)
                            .strokeBorder(tint.opacity(0.24), lineWidth: 0.7)
                    )
                Text(title)
                    .font(.system(size: 10.5, weight: .semibold))
                    .tracking(0.3)
                    .foregroundStyle(SplashPalette.muted)
                    .lineLimit(1)
                Spacer(minLength: 0)
            }

            HStack(alignment: .firstTextBaseline, spacing: 4) {
                Text(value)
                    .font(.system(size: 22, weight: .bold, design: .rounded))
                    .monospacedDigit()
                    .lineLimit(1)
                    .minimumScaleFactor(0.55)
                if !unit.isEmpty {
                    Text(unit)
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(SplashPalette.faint)
                        .lineLimit(1)
                }
                Spacer(minLength: 0)
            }

            if !history.isEmpty {
                Sparkline(values: history, tint: tint, upperBound: chartMax)
                    .frame(height: 20)
                    .padding(.top, 1)
            } else if let progress {
                Meter(fraction: progress, tint: tint)
                    .padding(.top, 2)
            }

            if let note = progressCaption ?? (caption.isEmpty ? nil : caption) {
                Text(note)
                    .font(.system(size: 9.5))
                    .foregroundStyle(SplashPalette.faint)
                    .lineLimit(1)
            }
        }
        .padding(13)
        .frame(maxWidth: .infinity, minHeight: TileMetrics.cardHeight,
               maxHeight: TileMetrics.cardHeight, alignment: .topLeading)
        .glassPanel(radius: TileMetrics.corner, tint: tint)
        // No hover lift: a scale, an offset and an animated shadow would put
        // every card into an offscreen pass twice a second on a board of
        // dozens, for an effect only the pointer follows.
    }
}

/// The bar under a card's value: how much of a known total it uses. A metric
/// that only counts (requests, failures) keeps a plain caption instead.
struct Meter: View {
    let fraction: Double
    let tint: Color

    var body: some View {
        GeometryReader { proxy in
            ZStack(alignment: .leading) {
                Capsule().fill(tint.opacity(0.13))
                Capsule()
                    .fill(
                        LinearGradient(
                            colors: [tint.opacity(0.65), tint],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .frame(width: max(3, proxy.size.width * min(1, max(0, fraction))))
            }
        }
        .frame(height: 5)
    }
}

/// The headline tile: the number is the point, so it is set large and the
/// icon and label stay quiet around it. `history` draws the last minute behind
/// the number; `note` is one quiet line under it, for the figure the headline
/// hides — the peak behind an average.
/// Equatable like the card it headlines, for the same reason.
struct HeroCard: View, Equatable {
    let title: String
    let value: String
    let unit: String
    let systemImage: String
    let tint: Color
    var progress: Double?
    var note = ""
    var history: [Double] = []
    var chartMax: Double?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Image(systemName: systemImage)
                    .font(.system(size: 12, weight: .bold))
                    .foregroundStyle(tint)
                    .frame(width: 28, height: 28)
                    .background(tint.opacity(0.14), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .strokeBorder(tint.opacity(0.26), lineWidth: 0.8)
                    )
                Text(title.uppercased())
                    .font(.system(size: 10, weight: .bold))
                    .tracking(1.1)
                    .foregroundStyle(SplashPalette.muted)
                    .lineLimit(1)
                Spacer(minLength: 0)
                if let progress {
                    Text(String(format: "%.0f%%", progress * 100))
                        .font(.system(size: 10, weight: .bold, design: .rounded))
                        .foregroundStyle(tint)
                        .padding(.horizontal, 7)
                        .padding(.vertical, 3)
                        .background(tint.opacity(0.13), in: Capsule())
                }
            }

            HStack(alignment: .firstTextBaseline, spacing: 5) {
                Text(value)
                    .font(.system(size: 38, weight: .bold, design: .rounded))
                    .monospacedDigit()
                    .foregroundStyle(tint)
                    .lineLimit(1)
                    .minimumScaleFactor(0.5)
                Text(unit)
                    .font(.system(size: 10.5, weight: .medium))
                    .foregroundStyle(SplashPalette.faint)
                    .lineLimit(1)
            }

            if !history.isEmpty {
                Sparkline(values: history, tint: tint, upperBound: chartMax)
                    .frame(height: 34)
            } else if let progress {
                Meter(fraction: progress, tint: tint)
            }

            Text(note.isEmpty ? " " : note)
                .font(.system(size: 9.5))
                .foregroundStyle(SplashPalette.faint)
                .lineLimit(1)
        }
        .padding(16)
        .frame(maxWidth: .infinity, minHeight: TileMetrics.heroHeight,
               maxHeight: TileMetrics.heroHeight, alignment: .topLeading)
        .glassPanel(radius: Theme.cardRadius, tint: tint, raised: true)
    }
}

/// A minute of readings as a line, with a soft gradient beneath it and a lit
/// dot at the newest sample.
struct Sparkline: View {
    let values: [Double]
    let tint: Color
    var lineWidth: CGFloat = 1.8
    var upperBound: Double?

    var body: some View {
        Canvas { context, size in
            guard values.count > 1, size.width > 1, size.height > 1 else { return }
            let peak = max(upperBound ?? values.max() ?? 1, 0.0001)
            let step = size.width / CGFloat(values.count - 1)
            func point(_ index: Int) -> CGPoint {
                let value = min(max(values[index], 0), peak)
                let y = size.height - CGFloat(value / peak) * (size.height - 3) - 1.5
                return CGPoint(x: CGFloat(index) * step, y: y)
            }

            var line = Path()
            line.move(to: point(0))
            for index in 1..<values.count {
                line.addLine(to: point(index))
            }

            var area = line
            area.addLine(to: CGPoint(x: size.width, y: size.height))
            area.addLine(to: CGPoint(x: 0, y: size.height))
            area.closeSubpath()

            context.fill(
                area,
                with: .linearGradient(
                    Gradient(colors: [tint.opacity(0.28), tint.opacity(0)]),
                    startPoint: CGPoint(x: 0, y: 0),
                    endPoint: CGPoint(x: 0, y: size.height)
                )
            )
            context.stroke(
                line,
                with: .linearGradient(
                    Gradient(colors: [tint.opacity(0.5), tint]),
                    startPoint: CGPoint(x: 0, y: 0),
                    endPoint: CGPoint(x: size.width, y: 0)
                ),
                style: StrokeStyle(lineWidth: lineWidth, lineCap: .round, lineJoin: .round)
            )

            let last = point(values.count - 1)
            context.fill(
                Path(ellipseIn: CGRect(x: last.x - 2.4, y: last.y - 2.4, width: 4.8, height: 4.8)),
                with: .color(tint)
            )
        }
        // No value animation: the Canvas redraws the whole line each sample
        // anyway, and an animated Canvas would snapshot and blend two draws
        // every second on every card that keeps a history.
    }
}

/// A ring gauge: the share of a budget in use, with the value in the middle.
/// Used where one number is the page's answer — the Metal memory the engine
/// holds against the budget it was planned with.
struct RingGauge: View {
    let fraction: Double
    var tint: Color = SplashPalette.blue
    var lineWidth: CGFloat = 10
    var label: String?
    var detail: String?

    private var clamped: Double { min(1, max(0, fraction)) }

    var body: some View {
        ZStack {
            Circle()
                .stroke(tint.opacity(0.12), lineWidth: lineWidth)
            Circle()
                .trim(from: 0, to: clamped)
                .stroke(
                    AngularGradient(
                        colors: [tint.opacity(0.55), tint, SplashPalette.cyan],
                        center: .center
                    ),
                    style: StrokeStyle(lineWidth: lineWidth, lineCap: .round)
                )
                .rotationEffect(.degrees(-90))

            VStack(spacing: 2) {
                if let label {
                    Text(label)
                        .font(.system(size: 21, weight: .bold, design: .rounded))
                        .monospacedDigit()
                        .lineLimit(1)
                        .minimumScaleFactor(0.6)
                }
                if let detail {
                    Text(detail)
                        .font(.system(size: 9))
                        .foregroundStyle(SplashPalette.faint)
                        .lineLimit(1)
                }
            }
            .padding(lineWidth + 4)
        }
    }
}
