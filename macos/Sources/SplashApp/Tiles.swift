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
    static let heroHeight: CGFloat = 104
    static let cardHeight: CGFloat = 84
    static let corner: CGFloat = 12
}

struct MetricSection: View {
    let title: String
    let systemImage: String
    let cards: [MetricCard]

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            Label(title, systemImage: systemImage)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.secondary)
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 190), spacing: 10)], spacing: 10) {
                ForEach(cards) { card in
                    card
                }
            }
        }
    }
}

struct MetricCard: View, Identifiable {
    let id = UUID()
    let title: String
    let value: String
    var caption = ""
    var systemImage = "circle"
    var tint: Color = .accentColor
    /// A 0...1 share of a known total; nil leaves the card without a bar.
    var progress: Double?
    var progressCaption: String?

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: systemImage)
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(tint)
                .frame(width: 28, height: 28)
                .background(tint.opacity(0.14), in: RoundedRectangle(cornerRadius: 8))
            VStack(alignment: .leading, spacing: 3) {
                Text(title).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                Text(value)
                    .font(.title3.weight(.semibold).monospacedDigit())
                    .lineLimit(1)
                    .minimumScaleFactor(0.6)
                if let progress {
                    Meter(fraction: progress, tint: tint)
                        .padding(.top, 3)
                }
                // One caption line, preferring the live share when there is one
                // so no card is a line taller than its neighbours.
                if let note = progressCaption ?? (caption.isEmpty ? nil : caption) {
                    Text(note).font(.caption2).foregroundStyle(.tertiary).lineLimit(1)
                }
            }
            Spacer(minLength: 0)
        }
        .padding(10)
        .frame(maxWidth: .infinity, minHeight: TileMetrics.cardHeight,
               maxHeight: TileMetrics.cardHeight, alignment: .topLeading)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: TileMetrics.corner))
        .overlay(
            RoundedRectangle(cornerRadius: TileMetrics.corner)
                .strokeBorder(.separator.opacity(0.5))
        )
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
                Capsule().fill(tint.opacity(0.15))
                Capsule()
                    .fill(tint)
                    .frame(width: proxy.size.width * min(1, max(0, fraction)))
            }
        }
        .frame(height: 4)
        .animation(.easeOut(duration: 0.25), value: fraction)
    }
}

/// The headline tile: the number is the point, so it is set large and the
/// icon and label stay quiet around it.
struct HeroCard: View {
    let title: String
    let value: String
    let unit: String
    let systemImage: String
    let tint: Color
    var progress: Double?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 7) {
                Image(systemName: systemImage)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(tint)
                Text(title)
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            HStack(alignment: .firstTextBaseline, spacing: 5) {
                Text(value)
                    .font(.system(size: 34, weight: .semibold))
                    .monospacedDigit()
                    .foregroundStyle(tint)
                    .lineLimit(1)
                    .minimumScaleFactor(0.5)
                Text(unit)
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
            }
            if let progress {
                Meter(fraction: progress, tint: tint)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, minHeight: TileMetrics.heroHeight,
               maxHeight: TileMetrics.heroHeight, alignment: .topLeading)
        .background(tint.opacity(0.08), in: RoundedRectangle(cornerRadius: 14))
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .strokeBorder(tint.opacity(0.18))
        )
    }
}
