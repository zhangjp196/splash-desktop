import SwiftUI

// MARK: - Rail

/// What a rail of sections needs from its sections: the name, the one-line
/// explanation, the icon, and the tint the rail paints them with. The
/// configuration editor's sections and the app-wide settings' ones both
/// qualify, so one rail item serves both pages.
protocol RailSection: Identifiable {
    var title: String { get }
    var summary: String { get }
    var systemImage: String { get }
    var tint: Color { get }
}

/// One section in a rail: an icon, a name, a one-line explanation, and — where
/// the caller tracks a draft — a dot when the section differs from the value it
/// opened on. The selected row wears a flat tint wash with a hairline of the
/// same tint, and a hover lays a quiet surface wash under the pointer; there is
/// no hover animation, matching the flattened chrome the live board uses.
struct RailItem<Section: RailSection>: View {
    let section: Section
    let selected: Bool
    var changed = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                Image(systemName: section.systemImage)
                    .font(.system(size: 11.5, weight: .semibold))
                    .foregroundStyle(selected ? .white : section.tint)
                    .frame(width: 26, height: 26)
                    .background(
                        // A flat tint fill rather than a gradient: the
                        // selection changes a few times a minute, and a
                        // gradient would blend an extra layer for a
                        // two-state chip.
                        selected
                            ? AnyShapeStyle(section.tint.opacity(0.85))
                            : AnyShapeStyle(section.tint.opacity(0.13)),
                        in: RoundedRectangle(cornerRadius: 7, style: .continuous)
                    )
                VStack(alignment: .leading, spacing: 2) {
                    Text(section.title)
                        .font(.system(size: 12, weight: selected ? .semibold : .medium))
                        .foregroundStyle(.primary)
                    Text(section.summary)
                        .font(.system(size: 9.5))
                        .foregroundStyle(SplashPalette.muted)
                        .lineLimit(2)
                        .multilineTextAlignment(.leading)
                }
                Spacer(minLength: 0)
                if changed {
                    Circle()
                        .fill(section.tint)
                        .frame(width: 5, height: 5)
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 7)
            .hoverHighlight(selected: selected, tint: section.tint)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Hover

/// The quiet hover chrome a selectable row shares: a surface wash under the
/// pointer, and where the row is the selected one, a flat tint wash with a
/// hairline of the same tint. The fill appears as the pointer arrives, with no
/// animation of its own — an animated hover keeps a row re-rendering for a
/// shade the pointer has already passed.
private struct HoverHighlight: ViewModifier {
    var selected: Bool
    var tint: Color
    var radius: CGFloat

    @State private var hovering = false

    func body(content: Content) -> some View {
        content
            .background(
                selected
                    ? AnyShapeStyle(tint.opacity(0.14))
                    : AnyShapeStyle(hovering ? SplashPalette.surface.opacity(0.5) : Color.clear),
                in: RoundedRectangle(cornerRadius: radius, style: .continuous)
            )
            .overlay(
                RoundedRectangle(cornerRadius: radius, style: .continuous)
                    .strokeBorder(selected ? tint.opacity(0.38) : Color.clear, lineWidth: 0.8)
            )
            .onHover { hovering = $0 }
    }
}

extension View {
    /// Applies the shared hover-and-selection chrome: a flat tint wash and a
    /// hairline while `selected`, a surface wash while the pointer is over the
    /// view otherwise.
    func hoverHighlight(selected: Bool, tint: Color, radius: CGFloat = 10) -> some View {
        modifier(HoverHighlight(selected: selected, tint: tint, radius: radius))
    }
}
