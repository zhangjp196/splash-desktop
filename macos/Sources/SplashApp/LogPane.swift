import SwiftUI

/// The server's log, in the window rather than a terminal: monospaced, live,
/// and scrolled to the end as it grows. Lines render as one `Text` each in a
/// `LazyVStack`, so a burst of output lays out the rows that appeared, not
/// the whole transcript again. The console keeps its own dark glass whatever
/// the window appearance, so long output is easy on the eyes.
struct LogPane: View {
    @EnvironmentObject private var model: AppModel
    /// True while the view follows the transcript's tail: new lines then
    /// scroll themselves into view. Scrolling up clears it — a reader of
    /// history must not be yanked back by live output — and reaching the
    /// bottom again, or the jump button, restores it.
    @State private var pinnedToBottom = true
    /// The transcript's bottom edge and height, and the viewport's height,
    /// as the geometry probes report them. Together they separate "new
    /// lines arrived" from "the reader scrolled away", which is what the
    /// pin above turns on.
    @State private var contentBottom: CGFloat = 0
    @State private var contentHeight: CGFloat = 0
    @State private var lastContentHeight: CGFloat = 0
    @State private var viewportHeight: CGFloat = 1

    /// How far past the viewport's bottom the content may sit and still
    /// count as pinned: slack for the rounding of a scroll in flight.
    private static let tailSlack: CGFloat = 24
    /// The scroll view's coordinate space, and the row the tail scrolls to.
    private static let tailSpace = "log.tail.space"
    private static let tailAnchor = "log.tail.anchor"

    var body: some View {
        VStack(spacing: 0) {
            PaneHeader(title: L10n.string("view.log"), subtitle: logSubtitle, eyebrow: L10n.string("log.eyebrow")) {
                HStack(spacing: 8) {
                    if model.isRunning {
                        HStack(spacing: 5) {
                            StatusBeacon(color: model.phase == .ready ? SplashPalette.mint : SplashPalette.amber, active: true)
                                .scaleEffect(0.75)
                            Text(L10n.string("log.live").uppercased())
                                .font(.system(size: 9, weight: .bold))
                                .tracking(1)
                                .foregroundStyle(SplashPalette.mint)
                        }
                        .padding(.horizontal, 9)
                        .padding(.vertical, 5)
                        .background(SplashPalette.mint.opacity(0.10), in: Capsule())
                    }
                    Button {
                        model.clearLog()
                    } label: {
                        Label(L10n.string("log.clear"), systemImage: "trash")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .buttonStyle(.borderless)
                    .disabled(logIsEmpty)
                }
            }
            .padding(.horizontal, Theme.pagePadding)
            .padding(.top, Theme.pagePadding)
            .padding(.bottom, 16)

            console
                .padding(.horizontal, Theme.pagePadding)
                .padding(.bottom, Theme.pagePadding)
        }
        .background(Color.clear)
    }

    /// A transcript with only an open line — a progress bar mid-rewrite —
    /// is not empty: the pane shows it and Clear works all the same.
    private var logIsEmpty: Bool {
        model.logLines.isEmpty && model.logPendingLine.isEmpty
    }

    private var console: some View {
        VStack(spacing: 0) {
            HStack(spacing: 6) {
                ForEach([Color(red: 1, green: 0.37, blue: 0.34), Color(red: 1, green: 0.74, blue: 0.18), Color(red: 0.16, green: 0.79, blue: 0.25)], id: \.self) { color in
                    Circle().fill(color.opacity(0.85)).frame(width: 7, height: 7)
                }
                Spacer()
                Text(verbatim: "splash — live output")
                    .font(.system(size: 9, weight: .medium, design: .monospaced))
                    .foregroundStyle(Color.white.opacity(0.32))
                Spacer()
                Text(verbatim: "\(model.logCharacterCount.formatted()) chars")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(Color.white.opacity(0.25))
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 9)
            .background(Color.white.opacity(0.025))

            Rectangle()
                .fill(Color.white.opacity(0.06))
                .frame(height: 1)

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 0) {
                        if logIsEmpty {
                            Text(L10n.string("log.empty"))
                                .font(.system(size: 11.5, design: .monospaced))
                                .foregroundStyle(Color.white.opacity(0.35))
                        } else {
                            ForEach(model.logLines.indices, id: \.self) { index in
                                logRow(model.logLines[index])
                            }
                            if !model.logPendingLine.isEmpty {
                                logRow(model.logPendingLine)
                            }
                        }
                        Color.clear
                            .frame(height: 1)
                            .id(Self.tailAnchor)
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 14)
                    .background(contentProbe)
                }
                .coordinateSpace(name: Self.tailSpace)
                .background(viewportProbe)
                .onPreferenceChange(LogContentMetricsKey.self) { metrics in
                    if metrics.bottom != contentBottom { contentBottom = metrics.bottom }
                    if metrics.height != contentHeight { contentHeight = metrics.height }
                    syncPin()
                }
                .onPreferenceChange(LogViewportHeightKey.self) { height in
                    if height != viewportHeight { viewportHeight = height }
                    syncPin()
                }
                .overlay(alignment: .bottom) {
                    if !pinnedToBottom {
                        jumpButton(proxy)
                    }
                }
                .animation(.easeOut(duration: 0.15), value: pinnedToBottom)
                .onChange(of: model.logLines) { _, _ in
                    followTail(proxy)
                }
                .onChange(of: model.logPendingLine) { _, _ in
                    followTail(proxy)
                }
                .onAppear {
                    // A fresh pane opens on the tail whatever the
                    // transcript's length, and starts pinned to keep it
                    // there while output arrives.
                    pinnedToBottom = true
                    proxy.scrollTo(Self.tailAnchor, anchor: .bottom)
                }
            }
        }
        .background(
            RoundedRectangle(cornerRadius: Theme.cardRadius, style: .continuous)
                .fill(
                    LinearGradient(
                        colors: [
                            Color(red: 0.035, green: 0.052, blue: 0.075),
                            Color(red: 0.025, green: 0.035, blue: 0.055),
                        ],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
        )
        .overlay(
            RoundedRectangle(cornerRadius: Theme.cardRadius, style: .continuous)
                .strokeBorder(
                    LinearGradient(
                        colors: [Color.white.opacity(0.16), Color.white.opacity(0.03)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    ),
                    lineWidth: 0.8
                )
        )
        .clipShape(RoundedRectangle(cornerRadius: Theme.cardRadius, style: .continuous))
        // No hover lift and no large shadow: the console is a full-width pane,
        // and a scale or an offscreen shadow pass on it would cost more than
        // the whole metrics board.
    }

    /// One line of the transcript. It wraps like the single block it came
    /// from, and selects on its own: selection now spans one row rather
    /// than the whole transcript, the trade for never re-laying-out all of
    /// it.
    private func logRow(_ line: String) -> some View {
        Text(verbatim: line)
            .font(.system(size: 11.5, design: .monospaced))
            .foregroundStyle(Color(red: 0.78, green: 0.94, blue: 0.88))
            .textSelection(.enabled)
            .lineSpacing(3)
            .fixedSize(horizontal: false, vertical: true)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// The transcript's own geometry, read at its background: where its
    /// bottom edge sits in the viewport's space, and how tall it is.
    private var contentProbe: some View {
        GeometryReader { geo in
            let frame = geo.frame(in: .named(Self.tailSpace))
            Color.clear.preference(
                key: LogContentMetricsKey.self,
                value: LogContentMetrics(bottom: frame.maxY, height: geo.size.height)
            )
        }
    }

    /// The viewport's height, read from the scroll view's own background,
    /// which does not scroll with the content.
    private var viewportProbe: some View {
        GeometryReader { geo in
            Color.clear.preference(key: LogViewportHeightKey.self, value: geo.size.height)
        }
    }

    /// Re-derive the pin from the latest geometry. New lines push the
    /// content's bottom down by exactly their own height before the follow
    /// scroll lands, so growth that deep is not the reader leaving: only
    /// distance beyond what the new lines explain is a scroll away, and
    /// distance back inside the slack is the reader at the bottom again.
    private func syncPin() {
        let distance = contentBottom - viewportHeight
        let growth = max(0, contentHeight - lastContentHeight)
        if contentHeight != lastContentHeight { lastContentHeight = contentHeight }
        if distance > Self.tailSlack + growth {
            if pinnedToBottom { pinnedToBottom = false }
        } else if distance <= Self.tailSlack {
            if !pinnedToBottom { pinnedToBottom = true }
        }
    }

    /// Keep the tail in view while pinned. The scroll is deliberately not
    /// animated: an animated follow would sample as "away from the bottom"
    /// mid-flight and read like the reader leaving.
    private func followTail(_ proxy: ScrollViewProxy) {
        guard pinnedToBottom else { return }
        proxy.scrollTo(Self.tailAnchor, anchor: .bottom)
    }

    /// The way back to the tail once the reader has scrolled away: a quiet
    /// round button over the console's bottom edge. Icon only — the pane's
    /// strings are localized, and this glyph needs no words.
    private func jumpButton(_ proxy: ScrollViewProxy) -> some View {
        Button {
            proxy.scrollTo(Self.tailAnchor, anchor: .bottom)
        } label: {
            Image(systemName: "arrow.down.to.line")
                .font(.system(size: 11, weight: .bold))
                .foregroundStyle(Color.white.opacity(0.8))
                .padding(9)
                .background(Circle().fill(Color.black.opacity(0.55)))
                .overlay(Circle().strokeBorder(Color.white.opacity(0.18), lineWidth: 0.8))
        }
        .buttonStyle(.plain)
        .padding(.bottom, 12)
        .transition(.opacity)
    }

    private var logSubtitle: String? {
        guard model.isRunning else { return nil }
        return logIsEmpty ? nil : L10n.string("log.live")
    }
}

/// The transcript's bottom edge in the viewport's space, and its height:
/// the pair separates "new lines arrived" from "the reader scrolled away".
private struct LogContentMetrics: Equatable {
    var bottom: CGFloat
    var height: CGFloat
}

private struct LogContentMetricsKey: PreferenceKey {
    static let defaultValue = LogContentMetrics(bottom: 0, height: 0)
    static func reduce(value: inout LogContentMetrics, nextValue: () -> LogContentMetrics) {
        value = nextValue()
    }
}

/// How tall the viewport is, read from the scroll view's own background.
private struct LogViewportHeightKey: PreferenceKey {
    static let defaultValue: CGFloat = 1
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = nextValue()
    }
}
