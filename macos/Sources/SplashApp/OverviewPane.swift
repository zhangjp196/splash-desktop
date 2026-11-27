import Charts
import SwiftUI

struct OverviewPane: View {
    @EnvironmentObject private var model: AppModel
    let selection: Binding<RootView.Section>

    private var modelName: String {
        model.runningConfig?.displayName ?? model.live.modelID ?? model.activeConfig?.displayName ?? L10n.string("models.none")
    }
    private var endpoint: String { "http://127.0.0.1:\(model.global.port)/v1" }
    private var requestTotal: Int? { model.live.updated ? model.live.completed : nil }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                HStack(alignment: .bottom) {
                    VStack(alignment: .leading, spacing: 5) {
                        Text(L10n.string("view.overview")).font(.system(size: 30, weight: .bold, design: .rounded)).tracking(-0.5)
                        Text(L10n.string("overview.description")).font(.system(size: 12)).foregroundStyle(SplashPalette.muted)
                    }
                    Spacer()
                    Button { model.copyToClipboard(endpoint) } label: { Label(L10n.string("overview.copy_api"), systemImage: "doc.on.doc") }
                        .buttonStyle(.bordered).tint(SplashPalette.blue)
                }
                hero
                LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible()), GridItem(.flexible()), GridItem(.flexible())], spacing: 11) {
                    metric(L10n.string("overview.metric.decode"), model.live.updated ? model.live.decodeTokensPerSecond.map { String(format: "%.1f", $0) } : nil, "tok/s", "bolt.fill")
                    metric(L10n.string("overview.metric.ttft"), model.live.ttftSamples > 0 ? model.live.ttftP50Milliseconds.map { String(format: "%.0f", $0) } : nil, "ms", "clock")
                    metric(L10n.string("overview.metric.memory"), model.live.updated ? model.live.physicalBytes.map { ByteCountFormatter.string(fromByteCount: Int64($0), countStyle: .memory) } : nil, "", "memorychip")
                    metric(L10n.string("overview.metric.completed"), requestTotal.map(String.init), "", "arrow.trianglehead.2.clockwise")
                }
                HStack(alignment: .top, spacing: 14) {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            VStack(alignment: .leading, spacing: 4) { Text(L10n.string("overview.chart.title")).font(.headline); Text(L10n.string("overview.chart.caption")).font(.caption).foregroundStyle(SplashPalette.muted) }
                            Spacer()
                            Button(L10n.string("overview.chart.details")) { selection.wrappedValue = .live }.buttonStyle(.borderless).foregroundStyle(SplashPalette.blue)
                        }
                        if model.decodeHistory.isEmpty {
                            ContentUnavailableView(L10n.string("overview.chart.empty.title"), systemImage: "waveform.path", description: Text(L10n.string("overview.chart.empty.description")))
                                .frame(height: 160)
                        } else {
                            Chart(Array(model.decodeHistory.enumerated()), id: \.offset) { item in
                                AreaMark(x: .value(L10n.string("overview.chart.sample"), item.offset), y: .value("tok/s", item.element)).foregroundStyle(SplashPalette.mint.opacity(0.18))
                                LineMark(x: .value(L10n.string("overview.chart.sample"), item.offset), y: .value("tok/s", item.element)).foregroundStyle(SplashPalette.mint).lineStyle(StrokeStyle(lineWidth: 2.5, lineCap: .round))
                            }
                            .chartYScale(domain: 0...max(model.decodeHistory.max() ?? 1, 1))
                            .chartXAxis(.hidden).chartYAxis { AxisMarks(position: .leading) }
                            .frame(height: 165).padding(.top, 4)
                        }
                    }
                    .padding(16).frame(maxWidth: .infinity, alignment: .leading).glassPanel(radius: 16)
                    environmentCard.frame(width: 255)
                }
                activityCard
            }
            .padding(26).frame(maxWidth: 1160, alignment: .leading).frame(maxWidth: .infinity)
        }
        .background(SplashPalette.canvas)
    }

    private var hero: some View {
        HStack(spacing: 18) {
            ZStack { RoundedRectangle(cornerRadius: 18).fill(.white.opacity(0.65)).frame(width: 64, height: 64); Image(systemName: "cpu.fill").font(.system(size: 27)).foregroundStyle(SplashPalette.blue) }
            VStack(alignment: .leading, spacing: 7) {
                Label(L10n.format("overview.server", model.statusText), systemImage: model.isRunning ? "circle.fill" : "circle")
                    .font(.system(size: 11, weight: .semibold)).foregroundStyle(model.isRunning ? SplashPalette.mint : SplashPalette.muted)
                Text(modelName).font(.system(size: 22, weight: .bold, design: .rounded)).lineLimit(1)
                HStack(spacing: 8) { Text(endpoint).font(.system(size: 11, design: .monospaced)); Button { model.copyToClipboard(endpoint) } label: { Image(systemName: "doc.on.doc") }.buttonStyle(.borderless) }
                    .foregroundStyle(SplashPalette.muted)
            }
            Spacer(minLength: 8)
            Button { model.isRunning ? model.stop() : model.start() } label: {
                Label(L10n.string(model.isRunning ? "overview.stop" : "overview.start"), systemImage: model.isRunning ? "stop.fill" : "play.fill")
                    .font(.system(size: 12, weight: .semibold)).padding(.horizontal, 13).padding(.vertical, 9)
            }.buttonStyle(.borderedProminent).tint(model.isRunning ? SplashPalette.sidebar : SplashPalette.blue).disabled(model.isBusy || (!model.isRunning && !model.canStart))
        }
        .padding(20).background(LinearGradient(colors: [Color(red: 0.91, green: 0.95, blue: 1.0), Color(red: 0.78, green: 0.87, blue: 1.0)], startPoint: .leading, endPoint: .trailing), in: RoundedRectangle(cornerRadius: 18))
        .overlay(RoundedRectangle(cornerRadius: 18).stroke(SplashPalette.mint.opacity(0.4), lineWidth: 1))
    }

    private func metric(_ title: String, _ value: String?, _ unit: String, _ icon: String) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Label(title, systemImage: icon).font(.system(size: 11, weight: .medium)).foregroundStyle(SplashPalette.muted)
            HStack(alignment: .firstTextBaseline, spacing: 4) { Text(value ?? "—").font(.system(size: 25, weight: .bold, design: .rounded)).monospacedDigit(); if !unit.isEmpty { Text(unit).font(.caption).foregroundStyle(SplashPalette.muted) } }
        }.frame(maxWidth: .infinity, alignment: .leading).padding(14).glassPanel(radius: 14)
    }

    private var environmentCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(L10n.string("overview.environment")).font(.headline)
            rowValue(L10n.string("overview.model"), modelName)
            rowValue(L10n.string("overview.backend"), model.live.updated ? L10n.string("overview.backend.metal") : "—")
            rowValue(L10n.string("overview.speculation"), model.live.speculationProvider ?? "—")
            Button(L10n.string("overview.view_models")) { selection.wrappedValue = .models }.buttonStyle(.borderless).foregroundStyle(SplashPalette.blue).padding(.top, 4)
        }.padding(16).frame(maxWidth: .infinity, alignment: .leading).glassPanel(radius: 16)
    }
    private func rowValue(_ title: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline) { Text(title).foregroundStyle(SplashPalette.muted); Spacer(minLength: 6); Text(value).fontWeight(.semibold).lineLimit(1).truncationMode(.middle) }.font(.system(size: 11)).padding(.vertical, 4)
    }

    private var activityCard: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack { Text(L10n.string("overview.recent_activity")).font(.headline); Spacer(); Button(L10n.string("overview.view_log")) { selection.wrappedValue = .log }.buttonStyle(.borderless).foregroundStyle(SplashPalette.blue) }
            if model.recentEvents.isEmpty {
                Text(L10n.string("overview.no_activity")).font(.system(size: 12)).foregroundStyle(SplashPalette.muted).frame(maxWidth: .infinity, alignment: .leading).padding(.vertical, 12)
            } else {
                ForEach(model.recentEvents.prefix(4)) { event in
                    HStack(spacing: 9) {
                        Circle().fill(event.kind == .completed ? SplashPalette.mint : SplashPalette.amber).frame(width: 7, height: 7)
                        Text(L10n.string(event.kind == .completed ? "overview.request.completed" : event.kind == .failed ? "overview.request.failed" : "overview.request.cancelled")).font(.system(size: 12, weight: .medium))
                        Spacer(); if let rate = event.tokensPerSecond { Text(String(format: "%.1f tok/s", rate)).font(.system(size: 10, design: .monospaced)).foregroundStyle(SplashPalette.muted) }
                        Text(event.date, style: .time).font(.system(size: 10, design: .monospaced)).foregroundStyle(SplashPalette.muted)
                    }.padding(.vertical, 5)
                }
            }
        }.padding(16).glassPanel(radius: 16)
    }
}
