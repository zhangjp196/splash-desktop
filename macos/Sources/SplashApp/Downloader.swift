import AppKit
import Foundation
import SwiftUI

/// Locates an existing prepared model through the bundled server.
@MainActor
final class Downloader: ObservableObject {
    enum Phase: Equatable {
        case idle
        case running
        case done(prepared: String)
        case failed(String)
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var log = ""

    private var process: Process?
    private let logLimit = 6000

    var isRunning: Bool { phase == .running }

    func start(model: String) {
        guard !isRunning else { return }
        let identifier = model.trimmingCharacters(in: .whitespaces)
        guard !identifier.isEmpty else { return }
        guard let server = Runtime.server else {
            phase = .failed(L10n.string("download.no_runtime"))
            return
        }
        log = ""
        phase = .running
        run(server: server, model: identifier)
    }

    func cancel() {
        guard let process, process.isRunning else { return }
        process.interrupt()
    }

    func reset() {
        guard !isRunning else { return }
        phase = .idle
        log = ""
    }

    private func run(server: URL, model: String) {
        let process = Process()
        process.executableURL = server
        process.arguments = ["model-path", model]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        process.terminationHandler = { [weak self] finished in
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            let output = String(decoding: data, as: UTF8.self)
            Task { @MainActor in self?.finished(finished, output: output) }
        }
        do {
            try process.run()
        } catch {
            phase = .failed(error.localizedDescription)
            return
        }
        self.process = process
    }

    private func finished(_ process: Process, output: String) {
        log = String(output.suffix(logLimit))
        self.process = nil
        guard isRunning else { return }
        if process.terminationStatus != 0 {
            let tail = log.split(separator: "\n").suffix(3).joined(separator: "\n")
            phase = .failed(tail.isEmpty
                ? L10n.format("download.failed", Int(process.terminationStatus))
                : tail)
            return
        }
        let prepared = log.trimmingCharacters(in: .whitespacesAndNewlines)
        phase = .done(prepared: prepared)
    }

}

/// Finds a model that has already been prepared on this Mac.
struct DownloadSheet: View {
    @ObservedObject var downloader: Downloader
    var onClose: () -> Void

    @State private var identifier = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 10) {
                Image(systemName: "magnifyingglass.circle.fill")
                    .font(.system(size: 22, weight: .semibold))
                    .foregroundStyle(SplashPalette.cyan)
                VStack(alignment: .leading, spacing: 2) {
                    Text(L10n.string("download.title"))
                        .font(.system(size: 15, weight: .semibold))
                    Text(L10n.string("download.subtitle"))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }

            requestCard

            if downloader.isRunning {
                ProgressView()
                    .progressViewStyle(.linear)
            }

            if !downloader.log.isEmpty {
                ScrollViewReader { proxy in
                    ScrollView {
                        Text(downloader.log)
                            .font(.system(size: 10.5, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                            .id("tail")
                    }
                    .frame(height: 120)
                    .onChange(of: downloader.log) { _, _ in
                        proxy.scrollTo("tail", anchor: .bottom)
                    }
                }
                .background(SplashPalette.surface.opacity(0.4), in: RoundedRectangle(cornerRadius: 10))
            }

            outcome
            footer
        }
        .padding(18)
        .frame(width: 620)
    }

    private var requestCard: some View {
        HStack(spacing: 8) {
            Text(L10n.string("field.model_id"))
                .font(.system(size: 11))
                .frame(width: 96, alignment: .leading)
            EditorField(text: $identifier, prompt: "owner/repo[:variant]", monospaced: true)
                .disabled(downloader.isRunning)
            if !Runtime.catalog.isEmpty {
                Menu {
                    ForEach(Runtime.catalog, id: \.self) { known in
                        Button(known) { identifier = known }
                    }
                } label: {
                    Image(systemName: "sparkles")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(SplashPalette.cyan)
                        .frame(width: 28, height: 28)
                        .background(
                            SplashPalette.cyan.opacity(0.13),
                            in: RoundedRectangle(cornerRadius: 7, style: .continuous)
                        )
                }
                .menuStyle(.borderlessButton)
                .menuIndicator(.hidden)
                .fixedSize()
                .help(L10n.string("examples"))
            }
        }
    }

    @ViewBuilder private var outcome: some View {
        switch downloader.phase {
        case .done(let prepared):
            VStack(alignment: .leading, spacing: 4) {
                Label(L10n.string("download.done"), systemImage: "checkmark.circle.fill")
                    .font(.system(size: 11.5, weight: .semibold))
                    .foregroundStyle(SplashPalette.mint)
                if !prepared.isEmpty {
                    Text(verbatim: prepared)
                        .font(.system(size: 10.5, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                        .textSelection(.enabled)
                }
                Text(L10n.string("download.done.help"))
                    .font(.system(size: 10))
                    .foregroundStyle(.secondary)
            }
        case .failed(let message):
            Label(message, systemImage: "exclamationmark.triangle.fill")
                .font(.system(size: 11))
                .foregroundStyle(SplashPalette.rose)
                .lineLimit(3)
        default:
            EmptyView()
        }
    }

    private var footer: some View {
        HStack(spacing: 10) {
            Spacer()
            if downloader.isRunning {
                Button(L10n.string("download.stop")) { downloader.cancel() }
            } else {
                Button(L10n.string("config.cancel")) { onClose() }
                Button(L10n.string("download.start")) {
                    downloader.start(model: identifier)
                }
                .buttonStyle(.borderedProminent)
                .tint(SplashPalette.blue)
                .disabled(identifier.trimmingCharacters(in: .whitespaces).isEmpty)
            }
        }
    }
}
