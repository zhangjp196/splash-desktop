import AppKit
import Foundation
import SwiftUI

/// The explicit downloader: it prepares a Hub model — and, when one is named,
/// its DFlash2 draft — into the base folder's cache, without starting a
/// server. Preparing is the launcher's own installer (`install/models.py`),
/// run with the app-level source, so a serve of the same model afterwards
/// finds everything ready.
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
    /// The last percentage a Hugging Face progress bar reported, when it did.
    @Published private(set) var progress: Double?

    private var process: Process?
    private var source = SourceSettings()
    private var steps: [[String]] = []
    private var step = 0
    private var prepared = ""
    private let logLimit = 6000

    var isRunning: Bool { phase == .running }

    /// Runs the installer's own two steps: `prepare` downloads and lays the
    /// files out, `link` publishes the root the launcher serves from.
    func start(model: String, source: SourceSettings) {
        guard !isRunning else { return }
        let identifier = model.trimmingCharacters(in: .whitespaces)
        guard !identifier.isEmpty else { return }
        guard let root = Runtime.root, let python = Runtime.python else {
            phase = .failed(L10n.string("download.no_runtime"))
            return
        }
        let installer = root.appendingPathComponent("install/models.py").path
        let options = ["--model", identifier]
        steps = [
            [installer] + options + ["prepare"],
            [installer] + options + ["link"],
        ]
        step = 0
        prepared = ""
        log = ""
        progress = nil
        self.source = source
        phase = .running
        run(python: python.path, root: root.path)
    }

    func cancel() {
        guard let process, process.isRunning else { return }
        process.interrupt()
    }

    func reset() {
        guard !isRunning else { return }
        phase = .idle
        log = ""
        progress = nil
    }

    private func run(python: String, root: String) {
        guard step < steps.count else {
            phase = .done(prepared: prepared)
            return
        }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: python)
        process.arguments = steps[step]
        process.currentDirectoryURL = URL(fileURLWithPath: root, isDirectory: true)
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["TRANSFORMERS_VERBOSITY"] = "error"
        // A download always asks the Hub: the offline switch a serve may want
        // is not this.
        environment.removeValue(forKey: "HF_HUB_OFFLINE")
        environment.merge(source.environment) { _, setting in setting }
        process.environment = environment
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor in self?.append(text) }
        }
        process.terminationHandler = { [weak self] finished in
            Task { @MainActor in self?.finished(finished) }
        }
        do {
            try process.run()
        } catch {
            phase = .failed(error.localizedDescription)
            return
        }
        self.process = process
    }

    private func finished(_ process: Process) {
        self.process = nil
        guard isRunning else { return }
        if process.terminationStatus != 0 {
            let tail = log.split(separator: "\n").suffix(3).joined(separator: "\n")
            phase = .failed(tail.isEmpty
                ? L10n.format("download.failed", Int(process.terminationStatus))
                : tail)
            return
        }
        // `link` answers the root it published, one line, which a later serve
        // of the same model reuses.
        if step == 1, let last = log.split(separator: "\n").last {
            prepared = String(last).trimmingCharacters(in: .whitespaces)
        }
        step += 1
        guard step < steps.count, let python = Runtime.python, let root = Runtime.root else {
            phase = .done(prepared: prepared)
            return
        }
        run(python: python.path, root: root.path)
    }

    private func append(_ text: String) {
        log += text
        if log.count > logLimit { log = String(log.suffix(logLimit)) }
        if let match = text.range(of: #"(\d{1,3})%\|"#, options: .regularExpression) {
            let digits = text[match].dropLast(2)
            if let value = Double(digits) { progress = min(max(value / 100.0, 0), 1) }
        }
    }

}

/// The downloader's own sheet: the app-level source above, the model to fetch
/// below, and the progress of both installer steps in between.
struct DownloadSheet: View {
    @EnvironmentObject private var model: AppModel
    @ObservedObject var downloader: Downloader
    var onClose: () -> Void

    @State private var identifier = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 10) {
                Image(systemName: "arrow.down.circle.fill")
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

            sourceCard
            requestCard

            if let progress = downloader.progress, downloader.isRunning {
                ProgressView(value: progress)
                    .progressViewStyle(.linear)
                    .tint(SplashPalette.cyan)
            } else if downloader.isRunning {
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

    /// The base folder is chosen, not typed: an NSOpenPanel path, the Hugging
    /// Face default while it is empty.
    private var sourceCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(L10n.string("download.source.title"))
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(SplashPalette.cyan)
            HStack(spacing: 8) {
                Text(L10n.string("download.base"))
                    .font(.system(size: 11))
                    .frame(width: 96, alignment: .leading)
                Text(verbatim: model.source.baseDirectory.isEmpty
                     ? L10n.string("download.base.default")
                     : (model.source.baseDirectory as NSString).expandingTildeInPath)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(model.source.baseDirectory.isEmpty ? .secondary : .primary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 9)
                    .padding(.vertical, 6)
                    .background(SplashPalette.surface.opacity(0.55),
                                in: RoundedRectangle(cornerRadius: 7, style: .continuous))
                Button { chooseBase() } label: {
                    Image(systemName: "folder")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(SplashPalette.cyan)
                        .frame(width: 28, height: 28)
                        .background(SplashPalette.surface.opacity(0.72),
                                    in: RoundedRectangle(cornerRadius: 7, style: .continuous))
                }
                .buttonStyle(.plain)
                .disabled(downloader.isRunning)
                .help(L10n.string("download.base.choose"))
                if !model.source.baseDirectory.isEmpty {
                    Button {
                        model.source.baseDirectory = ""
                        model.saveSource()
                    } label: {
                        Image(systemName: "arrow.counterclockwise")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundStyle(SplashPalette.muted)
                            .frame(width: 28, height: 28)
                            .background(SplashPalette.surface.opacity(0.72),
                                        in: RoundedRectangle(cornerRadius: 7, style: .continuous))
                    }
                    .buttonStyle(.plain)
                    .disabled(downloader.isRunning)
                    .help(L10n.string("download.base.reset"))
                }
            }
            HStack(spacing: 8) {
                Text(L10n.string("download.endpoint"))
                    .font(.system(size: 11))
                    .frame(width: 96, alignment: .leading)
                EditorField(text: $model.source.hubEndpoint,
                            prompt: "https://huggingface.co", monospaced: true)
                    .disabled(downloader.isRunning)
            }
            Text(L10n.string("download.source.help"))
                .font(.system(size: 10))
                .foregroundStyle(.secondary)
        }
        .padding(11)
        .background(SplashPalette.surface.opacity(0.4), in: RoundedRectangle(cornerRadius: 11))
        .onChange(of: model.source) { _, _ in model.saveSource() }
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
                    downloader.start(model: identifier, source: model.source)
                }
                .buttonStyle(.borderedProminent)
                .tint(SplashPalette.blue)
                .disabled(identifier.trimmingCharacters(in: .whitespaces).isEmpty)
            }
        }
    }

    private func chooseBase() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = L10n.string("field.browse")
        if panel.runModal() == .OK, let url = panel.urls.first {
            model.source.baseDirectory = url.path
            model.saveSource()
        }
    }
}
