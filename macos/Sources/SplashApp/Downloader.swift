import AppKit
import Foundation
import SwiftUI

/// Locates an existing prepared model through the bundled Python installer.
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
        guard let python = Runtime.python, let root = Runtime.root else {
            phase = .failed(L10n.string("download.no_runtime"))
            return
        }
        let installer = root.appendingPathComponent("install/models.py")
        log = ""
        phase = .running
        run(python: python, installer: installer, model: identifier)
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

    private func run(python: URL, installer: URL, model: String) {
        let process = Process()
        process.executableURL = python
        process.arguments = [installer.path, "--model", model, "link"]
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONUNBUFFERED"] = "1"
        process.environment = environment
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

/// Finds a model that has already been prepared on this Mac: the suggested
/// ones as a list, GGUF and MLX apart, or an ID typed by hand.
struct DownloadSheet: View {
    @ObservedObject var downloader: Downloader
    var onClose: () -> Void

    @State private var identifier = ""
    @State private var kind = SuggestedModel.ModelKind.gguf

    private var suggested: [SuggestedModel] {
        Runtime.catalog.filter { $0.kind == kind }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 10) {
                Image(systemName: "magnifyingglass.circle.fill")
                    .font(.system(size: 22, weight: .semibold))
                    .foregroundStyle(SplashPalette.cyan)
                VStack(alignment: .leading, spacing: 2) {
                    Text(L10n.string("download.title"))
                        .font(.system(size: 15, weight: .semibold))
                }
                Spacer()
            }

            if !Runtime.catalog.isEmpty {
                Picker("", selection: $kind) {
                    ForEach(SuggestedModel.ModelKind.allCases) { option in
                        Text(option.title).tag(option)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .disabled(downloader.isRunning)

                modelList
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

    /// The suggested models of the chosen kind, one row each: what choosing it
    /// downloads, and what it needs.
    private var modelList: some View {
        ScrollView {
            LazyVStack(spacing: 0) {
                ForEach(Array(suggested.enumerated()), id: \.element.id) { index, model in
                    if index > 0 {
                        Divider().opacity(0.5)
                    }
                    ModelRow(
                        model: model,
                        selected: identifier == model.id,
                        onChoose: { identifier = model.id }
                    )
                }
            }
        }
        .frame(height: 208)
        .background(SplashPalette.surface.opacity(0.4),
                    in: RoundedRectangle(cornerRadius: 10))
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .strokeBorder(SplashPalette.surfaceRaised.opacity(0.6), lineWidth: 1)
        )
    }

    private var requestCard: some View {
        HStack(spacing: 8) {
            Text(L10n.string("field.model_id"))
                .font(.system(size: 11))
                .frame(width: 96, alignment: .leading)
            EditorField(text: $identifier, prompt: "owner/repo[:variant]", monospaced: true)
                .disabled(downloader.isRunning)
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

/// One suggested model in the sheet's list: its name and variant, what
/// choosing it needs, and a few words for choosing it.
private struct ModelRow: View {
    let model: SuggestedModel
    let selected: Bool
    var onChoose: () -> Void

    @State private var hovering = false

    var body: some View {
        Button(action: onChoose) {
            HStack(spacing: 10) {
                Image(systemName: selected ? "largecircle.fill.circle" : "circle")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(selected ? SplashPalette.blue : .secondary)
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text(verbatim: model.name)
                            .font(.system(size: 12.5, weight: .medium))
                            .lineLimit(1)
                        if !model.variant.isEmpty {
                            Text(verbatim: model.variant)
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundStyle(.secondary)
                                .padding(.horizontal, 5)
                                .padding(.vertical, 1)
                                .background(
                                    SplashPalette.surfaceRaised.opacity(0.7),
                                    in: RoundedRectangle(cornerRadius: 4, style: .continuous)
                                )
                        }
                    }
                    if !model.about.isEmpty {
                        Text(verbatim: model.about)
                            .font(.system(size: 10.5))
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                }
                Spacer(minLength: 8)
                Text(verbatim: summary)
                    .font(.system(size: 10.5, design: .monospaced))
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .contentShape(Rectangle())
            .background(
                (selected || hovering) ? SplashPalette.blue.opacity(0.09) : .clear
            )
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .help(model.id)
    }

    private var summary: String {
        var parts: [String] = []
        if model.memoryGB > 0 {
            parts.append(L10n.format("download.model.memory", model.memoryGB))
        }
        if model.sizeGB > 0 {
            parts.append(L10n.format("download.model.size", model.sizeGB))
        }
        return parts.joined(separator: " · ")
    }
}
