import Foundation

/// The live board's formatting helpers, shared by the metric sections and the
/// cards: counts, byte figures, rates and the captions that give a number its
/// context. They read nothing but their arguments, so they sit beside the
/// board as an extension and every caller keeps the call it already had.
extension LivePane {

    /// "max 123 tokens/s", or nothing while the panel has not seen a reading.
    func peakNote(_ key: String, _ value: Double?, percent: Bool = false) -> String {
        guard let value, value > 0 else { return "" }
        return L10n.format(key, percent ? self.percent(value) : rate(value))
    }
    func speculationLabel(_ provider: String?) -> String {
        switch provider {
        case "dflash2": return "DFlash2"
        case "none": return L10n.string("speculation.target_only")
        default: return provider ?? "—"
        }
    }

    /// The rows a cycle verifies and, when a requested draft fell back, why.
    func speculationCaption(_ live: LiveStatus) -> String {
        if let fallback = live.speculationFallback, !fallback.isEmpty {
            return L10n.format("live.caption.speculation_fallback", fallback)
        }
        var parts: [String] = []
        if let rows = live.speculationRows {
            parts.append(L10n.format("live.caption.speculation_rows", rows))
        }
        return parts.joined(separator: " · ")
    }
    /// The weights' idle release as one card, beside the memory they hold:
    /// resident, released (the next request pays to take them back), or
    /// restoring now. The caption names the timer and how many restores the
    /// engine has made, so a released value does not look like a fault.
    func weightsState(_ live: LiveStatus) -> String {
        guard live.weightsKnown else { return "—" }
        if live.weightsRestoring { return L10n.string("live.weights.restoring") }
        if live.weightsReleased { return L10n.string("live.weights.released") }
        return L10n.string("live.weights.resident")
    }

    func weightsCaption(_ live: LiveStatus) -> String {
        guard live.weightsKnown else { return "" }
        guard let idle = live.idleReleaseSeconds else {
            return L10n.string("live.weights.caption_off")
        }
        let release = L10n.format("live.weights.caption", Int(idle.rounded()))
        guard live.weightsRestores > 0 else { return release }
        return L10n.format("live.weights.caption_restores", release, live.weightsRestores)
    }

    /// "96 of 128 tokens": the counts behind the disk reuse rate, so a rate
    /// read alone does not hide how much was reused at all.
    func diskReuseCaption(_ live: LiveStatus) -> String {
        guard let reused = live.reusedTokens, reused > 0 else { return "" }
        return L10n.format("live.caption.share", number(live.kvDiskHitTokens), number(reused))
    }

    func milliseconds(_ value: Double?) -> String {
        guard let value else { return "—" }
        return value < 10 ? String(format: "%.1f", value) : String(format: "%.0f", value)
    }

    func latencyCaption(p95: Double?, samples: Int) -> String {
        guard samples > 0, let p95 else {
            return L10n.string("live.caption.latency_empty")
        }
        return L10n.format("live.caption.latency", milliseconds(p95), samples)
    }

    /// The bar's note under the physical memory card: the share of the budget
    /// and the peak it has reached, either of which can be missing without
    /// taking the other with it.
    func heldNote(_ share: String?, _ peak: UInt64?) -> String? {
        let notes = [share, peak.map { L10n.format("live.caption.peak_of", bytes($0)) }]
            .compactMap { $0 }
        return notes.isEmpty ? nil : notes.joined(separator: " · ")
    }

    /// A used-of-total pair as a fraction and a caption, either of which is nil
    /// when the total is unknown so nothing renders a misleading bar.
    func share(
        _ used: UInt64?, of total: UInt64?
    ) -> (value: Double?, caption: String?) {
        guard let used, let total, total > 0 else { return (nil, nil) }
        let fraction = min(1, Double(used) / Double(total))
        return (fraction, L10n.format("live.caption.share", percent(fraction), bytes(total)))
    }

    func tokens(_ value: Int?) -> String {
        number(value)
    }

    /// A count with grouping separators, or a dash when the status left it out.
    func number(_ value: Int?) -> String {
        value.map { $0.formatted() } ?? "—"
    }

    func bytes(_ value: UInt64?) -> String {
        guard let value else { return "—" }
        return ByteCountFormatter.string(fromByteCount: Int64(value), countStyle: .memory)
    }

    func rate(_ value: Double?) -> String {
        guard let value else { return "—" }
        return String(format: "%.0f", value)
    }

    func percent(_ value: Double?) -> String {
        guard let value else { return "—" }
        return String(format: "%.1f%%", value * 100)
    }

    func disk(_ used: UInt64?, _ capacity: UInt64?) -> String {
        switch (used, capacity) {
        case (nil, nil): return "—"
        case (let used?, nil): return bytes(used)
        default:
            let percentValue = capacity.map { 100.0 * Double(used!) / Double($0) } ?? 0
            return String(format: "%@ · %.0f%%", bytes(used), percentValue)
        }
    }

    func idleOffloadCaption(_ live: LiveStatus) -> String {
        guard live.idleOffloadPasses > 0 || live.idleOffloadBytes != nil else { return "" }
        let offloadBytes = live.idleOffloadBytes.map {
            ByteCountFormatter.string(fromByteCount: Int64($0), countStyle: .memory)
        } ?? "—"
        return L10n.format("live.caption.idle_offload", live.idleOffloadPasses, offloadBytes)
    }
}
