import Foundation

/// What the picked model can use, read from its name before it is loaded. The
/// editor shows only the launch options the model can take, and a start
/// passes only those, so a value kept for another model cannot reach this one.
/// A model we cannot place keeps every option: nothing is hidden by a guess.
struct ModelProfile: Equatable {
    /// Routed experts: the expert store's memory option.
    var experts: Bool
    /// Qwen3.8-Flash-Next's PLE n-gram table.
    var ple: Bool
    /// A vision projector.
    var vision: Bool
    /// Thinking effort and a thinking token budget.
    var thinking: Bool
    /// Block-sparse attention: the sparse KV cache's memory option.
    var sparseKv: Bool

    static let everything = ModelProfile(
        experts: true, ple: true, vision: true, thinking: true, sparseKv: true
    )

    static func of(modelID: String, directory: String) -> ModelProfile {
        let name = "\(modelID) \(directory)".lowercased()
        guard !name.trimmingCharacters(in: .whitespaces).isEmpty else {
            return .everything
        }
        // Flint is text only and has no thinking mode.
        if name.contains("flint") {
            return ModelProfile(
                experts: false, ple: false, vision: false, thinking: false,
                sparseKv: false
            )
        }
        // Flash-Next is the sparse model with a PLE table.
        if name.contains("flash-next") || name.contains("flash_next") {
            return ModelProfile(
                experts: true, ple: true, vision: true, thinking: true,
                sparseKv: true
            )
        }
        // Qwen3.6-35B-A3B is MoE without a PLE table.
        if name.contains("a3b") || name.contains("qwen3.6-35b") {
            return ModelProfile(
                experts: true, ple: false, vision: true, thinking: true,
                sparseKv: false
            )
        }
        // Qwen3.8-27B and Ternary-Bonsai are dense: no expert store.
        if name.contains("27b") || name.contains("ternary-bonsai") {
            return ModelProfile(
                experts: false, ple: false, vision: true, thinking: true,
                sparseKv: false
            )
        }
        return .everything
    }
}
