# Explicit CPU MoE expert path

Set `SPLASH_CPU_BF16_EXPERTS=1` to opt into the host expert path. The default and every other value retain Metal execution. Routing, grouped gather, route rows, and merge remain the existing MoE operations.

The resident CPU path accepts either all six expert segments as Shared BF16 Rows or all six as host-readable native Q4_0 Rows: routed and shared gate, up, and down. Q4_0 uses original GGUF row blocks in `QuantizedSegment::Rows` (32 values per block: little-endian fp16 scale then 16 nibble bytes), with every matrix input size divisible by 32. The streamed path accepts native GGUF Q4_0 routed rows plus resident Shared Q4_0 rows for the shared expert. `ExpertStore` captures the routed experts selected by the step from their original source rows; CPU dispatch checks that every tile's routed expert has a captured entry. Mixed BF16/Q4_0 weights, Metal-repacked planes, inaccessible rows, unsupported source mappings, rotated experts, and separately scheduled routed steps use the complete Metal expert pass. Q4_0 decode and dot products accumulate in FP32; grouped activations and final output retain the existing BF16 contract.

## Metal synchronization

CPU access to Shared activation and weight buffers occurs in the graph host callback, after preceding GPU work completes. The captured weight bundle keeps row storage and views alive until that callback finishes. Private buffers are not CPU-addressable. CPU eligibility is decided before graph work is appended; unsupported segments therefore leave the expert path on Metal.
