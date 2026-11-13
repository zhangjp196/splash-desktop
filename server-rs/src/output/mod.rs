//! Incremental model-output parsing, ported from `server/output.py`: the
//! stop sequences' search, the reasoning's splitter, and the tool calls'
//! projector.

pub mod projector;
pub mod reasoning;
pub mod stop;

pub use projector::{
    convert_value, ProjectorEvent, StaticTypes, StreamingToolCallProjector, ToolCall, Types,
};
pub use reasoning::{ReasoningPart, ReasoningSplitter};
pub use stop::{hold_partial, StopSearch};
