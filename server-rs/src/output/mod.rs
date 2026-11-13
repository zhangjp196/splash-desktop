//! Incremental model-output parsing, ported from `server/output.py`: the
//! stop sequences' search, the reasoning's splitter, and the blocks a
//! response is read into. The tool-call projector is the next slice.

pub mod reasoning;
pub mod stop;

pub use reasoning::{ReasoningPart, ReasoningSplitter};
pub use stop::{hold_partial, StopSearch};
