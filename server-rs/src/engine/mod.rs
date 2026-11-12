//! The engine client's state machine and its subprocess wrapper.

pub mod call;
pub mod client;

pub use call::{
    wire_deadline, Call, CompletionCallback, EngineError, EventCallback, GenerationRequest,
    MaskProvider, RequestFailed,
};
pub use client::{EngineClient, STATUS_ANSWER_LIMIT};
