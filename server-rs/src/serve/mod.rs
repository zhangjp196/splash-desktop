//! The serving pipeline: one request from HTTP to the engine and its
//! output back, over the modules the port has built. This is the
//! end-to-end path the Python's frontend, backend and dialects thread
//! together. The exchange's remaining job machinery and the other
//! dialects build on it.

pub mod pipeline;
mod preparation;
pub mod server;
pub mod stream;

pub use pipeline::Pipeline;
pub use server::Server;
