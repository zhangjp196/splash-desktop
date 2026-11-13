//! The serving pipeline: one request from HTTP to the engine and its
//! output back, over the modules the port has built. This is the
//! end-to-end path the Python's frontend, backend and dialects thread
//! together; its remaining pieces (the exchange's job machinery, the
//! other dialects, the security layer) build on it.

pub mod pipeline;

pub use pipeline::Pipeline;
