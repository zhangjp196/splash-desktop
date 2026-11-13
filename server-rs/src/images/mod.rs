//! Image input: decode and resize to patch-aligned RGB, and the prompt
//! pieces the placeholders become. Ported from `server/images.py` and
//! `server/image_prompts.py` (module 12, images only).

pub mod images;
pub mod prompts;

pub use images::{
    ImageCache, ImageError, PreparedImage, RequestImages, MAX_IMAGE_BYTES, MAX_PIXELS,
    MAX_SOURCE_PIXELS, MIN_PIXELS,
};
