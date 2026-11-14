//! Decode and resize images to patch-aligned uint8 RGB within the pixel
//! budget. Ported from `server/images.py`.
//!
//! The engine handles normalization and vision inference. Grid geometry and
//! a content digest identify image prefixes; a byte-budgeted cache reuses
//! images resent in chat history.

use std::fmt;
use std::io::Cursor;
use std::sync::{Arc, Mutex};

use base64::Engine as _;
use image::metadata::Orientation;
use image::{DynamicImage, ImageDecoder, ImageReader};

use crate::support::lru::LruCache;
use crate::support::sha256;

pub const PATCH: u32 = 16;
pub const MERGE: u32 = 2;
pub const FACTOR: u32 = PATCH * MERGE;
/// Qwen3-VL's shortest_edge floor; the serving cap below is far below the
/// model's.
pub const MIN_PIXELS: u32 = 65_536;
/// Serving default and largest cap: 16,384 patches, the protocol's
/// per-image ceiling.
pub const MAX_PIXELS: u32 = 4_194_304;
/// Bound codec work before full-resolution decode, independently of the
/// model's resized input budget. Ordinary screenshots and up to 32 MP
/// photos fit here.
pub const MAX_SOURCE_PIXELS: u64 = 32 * 1024 * 1024;
pub const MAX_IMAGE_BYTES: usize = 32 * 1024 * 1024;

/// A payload or geometry the image pipeline refuses. [`ImageError::Capacity`]
/// marks the in-flight memory budget, which a retry may fit; the rest are
/// the request's own faults.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageError {
    Capacity(String),
    Invalid(String),
}

impl ImageError {
    fn invalid(message: impl Into<String>) -> ImageError {
        ImageError::Invalid(message.into())
    }

    /// The message the Python's `ValueError` carries.
    pub fn message(&self) -> &str {
        match self {
            ImageError::Capacity(message) | ImageError::Invalid(message) => message,
        }
    }
}

impl fmt::Display for ImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.message().fmt(f)
    }
}

impl std::error::Error for ImageError {}

/// One image, ready for the engine: its patch grid, the row-major uint8
/// RGB pixels of the resized image, and the first two little-endian words
/// of the pixels' digest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedImage {
    pub grid_height: u32,
    pub grid_width: u32,
    pub pixels: Vec<u8>,
    pub digest_lo: u64,
    pub digest_hi: u64,
}

impl PreparedImage {
    /// Merged language tokens the image occupies.
    pub fn tokens(&self) -> u32 {
        (self.grid_height / MERGE) * (self.grid_width / MERGE)
    }
}

/// Only `data:` URLs, `;base64`-encoded, within the encoded and decoded
/// size limits, with strict base64 validation.
pub fn decode_data_url(url: &str) -> Result<Vec<u8>, ImageError> {
    decode_data_url_limited(url, MAX_IMAGE_BYTES)
}

/// [`decode_data_url`] against its own limit, which the tests shrink.
fn decode_data_url_limited(url: &str, max_image_bytes: usize) -> Result<Vec<u8>, ImageError> {
    let rest = url
        .strip_prefix("data:")
        .ok_or_else(|| ImageError::invalid("only data: image URLs are supported"))?;
    let Some((header, encoded)) = rest.split_once(',') else {
        return Err(ImageError::invalid("image data URLs must be base64-encoded"));
    };
    if !header.contains(";base64") {
        return Err(ImageError::invalid("image data URLs must be base64-encoded"));
    }
    if encoded.len() > max_image_bytes * 4 / 3 + 4 {
        return Err(ImageError::invalid("image exceeds the size limit"));
    }
    let raw = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(|_| ImageError::invalid("image data URL is not valid base64"))?;
    if raw.len() > max_image_bytes {
        return Err(ImageError::invalid("image exceeds the size limit"));
    }
    Ok(raw)
}

/// The most patches an image resized within `max_pixels` can have: its
/// sides are whole merge units, so its patch count is a multiple of 4.
pub fn max_patches(max_pixels: u32) -> u32 {
    max_pixels / (PATCH * PATCH) / 4 * 4
}

/// Python's `round`: ties go to the even neighbor, which Rust's `f64::round`
/// does not do.
fn python_round(value: f64) -> f64 {
    let floor = value.floor();
    let fraction = value - floor;
    if fraction > 0.5 {
        floor + 1.0
    } else if fraction < 0.5 {
        floor
    } else if (floor as i64) % 2 == 0 {
        floor
    } else {
        floor + 1.0
    }
}

/// Upstream Qwen2VL smart-resize: patch/merge-aligned sides inside the
/// pixel budget with the aspect ratio preserved as closely as possible.
pub fn smart_resize(height: u32, width: u32, max_pixels: u32) -> Result<(u32, u32), ImageError> {
    if height == 0 || width == 0 {
        return Err(ImageError::invalid("image dimensions must be positive"));
    }
    if !(MIN_PIXELS..=MAX_PIXELS).contains(&max_pixels) {
        return Err(ImageError::invalid("invalid image pixel budget"));
    }
    let (height, width) = (f64::from(height), f64::from(width));
    let max_pixels = f64::from(max_pixels);
    if height.max(width) / height.min(width) > 200.0 {
        return Err(ImageError::invalid(
            "absolute aspect ratio must be smaller than 200",
        ));
    }
    let factor = f64::from(FACTOR);
    let mut h_bar = python_round(height / factor) * factor;
    let mut w_bar = python_round(width / factor) * factor;
    if h_bar * w_bar > max_pixels {
        let beta = (height * width / max_pixels).sqrt();
        h_bar = factor.max((height / beta / factor).floor() * factor);
        w_bar = factor.max((width / beta / factor).floor() * factor);
    } else if h_bar * w_bar < f64::from(MIN_PIXELS) {
        let beta = (f64::from(MIN_PIXELS) / (height * width)).sqrt();
        h_bar = (height * beta / factor).ceil() * factor;
        w_bar = (width * beta / factor).ceil() * factor;
    }
    // Upscaling to the minimum can also overshoot a small serving cap after
    // alignment. The hard cap takes precedence over that soft minimum.
    while h_bar * w_bar > max_pixels {
        if h_bar >= w_bar && h_bar > factor {
            h_bar -= factor;
        } else {
            w_bar -= factor;
        }
    }
    Ok((h_bar as u32, w_bar as u32))
}

/// Decode, convert to RGB, resize into the grid, and digest the pixels.
pub fn prepare(payload: &[u8], max_pixels: u32) -> Result<PreparedImage, ImageError> {
    prepare_limited(payload, max_pixels, MAX_SOURCE_PIXELS)
}

/// [`prepare`] against its own source-pixel limit, which the tests shrink.
fn prepare_limited(
    payload: &[u8],
    max_pixels: u32,
    source_limit: u64,
) -> Result<PreparedImage, ImageError> {
    let (height, width, pixels) = decode_and_resize(payload, max_pixels, source_limit)?;
    let grid_height = height / PATCH;
    let grid_width = width / PATCH;
    let mut digest_input = Vec::with_capacity(8 + pixels.len());
    digest_input.extend_from_slice(&grid_height.to_le_bytes());
    digest_input.extend_from_slice(&grid_width.to_le_bytes());
    digest_input.extend_from_slice(&pixels);
    let digest = sha256::digest(&digest_input);
    let digest_lo = u64::from_le_bytes(digest[..8].try_into().expect("digest head"));
    let digest_hi = u64::from_le_bytes(digest[8..16].try_into().expect("digest head"));
    Ok(PreparedImage {
        grid_height,
        grid_width,
        pixels,
        digest_lo,
        digest_hi,
    })
}

/// The decoded image, the resize target its orientation may swap, and the
/// final row-major RGB bytes.
fn decode_and_resize(
    payload: &[u8],
    max_pixels: u32,
    source_limit: u64,
) -> Result<(u32, u32, Vec<u8>), ImageError> {
    let decoded = decode(payload, source_limit)?;
    let (mut height, mut width) = smart_resize(decoded.height(), decoded.width(), max_pixels)?;
    let mut image = decoded;
    // Cameras store the sensor's orientation and an EXIF tag that turns it
    // upright for display; show the model the upright one. A malformed tag
    // leaves the stored orientation.
    if let Some(orientation) = exif_orientation(payload) {
        // A quarter turn swaps the stored sides, so the resize target
        // swaps with them, as the Python's stored-size comparison does.
        let swaps_sides = matches!(
            orientation,
            Orientation::Rotate90 | Orientation::Rotate270
                | Orientation::Rotate90FlipH | Orientation::Rotate270FlipH
        );
        image.apply_orientation(orientation);
        if swaps_sides {
            std::mem::swap(&mut height, &mut width);
        }
    }
    let rgb = if image.color().has_alpha() {
        // Composite onto white as Qwen's preprocessing does: transparent
        // pixels usually store black, which hides dark content.
        let rgba = image.to_rgba8();
        let mut white =
            image::RgbaImage::from_pixel(rgba.width(), rgba.height(), image::Rgba([255, 255, 255, 255]));
        image::imageops::overlay(&mut white, &rgba, 0, 0);
        DynamicImage::ImageRgba8(white).to_rgb8()
    } else {
        image.to_rgb8()
    };
    let rgb = if (rgb.height(), rgb.width()) != (height, width) {
        image::RgbImage::from_raw(
            width,
            height,
            resize_bicubic_rgb(&rgb, rgb.width(), rgb.height(), width, height),
        )
        .expect("resized RGB buffer has target dimensions")
    } else {
        rgb
    };
    Ok((height, width, rgb.into_raw()))
}

const PILLOW_RESAMPLE_BITS: u32 = 22;
const PILLOW_RESAMPLE_SCALE: f64 = (1u64 << PILLOW_RESAMPLE_BITS) as f64;

#[derive(Debug)]
struct ResampleCoefficients {
    start: usize,
    weights: Vec<i32>,
}

/// Pillow's 8-bit BICUBIC resampler. Coefficients use its 22-bit fixed-point
/// representation and each separable pass rounds and clips back to u8.
/// Used by `decode_and_resize` for Pillow-compatible RGB resizing.
fn resize_bicubic_rgb(
    input: &[u8],
    in_width: u32,
    in_height: u32,
    out_width: u32,
    out_height: u32,
) -> Vec<u8> {
    assert!(in_width > 0 && in_height > 0 && out_width > 0 && out_height > 0);
    assert_eq!(input.len(), in_width as usize * in_height as usize * 3);

    let horizontal_first = !((i64::from(in_height) - i64::from(out_height) > 0)
        && (i64::from(in_height) - i64::from(out_height))
            > (i64::from(in_width) - i64::from(out_width)) * 2);
    if horizontal_first {
        let horizontal = resample_horizontal(input, in_width, in_height, out_width);
        resample_vertical(&horizontal, out_width, in_height, out_height)
    } else {
        let vertical = resample_vertical(input, in_width, in_height, out_height);
        resample_horizontal(&vertical, in_width, out_height, out_width)
    }
}

fn bicubic_coefficients(in_size: u32, out_size: u32) -> Vec<ResampleCoefficients> {
    assert!(in_size > 0 && out_size > 0);
    let scale = f64::from(in_size) / f64::from(out_size);
    let filter_scale = scale.max(1.0);
    let support = 2.0 * filter_scale;
    (0..out_size)
        .map(|out| {
            let center = (f64::from(out) + 0.5) * scale;
            // C's cast to int truncates toward zero. Clamping follows it, as
            // Pillow does at the first and last source pixels.
            let first = ((center - support + 0.5) as i64).max(0) as usize;
            let last = ((center + support + 0.5) as i64).min(i64::from(in_size)) as usize;
            let values = (first..last)
                .map(|source| {
                    bicubic_filter((source as f64 + 0.5 - center) / filter_scale)
                })
                .collect::<Vec<_>>();
            let total = values.iter().sum::<f64>();
            let weights = values
                .into_iter()
                .map(|value| {
                    // Pillow rounds signed normalized coefficients by adding
                    // 0.5 and flooring, then stores them in a 22-bit table.
                    ((value / total * PILLOW_RESAMPLE_SCALE) + 0.5).floor() as i32
                })
                .collect();
            ResampleCoefficients { start: first, weights }
        })
        .collect()
}

fn bicubic_filter(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        ((1.5 * x - 2.5) * x) * x + 1.0
    } else if x < 2.0 {
        ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
    } else {
        0.0
    }
}

fn rounded_u8(sum: i64) -> u8 {
    ((sum + (1i64 << (PILLOW_RESAMPLE_BITS - 1))) >> PILLOW_RESAMPLE_BITS)
        .clamp(0, 255) as u8
}

fn resample_horizontal(input: &[u8], in_width: u32, height: u32, out_width: u32) -> Vec<u8> {
    let coefficients = bicubic_coefficients(in_width, out_width);
    let mut output = vec![0; out_width as usize * height as usize * 3];
    let in_stride = in_width as usize * 3;
    let out_stride = out_width as usize * 3;
    for y in 0..height as usize {
        let input_row = y * in_stride;
        let output_row = y * out_stride;
        for (x, coefficients) in coefficients.iter().enumerate() {
            let mut sums = [0i64; 3];
            for (index, weight) in coefficients.weights.iter().enumerate() {
                let source = input_row + (coefficients.start + index) * 3;
                let weight = i64::from(*weight);
                sums[0] += i64::from(input[source]) * weight;
                sums[1] += i64::from(input[source + 1]) * weight;
                sums[2] += i64::from(input[source + 2]) * weight;
            }
            let destination = output_row + x * 3;
            output[destination] = rounded_u8(sums[0]);
            output[destination + 1] = rounded_u8(sums[1]);
            output[destination + 2] = rounded_u8(sums[2]);
        }
    }
    output
}

fn resample_vertical(input: &[u8], width: u32, in_height: u32, out_height: u32) -> Vec<u8> {
    let coefficients = bicubic_coefficients(in_height, out_height);
    let stride = width as usize * 3;
    let mut output = vec![0; width as usize * out_height as usize * 3];
    for (y, coefficients) in coefficients.iter().enumerate() {
        let output_row = y * stride;
        for x in 0..stride {
            let sum = coefficients
                .weights
                .iter()
                .enumerate()
                .map(|(index, weight)| {
                    i64::from(input[(coefficients.start + index) * stride + x])
                        * i64::from(*weight)
                })
                .sum();
            output[output_row + x] = rounded_u8(sum);
        }
    }
    output
}

/// Decode the payload's first frame (a GIF's first, for one) after the
/// header's dimensions pass the source-pixel limit, so codec work is
/// bounded before full-resolution decode.
fn decode(payload: &[u8], source_limit: u64) -> Result<DynamicImage, ImageError> {
    let decoded = (|| {
        let decoder = ImageReader::new(Cursor::new(payload))
            .with_guessed_format()
            .map_err(|_| ImageError::invalid("image could not be decoded"))?
            .into_decoder()
            .map_err(|_| ImageError::invalid("image could not be decoded"))?;
        let (width, height) = decoder.dimensions();
        if u64::from(width) * u64::from(height) > source_limit {
            return Err(ImageError::invalid("source image exceeds the pixel limit"));
        }
        DynamicImage::from_decoder(decoder).map_err(|_| ImageError::invalid("image could not be decoded"))
    })();
    decoded
}

/// The EXIF orientation the payload declares, as the image crate's
/// transforms spell them; `None` without a readable one.
fn exif_orientation(payload: &[u8]) -> Option<Orientation> {
    let exif = exif::Reader::new()
        .read_from_container(&mut Cursor::new(payload))
        .ok()?;
    let field = exif.get_field(exif::Tag::Orientation, exif::In::PRIMARY)?;
    let value = field.value.get_uint(0)?;
    Orientation::from_exif(value as u8)
}

/// The image cache's counters, as the status document reports them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageCacheStats {
    pub entries: usize,
    pub bytes: usize,
    pub budget_bytes: usize,
    pub request_bytes: usize,
    pub request_budget_bytes: usize,
}

/// Cached prepared images, and the images of requests in flight.
struct CacheState {
    entries: LruCache<[u8; 32], PreparedImage>,
    request_bytes: usize,
    request_budget_bytes: usize,
}

/// Byte-budgeted LRU of prepared images keyed by the raw image bytes.
///
/// The state sits behind one mutex the batches share, so a batch's charge
/// and its release on drop travel with the same lock the cache inserts
/// under; nothing re-enters it, as the Python's re-entrant lock guarded
/// against a finalizer running mid-insertion.
pub struct ImageCache {
    state: Arc<Mutex<CacheState>>,
}

impl Default for ImageCache {
    fn default() -> ImageCache {
        ImageCache::new()
    }
}

impl ImageCache {
    pub const BUDGET_BYTES: usize = 256 * 1024 * 1024;
    pub const REQUEST_BUDGET_BYTES: usize = 256 * 1024 * 1024;

    pub fn new() -> ImageCache {
        ImageCache::with_budgets(Self::BUDGET_BYTES, Self::REQUEST_BUDGET_BYTES)
    }

    /// A cache with its own budgets, which the tests shrink.
    pub fn with_budgets(budget_bytes: usize, request_budget_bytes: usize) -> ImageCache {
        ImageCache {
            state: Arc::new(Mutex::new(CacheState {
                entries: LruCache::new(budget_bytes, None),
                request_bytes: 0,
                request_budget_bytes,
            })),
        }
    }

    /// The prepared image for `payload` at `max_pixels`, cached by
    /// sha256(le-u32 max_pixels ++ payload); a miss decodes outside the
    /// lock and inserts under it.
    pub fn prepare(&self, payload: &[u8], max_pixels: u32) -> Result<PreparedImage, ImageError> {
        let mut key_input = Vec::with_capacity(4 + payload.len());
        key_input.extend_from_slice(&max_pixels.to_le_bytes());
        key_input.extend_from_slice(payload);
        let key = sha256::digest(&key_input);
        {
            let mut state = self.state.lock().expect("image cache lock");
            if let Some(cached) = state.entries.get(&key) {
                return Ok(cached.clone());
            }
        }
        let prepared = prepare(payload, max_pixels)?;
        let mut state = self.state.lock().expect("image cache lock");
        state.entries.put(key, prepared.clone(), prepared.pixels.len());
        Ok(prepared)
    }

    /// An append-only batch for one request; its owner travels with the
    /// pixel payload.
    pub fn request_batch(&self) -> RequestImages {
        RequestImages {
            state: Arc::clone(&self.state),
            images: Vec::new(),
            bytes: 0,
        }
    }

    pub fn stats(&self) -> ImageCacheStats {
        let state = self.state.lock().expect("image cache lock");
        ImageCacheStats {
            entries: state.entries.len(),
            bytes: state.entries.bytes,
            budget_bytes: state.entries.budget_bytes,
            request_bytes: state.request_bytes,
            request_budget_bytes: state.request_budget_bytes,
        }
    }
}

/// Append-only request batch; its owner travels with the pixel payload.
///
/// Charge each occurrence, even cache hits: concatenation and native
/// transport repeat those pixels. Release only when the batch drops, not
/// when HTTP disconnects while native work or a mask task still owns it.
pub struct RequestImages {
    state: Arc<Mutex<CacheState>>,
    images: Vec<PreparedImage>,
    bytes: usize,
}

impl fmt::Debug for RequestImages {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RequestImages")
            .field("images", &self.images.len())
            .field("bytes", &self.bytes)
            .finish()
    }
}

impl RequestImages {
    /// Charge `image` against the request budget, then keep it.
    pub fn append(&mut self, image: PreparedImage) -> Result<(), ImageError> {
        let size = image.pixels.len();
        {
            let mut state = self.state.lock().expect("image cache lock");
            if size > state.request_budget_bytes - state.request_bytes {
                return Err(ImageError::Capacity(
                    "in-flight image memory budget is full".to_string(),
                ));
            }
            state.request_bytes += size;
        }
        self.bytes += size;
        self.images.push(image);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.images.len()
    }

    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, PreparedImage> {
        self.images.iter()
    }

    pub fn images(&self) -> &[PreparedImage] {
        &self.images
    }
}

impl Drop for RequestImages {
    fn drop(&mut self) {
        let mut state = self.state.lock().expect("image cache lock");
        state.request_bytes -= self.bytes;
    }
}

impl<'a> IntoIterator for &'a RequestImages {
    type Item = &'a PreparedImage;
    type IntoIter = std::slice::Iter<'a, PreparedImage>;

    fn into_iter(self) -> Self::IntoIter {
        self.images.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageEncoder, Rgb, RgbImage, Rgba, RgbaImage};

    /// A solid-color PNG the image crate encodes, as the Python tests'
    /// `png_bytes` does with PIL.
    fn png_bytes(width: u32, height: u32, color: [u8; 3]) -> Vec<u8> {
        let image = RgbImage::from_pixel(width, height, Rgb(color));
        let mut buffer = Cursor::new(Vec::new());
        image::codecs::png::PngEncoder::new(&mut buffer)
            .write_image(
                image.as_raw(),
                width,
                height,
                image::ExtendedColorType::Rgb8,
            )
            .expect("png encodes");
        buffer.into_inner()
    }

    fn rgba_png_bytes(width: u32, height: u32, top: [u8; 4], bottom: [u8; 4]) -> Vec<u8> {
        let mut image = RgbaImage::new(width, height);
        for (_x, y, pixel) in image.enumerate_pixels_mut() {
            *pixel = if y < height / 2 { Rgba(top) } else { Rgba(bottom) };
        }
        let mut buffer = Cursor::new(Vec::new());
        image::codecs::png::PngEncoder::new(&mut buffer)
            .write_image(
                image.as_raw(),
                width,
                height,
                image::ExtendedColorType::Rgba8,
            )
            .expect("png encodes");
        buffer.into_inner()
    }

    /// A PNG with an `eXIf` chunk the test builds by hand: the image
    /// crate's encoder writes no EXIF, so the chunk is spliced in before
    /// IEND, CRC included.
    fn png_with_exif_chunk(png: &[u8], chunk: &[u8]) -> Vec<u8> {
        let end = png.len() - 12; // before IEND's length, type and CRC
        let mut with_chunk = png[..end].to_vec();
        with_chunk.extend_from_slice(&(chunk.len() as u32).to_be_bytes());
        with_chunk.extend_from_slice(b"eXIf");
        with_chunk.extend_from_slice(chunk);
        let mut crc_input = b"eXIf".to_vec();
        crc_input.extend_from_slice(chunk);
        with_chunk.extend_from_slice(&crc32(&crc_input));
        with_chunk.extend_from_slice(&png[end..]);
        with_chunk
    }

    /// The PNG chunk CRC: the standard CRC-32 of the type and the data.
    fn crc32(input: &[u8]) -> [u8; 4] {
        let mut crc = 0xFFFF_FFFFu32;
        for byte in input {
            crc ^= u32::from(*byte);
            for _ in 0..8 {
                let mask = (crc & 1).wrapping_neg();
                crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
            }
        }
        (!crc).to_be_bytes()
    }

    /// A little-endian TIFF whose primary IFD carries one Orientation tag.
    fn exif_tiff(orientation: u16) -> Vec<u8> {
        let mut tiff = Vec::new();
        tiff.extend_from_slice(b"II\x2a\x00");
        tiff.extend_from_slice(&8u32.to_le_bytes()); // IFD at 8
        tiff.extend_from_slice(&1u16.to_le_bytes()); // one entry
        tiff.extend_from_slice(&0x0112u16.to_le_bytes()); // Orientation
        tiff.extend_from_slice(&3u16.to_le_bytes()); // SHORT
        tiff.extend_from_slice(&1u32.to_le_bytes()); // one value
        tiff.extend_from_slice(&orientation.to_le_bytes());
        tiff.extend_from_slice(&0u16.to_le_bytes()); // value padding
        tiff.extend_from_slice(&0u32.to_le_bytes()); // no next IFD
        tiff
    }

    #[test]
    fn sides_stay_patch_and_merge_aligned_inside_the_budget() {
        for (height, width) in [(100, 300), (10_000, 300), (17, 3_000), (3000, 4000)] {
            let (resized_height, resized_width) = smart_resize(height, width, MAX_PIXELS).unwrap();
            assert_eq!(resized_height % 32, 0);
            assert_eq!(resized_width % 32, 0);
            assert!(resized_height * resized_width >= MIN_PIXELS);
            assert!(resized_height * resized_width <= MAX_PIXELS);
        }
    }

    #[test]
    fn small_images_are_upscaled_to_the_minimum_budget() {
        assert_eq!(smart_resize(64, 64, MAX_PIXELS).unwrap(), (256, 256));
    }

    #[test]
    fn serving_cap_bounds_large_images() {
        let (height, width) = smart_resize(2880, 1800, 1_000_000).unwrap();
        assert!(height * width <= 1_000_000);
        assert!((height as f64 / width as f64 - 2880.0 / 1800.0).abs() < 0.1);
    }

    #[test]
    fn extreme_aspect_ratio_is_rejected() {
        let error = smart_resize(10, 3000, MAX_PIXELS).unwrap_err();
        assert!(error.message().contains("aspect ratio"));
    }

    #[test]
    fn alignment_never_exceeds_a_small_serving_cap() {
        // A seeded LCG stands in for the Python test's random.Random(42).
        let mut seed = 42u64;
        let mut next = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        let mut shapes = vec![(1, 100), (100, 1), (1, 200), (200, 1)];
        for _ in 0..300 {
            shapes.push((1 + next() % 10_000, 1 + next() % 10_000));
        }
        for (height, width) in shapes {
            if height.max(width) / height.min(width) > 200 {
                continue;
            }
            for cap in [MIN_PIXELS, 70_000, 100_000, MAX_PIXELS] {
                let (h, w) = smart_resize(height, width, cap).unwrap();
                assert!(h.min(w) >= FACTOR, "{height}x{width} at {cap}: {h}x{w}");
                assert_eq!(h % FACTOR, 0, "{height}x{width} at {cap}");
                assert_eq!(w % FACTOR, 0, "{height}x{width} at {cap}");
                assert!(h * w <= cap, "{height}x{width} at {cap}: {h}x{w}");
            }
        }
    }

    #[test]
    fn prepare_returns_grid_pixels_and_content_digest() {
        let prepared = prepare(&png_bytes(64, 64, [200, 30, 30]), MAX_PIXELS).unwrap();
        // 64x64 upscales to the 256x256 minimum: a 16x16 patch grid.
        assert_eq!((prepared.grid_height, prepared.grid_width), (16, 16));
        assert_eq!(prepared.tokens(), 64);
        assert_eq!(prepared.pixels.len(), 256 * 256 * 3);
        assert_eq!(&prepared.pixels[..3], &[200, 30, 30]);
        let mut digest_input = 16u32.to_le_bytes().to_vec();
        digest_input.extend_from_slice(&16u32.to_le_bytes());
        digest_input.extend_from_slice(&prepared.pixels);
        let digest = sha256::digest(&digest_input);
        assert_eq!(
            (prepared.digest_lo, prepared.digest_hi),
            (
                u64::from_le_bytes(digest[..8].try_into().unwrap()),
                u64::from_le_bytes(digest[8..16].try_into().unwrap())
            )
        );
    }

    #[test]
    fn bicubic_rgb_matches_pillow_fixed_fixture() {
        let (width, height, out_width, out_height) = (7u32, 5u32, 32u32, 32u32);
        let source = (0..width as usize * height as usize * 3)
            .map(|index| ((index * 73 + (index / 3) * 19 + 11) % 256) as u8)
            .collect::<Vec<_>>();
        let resized = resize_bicubic_rgb(&source, width, height, out_width, out_height);
        let digest = sha256::digest(&resized);
        let digest = digest.iter().map(|byte| format!("{byte:02x}")).collect::<String>();
        assert_eq!(digest, "94ce87451733d5d51ed98b16a4d4c6d34fd94447e22311dd7f5f8971119fc3c2");
        assert_eq!(&resized[..18], &[0, 71, 174, 0, 70, 173, 2, 69, 172, 51, 66, 170, 117, 62, 167, 190, 57, 162]);
    }

    #[test]
    fn decode_and_resize_uses_pillow_bicubic_fixture() {
        let (width, height) = (320u32, 320u32);
        let pixels = (0..width as usize * height as usize * 3)
            .map(|index| ((index * 73 + (index / 3) * 19 + 11) % 256) as u8)
            .collect::<Vec<_>>();
        let mut encoded = Cursor::new(Vec::new());
        image::codecs::png::PngEncoder::new(&mut encoded)
            .write_image(
                &pixels,
                width,
                height,
                image::ExtendedColorType::Rgb8,
            )
            .expect("PNG encodes");

        let (height, width, resized) =
            decode_and_resize(encoded.get_ref(), MIN_PIXELS, MAX_SOURCE_PIXELS).unwrap();
        assert_eq!((height, width), (256, 256));
        let digest = sha256::digest(&resized);
        let digest = digest.iter().map(|byte| format!("{byte:02x}")).collect::<String>();
        assert_eq!(digest, "a949a563c7a57e67da32eef7cf3163d2ef9feee925c69f49948b29ec203fddf5");
    }

    #[test]
    fn bicubic_rgb_matches_pillow_thousand_case_oracle() {
        // Pillow 13.0.0 oracle: 1,000 deterministic RGB sources, varied
        // up/down/mixed resizing, SHA-256 over dimensions and output bytes.
        let mut seed = 9u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        let mut oracle_input = Vec::new();
        for case in 0..1_000usize {
            let width = 2 + next() % 79;
            let height = 2 + next() % 79;
            let out_width = 2 + next() % 79;
            let out_height = 2 + next() % 79;
            let source = (0..width as usize * height as usize * 3)
                .map(|index| ((index * 73 + (index / 3) * 19 + case * 29 + 11) % 256) as u8)
                .collect::<Vec<_>>();
            oracle_input.extend_from_slice(&width.to_le_bytes());
            oracle_input.extend_from_slice(&height.to_le_bytes());
            oracle_input.extend_from_slice(&out_width.to_le_bytes());
            oracle_input.extend_from_slice(&out_height.to_le_bytes());
            oracle_input.extend_from_slice(&resize_bicubic_rgb(&source, width, height, out_width, out_height));
        }
        let digest = sha256::digest(&oracle_input);
        let digest = digest.iter().map(|byte| format!("{byte:02x}")).collect::<String>();
        assert_eq!(digest, "d24895ab8180eddd9ddc51fcc4a1c8c2f1feebe222d84dc6bdeb8e458b072cf3");
    }

    #[test]
    fn digest_depends_on_content_not_encoding() {
        let red = prepare(&png_bytes(64, 64, [200, 30, 30]), MAX_PIXELS).unwrap();
        let red_again = prepare(&png_bytes(64, 64, [200, 30, 30]), MAX_PIXELS).unwrap();
        let blue = prepare(&png_bytes(64, 64, [30, 30, 200]), MAX_PIXELS).unwrap();
        assert_eq!((red.digest_lo, red.digest_hi), (red_again.digest_lo, red_again.digest_hi));
        assert_ne!((red.digest_lo, red.digest_hi), (blue.digest_lo, blue.digest_hi));
    }

    #[test]
    fn undecodable_payload_is_an_image_error() {
        let error = prepare(b"not an image", MAX_PIXELS).unwrap_err();
        assert_eq!(error.message(), "image could not be decoded");
    }

    #[test]
    fn source_limit_and_aspect_ratio_are_checked_before_decode() {
        let normal = png_bytes(64, 64, [200, 30, 30]);
        let wide = png_bytes(201, 1, [200, 30, 30]);
        let prepared = prepare_limited(&normal, MAX_PIXELS, 64 * 64).unwrap();
        assert_eq!(prepared.tokens(), 64);
        let error = prepare_limited(&normal, MAX_PIXELS, 64 * 64 - 1).unwrap_err();
        assert!(error.message().contains("source image exceeds"));
        let error = prepare(&wide, MAX_PIXELS).unwrap_err();
        assert!(error.message().contains("aspect ratio"));
    }

    #[test]
    fn exif_orientation_is_applied_before_resizing() {
        // Orientation 6 displays the stored image turned a quarter
        // clockwise; the upright one is the same pixels pre-rotated.
        let stored = {
            let mut image = RgbImage::from_pixel(80, 40, Rgb([30, 30, 200]));
            for x in 0..80 {
                image.put_pixel(x, 0, Rgb([200, 30, 30]));
            }
            image
        };
        let upright = image::imageops::rotate90(&stored);
        let encode = |image: &RgbImage| {
            let mut buffer = Cursor::new(Vec::new());
            image::codecs::png::PngEncoder::new(&mut buffer)
                .write_image(image.as_raw(), image.width(), image.height(), image::ExtendedColorType::Rgb8)
                .expect("png encodes");
            buffer.into_inner()
        };
        let tagged = png_with_exif_chunk(&encode(&stored), &exif_tiff(6));
        let prepared_tagged = prepare(&tagged, MAX_PIXELS).unwrap();
        let prepared_upright = prepare(&encode(&upright), MAX_PIXELS).unwrap();
        assert_eq!(
            (
                prepared_tagged.grid_height,
                prepared_tagged.grid_width,
                prepared_tagged.digest_lo
            ),
            (
                prepared_upright.grid_height,
                prepared_upright.grid_width,
                prepared_upright.digest_lo
            )
        );
        // A malformed tag leaves the image as stored instead of failing it.
        let malformed = png_with_exif_chunk(&encode(&stored), b"Exif\x00\x00not a tiff header");
        let prepared_malformed = prepare(&malformed, MAX_PIXELS).unwrap();
        let prepared_stored = prepare(&encode(&stored), MAX_PIXELS).unwrap();
        assert_eq!(prepared_malformed.digest_lo, prepared_stored.digest_lo);
        assert_eq!(
            (prepared_malformed.grid_height, prepared_malformed.grid_width),
            (prepared_stored.grid_height, prepared_stored.grid_width)
        );
    }

    #[test]
    fn transparent_pixels_are_composited_onto_white() {
        // Opaque black on the top half, transparent pixels storing black
        // below.
        let rgba = rgba_png_bytes(256, 256, [0, 0, 0, 255], [0, 0, 0, 0]);
        let mut gray_alpha = image::ImageBuffer::<image::LumaA<u8>, Vec<u8>>::new(256, 256);
        for (_x, y, pixel) in gray_alpha.enumerate_pixels_mut() {
            *pixel = if y < 128 { image::LumaA([0, 255]) } else { image::LumaA([0, 0]) };
        }
        let mut la_buffer = Cursor::new(Vec::new());
        image::codecs::png::PngEncoder::new(&mut la_buffer)
            .write_image(gray_alpha.as_raw(), 256, 256, image::ExtendedColorType::La8)
            .expect("png encodes");
        let expected = vec![0u8; 256 * 128 * 3]
            .into_iter()
            .chain(std::iter::repeat(0xff).take(256 * 128 * 3))
            .collect::<Vec<u8>>();
        for payload in [rgba, la_buffer.into_inner()] {
            let prepared = prepare(&payload, MAX_PIXELS).unwrap();
            assert_eq!(prepared.pixels, expected);
        }
    }

    #[test]
    fn small_image_preparation_obeys_cap_after_upscale() {
        let prepared = prepare(&png_bytes(1, 100, [200, 30, 30]), MIN_PIXELS).unwrap();
        assert!(prepared.pixels.len() <= 3 * MIN_PIXELS as usize);
    }

    #[test]
    fn base64_data_urls_decode_and_others_are_rejected() {
        let payload = png_bytes(64, 64, [200, 30, 30]);
        let encoded = base64::engine::general_purpose::STANDARD.encode(&payload);
        let url = format!("data:image/png;base64,{encoded}");
        assert_eq!(decode_data_url(&url).unwrap(), payload);
        for invalid in [
            "https://example.com/x.png".to_string(),
            "data:image/png,rawbytes".to_string(),
            "data:image/png;base64,not*base64".to_string(),
        ] {
            assert!(decode_data_url(&invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn data_url_size_limits_are_enforced() {
        // The encoded-length limit fires before any decode work.
        let long = "A".repeat(64 * 4 / 3 + 5);
        let error = decode_data_url_limited(&format!("data:image/png;base64,{long}"), 64).unwrap_err();
        assert_eq!(error.message(), "image exceeds the size limit");
        // The decoded-length limit fires after a valid encoding.
        let payload = vec![0u8; 65];
        let encoded = base64::engine::general_purpose::STANDARD.encode(&payload);
        let error = decode_data_url_limited(&format!("data:image/png;base64,{encoded}"), 64).unwrap_err();
        assert_eq!(error.message(), "image exceeds the size limit");
        assert_eq!(decode_data_url_limited(&format!("data:image/png;base64,{encoded}"), 65).unwrap(), payload);
    }

    #[test]
    fn request_budget_counts_repeated_images_and_all_live_batches() {
        let cache = ImageCache::with_budgets(0, 12);
        let image = PreparedImage {
            grid_height: 2,
            grid_width: 2,
            pixels: b"abcd".to_vec(),
            digest_lo: 0,
            digest_hi: 0,
        };
        let mut first = cache.request_batch();
        let mut second = cache.request_batch();
        first.append(image.clone()).unwrap();
        first.append(image.clone()).unwrap();
        second.append(image.clone()).unwrap();
        assert_eq!(cache.stats().request_bytes, 12);
        assert!(matches!(
            second.append(image.clone()).unwrap_err(),
            ImageError::Capacity(_)
        ));
        assert_eq!(cache.stats().request_bytes, 12);
        // Dropping a batch releases its charge, even one that failed.
        drop(first);
        assert_eq!(cache.stats().request_bytes, 4);
        second.append(image.clone()).unwrap();
        drop(second);
        assert_eq!(cache.stats().request_bytes, 0);
    }

    #[test]
    fn request_budget_is_returned_after_exception() {
        let cache = ImageCache::with_budgets(0, 4);
        let image = PreparedImage {
            grid_height: 2,
            grid_width: 2,
            pixels: b"abcd".to_vec(),
            digest_lo: 0,
            digest_hi: 0,
        };
        let prepare_then_fail = || -> Result<(), ImageError> {
            let mut batch = cache.request_batch();
            batch.append(image.clone())?;
            Err(ImageError::Invalid("injected render failure".to_string()))
        };
        assert!(prepare_then_fail().is_err());
        assert_eq!(cache.stats().request_bytes, 0);
    }

    #[test]
    fn concurrent_request_batches_do_not_oversell_image_budget() {
        let cache = ImageCache::with_budgets(0, 12);
        let image = PreparedImage {
            grid_height: 2,
            grid_width: 2,
            pixels: b"abcd".to_vec(),
            digest_lo: 0,
            digest_hi: 0,
        };
        let mut batches: Vec<_> = (0..16).map(|_| cache.request_batch()).collect();
        let admitted = std::sync::Mutex::new(0usize);
        std::thread::scope(|scope| {
            let image = &image;
            let admitted = &admitted;
            let handles: Vec<_> = batches
                .iter_mut()
                .map(|batch| {
                    scope.spawn(move || {
                        match batch.append(image.clone()) {
                            Ok(()) => *admitted.lock().unwrap() += 1,
                            Err(ImageError::Capacity(_)) => {}
                            Err(error) => panic!("unexpected error: {error}"),
                        }
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        });
        assert_eq!(*admitted.lock().unwrap(), 3);
        assert_eq!(cache.stats().request_bytes, 12);
        drop(batches);
        assert_eq!(cache.stats().request_bytes, 0);
    }

    #[test]
    fn cache_reuses_prepared_images_and_evicts_by_bytes() {
        let budget = 2 * 256 * 256 * 3;
        let cache = ImageCache::with_budgets(budget, ImageCache::REQUEST_BUDGET_BYTES);
        let first = cache.prepare(&png_bytes(64, 64, [200, 30, 30]), MAX_PIXELS).unwrap();
        let again = cache.prepare(&png_bytes(64, 64, [200, 30, 30]), MAX_PIXELS).unwrap();
        assert_eq!(first, again);
        assert_eq!(cache.stats().entries, 1);
        cache.prepare(&png_bytes(64, 64, [0, 200, 0]), MAX_PIXELS).unwrap();
        cache.prepare(&png_bytes(64, 64, [0, 0, 200]), MAX_PIXELS).unwrap();
        assert_eq!(cache.stats().entries, 2);
        assert!(cache.stats().bytes <= budget);
        // A different serving cap is a different preparation.
        cache.prepare(&png_bytes(64, 64, [200, 30, 30]), MIN_PIXELS).unwrap();
        assert_eq!(cache.stats().entries, 2);
    }
}
