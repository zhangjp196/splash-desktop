//! Image placeholders in a prepared prompt: preparation, rendering,
//! expansion. Ported from `server/image_prompts.py`.
//!
//! The chat template emits one pad token per image; these functions prepare
//! the images a request carries, find the pad tokens the template rendered,
//! and widen each of them to the image's merged token count with the pixels
//! the engine injects there.

use std::collections::HashSet;
use std::sync::OnceLock;
use std::io::Read;

use serde_json::Value;

use crate::chat_messages::IMAGE_PAD_TOKEN;
use crate::images::images::{decode_data_url, ImageCache, ImageError, PreparedImage, RequestImages};
use crate::protocol::{
    ImageSpan, IMAGE_SPAN_BYTES, MAX_FRAME_PAYLOAD_BYTES, MAX_IMAGE_SPANS,
    MAX_THINKING_END_TOKENS, REQUEST_FIXED_BYTES,
};
use crate::support::errors::{ApiError, ContextLength};
use crate::templates::render::ChatTemplateError;
use crate::tokenization::fast::{FastTokenizer, TokenizerError};

/// The pad token's id, which every image span carries; the caller refuses
/// the request when the tokenizer defines none.
pub fn image_pad_token_id(tokenizer: &FastTokenizer) -> Result<u32, ApiError> {
    tokenizer.token_to_id(IMAGE_PAD_TOKEN).ok_or_else(|| {
        ApiError::new(
            400,
            "the tokenizer does not define the image pad token",
            "invalid_request_error",
        )
    })
}

/// An image error as the API refuses it: a full in-flight budget is an
/// overload, the rest are the request's own faults.
fn image_error_to_api(error: ImageError) -> ApiError {
    match error {
        ImageError::Capacity(message) => {
            ApiError::new(503, message, "frontend_overloaded")
        }
        ImageError::Invalid(message) => ApiError::new(400, message, "invalid_request_error"),
    }
}

/// Whether the images so far fit the context and the request frame. The
/// context check is the caller's switch, as the Python's `check_context`
/// is; a thinking budget's end tokens follow, at most this many.
pub fn check_image_request_size(
    tokens: u64,
    image_count: usize,
    pixel_bytes: usize,
    max_context: u64,
    check_context: bool,
    image_tokens_only: bool,
) -> Result<(), ApiError> {
    check_image_request_size_limited(
        tokens,
        image_count,
        pixel_bytes,
        max_context,
        check_context,
        image_tokens_only,
        MAX_FRAME_PAYLOAD_BYTES,
    )
}

/// [`check_image_request_size`] against its own frame limit, which the
/// tests shrink.
fn check_image_request_size_limited(
    tokens: u64,
    image_count: usize,
    pixel_bytes: usize,
    max_context: u64,
    check_context: bool,
    image_tokens_only: bool,
    max_frame_bytes: usize,
) -> Result<(), ApiError> {
    if check_context && tokens >= max_context {
        return Err(ContextLength {
            input_tokens: tokens,
            maximum_input_tokens: max_context - 1,
            image_tokens_only,
        }
        .into_api_error());
    }
    let frame_bytes = REQUEST_FIXED_BYTES as u64
        + 4 * tokens
        + IMAGE_SPAN_BYTES as u64 * image_count as u64
        + pixel_bytes as u64
        + 4 * MAX_THINKING_END_TOKENS as u64;
    if frame_bytes > max_frame_bytes as u64 {
        return Err(ApiError::new(
            400,
            "images exceed the request size limit",
            "invalid_request_error",
        ));
    }
    Ok(())
}

/// Prepared images in template render order: content parts in message
/// order, images in document order.
///
/// `check_deadline` is the caller's `remaining_request_time`: an image can
/// take a few hundred milliseconds to decode, and an expired request
/// starts no more of them. It returns `Err` when the request has none
/// left, which stops here.
pub fn prepare_images(
    cache: &ImageCache,
    messages: &[Value],
    max_image_pixels: u32,
    max_context: u64,
    check_context: bool,
    check_deadline: &mut dyn FnMut() -> Result<(), ApiError>,
) -> Result<RequestImages, ApiError> {
    prepare_images_limited(
        cache,
        messages,
        max_image_pixels,
        max_context,
        check_context,
        check_deadline,
        MAX_FRAME_PAYLOAD_BYTES,
    )
}

/// [`prepare_images`] against its own frame limit, which the tests shrink.
fn prepare_images_limited(
    cache: &ImageCache,
    messages: &[Value],
    max_image_pixels: u32,
    max_context: u64,
    check_context: bool,
    check_deadline: &mut dyn FnMut() -> Result<(), ApiError>,
    max_frame_bytes: usize,
) -> Result<RequestImages, ApiError> {
    let urls: Vec<Option<&str>> = messages
        .iter()
        .filter_map(|message| message.get("content").and_then(|content| content.as_array()))
        .flat_map(|parts| parts.iter())
        .filter(|part| part.get("type").and_then(|kind| kind.as_str()) == Some("image_url"))
        .map(|part| {
            part.get("image_url")
                .and_then(|image_url| image_url.get("url"))
                .and_then(|url| url.as_str())
        })
        .collect();
    let limit = MAX_IMAGE_SPANS;
    if urls.len() > limit {
        return Err(ApiError::new(
            400,
            format!("requests support at most {limit} images"),
            "invalid_request_error",
        ));
    }
    let mut prepared = cache.request_batch();
    let mut tokens: u64 = 0;
    let mut pixel_bytes: usize = 0;
    for url in urls {
        // An image can take a few hundred milliseconds to decode; an
        // expired request starts no more of them.
        check_deadline()?;
        // A missing or non-string url is refused like a non-data one, as
        // the Python's isinstance check does.
        let payload = decode_data_url(url.unwrap_or("")).map_err(image_error_to_api)?;
        let image = cache.prepare(&payload, max_image_pixels).map_err(image_error_to_api)?;
        tokens += u64::from(image.tokens());
        pixel_bytes += image.pixels.len();
        check_image_request_size_limited(
            tokens,
            prepared.len() + 1,
            pixel_bytes,
            max_context,
            check_context,
            true,
            max_frame_bytes,
        )?;
        prepared.append(image).map_err(image_error_to_api)?;
    }
    Ok(prepared)
}

/// A cryptographically random process-wide marker, matching Python's module
/// constant. Keeping it stable also lets the template renderer reuse its
/// compiled version across requests.
fn render_marker() -> Result<&'static str, ApiError> {
    static MARKER: OnceLock<String> = OnceLock::new();
    if let Some(marker) = MARKER.get() {
        return Ok(marker.as_str());
    }
    let mut random = [0; 16];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut file| file.read_exact(&mut random))
        .map_err(|_| ApiError::new(500, "image marker randomness is unavailable", "internal_server_error"))?;
    let marker = format!("__splash_image_{}__", random.iter().map(|byte| format!("{byte:02x}")).collect::<String>());
    let _ = MARKER.set(marker);
    Ok(MARKER.get().expect("image marker initialized").as_str())
}

/// Track placeholders emitted by the template, not quoted in input text.
///
/// A temporary render marker is removed before tokenization, so the pinned
/// template's final text and token IDs remain unchanged. Token offsets tie
/// each real image to its placeholder even when a coding agent has read
/// documentation or source containing literal vision tokens.
///
/// `render` renders one template source with the request's own messages,
/// tools and options — the caller's `_apply_chat_template` with the source
/// given. `tokenize` encodes with no special token added and the offsets,
/// as [`FastTokenizer::encode_with_offsets`] does. The offsets are byte
/// spans, and the recorded placeholder spans are byte spans of the same
/// text, where the Python records character spans of its own.
pub fn render_image_tokens(
    template_source: &str,
    render: &dyn Fn(&str) -> Result<String, ChatTemplateError>,
    tokenize: &dyn Fn(&str) -> Result<(Vec<u32>, Vec<(usize, usize)>), TokenizerError>,
) -> Result<(Vec<u32>, Vec<usize>, String), ApiError> {
    let marker = render_marker()?;
    let marked = template_source.replace(IMAGE_PAD_TOKEN, &marker);
    let rendered = render(&marked).map_err(|error| {
        // A template's own refusal reads as the Python's caller answers a
        // TemplateError; the rest keep the plain refusal.
        ApiError::new(
            400,
            format!("messages could not be rendered: {error}"),
            "invalid_request_error",
        )
    })?;
    // Split on the marker to find each pad's span, walking the parts as
    // the Python does: every pad after the first sits 36 bytes earlier
    // than its marker was, the marker being the longer string.
    let parts: Vec<&str> = rendered.split(&marker).collect();
    let pad_span = IMAGE_PAD_TOKEN.len();
    let mut starts: Vec<usize> = Vec::with_capacity(parts.len().saturating_sub(1));
    let mut offset = 0usize;
    for part in &parts[..parts.len().saturating_sub(1)] {
        offset += part.len();
        starts.push(offset);
        offset += pad_span;
    }
    let joined = parts.join(IMAGE_PAD_TOKEN);
    let (tokens, offsets) = tokenize(&joined).map_err(|_| {
        ApiError::new(400, "messages could not be rendered", "invalid_request_error")
    })?;
    let spans: HashSet<(usize, usize)> = starts
        .iter()
        .map(|&start| (start, start + pad_span))
        .collect();
    let positions: Vec<usize> = offsets
        .iter()
        .enumerate()
        .filter(|(_, span)| spans.contains(span))
        .map(|(index, _)| index)
        .collect();
    Ok((tokens, positions, joined))
}

/// The prompt's token count with every image widened, checked against the
/// rendered template: one position per image, each holding the pad token.
pub fn image_token_count(
    prompt_tokens: &[u32],
    prepared: &[PreparedImage],
    positions: &[usize],
    pad_id: u32,
) -> Result<u64, ApiError> {
    if positions.len() != prepared.len()
        || positions
            .iter()
            .any(|&position| prompt_tokens.get(position) != Some(&pad_id))
    {
        return Err(ApiError::new(
            400,
            "image count does not match the rendered template",
            "invalid_request_error",
        ));
    }
    let extra: u64 = prepared
        .iter()
        .map(|image| u64::from(image.tokens()) - 1)
        .sum();
    Ok(prompt_tokens.len() as u64 + extra)
}

/// Widens the template's single placeholder per image to the image's
/// merged token count and returns the spans the engine injects into.
pub fn expand_image_pads(
    prompt_tokens: &[u32],
    prepared: &[PreparedImage],
    positions: &[usize],
    pad_id: u32,
    max_context: u64,
    check_context: bool,
) -> Result<(Vec<u32>, Vec<ImageSpan>, Vec<u8>), ApiError> {
    expand_image_pads_limited(
        prompt_tokens,
        prepared,
        positions,
        pad_id,
        max_context,
        check_context,
        MAX_FRAME_PAYLOAD_BYTES,
    )
}

/// [`expand_image_pads`] against its own frame limit, which the tests
/// shrink.
fn expand_image_pads_limited(
    prompt_tokens: &[u32],
    prepared: &[PreparedImage],
    positions: &[usize],
    pad_id: u32,
    max_context: u64,
    check_context: bool,
    max_frame_bytes: usize,
) -> Result<(Vec<u32>, Vec<ImageSpan>, Vec<u8>), ApiError> {
    let token_count = image_token_count(prompt_tokens, prepared, positions, pad_id)?;
    // Validate lengths before expanding tokens or copying repeated pixels.
    // HTTP body and image-cache limits do not bound decoded request size.
    let pixel_bytes: usize = prepared.iter().map(|image| image.pixels.len()).sum();
    check_image_request_size_limited(
        token_count,
        prepared.len(),
        pixel_bytes,
        max_context,
        check_context,
        false,
        max_frame_bytes,
    )?;
    let mut expanded: Vec<u32> = Vec::new();
    let mut spans: Vec<ImageSpan> = Vec::new();
    let mut pixels: Vec<u8> = Vec::with_capacity(pixel_bytes);
    let mut cursor = 0usize;
    for (position, image) in positions.iter().zip(prepared.iter()) {
        expanded.extend_from_slice(&prompt_tokens[cursor..*position]);
        spans.push(ImageSpan {
            offset: expanded.len() as u32,
            tokens: image.tokens(),
            grid_height: image.grid_height,
            grid_width: image.grid_width,
            digest_lo: image.digest_lo,
            digest_hi: image.digest_hi,
        });
        expanded.extend(std::iter::repeat(pad_id).take(image.tokens() as usize));
        pixels.extend_from_slice(&image.pixels);
        cursor = position + 1;
    }
    expanded.extend_from_slice(&prompt_tokens[cursor..]);
    Ok((expanded, spans, pixels))
}

#[cfg(test)]
mod tests {
    use base64::Engine as _;
    use serde_json::json;

    use super::*;
    use crate::templates::render::ChatTemplateRenderer;
    use image::ImageEncoder;

    const TEMPLATE: &str = "{% for message in messages %}{{ message.role }}: \
        {% if message.content is string %}{{ message.content }}\
        {% else %}{% for part in message.content %}\
        {% if part.type == 'image_url' %}\
        {{ '<|vision_start|><|image_pad|><|vision_end|>' }}\
        {% else %}{{ part.text }}{% endif %}{% endfor %}{% endif %}\
        {{ '\\n' }}{% endfor %}";

    #[test]
    fn image_render_marker_is_128_bit_and_stable_across_requests() {
        let first = render_marker().unwrap().to_owned();
        assert!(first.starts_with("__splash_image_") && first.ends_with("__"));
        assert_eq!(first.len(), "__splash_image_".len() + 32 + 2);
        assert!(first["__splash_image_".len()..first.len() - 2]
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()));
        assert_eq!(render_marker().unwrap(), first);
    }

    /// A WordLevel tokenizer with the vision tokens added, written to a
    /// fresh directory the test owns.
    fn pad_tokenizer() -> (FastTokenizer, tempdir::TempDir) {
        let document = json!({
            "version": "1.0",
            "added_tokens": [
                {"id": 50, "content": "<|image_pad|>", "single_word": false,
                 "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 51, "content": "<|vision_start|>", "single_word": false,
                 "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 52, "content": "<|vision_end|>", "single_word": false,
                 "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            ],
            "normalizer": null,
            "pre_tokenizer": {"type": "WhitespaceSplit"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel", "unk_token": "[UNK]",
                "vocab": {"[UNK]": 0, "hello": 1, "look": 2, "at": 3, "this": 4},
            },
        });
        let directory = tempdir::TempDir::new("splash-images-test");
        std::fs::write(
            directory.path.join("tokenizer.json"),
            serde_json::to_string(&document).unwrap(),
        )
        .unwrap();
        let tokenizer = FastTokenizer::from_file(&directory.path).unwrap();
        (tokenizer, directory)
    }

    /// A minimal temporary directory, std only.
    mod tempdir {
        use std::path::PathBuf;
        use std::sync::atomic::{AtomicU64, Ordering};

        static NEXT_ID: AtomicU64 = AtomicU64::new(0);

        pub struct TempDir {
            pub path: PathBuf,
        }

        impl TempDir {
            pub fn new(name: &str) -> TempDir {
                let path = std::env::temp_dir().join(format!(
                    "{name}-{}-{}-{}",
                    std::process::id(),
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|since| since.as_nanos())
                        .unwrap_or(0),
                    NEXT_ID.fetch_add(1, Ordering::Relaxed)
                ));
                std::fs::create_dir(&path).unwrap();
                TempDir { path }
            }
        }

        impl Drop for TempDir {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.path);
            }
        }
    }

    fn render_with(messages: &Value) -> impl Fn(&str) -> Result<String, ChatTemplateError> + '_ {
        move |source| {
            ChatTemplateRenderer::new(source)?
                .render(messages, None, None, false, &Default::default())
        }
    }

    fn quoted_messages() -> Value {
        let quoted = "中文 📷 <|vision_start|><|image_pad|><|vision_end|>";
        json!([
            {"role": "system", "content": format!("Document: {IMAGE_PAD_TOKEN}")},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "unused"}},
                {"type": "text", "text": quoted},
            ]},
            {"role": "tool", "content": [
                {"type": "image_url", "image_url": {"url": "unused"}},
                {"type": "text", "text": quoted},
            ]},
        ])
    }

    fn fake_prepared(index: u64) -> PreparedImage {
        PreparedImage {
            grid_height: 4,
            grid_width: 4,
            pixels: vec![index as u8],
            digest_lo: index,
            digest_hi: 0,
        }
    }

    #[test]
    fn image_positions_ignore_quoted_vision_tokens() {
        let (tokenizer, _directory) = pad_tokenizer();
        let messages = quoted_messages();
        let messages_ref = &messages;
        let tokenize = |text: &str| tokenizer.encode_with_offsets(text, false);
        let render = render_with(messages_ref);

        // The baseline render, with no marker, carries every pad the text
        // holds: the template's two and the quoted three.
        let baseline_text = render(TEMPLATE).unwrap();
        let (baseline, _) = tokenize(&baseline_text).unwrap();
        let pad_id = tokenizer.token_to_id(IMAGE_PAD_TOKEN).unwrap();
        let all_pads: Vec<usize> = baseline
            .iter()
            .enumerate()
            .filter(|(_, token)| **token == pad_id)
            .map(|(index, _)| index)
            .collect();
        assert_eq!(all_pads.len(), 5);

        let (tokens, positions, rendered) =
            render_image_tokens(TEMPLATE, &render, &tokenize).unwrap();
        assert_eq!(tokens, baseline);
        assert_eq!(rendered, baseline_text);
        assert_eq!(positions, [all_pads[1], all_pads[3]]);

        // A repeat renders the same text and tokens, whatever marker it
        // draws.
        let repeated = render_image_tokens(TEMPLATE, &render, &tokenize).unwrap();
        assert_eq!(repeated, (tokens.clone(), positions.clone(), rendered.clone()));

        let prepared = [fake_prepared(1), fake_prepared(2)];
        let (expanded, spans, pixels) =
            expand_image_pads(&tokens, &prepared, &positions, pad_id, 1024, true).unwrap();
        let mut expected = baseline.clone();
        for position in positions.iter().rev() {
            expected.splice(*position..*position + 1, std::iter::repeat(pad_id).take(4));
        }
        assert_eq!(expanded, expected);
        assert_eq!(spans[0].offset as usize, positions[0]);
        assert_eq!(spans[1].offset as usize, positions[1] + 3);
        assert_eq!(pixels, vec![1u8, 2]);
        let error =
            expand_image_pads(&tokens, &prepared[..1], &positions, pad_id, 1024, true).unwrap_err();
        assert!(error.message.contains("image count"));
    }

    #[test]
    fn image_count_is_checked_before_decoding() {
        let limit = MAX_IMAGE_SPANS;
        let part = json!({"type": "image_url", "image_url": {"url": "unused"}});
        let messages = vec![
            json!({"role": "user", "content": vec![part.clone(); limit]}),
            json!({"role": "tool", "content": [part]}),
        ];
        let cache = ImageCache::new();
        let mut no_deadline = || -> Result<(), ApiError> { Ok(()) };
        let error = prepare_images(
            &cache,
            &messages,
            crate::images::images::MAX_PIXELS,
            1024,
            false,
            &mut no_deadline,
        )
        .unwrap_err();
        assert!(error.message.contains(&format!("at most {limit} images")));
        assert_eq!(error.status, 400);
        assert_eq!(cache.stats().request_bytes, 0);
        // One part past the count check reaches the decode, which refuses
        // a non-data url before any image work.
        let single = vec![json!({"role": "user", "content": [part]})];
        let error = prepare_images(
            &cache,
            &single,
            crate::images::images::MAX_PIXELS,
            1024,
            false,
            &mut no_deadline,
        )
        .unwrap_err();
        assert_eq!(error.message, "only data: image URLs are supported");
        assert_eq!(cache.stats().request_bytes, 0);
    }

    #[test]
    fn image_preparation_stops_at_aggregate_budget() {
        let png = |color: [u8; 3]| {
            let image = image::RgbImage::from_pixel(2, 2, image::Rgb(color));
            let mut buffer = std::io::Cursor::new(Vec::new());
            image::codecs::png::PngEncoder::new(&mut buffer)
                .write_image(image.as_raw(), 2, 2, image::ExtendedColorType::Rgb8)
                .unwrap();
            format!(
                "data:image/png;base64,{}",
                base64::engine::general_purpose::STANDARD.encode(buffer.into_inner())
            )
        };
        let messages = vec![json!({"content": [
            {"type": "image_url", "image_url": {"url": png([200, 30, 30])}},
            {"type": "image_url", "image_url": {"url": png([30, 30, 200])}},
            {"type": "image_url", "image_url": {"url": png([30, 200, 30])}},
        ]})];
        let cache = ImageCache::new();
        let mut no_deadline = || -> Result<(), ApiError> { Ok(()) };
        // One image's frame, plus one byte: the second image's check
        // refuses the request, so the third never decodes.
        let first = cache
            .prepare(
                &decode_data_url(&png([200, 30, 30])).unwrap(),
                crate::images::images::MAX_PIXELS,
            )
            .unwrap();
        let per_image = REQUEST_FIXED_BYTES as u64
            + 4 * u64::from(first.tokens())
            + IMAGE_SPAN_BYTES as u64
            + first.pixels.len() as u64
            + 4 * MAX_THINKING_END_TOKENS as u64;
        let error = prepare_images_limited(
            &cache,
            &messages,
            crate::images::images::MAX_PIXELS,
            u64::MAX,
            false,
            &mut no_deadline,
            (per_image + 1) as usize,
        )
        .unwrap_err();
        assert!(error.message.contains("request size limit"));
        assert_eq!(error.status, 400);
        // Two images prepared, the third never started.
        assert_eq!(cache.stats().entries, 2);
        assert_eq!(cache.stats().request_bytes, 0);
    }

    #[test]
    fn image_preparation_stops_once_the_request_expires() {
        let png = |color: [u8; 3]| {
            let image = image::RgbImage::from_pixel(2, 2, image::Rgb(color));
            let mut buffer = std::io::Cursor::new(Vec::new());
            image::codecs::png::PngEncoder::new(&mut buffer)
                .write_image(image.as_raw(), 2, 2, image::ExtendedColorType::Rgb8)
                .unwrap();
            format!(
                "data:image/png;base64,{}",
                base64::engine::general_purpose::STANDARD.encode(buffer.into_inner())
            )
        };
        let part = |color: [u8; 3]| {
            json!({"type": "image_url", "image_url": {"url": png(color)}})
        };
        let messages = vec![json!({"content": [
            part([200, 30, 30]),
            part([30, 30, 200]),
            part([30, 200, 30]),
        ]})];
        let cache = ImageCache::new();
        let mut calls = 0;
        let mut deadline = || -> Result<(), ApiError> {
            calls += 1;
            if calls > 1 {
                // Each image takes a moment; the second starts none.
                Err(ApiError::new(504, "request timed out", "request_timeout"))
            } else {
                Ok(())
            }
        };
        let error = prepare_images(
            &cache,
            &messages,
            crate::images::images::MAX_PIXELS,
            u64::MAX,
            false,
            &mut deadline,
        )
        .unwrap_err();
        assert_eq!((error.status, error.code.as_str()), (504, "request_timeout"));
        assert_eq!(calls, 2);
        assert_eq!(cache.stats().request_bytes, 0);
    }

    #[test]
    fn image_preparation_collects_parts_in_order_and_releases_on_error() {
        // Two 2x2 PNGs, red then blue, in a user turn and a tool result.
        let png = |color: [u8; 3]| {
            let image = image::RgbImage::from_pixel(2, 2, image::Rgb(color));
            let mut buffer = std::io::Cursor::new(Vec::new());
            image::codecs::png::PngEncoder::new(&mut buffer)
                .write_image(image.as_raw(), 2, 2, image::ExtendedColorType::Rgb8)
                .unwrap();
            let payload = buffer.into_inner();
            format!(
                "data:image/png;base64,{}",
                base64::engine::general_purpose::STANDARD.encode(&payload)
            )
        };
        let messages = vec![
            json!({"role": "user", "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": png([200, 30, 30])}},
            ]}),
            json!({"role": "tool", "content": [
                {"type": "image_url", "image_url": {"url": png([30, 30, 200])}},
            ]}),
        ];
        let cache = ImageCache::new();
        let mut no_deadline = || -> Result<(), ApiError> { Ok(()) };
        let prepared = prepare_images(
            &cache,
            &messages,
            crate::images::images::MAX_PIXELS,
            u64::MAX,
            false,
            &mut no_deadline,
        )
        .unwrap();
        assert_eq!(prepared.len(), 2);
        // 2x2 upscales to the 256x256 minimum: 64 tokens, 196,608 bytes.
        assert_eq!(prepared.images()[0].tokens(), 64);
        assert_eq!(prepared.images()[1].pixels[0..3], [30, 30, 200]);
        assert!(cache.stats().request_bytes > 0);
        drop(prepared);
        assert_eq!(cache.stats().request_bytes, 0);

        // A full in-flight budget answers 503 and releases the batch.
        let tight = ImageCache::with_budgets(
            ImageCache::BUDGET_BYTES,
            1024,
        );
        let error = prepare_images(
            &tight,
            &messages,
            crate::images::images::MAX_PIXELS,
            u64::MAX,
            false,
            &mut no_deadline,
        )
        .unwrap_err();
        assert_eq!((error.status, error.code.as_str()), (503, "frontend_overloaded"));
        assert!(error.message.contains("image memory budget"));
        assert_eq!(tight.stats().request_bytes, 0);
    }

    #[test]
    fn image_size_is_checked_before_pixel_concatenation() {
        let (tokenizer, _directory) = pad_tokenizer();
        let pad_id = tokenizer.token_to_id(IMAGE_PAD_TOKEN).unwrap();
        // The pixels are a small real buffer, but the frame budget the
        // check runs against is smaller still: the guard refuses on
        // lengths alone, before any output copy.
        let image = PreparedImage {
            grid_height: 4,
            grid_width: 4,
            pixels: vec![0u8; 2000],
            digest_lo: 0,
            digest_hi: 0,
        };
        let error = expand_image_pads_limited(
            &[pad_id],
            &[image],
            &[0],
            pad_id,
            u64::MAX,
            false,
            1024,
        )
        .unwrap_err();
        assert!(error.message.contains("request size limit"));
    }

    #[test]
    fn image_context_is_checked_before_expanding_placeholders() {
        let (tokenizer, _directory) = pad_tokenizer();
        let pad_id = tokenizer.token_to_id(IMAGE_PAD_TOKEN).unwrap();
        let image = PreparedImage {
            grid_height: 16,
            grid_width: 32,
            pixels: Vec::new(),
            digest_lo: 0,
            digest_hi: 0,
        };
        // 16x32 patches merge to 8x16 = 128 tokens, at a 128-token context.
        let error = expand_image_pads(&[pad_id], &[image], &[0], pad_id, 128, true).unwrap_err();
        assert!(error.message.contains("context window"));
        assert_eq!(error.code, "context_length_exceeded");
        assert!(!error.image_tokens_only);
    }

    #[test]
    fn image_token_count_checks_positions_and_pad_id() {
        let (tokenizer, _directory) = pad_tokenizer();
        let pad_id = tokenizer.token_to_id(IMAGE_PAD_TOKEN).unwrap();
        let prepared = [fake_prepared(1), fake_prepared(2)];
        // Two 4x4-patch grids merge to 4 tokens each: 3 extra per image.
        let prompt = [pad_id, 9, pad_id, 9];
        assert_eq!(
            image_token_count(&prompt, &prepared, &[0, 2], pad_id).unwrap(),
            4 + 6
        );
        // A count mismatch, and a position that holds another token.
        assert!(image_token_count(&prompt, &prepared[..1], &[0, 2], pad_id).is_err());
        assert!(image_token_count(&prompt, &prepared, &[0, 3], pad_id).is_err());
        assert!(image_token_count(&prompt, &prepared, &[0, 9], pad_id).is_err());
    }

    #[test]
    fn a_tokenizer_without_the_pad_token_is_refused() {
        let document = json!({
            "version": "1.0",
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "WhitespaceSplit"},
            "post_processor": null,
            "decoder": null,
            "model": {"type": "WordLevel", "unk_token": "[UNK]", "vocab": {"[UNK]": 0}},
        });
        let directory = tempdir::TempDir::new("splash-images-plain");
        std::fs::write(
            directory.path.join("tokenizer.json"),
            serde_json::to_string(&document).unwrap(),
        )
        .unwrap();
        let tokenizer = FastTokenizer::from_file(&directory.path).unwrap();
        let error = image_pad_token_id(&tokenizer).unwrap_err();
        assert_eq!(error.status, 400);
        assert!(error.message.contains("does not define the image pad token"));
    }
}
