"""Image placeholders in a prepared prompt: preparation, rendering, expansion.

The chat template emits one pad token per image; these functions prepare the
images a request carries, find the pad tokens the template rendered, and
widen each of them to the image's merged token count with the pixels the
engine injects there.
"""

import secrets

from . import images as image_input
from . import protocol as wire
from .backend import remaining_request_time
from .chat_messages import IMAGE_PAD_TOKEN
from .errors import APIError, ContextLengthError

# A stable marker lets repeated image requests reuse the compiled template.
IMAGE_RENDER_MARKER = f"__splash_image_{secrets.token_hex(16)}__"


def prepare_images(frontend, messages, deadline, *, check_context=True):
    """Prepared images in template render order: content parts in message
    order, images in document order."""
    parts = [
        part
        for message in messages
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if part.get("type") == "image_url"
    ]
    limit = wire.MAX_IMAGE_SPANS
    if len(parts) > limit:
        raise APIError(400, f"requests support at most {limit} images")
    prepared = frontend.images.request_batch()
    tokens = pixel_bytes = 0
    for part in parts:
        # An image can take a few hundred milliseconds to decode; an
        # expired request starts no more of them.
        remaining_request_time(deadline)
        try:
            payload = image_input.decode_data_url(part["image_url"]["url"])
            image = frontend.images.prepare(payload, frontend.max_image_pixels)
            tokens += image.tokens
            pixel_bytes += len(image.pixels)
            frontend._check_image_request_size(
                tokens,
                len(prepared) + 1,
                pixel_bytes,
                check_context=check_context,
                image_tokens_only=True,
            )
            prepared.append(image)
        except image_input.ImageCapacityError as error:
            raise APIError(503, str(error), "frontend_overloaded") from error
        except image_input.ImageError as error:
            raise APIError(400, str(error)) from error
    return prepared


def check_image_request_size(
    frontend,
    tokens,
    image_count,
    pixel_bytes,
    *,
    check_context=True,
    image_tokens_only=False,
):
    if check_context and tokens >= frontend.max_context:
        raise ContextLengthError(
            tokens, frontend.max_context - 1, image_tokens_only=image_tokens_only
        )
    # A thinking budget's end tokens follow, at most this many.
    frame_bytes = (
        wire.REQUEST_FIXED_BYTES
        + 4 * tokens
        + wire.IMAGE_SPAN_BYTES * image_count
        + pixel_bytes
        + 4 * wire.MAX_THINKING_END_TOKENS
    )
    if frame_bytes > wire.MAX_FRAME_PAYLOAD_BYTES:
        raise APIError(400, "images exceed the request size limit")


def render_image_tokens(frontend, messages, template):
    """Track placeholders emitted by the template, not quoted in input text.

    A temporary render marker is removed before tokenization, so the pinned
    template's final text and token IDs remain unchanged. Token offsets tie
    each real image to its placeholder even when a coding agent has read
    documentation or source containing literal vision tokens.
    """
    rendered = frontend._apply_chat_template(
        messages,
        {
            **template,
            "tokenize": False,
            "chat_template": template["chat_template"].replace(
                IMAGE_PAD_TOKEN, IMAGE_RENDER_MARKER
            ),
        },
    )
    parts = rendered.split(IMAGE_RENDER_MARKER)
    image_offsets = set()
    offset = 0
    for part in parts[:-1]:
        offset += len(part)
        image_offsets.add((offset, offset + len(IMAGE_PAD_TOKEN)))
        offset += len(IMAGE_PAD_TOKEN)
    rendered = IMAGE_PAD_TOKEN.join(parts)
    encoded = frontend._tokenize(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    positions = [
        index
        for index, span in enumerate(encoded["offset_mapping"])
        if tuple(span) in image_offsets
    ]
    return list(encoded["input_ids"]), positions, rendered


def image_token_count(frontend, prompt_tokens, prepared, positions):
    pad_id = frontend.tokenizer.convert_tokens_to_ids(IMAGE_PAD_TOKEN)
    if len(positions) != len(prepared) or any(
        prompt_tokens[position] != pad_id for position in positions
    ):
        raise APIError(400, "image count does not match the rendered template")
    return len(prompt_tokens) + sum(image.tokens - 1 for image in prepared)


def expand_image_pads(frontend, prompt_tokens, prepared, positions):
    """Widens the template's single placeholder per image to the image's
    merged token count and returns the spans the engine injects into."""
    token_count = frontend._image_token_count(prompt_tokens, prepared, positions)
    # Validate lengths before expanding tokens or copying repeated pixels.
    # HTTP body and image-cache limits do not bound decoded request size.
    frontend._check_image_request_size(
        token_count,
        len(prepared),
        sum(len(image.pixels) for image in prepared),
    )
    expanded, spans, cursor = [], [], 0
    pad_id = frontend.tokenizer.convert_tokens_to_ids(IMAGE_PAD_TOKEN)
    for position, image in zip(positions, prepared):
        expanded.extend(prompt_tokens[cursor:position])
        spans.append(
            wire.ImageSpan(
                len(expanded),
                image.tokens,
                image.grid_height,
                image.grid_width,
                image.digest_lo,
                image.digest_hi,
            )
        )
        expanded.extend([pad_id] * image.tokens)
        cursor = position + 1
    expanded.extend(prompt_tokens[cursor:])
    pixels = b"".join(image.pixels for image in prepared)
    return expanded, tuple(spans), pixels
