"""Decode and resize images to patch-aligned uint8 RGB within the pixel budget.

The engine handles normalization and vision inference. Grid geometry and a
content digest identify image prefixes; a byte-budgeted cache reuses images
resent in chat history.
"""

import base64
import hashlib
import io
import math
import struct
import threading
from dataclasses import dataclass

from .lru import LRUCache

PATCH = 16
MERGE = 2
FACTOR = PATCH * MERGE
# Qwen3-VL's shortest_edge floor; the serving cap below is far below the model's.
MIN_PIXELS = 65_536
# Serving default and largest cap: 16,384 patches, the protocol's per-image ceiling.
MAX_PIXELS = 4_194_304
MAX_ASPECT = 200
# Bound codec work before full-resolution decode, independently of the model's
# resized input budget. Ordinary screenshots and up to 32 MP photos fit here.
MAX_SOURCE_PIXELS = 32 * 1024 * 1024
MAX_IMAGE_BYTES = 32 * 1024 * 1024
IMAGE_FORMATS = ("JPEG", "PNG", "WEBP", "GIF")


class ImageError(ValueError):
    pass


class ImageCapacityError(ImageError):
    pass


@dataclass(frozen=True, slots=True)
class PreparedImage:
    grid_height: int
    grid_width: int
    pixels: bytes
    digest_lo: int
    digest_hi: int

    @property
    def tokens(self) -> int:
        """Merged language tokens the image occupies."""
        return (self.grid_height // MERGE) * (self.grid_width // MERGE)


def decode_data_url(url) -> bytes:
    if not isinstance(url, str) or not url.startswith("data:"):
        raise ImageError("only data: image URLs are supported")
    header, _, encoded = url.partition(",")
    if ";base64" not in header:
        raise ImageError("image data URLs must be base64-encoded")
    try:
        compact = "".join(encoded.split()).replace("-", "+").replace("_", "/")
        if len(compact) > MAX_IMAGE_BYTES * 4 // 3 + 4:
            raise ImageError("image exceeds the size limit")
        if len(compact) % 4 == 1:
            raise ValueError("invalid base64 length")
        compact += "=" * (-len(compact) % 4)
        raw = base64.b64decode(compact, validate=True)
    except ImageError:
        raise
    except (ValueError, TypeError) as error:
        raise ImageError("image data URL is not valid base64") from error
    if len(raw) > MAX_IMAGE_BYTES:
        raise ImageError("image exceeds the size limit")
    return raw


def max_patches(max_pixels: int) -> int:
    """The most patches an image resized within max_pixels can have: its
    sides are whole merge units, so its patch count is a multiple of 4."""
    return max_pixels // (PATCH * PATCH) // 4 * 4


def smart_resize(height: int, width: int, max_pixels: int) -> tuple[int, int]:
    """Upstream Qwen2VL smart-resize: patch/merge-aligned sides inside the
    pixel budget with the aspect ratio preserved as closely as possible."""
    if height <= 0 or width <= 0:
        raise ImageError("image dimensions must be positive")
    if not MIN_PIXELS <= max_pixels <= MAX_PIXELS:
        raise ImageError("invalid image pixel budget")
    if max(height, width) / min(height, width) > MAX_ASPECT:
        raise ImageError("absolute aspect ratio must be smaller than 200")
    h_bar = round(height / FACTOR) * FACTOR
    w_bar = round(width / FACTOR) * FACTOR
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(FACTOR, math.floor(height / beta / FACTOR) * FACTOR)
        w_bar = max(FACTOR, math.floor(width / beta / FACTOR) * FACTOR)
    elif h_bar * w_bar < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (height * width))
        h_bar = math.ceil(height * beta / FACTOR) * FACTOR
        w_bar = math.ceil(width * beta / FACTOR) * FACTOR
    # Upscaling to the minimum can also overshoot a small serving cap after
    # alignment. The hard cap takes precedence over that soft minimum.
    while h_bar * w_bar > max_pixels:
        if h_bar >= w_bar and h_bar > FACTOR:
            h_bar -= FACTOR
        else:
            w_bar -= FACTOR
    return h_bar, w_bar


def prepare_inkling(payload: bytes, max_pixels: int = MAX_PIXELS) -> "PreparedInklingImage":
    """Prepare Inkling-2's 40px tiles without adapting them to Qwen's grid.

    Each tile contains four 20px patches and becomes one language token. The
    raw RGB image is padded with -1 before rescaling and normalization;
    transparent pixels are composited over Inkling's 8px 204/153 checkerboard.
    Images exceeding the padded pixel budget are rejected, never resized.
    """
    from PIL import Image, ImageOps

    if not isinstance(max_pixels, int) or isinstance(max_pixels, bool) or not 0 < max_pixels <= MAX_PIXELS:
        raise ImageError("invalid image pixel budget")
    if len(payload) > MAX_IMAGE_BYTES:
        raise ImageError("image exceeds the size limit")
    try:
        with Image.open(io.BytesIO(payload), formats=IMAGE_FORMATS) as decoded:
            if decoded.height * decoded.width > MAX_SOURCE_PIXELS:
                raise ImageError("source image exceeds the pixel limit")
            try:
                ImageOps.exif_transpose(decoded, in_place=True)
            except Exception:
                pass
            width, height = decoded.size
            if (
                width <= 0
                or height <= 0
                or max(width, height) / min(width, height) > MAX_ASPECT
            ):
                raise ImageError("absolute aspect ratio must be smaller than 200")
            padded_width = (width + 39) // 40 * 40
            padded_height = (height + 39) // 40 * 40
            if padded_width * padded_height > max_pixels:
                raise ImageError("padded image exceeds the pixel limit")
            rgba = decoded.convert("RGBA")
            background = Image.new("RGB", (padded_width, padded_height))
            px = background.load()
            for y in range(padded_height):
                for x in range(padded_width):
                    value = 204 if ((x // 8 + y // 8) % 2 == 0) else 153
                    px[x, y] = (value, value, value)
            background.paste(rgba, (0, 0), rgba.getchannel("A"))
            # Padding is raw -1 before the 1/255 rescale and normalization.
            raw = bytearray(padded_width * padded_height * 3)
            source = background.tobytes()
            raw[:] = source
            import array
            values = array.array("f", ((channel / 255.0 - 0.5) / 0.5 for channel in raw))
            for y in range(height):
                for x in range(width, padded_width):
                    start = (y * padded_width + x) * 3
                    values[start : start + 3] = array.array("f", [(-1 / 255.0 - 0.5) / 0.5]) * 3
            for y in range(height, padded_height):
                start = y * padded_width * 3
                values[start : start + padded_width * 3] = array.array("f", [(-1 / 255.0 - 0.5) / 0.5]) * (padded_width * 3)
            # Upstream emits [grid rows, grid columns, 40, 40, RGB].
            if struct.pack("=I", 1) != struct.pack("<I", 1):
                values.byteswap()
            ordered = bytearray(len(values) * values.itemsize)
            source = memoryview(values).cast("B")
            destination = memoryview(ordered)
            row_bytes = 40 * 3 * values.itemsize
            offset = 0
            for tile_y in range(padded_height // 40):
                for tile_x in range(padded_width // 40):
                    for row in range(40):
                        start = ((tile_y * 40 + row) * padded_width + tile_x * 40) * 3 * values.itemsize
                        destination[offset:offset + row_bytes] = source[start:start + row_bytes]
                        offset += row_bytes
            pixels = bytes(ordered)
    except ImageError:
        raise
    except Exception as error:
        raise ImageError("image could not be decoded") from error
    grid_height, grid_width = padded_height // 40, padded_width // 40
    digest = hashlib.sha256(b"inkling2" + struct.pack("<II", grid_height, grid_width) + pixels).digest()
    digest_lo, digest_hi = struct.unpack_from("<QQ", digest)
    return PreparedInklingImage(grid_height, grid_width, pixels, digest_lo, digest_hi)


@dataclass(frozen=True, slots=True)
class PreparedInklingImage:
    grid_height: int
    grid_width: int
    pixels: bytes
    digest_lo: int
    digest_hi: int
    processor: str = "inkling2"

    @property
    def tile_height(self):
        return self.grid_height

    @property
    def tile_width(self):
        return self.grid_width

    @property
    def tokens(self):
        return self.tile_height * self.tile_width


def prepare(payload: bytes, max_pixels: int = MAX_PIXELS) -> PreparedImage:
    """Decode, convert to RGB, resize into the grid, and digest the pixels."""
    from PIL import Image, ImageOps

    try:
        with Image.open(io.BytesIO(payload), formats=IMAGE_FORMATS) as decoded:
            if decoded.height * decoded.width > MAX_SOURCE_PIXELS:
                raise ImageError("source image exceeds the pixel limit")
            stored = decoded.size
            height, width = smart_resize(decoded.height, decoded.width, max_pixels)
            try:
                # Cameras store the sensor's orientation and an EXIF tag that
                # turns it upright for display; show the model the upright one.
                ImageOps.exif_transpose(decoded, in_place=True)
            except Exception:
                pass  # A malformed tag leaves the stored orientation.
            if decoded.size != stored:
                height, width = width, height
            if decoded.has_transparency_data:
                # Composite onto white as Qwen's preprocessing does: transparent
                # pixels usually store black, which hides dark content.
                rgba = decoded.convert("RGBA")
                image = Image.new("RGB", rgba.size, (255, 255, 255))
                image.paste(rgba, mask=rgba)
            else:
                image = decoded.convert("RGB")
            if (image.height, image.width) != (height, width):
                image = image.resize((width, height), Image.Resampling.BICUBIC)
            pixels = image.tobytes()
    except ImageError:
        raise
    except Exception as error:
        raise ImageError("image could not be decoded") from error
    grid_height, grid_width = height // PATCH, width // PATCH
    digest = hashlib.sha256(struct.pack("<II", grid_height, grid_width) + pixels)
    digest_lo, digest_hi = struct.unpack_from("<QQ", digest.digest())
    return PreparedImage(grid_height, grid_width, pixels, digest_lo, digest_hi)


class PreparedImages(list):
    """Append-only request batch; its owner travels with the pixel payload.

    Charge each occurrence, even cache hits: concatenation and native transport
    repeat those pixels. Release only when every holder drops the batch, not
    when HTTP disconnects while native work or a mask task still owns it.
    """

    def __init__(self, cache):
        super().__init__()
        self._cache = cache
        self._bytes = 0

    def append(self, image):
        size = len(image.pixels)
        with self._cache._lock:
            if size > self._cache.REQUEST_BUDGET_BYTES - self._cache._request_bytes:
                raise ImageCapacityError("in-flight image memory budget is full")
            self._cache._request_bytes += size
            self._bytes += size
        super().append(image)

    def __del__(self):
        with self._cache._lock:
            self._cache._request_bytes -= self._bytes


class ImageCache:
    """Byte-budgeted LRU of prepared images keyed by the raw image bytes."""

    # Cached prepared images, and the images of requests in flight.
    BUDGET_BYTES = 256 * 1024 * 1024
    REQUEST_BUDGET_BYTES = 256 * 1024 * 1024

    def __init__(self):
        # GC may finalize a cancelled batch during a cache insertion on this
        # same thread; returning its byte charge must not deadlock the cache.
        self._lock = threading.RLock()
        self._entries = LRUCache(self.BUDGET_BYTES)
        self._request_bytes = 0

    def request_batch(self):
        return PreparedImages(self)

    def prepare(
        self, payload: bytes, max_pixels: int, processor="qwen"
    ) -> PreparedImage | PreparedInklingImage:
        key = hashlib.sha256(processor.encode() + struct.pack("<I", max_pixels) + payload).digest()
        with self._lock:
            cached = self._entries.get(key)
        if cached is not None:
            return cached
        if processor == "qwen":
            prepared = prepare(payload, max_pixels)
        elif processor == "inkling2":
            prepared = prepare_inkling(payload, max_pixels)
        else:
            raise ImageError("unsupported image processor")
        with self._lock:
            self._entries.put(key, prepared, len(prepared.pixels))
        return prepared

    def stats(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._entries),
                "bytes": self._entries.bytes,
                "budget_bytes": self._entries.budget_bytes,
                "request_bytes": self._request_bytes,
                "request_budget_bytes": self.REQUEST_BUDGET_BYTES,
            }
