"""The chat page and its assets, read from the package once at startup."""

from pathlib import Path

CHAT_HTML = Path(__file__).with_name("chat.html").read_bytes()
CHAT_ASSETS = {
    f"/chat-assets/{name}": (Path(__file__).parent / "chat-assets" / name).read_bytes()
    for name in (
        "markdown-it-15.0.2.min.js",
        "highlight-11.12.0.min.js",
        "github-11.12.0.min.css",
        "github-dark-11.12.0.min.css",
    )
}
# The chat page's brand mark, in its text colors, which the page shows too.
# Browsers, and other clients, ask for a site's icon at /favicon.ico.
FAVICON_SVG = Path(__file__).with_name("favicon.svg").read_bytes()
