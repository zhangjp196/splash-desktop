"""Each API's own module: how it reads its requests and writes its answers.

A dialect's functions take the HTTP handler (server.FrontendHandler) and use
its public methods: they answer a request or return the generation it prepared
and the function that writes the answer. The HTTP itself stays in the router.

The shared Chat message normalization every API reaches is server.chat_messages;
no dialect imports another, and the modules they all reach import none.
"""

from . import anthropic, openai, responses, scoring

__all__ = ("anthropic", "openai", "responses", "scoring")
