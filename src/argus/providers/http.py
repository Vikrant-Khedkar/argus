"""HTTPProvider — generic OpenAI-compatible chat-completion endpoint.

For wrapping a vendor's hosted model that follows the OpenAI Chat Completions
schema. Lets an AI vendor's CTO audit their own deployment by pointing Argus
at their endpoint:

    provider = HTTPProvider(
        url="https://api.acme-ai.com/v1/chat/completions",
        model="acme/their-model-v1",
        auth_header={"Authorization": "Bearer ..."},
    )
"""

from __future__ import annotations

import httpx

from .base import ChatProvider


class HTTPProvider(ChatProvider):
    """Generic OpenAI-compatible HTTPS chat-completion provider.

    Args:
        url: full chat-completions endpoint URL.
        model: model identifier the endpoint expects.
        auth_header: dict of headers to send (usually Authorization).
        extra_headers: any additional headers (User-Agent, etc.).
        timeout_s: request timeout.
        response_path: dotted path to the assistant content within the JSON
            response. Default ``"choices.0.message.content"`` matches OpenAI.
    """

    name = "http"

    def __init__(
        self,
        url: str,
        model: str,
        auth_header: dict | None = None,
        extra_headers: dict | None = None,
        timeout_s: float = 180.0,
        response_path: str = "choices.0.message.content",
    ):
        self.url = url
        self.model = model
        self.auth_header = auth_header or {}
        self.extra_headers = extra_headers or {}
        self.timeout_s = timeout_s
        self.response_path = response_path

    def chat(self, messages, max_tokens=512, temperature=0.7):
        headers = {"Content-Type": "application/json", **self.auth_header, **self.extra_headers}
        r = httpx.post(
            self.url,
            headers=headers,
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        return self._extract(r.json())

    def _extract(self, data: dict) -> str:
        node = data
        for part in self.response_path.split("."):
            if part.isdigit():
                node = node[int(part)]
            else:
                node = node[part]
        return str(node)


__all__ = ["HTTPProvider"]
