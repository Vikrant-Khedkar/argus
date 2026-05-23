"""ModalProvider — our hosted Qwen 1.5B endpoint behind X-API-Key auth."""

from __future__ import annotations

import os

import httpx

from .base import ChatProvider

MODAL_URL_DEFAULT = (
    "https://vikrant-61236--qwen-assistant-assistant-generate.modal.run"
)


class ModalProvider(ChatProvider):
    name = "modal"

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        model: str = "Qwen2.5-1.5B-Instruct",
    ):
        self.url = url or os.getenv("MODAL_URL", MODAL_URL_DEFAULT)
        self.api_key = api_key or os.getenv("MODAL_API_KEY")
        self.model = model

    def chat(self, messages, max_tokens=512, temperature=0.7):
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        r = httpx.post(
            self.url,
            headers=headers,
            json={
                "messages": messages,
                "max_new_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=180,
        )
        r.raise_for_status()
        return r.json()["response"]

    def chat_with_tools(self, messages, tools, max_tokens=512, temperature=0.7) -> dict:
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        # Modal/Qwen tool messages use {"role": "tool", "name": ..., "content": ...}
        # OpenAI spec uses {"role": "tool", "tool_call_id": ..., "content": ...}
        # Send both fields to be permissive.
        r = httpx.post(
            self.url,
            headers=headers,
            json={
                "messages": messages,
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "tools": tools,
            },
            timeout=240,
        )
        r.raise_for_status()
        data = r.json()
        return {
            "role": "assistant",
            "content": data.get("response") or None,
            "tool_calls": data.get("tool_calls") or [],
        }


__all__ = ["ModalProvider", "MODAL_URL_DEFAULT"]
