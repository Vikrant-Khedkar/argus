import os
from abc import ABC, abstractmethod
from typing import Literal

import httpx
from dotenv import load_dotenv

load_dotenv()

MODAL_URL_DEFAULT = "https://vikrant-61236--qwen-assistant-assistant-generate.modal.run"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class ChatProvider(ABC):
    name: str

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str: ...


class ModalProvider(ChatProvider):
    name = "modal"

    def __init__(self, url: str | None = None, api_key: str | None = None):
        self.url = url or os.getenv("MODAL_URL", MODAL_URL_DEFAULT)
        self.api_key = api_key or os.getenv("MODAL_API_KEY")

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


class OpenRouterProvider(ChatProvider):
    name = "openrouter"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")

    def chat(self, messages, max_tokens=512, temperature=0.7, tools=None):
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            body["tools"] = tools
        r = httpx.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=180,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def chat_with_tools(self, messages, tools, max_tokens=512, temperature=0.7) -> dict:
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "tools": tools,
        }
        r = httpx.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=180,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]


ProviderName = Literal["modal", "openrouter"]


def get_provider(name: ProviderName | None = None) -> ChatProvider:
    name = (name or os.getenv("PROVIDER", "modal")).lower()
    if name == "modal":
        return ModalProvider()
    if name == "openrouter":
        return OpenRouterProvider()
    raise ValueError(f"unknown provider: {name!r} (use 'modal' or 'openrouter')")


if __name__ == "__main__":
    import sys

    which = sys.argv[1] if len(sys.argv) > 1 else "modal"
    provider = get_provider(which)
    messages = [
        {"role": "system", "content": "You are a helpful personal assistant."},
        {"role": "user", "content": "Name three Python packages for testing."},
    ]
    print(f"[{provider.name}] {provider.chat(messages)}")
