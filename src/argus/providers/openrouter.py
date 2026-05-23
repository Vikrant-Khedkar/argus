"""OpenRouterProvider — proxies to anything on openrouter.ai/models.

Every call acquires the global OPENROUTER_SEMAPHORE before firing, so all
concurrent OpenRouter traffic (model-under-test inference + scorer judges)
shares one bounded concurrency budget. AIMD backoff handles 429s
automatically — capacity halves on rate-limit, additively grows on
success — and a small exponential retry layer absorbs transient
errors.
"""

from __future__ import annotations

import os
import random
import time

import httpx

from .base import ChatProvider
from ..concurrency import OPENROUTER_SEMAPHORE

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

_MAX_RETRIES = 5
_BASE_BACKOFF_S = 1.0


def _post_with_backoff(headers: dict, body: dict, timeout: float = 180.0) -> dict:
    """POST to OpenRouter under the global semaphore, retrying on 429 / 5xx."""
    last_err: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        with OPENROUTER_SEMAPHORE.acquire():
            try:
                r = httpx.post(OPENROUTER_URL, headers=headers, json=body, timeout=timeout)
            except httpx.TimeoutException as e:
                last_err = e
                # Network timeout — treat as transient
                sleep_s = _BASE_BACKOFF_S * (2 ** attempt) + random.random() * 0.5
                time.sleep(sleep_s)
                continue

            if r.status_code == 429:
                OPENROUTER_SEMAPHORE.signal_rate_limit()
                # Honour Retry-After when provided, else exponential
                retry_after = r.headers.get("retry-after")
                sleep_s = (
                    float(retry_after) if retry_after and retry_after.isdigit()
                    else _BASE_BACKOFF_S * (2 ** attempt) + random.random() * 0.5
                )
                time.sleep(sleep_s)
                last_err = httpx.HTTPStatusError(
                    f"429 rate-limited (attempt {attempt + 1})", request=r.request, response=r,
                )
                continue

            if 500 <= r.status_code < 600:
                last_err = httpx.HTTPStatusError(
                    f"{r.status_code} from OpenRouter", request=r.request, response=r,
                )
                time.sleep(_BASE_BACKOFF_S * (2 ** attempt) + random.random() * 0.5)
                continue

            if 400 <= r.status_code < 500:
                # 4xx (other than 429) is permanent — surface the body so
                # the caller can see *what* OpenRouter rejected. Don't retry.
                body_excerpt = (r.text or "")[:500]
                raise httpx.HTTPStatusError(
                    f"{r.status_code} from OpenRouter "
                    f"(model={body.get('model')}): {body_excerpt}",
                    request=r.request,
                    response=r,
                )

            r.raise_for_status()
            OPENROUTER_SEMAPHORE.signal_success()
            return r.json()

    if last_err is not None:
        raise last_err
    raise RuntimeError(f"OpenRouter call failed after {_MAX_RETRIES} retries with no error captured")


class OpenRouterProvider(ChatProvider):
    name = "openrouter"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat(self, messages, max_tokens=512, temperature=0.7, tools=None):
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            body["tools"] = tools
        data = _post_with_backoff(self._headers(), body)
        return data["choices"][0]["message"]["content"]

    def chat_with_tools(self, messages, tools, max_tokens=512, temperature=0.7) -> dict:
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "tools": tools,
        }
        data = _post_with_backoff(self._headers(), body)
        return data["choices"][0]["message"]


__all__ = ["OpenRouterProvider", "OPENROUTER_URL"]
