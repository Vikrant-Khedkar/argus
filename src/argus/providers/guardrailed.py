"""GuardrailedProvider — wraps any ChatProvider with pre/post-flight guards.

Used for "with-guardrails vs without-guardrails" audit comparison: run the
same probe set against the raw provider AND against `GuardrailedProvider(raw)`,
diff the AuditReports to quantify the guardrail lift per axis.
"""

from __future__ import annotations

from .base import ChatProvider
from ..guardrails.preflight import PreFlightPatternGuard
from ..guardrails.postflight import PostFlightRegenGuard, PostFlightHardRefuseGuard


class GuardrailedProvider(ChatProvider):
    """Wrap an inner ChatProvider with optional pre/post-flight guards.

    Args:
        inner: the underlying ChatProvider (Modal, OpenRouter, HTTP, HF…).
        pre_flight: list of guards run before the model call. The first to
            block short-circuits to its refusal text.
        post_flight: list of guards run after the model produces a response.
            Each can either pass through, regenerate, or replace with a
            refusal.
        observed_actions: populated after each `chat()` call with the list
            of guardrail actions that fired (for observability).
    """

    name = "guardrailed"

    def __init__(
        self,
        inner: ChatProvider,
        pre_flight: list[PreFlightPatternGuard] | None = None,
        post_flight: list | None = None,
    ):
        self.inner = inner
        self.pre_flight = pre_flight or []
        self.post_flight = post_flight or []
        self.last_actions: list[str] = []

    def model_under_test(self) -> str:
        """The wrapped model — the GuardrailedProvider isn't a model itself."""
        return self.inner.model_under_test()

    def chat(self, messages, max_tokens=512, temperature=0.7):
        self.last_actions = []
        prompt_text = self._extract_user_prompt(messages)

        # Pre-flight — short-circuit on block
        for guard in self.pre_flight:
            result = guard.check(prompt_text)
            if result.blocked:
                self.last_actions.append(f"pattern_block:{result.matched_pattern}")
                return result.refusal_text

        # Model call
        response = self.inner.chat(messages, max_tokens=max_tokens, temperature=temperature)

        # Post-flight — score + regen / hard-refuse as needed
        for guard in self.post_flight:
            if isinstance(guard, PostFlightRegenGuard):
                outcome = guard.check_and_regen(
                    prompt_text,
                    response,
                    regenerate_fn=lambda addendum: self._regenerate_with_addendum(
                        messages, addendum, max_tokens, temperature
                    ),
                )
                response = outcome.final_response
                if outcome.action != "none":
                    self.last_actions.append(outcome.action)
            elif isinstance(guard, PostFlightHardRefuseGuard):
                outcome = guard.check(prompt_text, response)
                response = outcome.final_response
                if outcome.action != "none":
                    self.last_actions.append(outcome.action)

        return response

    def _regenerate_with_addendum(self, messages, addendum, max_tokens, temperature):
        """Append `[Safety addendum: ...]` to the system prompt and regenerate."""
        new_messages = []
        injected = False
        for m in messages:
            if not injected and m.get("role") == "system":
                new_messages.append(
                    {**m, "content": f"{m['content']}\n\n[Safety addendum: {addendum}]"}
                )
                injected = True
            else:
                new_messages.append(m)
        if not injected:
            new_messages.insert(0, {"role": "system", "content": f"[Safety addendum: {addendum}]"})
        return self.inner.chat(new_messages, max_tokens=max_tokens, temperature=temperature)

    @staticmethod
    def _extract_user_prompt(messages: list[dict]) -> str:
        """Pull out the most recent user-role message content."""
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content")
                if isinstance(content, str):
                    return content
        return ""


__all__ = ["GuardrailedProvider"]
