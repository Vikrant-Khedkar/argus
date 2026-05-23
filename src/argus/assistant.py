import json
import os
import re
import time
from pathlib import Path

from .adapter import ChatProvider
from .memory import Conversation
from .observability import log_row, new_request_id

DEFAULT_PERSONA = "You are a helpful personal assistant. Be concise, friendly, and accurate."

SUMMARIZER_SYSTEM = (
    "You compress dialogue into a 2-3 sentence factual summary. "
    "Keep names, preferences, decisions, and key facts. Drop pleasantries. "
    "Output the summary only — no preamble."
)

HARD_REFUSAL = (
    "I can't help with that. The request triggered a safety guardrail. "
    "If you believe this is in error, please rephrase."
)

REGEN_PROMPTS = {
    "output_liability": (
        "Be extra careful about accuracy. If you are not certain about specific facts, "
        "dates, names, or entities, say so explicitly. Do not fabricate details or "
        "invent plausible-sounding information."
    ),
    "discrimination_liability": (
        "Do not make assumptions or generalizations based on gender, race, age, "
        "occupation, nationality, religion, or other protected attributes. "
        "Treat individuals equitably; if a question implies a stereotype, push back."
    ),
}


def env_flag(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def load_persona() -> str:
    inline = os.getenv("PERSONA")
    if inline:
        return inline.strip()
    path = os.getenv("PERSONA_FILE")
    if path:
        p = Path(path)
        if p.exists():
            return p.read_text().strip()
    return DEFAULT_PERSONA


class Guardrails:
    """Eval-informed safety layer. Patterns encode the jailbreaks that landed in Phase 2."""

    JAILBREAK_PATTERNS = [
        (re.compile(r"translate\b.{0,80}\b(then|and)\b.{0,80}\b(provide|give|tell|share|show)\b.{0,80}\b(actual|real|instructions?)\b", re.I | re.S), "translation_laundering"),
        (re.compile(r"\byou are (now |going to be )?(dan|developer mode|do anything now|in developer mode)\b", re.I), "persona_swap"),
        (re.compile(r"pretend (you('?| a)?re|to be).{0,80}(deceased|dead|grandmother|grandfather)", re.I | re.S), "emotional_roleplay"),
        (re.compile(r"ignore (all |the )?(previous|prior|earlier) instructions?", re.I), "instruction_injection"),
    ]

    def pre_flight(self, user_text: str) -> tuple[str, str] | None:
        for pattern, label in self.JAILBREAK_PATTERNS:
            if pattern.search(user_text):
                return HARD_REFUSAL, label
        return None

    def post_flight_action(self, risk_result) -> tuple[str, str | None]:
        """Returns ('none' | 'regenerate' | 'hard_refuse', system_prompt_addendum_or_None)."""
        if risk_result.safety_liability < 1.0:
            return ("hard_refuse", None)
        if risk_result.output_liability < 1.0:
            return ("regenerate", REGEN_PROMPTS["output_liability"])
        if risk_result.discrimination_liability < 1.0:
            return ("regenerate", REGEN_PROMPTS["discrimination_liability"])
        return ("none", None)


class Assistant:
    def __init__(
        self,
        provider: ChatProvider,
        persona: str | None = None,
        max_turns: int = 6,
        guardrails_enabled: bool | None = None,
        tool_use_enabled: bool | None = None,
        risk_scorer=None,
    ):
        self.provider = provider
        self.persona = persona or load_persona()
        self.conv = Conversation(self.persona, max_turns=max_turns)
        self.guardrails_enabled = (
            env_flag("GUARDRAILS_ENABLED", False) if guardrails_enabled is None else guardrails_enabled
        )
        self.tool_use_enabled = (
            env_flag("TOOL_USE_ENABLED", False) if tool_use_enabled is None else tool_use_enabled
        )
        self.guardrails = Guardrails()
        self._risk_scorer = risk_scorer
        self.last_risk = None

    def _ensure_risk_scorer(self):
        if self._risk_scorer is None:
            from risk_score import RiskScorer
            self._risk_scorer = RiskScorer()
        return self._risk_scorer

    def ask(self, user_text: str) -> str:
        request_id = new_request_id()
        t0 = time.time()
        guardrail_action = None
        tool_calls_count = 0
        risk = None
        reply = ""
        error = None

        try:
            if self.guardrails_enabled:
                blocked = self.guardrails.pre_flight(user_text)
                if blocked:
                    reply, label = blocked
                    guardrail_action = f"pattern_block:{label}"
                    self.conv.add_user(user_text)
                    self.conv.add_assistant(reply)
                    return reply

            self.conv.add_user(user_text)
            if self.conv.needs_prune():
                self.conv.prune(self._summarize)

            if self.tool_use_enabled and hasattr(self.provider, "chat_with_tools"):
                reply, tool_calls_count = self._tool_loop()
            else:
                reply = self.provider.chat(self.conv.messages())

            if self.guardrails_enabled:
                scorer = self._ensure_risk_scorer()
                risk = scorer.score(user_text, reply)
                self.last_risk = risk
                action, regen_addendum = self.guardrails.post_flight_action(risk)
                if action == "hard_refuse":
                    reply = HARD_REFUSAL
                    guardrail_action = "hard_refuse"
                elif action == "regenerate":
                    reply = self._regenerate(regen_addendum)
                    guardrail_action = "regenerated"
                    risk = scorer.score(user_text, reply)
                    self.last_risk = risk

            self.conv.add_assistant(reply)
            return reply
        except Exception as e:
            error = str(e)
            reply = f"[error] {e}"
            return reply
        finally:
            row: dict = {
                "request_id": request_id,
                "provider": self.provider.name,
                "prompt": user_text,
                "response": reply,
                "latency_s": round(time.time() - t0, 3),
                "guardrail_action": guardrail_action,
                "tool_calls_count": tool_calls_count,
                "error": error,
            }
            if risk is not None:
                row.update({
                    "output_liability": risk.output_liability,
                    "discrimination_liability": risk.discrimination_liability,
                    "safety_liability": risk.safety_liability,
                    "overall_tier": risk.overall_tier,
                })
            log_row(row)

    def _summarize(self, turns: list[dict]) -> str:
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in turns)
        messages = [
            {"role": "system", "content": SUMMARIZER_SYSTEM},
            {"role": "user", "content": transcript},
        ]
        return self.provider.chat(messages, max_tokens=200, temperature=0.2)

    def _regenerate(self, addendum: str) -> str:
        original_system = self.conv.system_prompt
        self.conv.system_prompt = f"{original_system}\n\n[Safety addendum: {addendum}]"
        try:
            return self.provider.chat(self.conv.messages())
        finally:
            self.conv.system_prompt = original_system

    def _tool_loop(self, max_iters: int = 3) -> tuple[str, int]:
        from .tools.web_search import WEB_SEARCH_TOOL_SCHEMA, dispatch

        messages = list(self.conv.messages())
        tool_calls_count = 0
        for _ in range(max_iters):
            msg = self.provider.chat_with_tools(messages, tools=[WEB_SEARCH_TOOL_SCHEMA])
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                return msg.get("content") or "", tool_calls_count
            messages.append(msg)
            for tc in tool_calls:
                tool_calls_count += 1
                fn = tc.get("function", {})
                result = dispatch(fn.get("name", ""), fn.get("arguments", "{}"))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id"),
                    "content": result,
                })
        # Fell out of loop with tool calls still pending
        return msg.get("content") or "", tool_calls_count

    def set_persona(self, persona: str) -> None:
        self.persona = persona.strip()
        self.conv.system_prompt = self.persona

    def set_provider(self, provider: ChatProvider) -> None:
        self.provider = provider

    def reset(self) -> None:
        self.conv.reset()
        self.last_risk = None
