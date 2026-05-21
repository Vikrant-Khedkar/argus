from typing import Callable


class Conversation:
    def __init__(self, system_prompt: str, max_turns: int = 6):
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.history: list[dict] = []
        self.summary: str | None = None

    def add_user(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})

    def messages(self) -> list[dict]:
        system = self.system_prompt
        if self.summary:
            system = f"{system}\n\n[Earlier context summary: {self.summary}]"
        return [{"role": "system", "content": system}] + self.history

    def needs_prune(self) -> bool:
        return len(self.history) > self.max_turns * 2

    def prune(self, summarizer: Callable[[list[dict]], str]) -> None:
        if len(self.history) < 4:
            return
        cut = len(self.history) // 2
        cut -= cut % 2
        old, kept = self.history[:cut], self.history[cut:]
        try:
            new_summary = summarizer(old)
        except Exception:
            new_summary = ""
        if new_summary:
            self.summary = (
                f"{self.summary} {new_summary}".strip() if self.summary else new_summary
            )
        self.history = kept

    def reset(self) -> None:
        self.history = []
        self.summary = None
