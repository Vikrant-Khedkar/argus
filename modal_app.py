import os

import modal
from fastapi import Header, HTTPException

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
CACHE_DIR = "/cache"

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "fastapi",
        "pydantic",
    )
    .env({"HF_HOME": CACHE_DIR})
)

app = modal.App("qwen-assistant")

model_volume = modal.Volume.from_name("qwen-model-cache", create_if_missing=True)
secrets = [modal.Secret.from_name("modal-auth")]


@app.cls(
    image=image,
    gpu="T4",
    timeout=600,
    scaledown_window=300,
    volumes={CACHE_DIR: model_volume},
    secrets=secrets,
)
@modal.concurrent(max_inputs=4)
class Assistant:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=CACHE_DIR,
        )
        self.model.eval()

    def _generate(self, messages, max_new_tokens, temperature, tools=None):
        import torch

        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if tools:
            template_kwargs["tools"] = tools

        text = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )

    def _parse_tool_calls(self, text: str):
        """Parse Qwen-format <tool_call>{...}</tool_call> blocks. Returns (content, tool_calls)."""
        import json
        import re

        calls = []
        for i, m in enumerate(re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)):
            try:
                payload = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            name = payload.get("name")
            args = payload.get("arguments", {})
            if not name:
                continue
            calls.append({
                "id": f"qwen_tc_{i}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
                },
            })
        clean_text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).strip()
        return clean_text, calls

    @modal.method()
    def chat(self, messages: list[dict], max_new_tokens: int = 300, temperature: float = 0.7) -> str:
        return self._generate(messages, max_new_tokens, temperature)

    @modal.fastapi_endpoint(method="POST", docs=True)
    def generate(self, request: dict, x_api_key: str | None = Header(default=None)):
        expected = os.environ.get("MODAL_AUTH_TOKEN")
        if expected and x_api_key != expected:
            raise HTTPException(status_code=401, detail="unauthorized")

        messages = request.get("messages")
        if not messages:
            prompt = request.get("prompt", "")
            messages = [
                {"role": "system", "content": "You are a helpful personal assistant."},
                {"role": "user", "content": prompt},
            ]
        tools = request.get("tools")
        raw = self._generate(
            messages,
            int(request.get("max_new_tokens", 300)),
            float(request.get("temperature", 0.7)),
            tools=tools,
        )
        if tools:
            content, tool_calls = self._parse_tool_calls(raw)
            return {"response": content, "tool_calls": tool_calls}
        return {"response": raw}


@app.local_entrypoint()
def main(prompt: str = "Hello, who are you?"):
    messages = [
        {"role": "system", "content": "You are a helpful personal assistant."},
        {"role": "user", "content": prompt},
    ]
    print(Assistant().chat.remote(messages))
