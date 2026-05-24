"""HuggingFaceProvider — locally hosted HF Transformers model.

For self-hosted / on-prem audits where the model isn't behind an HTTPS
endpoint. Uses the same `transformers` pipeline our Modal deployment uses
internally, just running in-process.
"""

from __future__ import annotations

from .base import ChatProvider


class HuggingFaceProvider(ChatProvider):
    """Wrap a HuggingFace causal LM in the ChatProvider interface.

    Args:
        model_id: HF model id (e.g. ``Qwen/Qwen2.5-1.5B-Instruct``).
        device: ``"cuda"``, ``"cpu"``, or ``"auto"``. Default ``"auto"``.
        torch_dtype: precision — ``"float16"``, ``"bfloat16"``, or ``"auto"``.
            Defaults to ``"float16"``.
        cache_dir: optional HF cache override.
    """

    name = "huggingface"

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        torch_dtype: str = "float16",
        cache_dir: str | None = None,
    ):
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.cache_dir = cache_dir
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": "auto",
        }
        dtype = dtype_map.get(self.torch_dtype, torch.float16)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map=self.device,
            cache_dir=self.cache_dir,
        )
        self._model.eval()

    def chat(self, messages, max_tokens=512, temperature=0.7):
        self._ensure_loaded()
        import torch

        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=0.9,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        return self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )


__all__ = ["HuggingFaceProvider"]
