"""Gemma 4 E4B-it generator wrapper.

Loads `google/gemma-4-E4B-it` once (text-only path) via AutoProcessor +
AutoModelForCausalLM. Chat messages are built with system/user roles and rendered
with `apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)`
— thinking OFF for grounded, low-latency answers. Generation uses temperature 0.3
(below the model-card default of 1.0) for faithfulness.

The model is loaded lazily and cached as a process singleton. A `MockLLM` with the
same interface is provided so the agent/guardrail/API tests never depend on the
multi-GB download or a working GPU.
"""

from __future__ import annotations

import logging
from typing import Protocol

from config import get_settings

logger = logging.getLogger(__name__)


class LLM(Protocol):
    def generate(self, system: str, user: str, max_new_tokens: int | None = None) -> str: ...


class GemmaLLM:
    """Real Gemma generator. Heavy — construct once, reuse."""

    def __init__(self, model_id: str | None = None, device: str | None = None):
        s = get_settings()
        self.model_id = model_id or s.generator_model
        self.device = device or s.resolve_device()
        self.temperature = s.gen_temperature
        self.max_new_tokens = s.gen_max_new_tokens
        self._model = None
        self._processor = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading generator %s (device=%s)", self.model_id, self.device)
        # google/gemma-4-E4B-it is multimodal; its AutoProcessor pulls an image
        # processor that requires torchvision. We only use the TEXT path, so we
        # load AutoTokenizer instead — no vision dependency needed.
        self._processor = AutoTokenizer.from_pretrained(self.model_id)
        device_map = "auto" if self.device.startswith("cuda") else None
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype="auto", device_map=device_map
        )
        if device_map is None:
            self._model = self._model.to(self.device)
        self._model.eval()

    def generate(self, system: str, user: str, max_new_tokens: int | None = None) -> str:
        import torch

        self._ensure()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        tmpl_kwargs = dict(
            add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        try:  # thinking OFF for grounded, low-latency answers (if the template supports it)
            inputs = self._processor.apply_chat_template(
                messages, enable_thinking=False, **tmpl_kwargs
            )
        except (TypeError, ValueError):
            inputs = self._processor.apply_chat_template(messages, **tmpl_kwargs)
        inputs = inputs.to(self._model.device)

        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
            )
        # Strip the prompt tokens, keep only the generated continuation.
        gen = out[0][inputs["input_ids"].shape[1] :]
        return self._processor.decode(gen, skip_special_tokens=True).strip()


class MockLLM:
    """Deterministic stand-in for tests / when the real model can't load.

    Returns a short grounded-looking answer that cites the first context line, so
    pipeline wiring (citations, guardrails, logging) can be exercised offline.
    """

    def __init__(self, canned: str | None = None):
        self.canned = canned

    def generate(self, system: str, user: str, max_new_tokens: int | None = None) -> str:
        if self.canned is not None:
            return self.canned
        # Faithfulness-judge prompt → return a GROUNDED verdict so the mock
        # pipeline produces cited answers instead of always refusing.
        if "judge" in system.lower():
            return "VERDICT: GROUNDED\nSCORE: 0.85"
        # Otherwise echo a deterministic answer referencing the prompt (with a [1]
        # citation) for traceability.
        first_line = next((ln for ln in user.splitlines() if ln.strip()), "")
        return f"[mock-answer] Based on the provided context [1]: {first_line[:160]}"


_llm: LLM | None = None


def get_llm() -> LLM:
    """Process-singleton accessor. Honors GENERATOR_MODEL env var.

    Values: "mock" → MockLLM (offline/tests), "hf" → HFRouterLLM (hosted),
    anything else → GemmaLLM (local checkpoint).
    """
    global _llm
    if _llm is None:
        model = get_settings().generator_model.lower()
        if model == "mock":
            logger.warning("Using MockLLM (GENERATOR_MODEL=mock)")
            _llm = MockLLM()
        elif model == "hf":
            logger.info("Using HFRouterLLM (GENERATOR_MODEL=hf)")
            from llm.hf_router import HFRouterLLM
            _llm = HFRouterLLM()
        else:
            _llm = GemmaLLM()
    return _llm


def set_llm(llm: LLM) -> None:
    """Inject an LLM (used by the API lifespan and by tests)."""
    global _llm
    _llm = llm
