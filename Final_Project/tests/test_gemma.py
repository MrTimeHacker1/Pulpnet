"""Tests for the Gemma wrapper using MockLLM (no model download / GPU needed).

The real GemmaLLM is exercised separately by tests/check_gemma_live.py, which is
NOT part of the default suite because it requires the multi-GB model + a working
accelerator.
"""

from __future__ import annotations

from llm.gemma import MockLLM, get_llm, set_llm


def test_mock_llm_interface():
    llm = MockLLM()
    out = llm.generate("You are helpful.", "What is CE212?\nContext: ...")
    assert isinstance(out, str) and out
    assert "What is CE212?" in out  # echoes first user line for traceability


def test_canned_mock():
    llm = MockLLM(canned="hello")
    assert llm.generate("s", "u") == "hello"


def test_set_and_get_llm_singleton():
    sentinel = MockLLM(canned="sentinel")
    set_llm(sentinel)
    assert get_llm() is sentinel
    assert get_llm().generate("s", "u") == "sentinel"
