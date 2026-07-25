"""テストは決定論的フォールバック経路で実行する（実LLMを呼ばない）。"""

import pytest


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    for env in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "MI_HUB_LLM_BASE_URL",
        "MI_HUB_LLM_PROVIDER",
        "MI_HUB_LLM_MODEL",
    ):
        monkeypatch.delenv(env, raising=False)
