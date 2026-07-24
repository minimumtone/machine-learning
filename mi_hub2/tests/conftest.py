"""テストは決定論的フォールバック経路で実行する（実LLMを呼ばない）。"""

import pytest


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
