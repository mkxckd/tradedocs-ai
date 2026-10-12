import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    """No real LLM calls in tests, and telemetry goes to a temporary database."""
    for name in ("LLM_API_KEY", "LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TRADEDOCS_DB", str(tmp_path / "telemetry.db"))
