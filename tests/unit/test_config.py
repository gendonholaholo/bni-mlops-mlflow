"""Config is loaded once from environment, frozen, and validates required vars."""

from __future__ import annotations

import pytest

from llmops._config import get_config, reset_config_cache
from llmops.exceptions import LLMOpsConfigError


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    reset_config_cache()


def test_config_reads_required_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    monkeypatch.delenv("LLMOPS_EXPERIMENT_NAME", raising=False)
    monkeypatch.delenv("LLMOPS_DISABLE_TRACING", raising=False)

    cfg = get_config()

    assert cfg.tracking_uri == "http://localhost:5001"
    assert cfg.experiment_name == "bni-agentic-prd"
    assert cfg.disable_tracing is False


def test_config_optional_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://x:5001")
    monkeypatch.setenv("LLMOPS_EXPERIMENT_NAME", "custom-exp")
    monkeypatch.setenv("LLMOPS_DISABLE_TRACING", "true")

    cfg = get_config()

    assert cfg.experiment_name == "custom-exp"
    assert cfg.disable_tracing is True


def test_config_missing_required_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    with pytest.raises(LLMOpsConfigError, match="MLFLOW_TRACKING_URI"):
        get_config()


def test_config_is_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://x:5001")
    cfg = get_config()
    with pytest.raises(Exception):  # noqa: B017  # FrozenInstanceError or AttributeError
        cfg.tracking_uri = "http://other"  # type: ignore[misc]


def test_config_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://x:5001")
    cfg1 = get_config()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://other:5001")
    cfg2 = get_config()
    assert cfg1 is cfg2  # same instance, env reread NOT triggered

    reset_config_cache()
    cfg3 = get_config()
    assert cfg3.tracking_uri == "http://other:5001"
