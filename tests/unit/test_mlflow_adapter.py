"""Adapter is the SOLE module that imports mlflow.
Tests use a fake mlflow injected via sys.modules to verify adapter calls correct APIs."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from llmops._config import Config


@pytest.fixture
def fake_mlflow(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Inject a mock `mlflow` module so the adapter can be exercised in isolation."""
    mod = types.ModuleType("mlflow")
    mod.set_tracking_uri = MagicMock()  # type: ignore[attr-defined]
    mod.set_experiment = MagicMock()  # type: ignore[attr-defined]
    mod.set_tag = MagicMock()  # type: ignore[attr-defined]
    mod.set_prompt_alias = MagicMock()  # type: ignore[attr-defined]

    genai = types.ModuleType("mlflow.genai")
    genai.register_prompt = MagicMock(  # type: ignore[attr-defined]
        return_value=types.SimpleNamespace(name="x", version=1)
    )
    genai.load_prompt = MagicMock(  # type: ignore[attr-defined]
        return_value=types.SimpleNamespace(name="x", version=1, template="t")
    )
    mod.genai = genai  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "mlflow", mod)
    monkeypatch.setitem(sys.modules, "mlflow.genai", genai)
    # Force reimport of adapter
    sys.modules.pop("llmops._mlflow_adapter", None)
    return mod


def test_adapter_initialise_sets_uri_and_experiment(fake_mlflow: MagicMock) -> None:
    from llmops._mlflow_adapter import MLflowAdapter

    cfg = Config(tracking_uri="http://x:5001", experiment_name="exp", disable_tracing=False)
    adapter = MLflowAdapter(cfg)
    adapter.initialise()

    fake_mlflow.set_tracking_uri.assert_called_once_with("http://x:5001")
    fake_mlflow.set_experiment.assert_called_once_with("exp")


def test_adapter_register_prompt_calls_genai(fake_mlflow: MagicMock) -> None:
    from llmops._mlflow_adapter import MLflowAdapter

    cfg = Config(tracking_uri="http://x", experiment_name="e", disable_tracing=False)
    a = MLflowAdapter(cfg)
    a.register_prompt(name="p", template="t", commit_message="c", tags={"k": "v"})

    fake_mlflow.genai.register_prompt.assert_called_once_with(
        name="p", template="t", commit_message="c", tags={"k": "v"}, model_config=None
    )


def test_adapter_load_prompt_uri_form(fake_mlflow: MagicMock) -> None:
    from llmops._mlflow_adapter import MLflowAdapter

    cfg = Config(tracking_uri="http://x", experiment_name="e", disable_tracing=False)
    a = MLflowAdapter(cfg)
    a.load_prompt("agent_tujuan", alias="production")

    fake_mlflow.genai.load_prompt.assert_called_once_with(
        name_or_uri="prompts:/agent_tujuan@production"
    )


def test_adapter_register_prompt_with_model_config(fake_mlflow: MagicMock) -> None:
    """Issue #2: adapter forwards model_config kwarg to mlflow.genai.register_prompt
    so generation hyperparameters are versioned with the prompt template."""
    from llmops._mlflow_adapter import MLflowAdapter

    cfg = Config(tracking_uri="http://x", experiment_name="e", disable_tracing=False)
    a = MLflowAdapter(cfg)
    a.register_prompt(
        name="p",
        template="t",
        commit_message="c",
        tags={"k": "v"},
        model_config={"temperature": 0.7, "top_k": 40, "num_ctx": 4096},
    )

    fake_mlflow.genai.register_prompt.assert_called_once_with(
        name="p",
        template="t",
        commit_message="c",
        tags={"k": "v"},
        model_config={"temperature": 0.7, "top_k": 40, "num_ctx": 4096},
    )


def test_adapter_set_alias_calls_root_namespace(fake_mlflow: MagicMock) -> None:
    """set_prompt_alias is on mlflow.* not mlflow.genai.* — adapter must use the right one."""
    from llmops._mlflow_adapter import MLflowAdapter

    cfg = Config(tracking_uri="http://x", experiment_name="e", disable_tracing=False)
    a = MLflowAdapter(cfg)
    a.set_alias("agent_tujuan", alias="staging", version=3)

    fake_mlflow.set_prompt_alias.assert_called_once_with("agent_tujuan", "staging", 3)
