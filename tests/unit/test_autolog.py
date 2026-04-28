"""autolog passthrough — verify provider validation, adapter init, and
delegation to mlflow.<provider>.autolog()."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fake_mlflow_for_autolog(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    """Inject a mock mlflow with provider submodules that expose autolog()."""
    mod = types.ModuleType("mlflow")
    mod.set_tracking_uri = MagicMock()  # type: ignore[attr-defined]
    mod.set_experiment = MagicMock()  # type: ignore[attr-defined]
    mod.set_tag = MagicMock()  # type: ignore[attr-defined]
    mod.set_prompt_alias = MagicMock()  # type: ignore[attr-defined]
    mod.MlflowClient = MagicMock()  # type: ignore[attr-defined]
    genai = types.ModuleType("mlflow.genai")
    genai.register_prompt = MagicMock()  # type: ignore[attr-defined]
    genai.load_prompt = MagicMock()  # type: ignore[attr-defined]
    mod.genai = genai  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "mlflow", mod)
    monkeypatch.setitem(sys.modules, "mlflow.genai", genai)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://x:5001")

    provider_mocks: dict[str, MagicMock] = {}
    for provider in ("openai", "anthropic", "langchain", "llama_index", "ag2"):
        sub = types.ModuleType(f"mlflow.{provider}")
        autolog_mock = MagicMock()
        sub.autolog = autolog_mock  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, f"mlflow.{provider}", sub)
        provider_mocks[provider] = autolog_mock

    for name in ("llmops._config", "llmops._mlflow_adapter", "llmops.tracing", "llmops._autolog"):
        sys.modules.pop(name, None)

    return {"mlflow": mod, "providers": provider_mocks}


def test_autolog_calls_provider_autolog(
    fake_mlflow_for_autolog: dict[str, MagicMock],
) -> None:
    from llmops._autolog import autolog

    autolog("openai")

    fake_mlflow_for_autolog["providers"]["openai"].assert_called_once_with()


def test_autolog_forwards_kwargs(
    fake_mlflow_for_autolog: dict[str, MagicMock],
) -> None:
    from llmops._autolog import autolog

    autolog("langchain", log_inputs_outputs=False, silent=True)

    fake_mlflow_for_autolog["providers"]["langchain"].assert_called_once_with(
        log_inputs_outputs=False, silent=True
    )


def test_autolog_initialises_adapter_first(
    fake_mlflow_for_autolog: dict[str, MagicMock],
) -> None:
    """Issue #1 corollary: autolog'd traces must also land in the configured
    experiment, so set_tracking_uri + set_experiment must run before the
    provider's autolog hook fires."""
    from llmops._autolog import autolog

    autolog("anthropic")

    mlflow_mod = fake_mlflow_for_autolog["mlflow"]
    mlflow_mod.set_tracking_uri.assert_called_once_with("http://x:5001")
    mlflow_mod.set_experiment.assert_called_once_with("bni-agentic-prd")


def test_autolog_unsupported_provider_raises(
    fake_mlflow_for_autolog: dict[str, MagicMock],
) -> None:
    from llmops._autolog import autolog

    with pytest.raises(ValueError, match="unsupported autolog provider"):
        autolog("nonexistent_framework")


def test_autolog_disabled_is_noop(
    fake_mlflow_for_autolog: dict[str, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLMOPS_DISABLE_TRACING", "true")
    sys.modules.pop("llmops._config", None)
    sys.modules.pop("llmops._autolog", None)

    from llmops._autolog import autolog

    autolog("openai")  # no-op
    fake_mlflow_for_autolog["providers"]["openai"].assert_not_called()


def test_supported_providers_covers_known_integrations() -> None:
    """Pin the supported-provider list so a regression catches accidental
    deletions. MLflow 3.11.1 ships these 17 integration submodules."""
    from llmops._autolog import SUPPORTED_PROVIDERS

    expected = {
        "ag2",
        "anthropic",
        "autogen",
        "bedrock",
        "crewai",
        "dspy",
        "gemini",
        "groq",
        "haystack",
        "langchain",
        "litellm",
        "llama_index",
        "mistral",
        "openai",
        "pydantic_ai",
        "semantic_kernel",
        "smolagents",
    }
    assert frozenset(expected) == SUPPORTED_PROVIDERS
