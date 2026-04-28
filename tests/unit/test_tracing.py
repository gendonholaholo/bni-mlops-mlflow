"""trace_agent: context manager + decorator (same callable)."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fake_mlflow_for_tracing(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    """Inject a mock mlflow with span lifecycle helpers."""
    span_mock = MagicMock()
    span_mock.span_id = "s1"
    span_mock.trace_id = "r1"
    client_mock = MagicMock()
    client_mock.start_trace = MagicMock(
        return_value=types.SimpleNamespace(trace_id="r1", span_id="s1")
    )
    client_mock.start_span = MagicMock(return_value=span_mock)
    client_mock.end_span = MagicMock()
    client_mock.end_trace = MagicMock()

    mod = types.ModuleType("mlflow")
    mod.set_tracking_uri = MagicMock()  # type: ignore[attr-defined]
    mod.set_experiment = MagicMock()  # type: ignore[attr-defined]
    mod.set_tag = MagicMock()  # type: ignore[attr-defined]
    mod.set_prompt_alias = MagicMock()  # type: ignore[attr-defined]
    mod.MlflowClient = MagicMock(return_value=client_mock)  # type: ignore[attr-defined]
    genai = types.ModuleType("mlflow.genai")
    genai.register_prompt = MagicMock()  # type: ignore[attr-defined]
    genai.load_prompt = MagicMock()  # type: ignore[attr-defined]
    mod.genai = genai  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "mlflow", mod)
    monkeypatch.setitem(sys.modules, "mlflow.genai", genai)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://x:5001")
    sys.modules.pop("llmops._mlflow_adapter", None)
    sys.modules.pop("llmops.tracing", None)
    sys.modules.pop("llmops._config", None)

    return {"mlflow": mod, "client": client_mock}


def test_trace_agent_as_context_manager_starts_and_ends_span(
    fake_mlflow_for_tracing: dict[str, MagicMock],
) -> None:
    from llmops.tracing import trace_agent

    with trace_agent("agent_tujuan"):
        pass

    client = fake_mlflow_for_tracing["client"]
    assert client.start_trace.called or client.start_span.called
    end_calls = client.end_span.call_count + client.end_trace.call_count
    assert end_calls >= 1


def test_trace_agent_as_decorator_wraps_function(
    fake_mlflow_for_tracing: dict[str, MagicMock],
) -> None:
    from llmops.tracing import trace_agent

    @trace_agent("agent_tujuan")
    def f(x: int) -> int:
        return x * 2

    result = f(3)
    assert result == 6

    client = fake_mlflow_for_tracing["client"]
    assert client.start_trace.called or client.start_span.called


def test_trace_agent_disabled_is_noop(
    fake_mlflow_for_tracing: dict[str, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLMOPS_DISABLE_TRACING", "true")
    sys.modules.pop("llmops._config", None)
    sys.modules.pop("llmops.tracing", None)

    from llmops.tracing import trace_agent

    with trace_agent("agent_x"):
        pass

    client = fake_mlflow_for_tracing["client"]
    assert client.start_trace.call_count == 0
    assert client.start_span.call_count == 0


def test_trace_agent_runtime_error_does_not_crash_caller(
    fake_mlflow_for_tracing: dict[str, MagicMock],
) -> None:
    """If MLflow itself raises, the user's code must still run (fail-soft runtime)."""
    client = fake_mlflow_for_tracing["client"]
    client.start_trace.side_effect = RuntimeError("network hiccup")
    client.start_span.side_effect = RuntimeError("network hiccup")

    from llmops.tracing import trace_agent

    with trace_agent("agent_x"):
        result = 42
    assert result == 42
