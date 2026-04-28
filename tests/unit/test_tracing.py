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


def test_nested_trace_agent_creates_child_span(
    fake_mlflow_for_tracing: dict[str, MagicMock],
) -> None:
    from llmops.tracing import trace_agent

    with trace_agent("orchestrator"):  # noqa: SIM117
        with trace_agent("agent_tujuan"):
            pass

    client = fake_mlflow_for_tracing["client"]
    # Outer = start_trace; inner = start_span with parent_id linkage
    assert client.start_trace.call_count == 1
    assert client.start_span.call_count == 1
    inner_call = client.start_span.call_args
    assert "parent_id" in inner_call.kwargs or len(inner_call.args) >= 3


def test_trace_agent_propagates_user_exception(
    fake_mlflow_for_tracing: dict[str, MagicMock],
) -> None:
    from llmops.tracing import trace_agent

    with pytest.raises(ValueError, match="boom"):  # noqa: SIM117
        with trace_agent("agent_x"):
            raise ValueError("boom")

    client = fake_mlflow_for_tracing["client"]
    # The end-of-trace call carried status=ERROR
    assert client.end_trace.called
    args, kwargs = client.end_trace.call_args
    assert kwargs.get("status") == "ERROR"


def test_trace_agent_initialises_adapter_without_prompt_calls(
    fake_mlflow_for_tracing: dict[str, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #1 regression: pure-tracing consumer (no load_prompt) must still
    trigger MLflowAdapter.initialise() — i.e. mlflow.set_tracking_uri AND
    mlflow.set_experiment must be called before MlflowClient.start_trace,
    so traces land in LLMOPS_EXPERIMENT_NAME and not 'Default'."""
    monkeypatch.setenv("LLMOPS_EXPERIMENT_NAME", "bni-agentic-prd")
    sys.modules.pop("llmops._config", None)
    sys.modules.pop("llmops._mlflow_adapter", None)
    sys.modules.pop("llmops.tracing", None)

    from llmops.tracing import trace_agent

    mlflow_mod = fake_mlflow_for_tracing["mlflow"]

    with trace_agent("chatbot"):
        pass

    # The adapter ran initialise() exactly once before the trace was started.
    mlflow_mod.set_tracking_uri.assert_called_once_with("http://x:5001")
    mlflow_mod.set_experiment.assert_called_once_with("bni-agentic-prd")

    client = fake_mlflow_for_tracing["client"]
    assert client.start_trace.called


def test_log_hyperparams_sets_span_attributes(
    fake_mlflow_for_tracing: dict[str, MagicMock],
) -> None:
    """Issue #2 Option 2: log_hyperparams writes llmops.hp.* attributes onto
    the currently-active span via Span.set_attributes."""
    from llmops.tracing import log_hyperparams, trace_agent

    client = fake_mlflow_for_tracing["client"]
    # The fake start_trace returns a SimpleNamespace; replace its set_attributes
    # with a MagicMock so we can assert the call.
    root_span = MagicMock()
    root_span.trace_id = "r1"
    root_span.span_id = "s1"
    client.start_trace.return_value = root_span

    with trace_agent("chatbot"):
        log_hyperparams(model="qwen2.5:7b", temperature=0.7, top_k=40, num_ctx=4096)

    root_span.set_attributes.assert_called_once()
    (attrs,) = root_span.set_attributes.call_args.args
    assert attrs == {
        "llmops.hp.model": "qwen2.5:7b",
        "llmops.hp.temperature": 0.7,
        "llmops.hp.top_k": 40,
        "llmops.hp.num_ctx": 4096,
    }


def test_log_hyperparams_attaches_to_innermost_span(
    fake_mlflow_for_tracing: dict[str, MagicMock],
) -> None:
    """When called inside a nested trace_agent, attributes attach to the
    inner span, not the outer trace root."""
    from llmops.tracing import log_hyperparams, trace_agent

    client = fake_mlflow_for_tracing["client"]
    root_span = MagicMock()
    root_span.trace_id = "r1"
    root_span.span_id = "s_root"
    client.start_trace.return_value = root_span

    inner_span = MagicMock()
    inner_span.trace_id = "r1"
    inner_span.span_id = "s_inner"
    client.start_span.return_value = inner_span

    with trace_agent("orchestrator"):  # noqa: SIM117
        with trace_agent("inner"):
            log_hyperparams(temperature=0.3)

    inner_span.set_attributes.assert_called_once_with({"llmops.hp.temperature": 0.3})
    root_span.set_attributes.assert_not_called()


def test_log_hyperparams_outside_trace_is_noop(
    fake_mlflow_for_tracing: dict[str, MagicMock],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Calling log_hyperparams without an active trace_agent must not raise —
    it logs a warning and returns."""
    from llmops.tracing import log_hyperparams

    log_hyperparams(temperature=0.7)  # no exception
    # The warning is emitted, but we don't pin the exact message text.
    # Verifying the no-raise behavior is the contract we care about.


def test_log_hyperparams_disabled_is_noop(
    fake_mlflow_for_tracing: dict[str, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLMOPS_DISABLE_TRACING", "true")
    sys.modules.pop("llmops._config", None)
    sys.modules.pop("llmops.tracing", None)

    from llmops.tracing import log_hyperparams

    log_hyperparams(temperature=0.7)  # no exception, no-op
    client = fake_mlflow_for_tracing["client"]
    assert client.start_trace.call_count == 0


# Tier 1.1: span_type ------------------------------------------------------

def test_trace_agent_applies_span_type_on_root(
    fake_mlflow_for_tracing: dict[str, MagicMock],
) -> None:
    from llmops.tracing import SpanType, trace_agent

    client = fake_mlflow_for_tracing["client"]
    root_span = MagicMock()
    root_span.trace_id = "r1"
    root_span.span_id = "s1"
    client.start_trace.return_value = root_span

    with trace_agent("chatbot", span_type=SpanType.AGENT):
        pass

    root_span.set_span_type.assert_called_once_with("AGENT")


def test_trace_agent_applies_span_type_on_nested_span(
    fake_mlflow_for_tracing: dict[str, MagicMock],
) -> None:
    from llmops.tracing import SpanType, trace_agent

    client = fake_mlflow_for_tracing["client"]
    root_span = MagicMock()
    root_span.trace_id = "r1"
    root_span.span_id = "s_root"
    client.start_trace.return_value = root_span

    inner_span = MagicMock()
    inner_span.trace_id = "r1"
    inner_span.span_id = "s_inner"
    client.start_span.return_value = inner_span

    with trace_agent("orch"):  # noqa: SIM117
        with trace_agent("retriever_step", span_type=SpanType.RETRIEVER):
            pass

    inner_span.set_span_type.assert_called_once_with("RETRIEVER")
    root_span.set_span_type.assert_not_called()


def test_trace_agent_no_span_type_skips_call(
    fake_mlflow_for_tracing: dict[str, MagicMock],
) -> None:
    """When span_type is omitted (default UNKNOWN), set_span_type is not called
    — MLflow's default UNKNOWN remains in effect without an extra round trip."""
    from llmops.tracing import trace_agent

    client = fake_mlflow_for_tracing["client"]
    root_span = MagicMock()
    root_span.trace_id = "r1"
    root_span.span_id = "s1"
    client.start_trace.return_value = root_span

    with trace_agent("chatbot"):
        pass

    root_span.set_span_type.assert_not_called()


def test_span_type_constants_are_strings() -> None:
    """SpanType class attrs match MLflow 3.11.x's enum values verbatim."""
    from llmops.tracing import SpanType

    assert SpanType.LLM == "LLM"
    assert SpanType.AGENT == "AGENT"
    assert SpanType.CHAT_MODEL == "CHAT_MODEL"
    assert SpanType.RETRIEVER == "RETRIEVER"
    assert SpanType.TOOL == "TOOL"
    assert SpanType.UNKNOWN == "UNKNOWN"


# Tier 1.2: set_trace_tags / set_trace_metadata ----------------------------

def test_set_trace_tags_writes_attributes_on_root_span(
    fake_mlflow_for_tracing: dict[str, MagicMock],
) -> None:
    from llmops.tracing import set_trace_tags, trace_agent

    client = fake_mlflow_for_tracing["client"]
    root_span = MagicMock()
    root_span.trace_id = "r1"
    root_span.span_id = "s1"
    client.start_trace.return_value = root_span

    inner_span = MagicMock()
    inner_span.trace_id = "r1"
    inner_span.span_id = "s2"
    client.start_span.return_value = inner_span

    with trace_agent("orch"):  # noqa: SIM117
        with trace_agent("inner"):
            set_trace_tags(team="alpha", env="prod")

    # Tags always land on root, not the active inner span.
    root_span.set_attributes.assert_called_once_with({"team": "alpha", "env": "prod"})
    inner_span.set_attributes.assert_not_called()


def test_set_trace_tags_outside_trace_is_noop(
    fake_mlflow_for_tracing: dict[str, MagicMock],
) -> None:
    from llmops.tracing import set_trace_tags

    set_trace_tags(team="alpha")  # no exception


def test_set_trace_metadata_uses_canonical_keys(
    fake_mlflow_for_tracing: dict[str, MagicMock],
) -> None:
    from llmops.tracing import set_trace_metadata, trace_agent

    client = fake_mlflow_for_tracing["client"]
    root_span = MagicMock()
    root_span.trace_id = "r1"
    root_span.span_id = "s1"
    client.start_trace.return_value = root_span

    with trace_agent("chatbot"):
        set_trace_metadata(session_id="sess-42", user_id="alice")

    root_span.set_attributes.assert_called_once_with(
        {"mlflow.trace.session": "sess-42", "mlflow.trace.user": "alice"}
    )


def test_set_trace_metadata_partial(
    fake_mlflow_for_tracing: dict[str, MagicMock],
) -> None:
    """Only set keys for the IDs that were actually provided."""
    from llmops.tracing import set_trace_metadata, trace_agent

    client = fake_mlflow_for_tracing["client"]
    root_span = MagicMock()
    root_span.trace_id = "r1"
    root_span.span_id = "s1"
    client.start_trace.return_value = root_span

    with trace_agent("chatbot"):
        set_trace_metadata(session_id="sess-42")

    root_span.set_attributes.assert_called_once_with({"mlflow.trace.session": "sess-42"})


def test_set_trace_metadata_no_args_is_noop(
    fake_mlflow_for_tracing: dict[str, MagicMock],
) -> None:
    from llmops.tracing import set_trace_metadata, trace_agent

    client = fake_mlflow_for_tracing["client"]
    root_span = MagicMock()
    root_span.trace_id = "r1"
    root_span.span_id = "s1"
    client.start_trace.return_value = root_span

    with trace_agent("chatbot"):
        set_trace_metadata()

    root_span.set_attributes.assert_not_called()


# Tier 1.3: set_outputs / decorator auto-capture ---------------------------

def test_trace_agent_set_outputs_passed_to_end_trace(
    fake_mlflow_for_tracing: dict[str, MagicMock],
) -> None:
    from llmops.tracing import trace_agent

    client = fake_mlflow_for_tracing["client"]

    with trace_agent("chatbot") as t:
        t.set_outputs({"answer": "halo"})

    end_kwargs = client.end_trace.call_args.kwargs
    assert end_kwargs["outputs"] == {"answer": "halo"}
    assert end_kwargs["status"] == "OK"


def test_trace_agent_no_set_outputs_omits_outputs_kwarg(
    fake_mlflow_for_tracing: dict[str, MagicMock],
) -> None:
    """Backward-compat: existing consumers that don't call set_outputs must
    not have outputs=None forced into end_trace."""
    from llmops.tracing import trace_agent

    client = fake_mlflow_for_tracing["client"]

    with trace_agent("chatbot"):
        pass

    end_kwargs = client.end_trace.call_args.kwargs
    assert "outputs" not in end_kwargs


def test_trace_agent_decorator_auto_captures_return_value(
    fake_mlflow_for_tracing: dict[str, MagicMock],
) -> None:
    from llmops.tracing import trace_agent

    @trace_agent("agent_tujuan")
    def f(x: int) -> dict[str, int]:
        return {"doubled": x * 2}

    assert f(3) == {"doubled": 6}

    client = fake_mlflow_for_tracing["client"]
    end_kwargs = client.end_trace.call_args.kwargs
    assert end_kwargs["outputs"] == {"doubled": 6}


def test_trace_agent_decorator_does_not_capture_when_function_raises(
    fake_mlflow_for_tracing: dict[str, MagicMock],
) -> None:
    """If the wrapped function raises, no outputs are recorded and the trace
    ends with status=ERROR."""
    from llmops.tracing import trace_agent

    @trace_agent("agent_tujuan")
    def f() -> int:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        f()

    client = fake_mlflow_for_tracing["client"]
    end_kwargs = client.end_trace.call_args.kwargs
    assert end_kwargs["status"] == "ERROR"
    assert "outputs" not in end_kwargs


def test_trace_agent_set_outputs_on_nested_span_passed_to_end_span(
    fake_mlflow_for_tracing: dict[str, MagicMock],
) -> None:
    from llmops.tracing import trace_agent

    client = fake_mlflow_for_tracing["client"]
    inner_span = MagicMock()
    inner_span.trace_id = "r1"
    inner_span.span_id = "s_inner"
    client.start_span.return_value = inner_span

    with trace_agent("orch"):  # noqa: SIM117
        with trace_agent("inner") as inner:
            inner.set_outputs(42)

    end_span_kwargs = client.end_span.call_args.kwargs
    assert end_span_kwargs["outputs"] == 42


