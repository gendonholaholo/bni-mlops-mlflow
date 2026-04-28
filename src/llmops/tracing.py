"""trace_agent — context manager AND decorator (single callable)."""

from __future__ import annotations

import json as _json
import logging
import threading
from contextlib import ContextDecorator
from typing import Any

from llmops._config import get_config
from llmops._mlflow_adapter import MLflowAdapter

_log = logging.getLogger(__name__)
_adapter: MLflowAdapter | None = None
_local = threading.local()

# Sentinel for "outputs not set" — distinct from a user explicitly passing None.
_UNSET = object()


class SpanType:
    """Canonical MLflow span type string constants (mirrors
    ``mlflow.entities.SpanType`` as of MLflow 3.11.x — 15 values). Use with
    ``trace_agent(span_type=SpanType.AGENT)`` for IDE-friendly autocomplete and
    typo safety. Users may also pass raw strings; MLflow accepts any value."""

    LLM = "LLM"
    CHAIN = "CHAIN"
    AGENT = "AGENT"
    TOOL = "TOOL"
    CHAT_MODEL = "CHAT_MODEL"
    RETRIEVER = "RETRIEVER"
    PARSER = "PARSER"
    EMBEDDING = "EMBEDDING"
    RERANKER = "RERANKER"
    MEMORY = "MEMORY"
    UNKNOWN = "UNKNOWN"
    WORKFLOW = "WORKFLOW"
    TASK = "TASK"
    GUARDRAIL = "GUARDRAIL"
    EVALUATOR = "EVALUATOR"


def _get_adapter() -> MLflowAdapter:
    global _adapter
    if _adapter is None:
        _adapter = MLflowAdapter(get_config())
        _adapter.initialise()
    return _adapter


def _stack() -> list[Any]:
    if not hasattr(_local, "spans"):
        _local.spans = []
    return _local.spans


def _current_span() -> Any | None:
    """Return the topmost live span on the thread-local stack, or None if no
    trace_agent is active. Span objects expose ``.trace_id`` and ``.span_id``
    directly, so callers don't need a wrapping dict."""
    stack = _stack()
    return stack[-1] if stack else None


def set_trace_tags(**tags: Any) -> None:
    """Attach mutable user-defined tags to the active trace.

    Writes to ``TraceInfo.tags`` via ``MlflowClient.set_trace_tag(trace_id, ...)``
    so the values land in the MLflow trace overview and are queryable via
    ``mlflow.search_traces(filter_string='tags."env" = \\'dev\\'')`` (quote
    keys that contain dots).

    Must be called inside an active ``trace_agent`` context. Outside one, the
    call is a warning + no-op (never raises). Disabled when
    ``LLMOPS_DISABLE_TRACING=true``. Values are coerced to ``str`` to satisfy
    MLflow's ``dict[str, str]`` tag contract.
    """
    if not tags or get_config().disable_tracing:
        return
    stack = _stack()
    if not stack:
        _log.warning("llmops.set_trace_tags called outside trace_agent — ignored")
        return
    trace_id = stack[0].trace_id
    try:
        from mlflow import MlflowClient  # noqa: TID253, PLC0415

        client = MlflowClient()
        for k, v in tags.items():
            client.set_trace_tag(trace_id, k, str(v))
    except Exception as e:  # noqa: BLE001
        _log.warning("llmops.set_trace_tags set_trace_tag failed: %r", e)


def set_trace_metadata(
    session_id: str | None = None, user_id: str | None = None
) -> None:
    """Attach canonical session/user identifiers to the active trace.

    Writes to ``TraceInfo.request_metadata`` under MLflow's reserved keys
    ``mlflow.trace.session`` and ``mlflow.trace.user`` via the in-memory trace
    manager (the same path ``mlflow.update_current_trace`` uses internally —
    we bypass the fluent active-span check because ``trace_agent`` builds
    traces with the low-level ``MlflowClient.start_trace`` API, which does
    not register a fluent active span). The MLflow UI's Sessions tab and
    ``mlflow.search_traces(filter_string='metadata."mlflow.trace.session"
    = ...')`` consume these keys for grouping and filtering.

    Must be called inside an active ``trace_agent`` context, before the root
    trace ends — metadata is **immutable once the trace is logged**.
    """
    meta: dict[str, str] = {}
    if session_id is not None:
        meta["mlflow.trace.session"] = session_id
    if user_id is not None:
        meta["mlflow.trace.user"] = user_id
    if not meta or get_config().disable_tracing:
        return
    stack = _stack()
    if not stack:
        _log.warning("llmops.set_trace_metadata called outside trace_agent — ignored")
        return
    trace_id = stack[0].trace_id
    try:
        from mlflow.tracing.trace_manager import InMemoryTraceManager  # noqa: TID253, PLC0415

        mgr = InMemoryTraceManager.get_instance()
        for k, v in meta.items():
            mgr.set_trace_metadata(trace_id, k, str(v))
    except Exception as e:  # noqa: BLE001
        _log.warning("llmops.set_trace_metadata failed: %r", e)


def log_hyperparams(**hyperparams: Any) -> None:
    """Record runtime hyperparameters as attributes on the currently-active span.

    Issue #2 (Option 2): a runtime escape hatch for consumers that want to log
    generation hyperparameters (temperature, top_k, num_ctx, model name, etc.)
    without going through the prompt registry. Values are written as span
    attributes under the ``llmops.hp.*`` namespace.

    Must be called inside an active ``trace_agent`` context. Outside one, the
    call is a warning + no-op (never raises). Disabled when
    ``LLMOPS_DISABLE_TRACING=true``.

    Example::

        with llmops.trace_agent("chatbot"):
            llmops.log_hyperparams(model="qwen2.5:7b", temperature=0.7, top_k=40)
            ollama.chat(...)
    """
    if get_config().disable_tracing:
        return

    span = _current_span()
    if span is None:
        _log.warning("llmops.log_hyperparams called outside trace_agent — ignored")
        return

    try:
        span.set_attributes({f"llmops.hp.{k}": v for k, v in hyperparams.items()})
    except Exception as e:  # noqa: BLE001
        _log.warning("llmops.log_hyperparams set_attributes failed: %r", e)


class trace_agent(ContextDecorator):  # noqa: N801 (intentional lowercase API surface)
    """Context manager + decorator — single object exposes both interfaces.

    Usage::

        with trace_agent("agent_tujuan"):
            ...

        @trace_agent("agent_rilis", span_type=SpanType.AGENT)
        def run_agent_rilis(...): ...

    The ``span_type`` parameter accepts any MLflow ``SpanType`` value (``LLM``,
    ``CHAT_MODEL``, ``RETRIEVER``, ``TOOL``, ``AGENT``, ``CHAIN``, ``EMBEDDING``,
    ``RERANKER``, ``PARSER``, ``MEMORY``, ``WORKFLOW``, ``TASK``, ``GUARDRAIL``,
    ``EVALUATOR``, ``UNKNOWN``). Use the re-exported ``llmops.SpanType`` enum
    rather than raw strings so renames stay safe.
    """

    def __init__(self, name: str, span_type: str | None = None, **attrs: Any) -> None:
        self.name = name
        self.span_type = span_type
        self.attrs = attrs
        self._span = None
        self._client = None
        self._trace_handle = None
        self._is_root = False
        self._outputs: Any = _UNSET

    def __enter__(self) -> trace_agent:
        cfg = get_config()
        if cfg.disable_tracing:
            return self

        # Issue #1: ensure MLflowAdapter.initialise() ran (set_tracking_uri +
        # set_experiment) before any MlflowClient call. Without this, traces
        # land in 'Default' for consumers that use trace_agent without first
        # touching the prompt registry. The adapter's _initialised flag makes
        # this a no-op after the first call.
        _get_adapter()

        try:
            from mlflow import MlflowClient  # noqa: TID253

            self._client = MlflowClient()
            stack = _stack()
            if not stack:
                handle = self._client.start_trace(name=self.name, inputs=self.attrs or None)
                self._trace_handle = handle
                self._is_root = True
                stack.append(handle)
                if self.span_type:
                    handle.set_span_type(self.span_type)
            else:
                parent = stack[-1]
                span = self._client.start_span(
                    name=self.name,
                    trace_id=parent.trace_id,
                    parent_id=parent.span_id,
                    inputs=self.attrs or None,
                )
                self._span = span
                self._is_root = False
                stack.append(span)
                if self.span_type:
                    span.set_span_type(self.span_type)
        except Exception as e:  # noqa: BLE001
            _log.warning("llmops trace_agent('%s') start failed: %r", self.name, e)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        cfg = get_config()
        if cfg.disable_tracing or self._client is None:
            return
        try:
            stack = _stack()
            if stack:
                stack.pop()
            status = "ERROR" if exc_type else "OK"
            end_kwargs: dict[str, Any] = {"status": status}
            if self._outputs is not _UNSET:
                end_kwargs["outputs"] = self._outputs
            if self._is_root and self._trace_handle is not None:
                # Flush accumulated prompt_versions BEFORE ending the trace,
                # so the tag attaches to the still-active run.
                self._flush_prompt_versions()
                self._client.end_trace(trace_id=self._trace_handle.trace_id, **end_kwargs)
            elif self._span is not None:
                self._client.end_span(
                    trace_id=self._span.trace_id,
                    span_id=self._span.span_id,
                    **end_kwargs,
                )
        except Exception as e:  # noqa: BLE001
            _log.warning("llmops trace_agent('%s') end failed: %r", self.name, e)

    def set_outputs(self, value: Any) -> None:
        """Record the trace/span output. Captured automatically when ``trace_agent``
        is used as a decorator (the wrapped function's return value). Call
        explicitly inside a ``with trace_agent(...) as t:`` block to record
        outputs in context-manager form."""
        self._outputs = value

    def __call__(self, func: Any) -> Any:
        """Decorator form — auto-captures the wrapped function's return value as
        the span's outputs, then re-raises any exception unchanged."""
        from functools import wraps

        @wraps(func)
        def inner(*args: Any, **kwargs: Any) -> Any:
            with self._recreate_cm() as cm:
                result = func(*args, **kwargs)
                cm.set_outputs(result)
                return result

        return inner

    def _flush_prompt_versions(self) -> None:
        """Serialize the thread-local prompt_versions dict and write it as a run tag.
        Always resets the accumulator at the end (success or failure)."""
        # Lazy import to avoid prompts <-> tracing circular import at module load
        from llmops import prompts as _prompts  # noqa: PLC0415

        try:
            versions = _prompts.get_loaded_versions()
            if versions:
                from mlflow import set_tag as _set_tag  # noqa: TID253, PLC0415

                _set_tag("llmops.prompt_versions", _json.dumps(versions, sort_keys=True))
        except Exception as e:  # noqa: BLE001
            _log.warning("prompt_versions tag write failed: %r", e)
        finally:
            import contextlib as _contextlib  # noqa: PLC0415

            with _contextlib.suppress(Exception):
                _prompts.reset_loaded_versions()
