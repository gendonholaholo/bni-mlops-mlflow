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


class trace_agent(ContextDecorator):  # noqa: N801 (intentional lowercase API surface)
    """Context manager + decorator — single object exposes both interfaces.

    Usage::

        with trace_agent("agent_tujuan"):
            ...

        @trace_agent("agent_rilis")
        def run_agent_rilis(...): ...
    """

    def __init__(self, name: str, **attrs: Any) -> None:
        self.name = name
        self.attrs = attrs
        self._span = None
        self._client = None
        self._trace_handle = None
        self._is_root = False

    def __enter__(self) -> trace_agent:
        cfg = get_config()
        if cfg.disable_tracing:
            return self

        try:
            from mlflow import MlflowClient  # noqa: TID253

            self._client = MlflowClient()
            stack = _stack()
            if not stack:
                handle = self._client.start_trace(name=self.name, inputs=self.attrs or None)
                self._trace_handle = handle
                self._is_root = True
                stack.append({"trace_id": handle.trace_id, "span_id": handle.span_id})
            else:
                parent = stack[-1]
                span = self._client.start_span(
                    name=self.name,
                    trace_id=parent["trace_id"],
                    parent_id=parent["span_id"],
                    inputs=self.attrs or None,
                )
                self._span = span
                self._is_root = False
                stack.append({"trace_id": parent["trace_id"], "span_id": span.span_id})
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
            if self._is_root and self._trace_handle is not None:
                # Flush accumulated prompt_versions BEFORE ending the trace,
                # so the tag attaches to the still-active run.
                self._flush_prompt_versions()
                self._client.end_trace(trace_id=self._trace_handle.trace_id, status=status)
            elif self._span is not None:
                self._client.end_span(
                    trace_id=self._span.trace_id,
                    span_id=self._span.span_id,
                    status=status,
                )
        except Exception as e:  # noqa: BLE001
            _log.warning("llmops trace_agent('%s') end failed: %r", self.name, e)

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
