"""trace_agent — context manager AND decorator (single callable)."""

from __future__ import annotations

import logging
from contextlib import ContextDecorator
from typing import Any

from llmops._config import get_config
from llmops._mlflow_adapter import MLflowAdapter

_log = logging.getLogger(__name__)
_adapter: MLflowAdapter | None = None


def _get_adapter() -> MLflowAdapter:
    global _adapter
    if _adapter is None:
        _adapter = MLflowAdapter(get_config())
        _adapter.initialise()
    return _adapter


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

    def __enter__(self) -> trace_agent:
        cfg = get_config()
        if cfg.disable_tracing:
            return self
        try:
            from mlflow import MlflowClient  # noqa: TID253 — done lazily; could also be in adapter

            self._client = MlflowClient()
            self._trace_handle = self._client.start_trace(name=self.name, inputs=self.attrs or None)
        except Exception as e:  # noqa: BLE001 — fail-soft for runtime errors
            _log.warning("llmops trace_agent('%s') start failed: %r", self.name, e)
            self._trace_handle = None
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._trace_handle is None or self._client is None:
            return
        try:
            status = "ERROR" if exc_type else "OK"
            self._client.end_trace(
                trace_id=self._trace_handle.trace_id,
                outputs=None,
                status=status,
            )
        except Exception as e:  # noqa: BLE001
            _log.warning("llmops trace_agent('%s') end failed: %r", self.name, e)
        # Returning None / False ensures original exception (if any) propagates
