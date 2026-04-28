"""Autolog passthrough — one-line auto-instrumentation for LLM/agent frameworks.

Wraps ``mlflow.{provider}.autolog()`` so consumers don't import mlflow directly,
and ensures the SDK's tracking URI and experiment are set first (Issue #1
applies to autolog too — without ``MLflowAdapter.initialise()`` the auto-captured
traces would land in 'Default').
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

from llmops._config import get_config
from llmops.tracing import _get_adapter

_log = logging.getLogger(__name__)

# MLflow 3.11.1 integration submodules with autolog support (verified against
# https://github.com/mlflow/mlflow/tree/v3.11.1/mlflow). When MLflow ships a new
# integration, add it here; SDK consumers don't have to upgrade mid-cycle.
SUPPORTED_PROVIDERS: frozenset[str] = frozenset(
    {
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
)


def autolog(provider: str, **kwargs: Any) -> None:
    """Enable automatic tracing for an LLM/agent framework.

    Example::

        llmops.autolog("openai")
        llmops.autolog("langchain", log_inputs_outputs=False)

    Passes ``**kwargs`` straight to ``mlflow.{provider}.autolog()`` — see each
    integration's MLflow doc for supported flags (``disable``, ``silent``,
    ``log_input_examples``, framework-specific toggles, etc.).

    Disabled when ``LLMOPS_DISABLE_TRACING=true``. Raises ``ValueError`` if
    ``provider`` is not in :data:`SUPPORTED_PROVIDERS`.
    """
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"unsupported autolog provider {provider!r}; "
            f"supported: {sorted(SUPPORTED_PROVIDERS)}"
        )
    if get_config().disable_tracing:
        return

    _get_adapter()
    mod = importlib.import_module(f"mlflow.{provider}")  # noqa: TID253
    mod.autolog(**kwargs)
