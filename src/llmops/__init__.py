"""bni-llmops — LLM Ops SDK for BNI.

Public API:
    llmops.trace_agent(name, **attrs)         # context manager + decorator
    llmops.load_prompt(ref)                   # 'name@alias' or 'name/version'
    llmops.register_prompt(name, template, ...)
    llmops.set_alias(name, alias, version, from_alias=None)

Configuration via environment variables only (read on first call):
    MLFLOW_TRACKING_URI       (required)
    LLMOPS_EXPERIMENT_NAME    (default: 'bni-agentic-prd')
    LLMOPS_DISABLE_TRACING    (default: 'false')
"""

from __future__ import annotations

from llmops.exceptions import (
    LLMOpsConfigError,
    LLMOpsError,
    LLMOpsPromptNotFoundError,
    LLMOpsValidationError,
)
from llmops.prompts import load_prompt, register_prompt, set_alias
from llmops.tracing import trace_agent

__all__ = [
    "LLMOpsConfigError",
    "LLMOpsError",
    "LLMOpsPromptNotFoundError",
    "LLMOpsValidationError",
    "load_prompt",
    "register_prompt",
    "set_alias",
    "trace_agent",
]
