"""bni-llmops — LLM Ops SDK for BNI.

Public API:
    llmops.trace_agent(name, span_type=None, **attrs)  # context manager + decorator
    llmops.SpanType                                    # canonical span type constants
    llmops.load_prompt(ref)                            # 'name@alias' or 'name/version'
    llmops.register_prompt(name, template, ...)
    llmops.set_alias(name, alias, version, from_alias=None)
    llmops.log_hyperparams(**hp)                       # span-level hyperparams
    llmops.set_trace_tags(**tags)                      # arbitrary trace tags
    llmops.set_trace_metadata(session_id=, user_id=)   # canonical session/user IDs
    llmops.autolog(provider, **kwargs)                 # mlflow.<provider>.autolog passthrough

Configuration via environment variables only (read on first call):
    MLFLOW_TRACKING_URI       (required)
    LLMOPS_EXPERIMENT_NAME    (default: 'bni-agentic-prd')
    LLMOPS_DISABLE_TRACING    (default: 'false')
"""

from __future__ import annotations

from llmops._autolog import autolog
from llmops.exceptions import (
    LLMOpsConfigError,
    LLMOpsError,
    LLMOpsPromptNotFoundError,
    LLMOpsValidationError,
)
from llmops.prompts import load_prompt, register_prompt, set_alias
from llmops.tracing import (
    SpanType,
    log_hyperparams,
    set_trace_metadata,
    set_trace_tags,
    trace_agent,
)

__all__ = [
    "LLMOpsConfigError",
    "LLMOpsError",
    "LLMOpsPromptNotFoundError",
    "LLMOpsValidationError",
    "SpanType",
    "autolog",
    "load_prompt",
    "log_hyperparams",
    "register_prompt",
    "set_alias",
    "set_trace_metadata",
    "set_trace_tags",
    "trace_agent",
]
