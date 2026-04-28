"""Frozen, env-loaded configuration. Read once on first access (lazy + cached)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from llmops.exceptions import LLMOpsConfigError

_DEFAULT_EXPERIMENT = "bni-agentic-prd"
_cached: Config | None = None


@dataclass(frozen=True, slots=True)
class Config:
    tracking_uri: str
    experiment_name: str
    disable_tracing: bool


def _read_env() -> Config:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise LLMOpsConfigError(
            "MLFLOW_TRACKING_URI is required. "
            "Set it (e.g., http://localhost:5001) before importing or using llmops."
        )
    return Config(
        tracking_uri=tracking_uri,
        experiment_name=os.environ.get("LLMOPS_EXPERIMENT_NAME", _DEFAULT_EXPERIMENT),
        disable_tracing=os.environ.get("LLMOPS_DISABLE_TRACING", "false").lower() == "true",
    )


def get_config() -> Config:
    """Return cached config; read env on first call."""
    global _cached
    if _cached is None:
        _cached = _read_env()
    return _cached


def reset_config_cache() -> None:
    """Test helper: drop cached config so next get_config() re-reads env."""
    global _cached
    _cached = None
