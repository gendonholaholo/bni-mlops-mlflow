"""Prompt registry SDK API: load, register, set_alias.

All public functions go through MLflowAdapter — never `import mlflow` here.
"""

from __future__ import annotations

import re
from typing import Any

from llmops._config import get_config
from llmops._mlflow_adapter import MLflowAdapter
from llmops.exceptions import LLMOpsPromptNotFoundError

_adapter: MLflowAdapter | None = None
_PROMPT_REF = re.compile(r"^([a-z][a-z0-9_]*)([@/])(.+)$")


def _get_adapter() -> MLflowAdapter:
    global _adapter
    if _adapter is None:
        _adapter = MLflowAdapter(get_config())
        _adapter.initialise()
    return _adapter


def _parse_ref(ref: str) -> tuple[str, str | None, int | None]:
    """Parse 'name@alias' or 'name/version' into (name, alias, version)."""
    m = _PROMPT_REF.match(ref)
    if not m:
        raise ValueError(
            f"Invalid prompt reference {ref!r}; expected 'name@alias' or 'name/version'"
        )
    name, sep, tail = m.group(1), m.group(2), m.group(3)
    if sep == "@":
        return name, tail, None
    return name, None, int(tail)


def load_prompt(ref: str) -> Any:
    """Load a prompt by 'name@alias' or 'name/version'.

    Returns an object with `.template` and `.format(**vars)`.
    Raises LLMOpsPromptNotFoundError if the prompt or alias does not exist.
    """
    name, alias, version = _parse_ref(ref)
    adapter = _get_adapter()
    try:
        return adapter.load_prompt(name=name, alias=alias, version=version)
    except Exception as e:  # noqa: BLE001
        msg = str(e).upper()
        if "DOES_NOT_EXIST" in msg or "NOT_FOUND" in msg or "RESOURCE_DOES_NOT_EXIST" in msg:
            raise LLMOpsPromptNotFoundError(name, alias=alias, version=version) from e
        raise


def register_prompt(
    name: str,
    template: str,
    commit_message: str | None = None,
    tags: dict[str, str] | None = None,
) -> Any:
    """Register a new prompt version. Idempotent (v1 strategy): if any of the
    `staging` or `production` aliases points at a version whose template equals
    the new template, return that version without creating a new one.

    Note: a more thorough check (search ALL versions) is v2 work — the search API
    surface needs validation against MLflow 3.11.x first.
    """
    adapter = _get_adapter()

    existing = None
    for try_alias in ("staging", "production"):
        try:
            existing = adapter.load_prompt(name=name, alias=try_alias)
            if existing.template == template:
                return existing
        except Exception:  # noqa: BLE001
            continue

    return adapter.register_prompt(
        name=name, template=template, commit_message=commit_message, tags=tags
    )
