"""Exception hierarchy: all SDK errors inherit from LLMOpsError."""

from __future__ import annotations

import pytest

from llmops.exceptions import (
    LLMOpsConfigError,
    LLMOpsError,
    LLMOpsPromptNotFoundError,
    LLMOpsValidationError,
)


def test_all_errors_inherit_from_base() -> None:
    for err in (LLMOpsConfigError, LLMOpsPromptNotFoundError, LLMOpsValidationError):
        assert issubclass(err, LLMOpsError)


def test_base_inherits_from_exception() -> None:
    assert issubclass(LLMOpsError, Exception)


def test_config_error_carries_message() -> None:
    with pytest.raises(LLMOpsConfigError, match="bad env"):
        raise LLMOpsConfigError("bad env")


def test_prompt_not_found_carries_name_and_alias() -> None:
    err = LLMOpsPromptNotFoundError("agent_x", alias="staging")
    assert "agent_x" in str(err)
    assert "staging" in str(err)
