"""SDK exception hierarchy. All errors raised by llmops inherit from LLMOpsError."""

from __future__ import annotations


class LLMOpsError(Exception):
    """Base for all SDK-raised errors."""


class LLMOpsConfigError(LLMOpsError):
    """Config / env-var problem detected at startup or first SDK call."""


class LLMOpsValidationError(LLMOpsError):
    """Schema or input validation error (e.g., invalid prompt YAML)."""


class LLMOpsPromptNotFoundError(LLMOpsError):
    """Requested prompt name/alias does not exist in the registry."""

    def __init__(self, name: str, alias: str | None = None, version: int | None = None) -> None:
        self.name = name
        self.alias = alias
        self.version = version
        ref = f"@{alias}" if alias else (f"/{version}" if version else "")
        super().__init__(f"Prompt not found: {name}{ref}")
