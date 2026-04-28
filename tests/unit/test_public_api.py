"""Public API surface — what `import llmops` exposes."""

from __future__ import annotations

import llmops


def test_public_api_surface() -> None:
    assert callable(llmops.trace_agent)
    assert callable(llmops.load_prompt)
    assert callable(llmops.register_prompt)
    assert callable(llmops.set_alias)
    assert hasattr(llmops, "LLMOpsError")
    assert hasattr(llmops, "LLMOpsConfigError")
    assert hasattr(llmops, "LLMOpsPromptNotFoundError")


def test_no_init_function() -> None:
    """Per spec: implicit-only initialization. There is no llmops.init()."""
    assert not hasattr(llmops, "init")
