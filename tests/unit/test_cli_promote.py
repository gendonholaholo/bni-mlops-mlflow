from __future__ import annotations

import sys
from unittest.mock import MagicMock

from typer.testing import CliRunner


def test_promote_calls_sdk_set_alias(monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://x:5001")
    mock_load = MagicMock(return_value=type("P", (), {"name": "agent_x", "version": 5})())
    mock_set = MagicMock()
    sys.modules.pop("llmops.cli", None)
    monkeypatch.setattr("llmops.prompts.load_prompt", mock_load)
    monkeypatch.setattr("llmops.prompts.set_alias", mock_set)

    from llmops.cli import app

    r = CliRunner().invoke(app, ["promote", "agent_x", "staging", "production"])
    assert r.exit_code == 0, r.stdout
    mock_set.assert_called_once_with("agent_x", alias="production", version=5, from_alias="staging")
