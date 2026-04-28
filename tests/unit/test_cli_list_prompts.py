from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

from typer.testing import CliRunner


def test_list_prompts_prints_table(monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://x:5001")
    fake_search = MagicMock(
        return_value=[
            types.SimpleNamespace(name="agent_x", aliases={"staging": 3, "production": 2}),
            types.SimpleNamespace(name="agent_y", aliases={"staging": 1}),
        ]
    )
    sys.modules.pop("llmops.cli", None)
    monkeypatch.setattr(
        "llmops._mlflow_adapter.MLflowAdapter.search_prompts", fake_search, raising=False
    )

    from llmops.cli import app

    r = CliRunner().invoke(app, ["list-prompts"])
    assert r.exit_code == 0, r.stdout
    assert "agent_x" in r.stdout
    assert "agent_y" in r.stdout
