from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

from typer.testing import CliRunner


def test_register_prompts_iterates_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://x:5001")

    # Prepare two prompt YAMLs
    (tmp_path / "a.yaml").write_text(
        "schema_version: 1\nname: a\ndescription: ten chars min\n"
        "template: '{{ x }}'\nvariables: [x]\ntags: {}\n"
    )
    (tmp_path / "b.yaml").write_text(
        "schema_version: 1\nname: b\ndescription: ten chars min\n"
        "template: '{{ y }}'\nvariables: [y]\ntags: {}\n"
    )

    # Mock the registration call
    mock_register = MagicMock(
        side_effect=[
            types.SimpleNamespace(name="a", version=1),
            types.SimpleNamespace(name="b", version=1),
        ]
    )
    sys.modules.pop("llmops.cli", None)
    monkeypatch.setattr("llmops.prompts.register_prompt", mock_register)
    monkeypatch.setattr("llmops.prompts.set_alias", MagicMock())

    from llmops.cli import app

    r = CliRunner().invoke(app, ["register-prompts", str(tmp_path)])
    assert r.exit_code == 0, r.stdout
    assert mock_register.call_count == 2
    assert "a" in r.stdout and "b" in r.stdout


def test_register_prompts_sets_staging_alias(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://x:5001")
    (tmp_path / "a.yaml").write_text(
        "schema_version: 1\nname: a\ndescription: ten chars min\n"
        "template: '{{ x }}'\nvariables: [x]\ntags: {}\n"
    )
    mock_register = MagicMock(return_value=types.SimpleNamespace(name="a", version=4))
    mock_set_alias = MagicMock()
    sys.modules.pop("llmops.cli", None)
    monkeypatch.setattr("llmops.prompts.register_prompt", mock_register)
    monkeypatch.setattr("llmops.prompts.set_alias", mock_set_alias)

    from llmops.cli import app

    r = CliRunner().invoke(app, ["register-prompts", str(tmp_path)])
    assert r.exit_code == 0, r.stdout
    mock_set_alias.assert_called_once_with("a", alias="staging", version=4)
