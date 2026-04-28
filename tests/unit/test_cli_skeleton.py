from __future__ import annotations

from typer.testing import CliRunner


def test_help() -> None:
    from llmops.cli import app

    r = CliRunner().invoke(app, ["--help"])
    assert r.exit_code == 0
    assert "llmops" in r.stdout.lower()


def test_version() -> None:
    from llmops.cli import app

    r = CliRunner().invoke(app, ["--version"])
    assert r.exit_code == 0
    # SemVer present
    import re

    assert re.search(r"\d+\.\d+\.\d+", r.stdout)
