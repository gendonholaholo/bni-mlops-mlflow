from __future__ import annotations

from typer.testing import CliRunner


def test_doctor_reports_missing_uri(monkeypatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    from llmops.cli import app

    r = CliRunner().invoke(app, ["doctor"])
    assert r.exit_code != 0
    assert "MLFLOW_TRACKING_URI" in r.stdout


def test_doctor_with_uri_reports_progress(monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    from llmops.cli import app

    r = CliRunner().invoke(app, ["doctor", "--no-network"])
    assert r.exit_code == 0
    assert "MLFLOW_TRACKING_URI" in r.stdout
    assert "configured" in r.stdout.lower()
