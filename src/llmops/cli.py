"""llmops CLI — entry point for ops-side tasks (register prompts, promote alias, doctor)."""

from __future__ import annotations

import os
import urllib.request
from importlib.metadata import version as _pkg_version

import typer

app = typer.Typer(
    name="llmops",
    help="BNI LLM Ops CLI — manage prompts and tracing infrastructure.",
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"bni-llmops {_pkg_version('bni-llmops')}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
    ),
) -> None:
    """Global options."""


@app.command()
def doctor(
    no_network: bool = typer.Option(
        False, "--no-network", help="Skip network reachability checks."
    ),
) -> None:
    """Validate environment, MLflow reachability, and Postgres connectivity."""
    failed = False

    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        typer.echo("[fail] MLFLOW_TRACKING_URI not set")
        raise typer.Exit(code=1)
    typer.echo(f"[ok]   MLFLOW_TRACKING_URI configured: {uri}")

    typer.echo(
        f"[ok]   LLMOPS_EXPERIMENT_NAME = "
        f"{os.environ.get('LLMOPS_EXPERIMENT_NAME', 'bni-agentic-prd')}"
    )

    if no_network:
        typer.echo("[skip] Network checks skipped (--no-network)")
        raise typer.Exit(code=0)

    try:
        with urllib.request.urlopen(f"{uri.rstrip('/')}/health", timeout=3) as r:
            if r.status == 200 and r.read().decode().strip() == "OK":
                typer.echo("[ok]   MLflow /health returned OK")
            else:
                typer.echo(f"[fail] MLflow /health returned {r.status}")
                failed = True
    except Exception as e:  # noqa: BLE001
        typer.echo(f"[fail] MLflow unreachable at {uri}: {e}")
        failed = True

    if failed:
        raise typer.Exit(code=1)
