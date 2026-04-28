"""llmops CLI — entry point for ops-side tasks (register prompts, promote alias, doctor)."""

from __future__ import annotations

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
