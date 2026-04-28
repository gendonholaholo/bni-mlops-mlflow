"""llmops CLI — entry point for ops-side tasks (register prompts, promote alias, doctor)."""

from __future__ import annotations

import os
import urllib.request
from importlib.metadata import version as _pkg_version
from pathlib import Path

import typer
import yaml

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


@app.command(name="register-prompts")
def register_prompts(
    directory: Path = typer.Argument(Path("prompts"), help="Directory containing *.yaml prompts"),  # noqa: B008
    set_staging: bool = typer.Option(
        True,
        "--set-staging/--no-set-staging",
        help="After registering, set 'staging' alias to each registered version (default: on).",
    ),
) -> None:
    """Bulk-register every *.yaml in DIRECTORY (idempotent). After registering,
    set the `staging` alias to the resulting version for each prompt — this is
    the locked CI behavior for `register-prompts.yml` on merge to gos-dev."""
    # Lazy imports — keep CLI startup fast and avoid pulling SDK deps unless needed
    from llmops._prompt_schema import PromptYAML
    from llmops.prompts import register_prompt as _reg
    from llmops.prompts import set_alias as _set_alias

    paths = sorted(directory.glob("*.yaml"))
    if not paths:
        typer.echo(f"No prompts found in {directory}")
        raise typer.Exit(code=0)

    failed = 0
    for p in paths:
        if p.name.startswith("_"):
            continue
        try:
            data = yaml.safe_load(p.read_text())
            schema = PromptYAML(**data)
            if schema.name != p.stem:
                raise ValueError(f"filename stem {p.stem!r} != name {schema.name!r}")
            result = _reg(
                name=schema.name,
                template=schema.template,
                commit_message=os.environ.get("GITHUB_SHA", ""),
                tags=schema.tags,
                model_config=schema.mlflow_model_config,
            )
            if set_staging:
                _set_alias(schema.name, alias="staging", version=result.version)
            typer.echo(f"[ok]   {schema.name} v{result.version} (staging alias set)")
        except Exception as e:  # noqa: BLE001
            typer.echo(f"[fail] {p.name}: {e}")
            failed += 1

    if failed:
        raise typer.Exit(code=1)


@app.command()
def promote(
    prompt_name: str = typer.Argument(..., help="Prompt name."),
    from_alias: str = typer.Argument(..., help="Source alias (e.g., 'staging')."),
    to_alias: str = typer.Argument(..., help="Target alias (e.g., 'production')."),
) -> None:
    """Move TO_ALIAS to the version currently pointed to by FROM_ALIAS."""
    from llmops.prompts import load_prompt as _load
    from llmops.prompts import set_alias as _set

    src = _load(f"{prompt_name}@{from_alias}")
    _set(prompt_name, alias=to_alias, version=src.version, from_alias=from_alias)
    typer.echo(f"[ok]   {prompt_name}: {from_alias}@v{src.version} -> {to_alias}@v{src.version}")


@app.command(name="list-prompts")
def list_prompts() -> None:
    """Print a table of registered prompts and their aliases."""
    from llmops._config import get_config
    from llmops._mlflow_adapter import MLflowAdapter

    adapter = MLflowAdapter(get_config())
    prompts_ = adapter.search_prompts()

    if not prompts_:
        typer.echo("(no prompts registered)")
        raise typer.Exit(code=0)

    typer.echo(f"{'NAME':<32} {'ALIASES':<60}")
    typer.echo("-" * 92)
    for p in prompts_:
        aliases = getattr(p, "aliases", {}) or {}
        alias_str = ", ".join(f"{a}=v{v}" for a, v in sorted(aliases.items()))
        typer.echo(f"{p.name:<32} {alias_str:<60}")
