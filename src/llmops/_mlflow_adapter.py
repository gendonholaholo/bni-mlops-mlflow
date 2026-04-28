"""Sole MLflow coupling point. ALL other llmops modules MUST go through this adapter.

CI rule (TID253) forbids `import mlflow` outside this file. See pyproject.toml.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import mlflow  # noqa: TID253 (this file is the SOLE allowed importer)
import mlflow.genai  # noqa: TID253

from llmops._config import Config


class _PromptObj(Protocol):
    name: str
    version: int
    template: str
    # NOTE: PromptVersion also exposes ``.model_config`` (MLflow 3.8+) when the
    # prompt was registered with one. We don't declare it on the Protocol because
    # not every consumer / mock path supplies it; callers use getattr fallback.

    def format(self, **vars: Any) -> str: ...  # noqa: A002


@dataclass
class MLflowAdapter:
    """Thin wrapper around mlflow client. Stateless except for `initialised` flag."""

    config: Config
    _initialised: bool = False

    def initialise(self) -> None:
        """Idempotent: set tracking URI + experiment once."""
        if self._initialised:
            return
        mlflow.set_tracking_uri(self.config.tracking_uri)
        mlflow.set_experiment(self.config.experiment_name)
        self._initialised = True

    # --- prompt registry ---

    def register_prompt(
        self,
        name: str,
        template: str,
        commit_message: str | None = None,
        tags: dict[str, str] | None = None,
        model_config: dict[str, Any] | None = None,
    ) -> _PromptObj:
        # ``model_config`` is the canonical MLflow 3.8+ field for versioning
        # generation hyperparameters alongside the prompt template (Issue #2).
        return mlflow.genai.register_prompt(
            name=name,
            template=template,
            commit_message=commit_message,
            tags=tags,
            model_config=model_config,
        )

    def load_prompt(
        self, name: str, alias: str | None = None, version: int | None = None
    ) -> _PromptObj:
        if alias is not None:
            uri = f"prompts:/{name}@{alias}"
        elif version is not None:
            uri = f"prompts:/{name}/{version}"
        else:
            raise ValueError("Either alias or version must be provided")
        return mlflow.genai.load_prompt(name_or_uri=uri)

    def set_alias(self, name: str, alias: str, version: int) -> None:
        # NOTE: this lives on mlflow.* (root namespace), NOT mlflow.genai.*
        mlflow.set_prompt_alias(name, alias, version)

    def write_prompt_version_tags(self, name: str, version: int, tags: dict[str, str]) -> None:
        """Write each (key, value) in `tags` as a tag on the given prompt version
        via MlflowClient.set_prompt_version_tag (verified API surface in mlflow 3.11.x).
        """
        from mlflow import MlflowClient  # noqa: TID253 — adapter is the only allowed importer

        client = MlflowClient()
        for k, v in tags.items():
            client.set_prompt_version_tag(name, version, k, str(v))

    def search_prompts(self) -> list[Any]:
        """Return registered prompts with their aliases populated.

        MLflow's Prompt entity does not expose aliases directly (only name, description,
        creation_timestamp, tags). To populate aliases for the v1 list-prompts CLI we
        probe each prompt for a fixed set of known alias names (staging, production)
        and aggregate them into a SimpleNamespace with `.name` and `.aliases: dict[str, int]`.

        v2 work: surface aliases via a dedicated MLflow API if/when one ships.
        """
        from types import SimpleNamespace

        from mlflow import MlflowClient  # noqa: TID253 — adapter is the only allowed importer

        client = MlflowClient()
        result: list[Any] = []
        for p in client.search_prompts():
            aliases: dict[str, int] = {}
            for alias_name in ("staging", "production"):
                try:
                    pv = client.get_prompt_version_by_alias(p.name, alias_name)
                    aliases[alias_name] = int(pv.version)
                except Exception:  # noqa: BLE001 — missing alias is expected
                    continue
            result.append(SimpleNamespace(name=p.name, aliases=aliases))
        return result

    # --- tracing primitives (used by tracing.py) ---

    def set_run_tag(self, key: str, value: str) -> None:
        mlflow.set_tag(key, value)
