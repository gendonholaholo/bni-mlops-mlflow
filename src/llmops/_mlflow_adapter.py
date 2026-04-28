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
    ) -> _PromptObj:
        return mlflow.genai.register_prompt(
            name=name,
            template=template,
            commit_message=commit_message,
            tags=tags,
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

    # --- tracing primitives (used by tracing.py) ---

    def set_run_tag(self, key: str, value: str) -> None:
        mlflow.set_tag(key, value)
