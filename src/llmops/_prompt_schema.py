"""Pydantic v2 model for prompt YAML files. Validation rules per spec Appendix B.

Schema versions:
    1: original — name, description, template, variables, tags
    2: + model_config (Issue #2) — optional dict[str, Any] of generation
       hyperparameters (temperature, top_k, num_ctx, model_name, etc.). Maps
       1:1 to ``mlflow.genai.register_prompt(model_config=...)`` (native MLflow
       3.8+ field, exposed back as ``prompt.model_config`` after load_prompt).

The ``mlflow_model_config`` field name is internal-only — its YAML alias is
``model_config``. The rename is forced by Pydantic v2's reservation of the
``model_config`` class attribute for ConfigDict.
"""

from __future__ import annotations

import re
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_VAR_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


class PromptYAML(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        populate_by_name=True,
    )

    schema_version: int = Field(..., ge=1, le=2)
    name: str
    description: str = Field(..., min_length=10)
    template: str
    variables: list[str]
    tags: dict[str, str] = Field(default_factory=dict)
    mlflow_model_config: dict[str, Any] | None = Field(default=None, alias="model_config")

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if not _NAME_RE.match(self.name):
            raise ValueError(f"name {self.name!r} must match {_NAME_RE.pattern}")

        in_template = set(_VAR_RE.findall(self.template))
        declared = set(self.variables)

        missing = in_template - declared
        if missing:
            raise ValueError(
                f"template references undeclared variables: {sorted(missing)} "
                f"(declare them in `variables:`)"
            )
        unused = declared - in_template
        if unused:
            raise ValueError(
                f"`variables:` declares unused entries: {sorted(unused)} "
                f"(every variable must appear as {{{{ var }}}} in the template)"
            )

        for k, v in self.tags.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise ValueError("tags must be flat dict[str, str]")

        if self.mlflow_model_config is not None and self.schema_version < 2:
            raise ValueError(
                "`model_config` requires schema_version >= 2 "
                "(bump `schema_version: 2` to use it)"
            )
        return self
