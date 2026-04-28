"""Pydantic v2 model for prompt YAML files. Validation rules per spec Appendix B."""

from __future__ import annotations

import re
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_VAR_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


class PromptYAML(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: int = Field(..., ge=1, le=1)
    name: str
    description: str = Field(..., min_length=10)
    template: str
    variables: list[str]
    tags: dict[str, str] = Field(default_factory=dict)

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
        return self
