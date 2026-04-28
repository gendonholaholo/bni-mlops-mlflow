from __future__ import annotations

import pytest

from prompts._schema import PromptYAML


def _good() -> dict:
    return {
        "schema_version": 1,
        "name": "agent_demo",
        "description": "ten-char-or-more description here",
        "template": "Hello {{ name }}, your input was: {{ input }}",
        "variables": ["name", "input"],
        "tags": {"domain": "demo"},
    }


def test_good_yaml_parses() -> None:
    p = PromptYAML(**_good())
    assert p.name == "agent_demo"


def test_schema_version_2_accepted_for_model_config() -> None:
    """Schema bumped to 2 (Issue #2) to allow optional model_config block."""
    p = PromptYAML(
        **(_good() | {"schema_version": 2, "model_config": {"temperature": 0.7, "top_k": 40}})
    )
    assert p.mlflow_model_config == {"temperature": 0.7, "top_k": 40}


def test_schema_version_3_rejected() -> None:
    bad = _good() | {"schema_version": 3}
    with pytest.raises(Exception):  # noqa: B017
        PromptYAML(**bad)


def test_model_config_requires_schema_version_2() -> None:
    """Setting model_config under schema_version: 1 must fail with a clear error."""
    bad = _good() | {"model_config": {"temperature": 0.7}}
    with pytest.raises(Exception, match="schema_version >= 2"):
        PromptYAML(**bad)


def test_v1_without_model_config_still_works() -> None:
    """Backward-compat: existing v1 prompts (no model_config) keep parsing."""
    p = PromptYAML(**_good())
    assert p.schema_version == 1
    assert p.mlflow_model_config is None


def test_name_regex_enforced() -> None:
    bad = _good() | {"name": "Agent_Tujuan"}  # uppercase forbidden
    with pytest.raises(Exception):  # noqa: B017
        PromptYAML(**bad)


def test_unused_variable_rejected() -> None:
    """Every entry in `variables` must appear in template."""
    bad = _good() | {"variables": ["name", "input", "extra"]}
    with pytest.raises(Exception, match="extra"):
        PromptYAML(**bad)


def test_undeclared_variable_in_template_rejected() -> None:
    bad = _good() | {"template": "Hello {{ name }}, {{ ghost }}"}
    with pytest.raises(Exception, match="ghost"):
        PromptYAML(**bad)


def test_short_description_rejected() -> None:
    bad = _good() | {"description": "tiny"}
    with pytest.raises(Exception):  # noqa: B017
        PromptYAML(**bad)


def test_tags_must_be_flat_strings() -> None:
    bad = _good() | {"tags": {"k": {"nested": "x"}}}
    with pytest.raises(Exception):  # noqa: B017
        PromptYAML(**bad)
