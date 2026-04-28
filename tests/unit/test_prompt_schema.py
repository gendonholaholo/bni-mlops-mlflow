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


def test_schema_version_must_be_1() -> None:
    bad = _good() | {"schema_version": 2}
    with pytest.raises(Exception):  # noqa: B017
        PromptYAML(**bad)


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
