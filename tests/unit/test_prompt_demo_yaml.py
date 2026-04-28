from __future__ import annotations

from pathlib import Path

import yaml

from prompts._schema import PromptYAML

REPO = Path(__file__).resolve().parents[2]


def test_agent_demo_yaml_validates() -> None:
    data = yaml.safe_load((REPO / "prompts" / "agent_demo.yaml").read_text())
    PromptYAML(**data)  # raises on invalid


def test_filename_matches_name_field() -> None:
    p = REPO / "prompts" / "agent_demo.yaml"
    data = yaml.safe_load(p.read_text())
    assert data["name"] == p.stem
