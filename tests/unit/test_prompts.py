from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from llmops.exceptions import LLMOpsPromptNotFoundError


@pytest.fixture
def fake_mlflow_for_prompts(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    mod = types.ModuleType("mlflow")
    mod.set_tracking_uri = MagicMock()  # type: ignore[attr-defined]
    mod.set_experiment = MagicMock()  # type: ignore[attr-defined]
    mod.set_tag = MagicMock()  # type: ignore[attr-defined]
    mod.set_prompt_alias = MagicMock()  # type: ignore[attr-defined]
    mod.MlflowClient = MagicMock()  # type: ignore[attr-defined]
    genai = types.ModuleType("mlflow.genai")
    genai.register_prompt = MagicMock()  # type: ignore[attr-defined]
    genai.load_prompt = MagicMock()  # type: ignore[attr-defined]
    mod.genai = genai  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "mlflow", mod)
    monkeypatch.setitem(sys.modules, "mlflow.genai", genai)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://x:5001")

    for name in ("llmops._config", "llmops._mlflow_adapter", "llmops.prompts"):
        sys.modules.pop(name, None)
    return {"mlflow": mod, "genai": genai}


def test_load_prompt_by_alias(fake_mlflow_for_prompts: dict[str, MagicMock]) -> None:
    from llmops.prompts import load_prompt

    fake_mlflow_for_prompts["genai"].load_prompt.return_value = types.SimpleNamespace(
        name="p", version=2, template="hi {{ name }}"
    )

    p = load_prompt("p@staging")
    assert p.template == "hi {{ name }}"
    fake_mlflow_for_prompts["genai"].load_prompt.assert_called_once_with(
        name_or_uri="prompts:/p@staging"
    )


def test_load_prompt_by_version(fake_mlflow_for_prompts: dict[str, MagicMock]) -> None:
    from llmops.prompts import load_prompt

    fake_mlflow_for_prompts["genai"].load_prompt.return_value = types.SimpleNamespace(
        name="p", version=3, template="x"
    )

    load_prompt("p/3")
    fake_mlflow_for_prompts["genai"].load_prompt.assert_called_once_with(name_or_uri="prompts:/p/3")


def test_load_prompt_missing_raises_typed_error(
    fake_mlflow_for_prompts: dict[str, MagicMock],
) -> None:
    from llmops.prompts import load_prompt

    fake_mlflow_for_prompts["genai"].load_prompt.side_effect = Exception(
        "RestException: RESOURCE_DOES_NOT_EXIST"
    )

    with pytest.raises(LLMOpsPromptNotFoundError):
        load_prompt("missing@production")
