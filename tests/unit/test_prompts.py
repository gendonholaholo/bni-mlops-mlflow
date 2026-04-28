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
    mod.update_current_trace = MagicMock()  # type: ignore[attr-defined]
    mod.MlflowClient = MagicMock()  # type: ignore[attr-defined]
    genai = types.ModuleType("mlflow.genai")
    genai.register_prompt = MagicMock()  # type: ignore[attr-defined]
    genai.load_prompt = MagicMock()  # type: ignore[attr-defined]
    genai.set_prompt_alias = MagicMock()  # type: ignore[attr-defined]
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


def test_register_prompt_idempotent_when_template_unchanged(
    fake_mlflow_for_prompts: dict[str, MagicMock],
) -> None:
    """If the latest version's template equals the new template, no new version is created."""
    from llmops.prompts import register_prompt

    genai = fake_mlflow_for_prompts["genai"]
    # First call: no existing prompt → register creates v1
    genai.load_prompt.side_effect = Exception("RESOURCE_DOES_NOT_EXIST")
    genai.register_prompt.return_value = types.SimpleNamespace(name="p", version=1)
    p1 = register_prompt(name="p", template="hello {{ x }}", commit_message="init")
    assert p1.version == 1
    assert genai.register_prompt.call_count == 1

    # Second call with same template: load returns the existing one; no new register
    genai.load_prompt.side_effect = None
    genai.load_prompt.return_value = types.SimpleNamespace(
        name="p", version=1, template="hello {{ x }}"
    )
    p2 = register_prompt(name="p", template="hello {{ x }}", commit_message="noop")
    assert p2.version == 1
    assert genai.register_prompt.call_count == 1  # NOT incremented


def test_register_prompt_creates_new_when_template_changes(
    fake_mlflow_for_prompts: dict[str, MagicMock],
) -> None:
    from llmops.prompts import register_prompt

    genai = fake_mlflow_for_prompts["genai"]
    genai.load_prompt.return_value = types.SimpleNamespace(
        name="p", version=1, template="old", model_config=None
    )
    genai.register_prompt.return_value = types.SimpleNamespace(name="p", version=2)

    p = register_prompt(name="p", template="new", commit_message="update")
    assert p.version == 2
    assert genai.register_prompt.called


def test_register_prompt_creates_new_when_model_config_changes(
    fake_mlflow_for_prompts: dict[str, MagicMock],
) -> None:
    """Issue #2: idempotence considers model_config — same template but
    different generation hyperparameters MUST create a new prompt version."""
    from llmops.prompts import register_prompt

    genai = fake_mlflow_for_prompts["genai"]
    # Existing version: template "t" with temperature=0.7
    genai.load_prompt.return_value = types.SimpleNamespace(
        name="p", version=1, template="t", model_config={"temperature": 0.7}
    )
    genai.register_prompt.return_value = types.SimpleNamespace(name="p", version=2)

    p = register_prompt(
        name="p",
        template="t",  # same template
        commit_message="tune",
        model_config={"temperature": 0.3},  # different hyperparams
    )
    assert p.version == 2
    genai.register_prompt.assert_called_once()
    assert genai.register_prompt.call_args.kwargs["model_config"] == {"temperature": 0.3}


def test_register_prompt_idempotent_when_template_and_model_config_match(
    fake_mlflow_for_prompts: dict[str, MagicMock],
) -> None:
    """Same template AND same model_config → idempotent (no new version)."""
    from llmops.prompts import register_prompt

    genai = fake_mlflow_for_prompts["genai"]
    genai.load_prompt.return_value = types.SimpleNamespace(
        name="p", version=4, template="t", model_config={"temperature": 0.7, "top_k": 40}
    )

    p = register_prompt(
        name="p",
        template="t",
        model_config={"temperature": 0.7, "top_k": 40},
    )
    assert p.version == 4
    assert genai.register_prompt.call_count == 0


def test_prompt_versions_tag_written_on_outer_trace_exit(
    fake_mlflow_for_prompts: dict[str, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """load_prompt accumulates name->version into thread-local; outermost
    trace_agent serializes to JSON and writes as `llmops.prompt_versions` tag."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://x:5001")
    sys.modules.pop("llmops._config", None)
    sys.modules.pop("llmops.tracing", None)
    sys.modules.pop("llmops.prompts", None)

    # tracing.py uses MlflowClient via the fake mlflow module; ensure MlflowClient
    # is wired so __enter__/__exit__ + the prompt-versions flush all hit the same client
    client_mock = MagicMock()
    client_mock.start_trace.return_value = types.SimpleNamespace(trace_id="r1", span_id="s1")
    fake_mlflow_for_prompts["mlflow"].MlflowClient = MagicMock(return_value=client_mock)

    from llmops.prompts import load_prompt
    from llmops.tracing import trace_agent

    genai = fake_mlflow_for_prompts["genai"]
    genai.load_prompt.side_effect = [
        types.SimpleNamespace(name="agent_tujuan", version=2, template="t"),
        types.SimpleNamespace(name="agent_rilis", version=5, template="t"),
    ]

    with trace_agent("orch"):
        with trace_agent("agent_tujuan"):
            load_prompt("agent_tujuan@production")
        with trace_agent("agent_rilis"):
            load_prompt("agent_rilis@production")

    # End-of-outer-trace must have written llmops.prompt_versions via mlflow.set_tag
    set_tag = fake_mlflow_for_prompts["mlflow"].set_tag
    calls = [c for c in set_tag.call_args_list if c.args and c.args[0] == "llmops.prompt_versions"]
    assert len(calls) == 1, (
        f"expected 1 prompt_versions tag, got {len(calls)}: {set_tag.call_args_list}"
    )
    payload = calls[0].args[1]
    import json as _j

    parsed = _j.loads(payload)
    assert parsed == {"agent_tujuan": 2, "agent_rilis": 5}


def test_set_alias_writes_audit_tags(
    fake_mlflow_for_prompts: dict[str, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """set_alias updates the alias pointer AND writes audit tags via
    MlflowClient.set_prompt_version_tag for each audit key."""
    monkeypatch.setenv("GITHUB_ACTOR", "alice")
    monkeypatch.setenv("GITHUB_SHA", "deadbeef")

    # Reroute MlflowClient() to a fixed mock so we can verify tag writes
    client_mock = MagicMock()
    fake_mlflow_for_prompts["mlflow"].MlflowClient = MagicMock(return_value=client_mock)

    from llmops.prompts import set_alias

    set_alias("agent_tujuan", alias="production", version=3, from_alias="staging")

    # The alias pointer was moved on the root mlflow namespace
    fake_mlflow_for_prompts["genai"].set_prompt_alias.assert_called_once_with(
        "agent_tujuan", "production", 3
    )

    # Audit tags were written one-per-key via set_prompt_version_tag(name, version, key, value)
    tag_calls = client_mock.set_prompt_version_tag.call_args_list
    assert len(tag_calls) >= 4, (
        f"expected at least 4 tag writes, got {len(tag_calls)}: {tag_calls!r}"
    )

    written: dict[str, str] = {}
    for call in tag_calls:
        # Accept positional or kwarg form
        if len(call.args) == 4:
            name, version, key, value = call.args
        else:
            name = call.kwargs.get("name", call.args[0] if call.args else None)
            version = call.kwargs.get("version", call.args[1] if len(call.args) > 1 else None)
            key = call.kwargs.get("key", call.args[2] if len(call.args) > 2 else None)
            value = call.kwargs.get("value", call.args[3] if len(call.args) > 3 else None)
        assert name == "agent_tujuan"
        assert version == 3
        written[key] = value

    assert written.get("promoted_to_alias") == "production"
    assert written.get("promoted_from_alias") == "staging"
    assert "promoted_at" in written and written["promoted_at"]  # ISO timestamp non-empty
    assert written.get("promoted_by") == "alice"
    assert written.get("promoted_git_sha") == "deadbeef"
