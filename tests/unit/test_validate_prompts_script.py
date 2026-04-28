from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_validate_passes_on_demo() -> None:
    r = subprocess.run(
        [sys.executable, "scripts/validate_prompts.py", "prompts/"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr


def test_validate_fails_on_invalid(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "schema_version: 1\nname: BAD_NAME\ndescription: short\n"
        "template: 'hi'\nvariables: []\ntags: {}\n"
    )
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "validate_prompts.py"), str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
    assert "BAD_NAME" in (r.stdout + r.stderr)
