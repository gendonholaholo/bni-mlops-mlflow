"""Smoke test: docker compose up brings stack to healthy state and MLflow /health responds."""
from __future__ import annotations

import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _docker_available() -> bool:
    return shutil.which("docker") is not None


@pytest.mark.skipif(not _docker_available(), reason="docker CLI not available")
@pytest.mark.skipif(os.environ.get("SKIP_COMPOSE_SMOKE") == "1", reason="explicitly skipped")
def test_compose_up_health() -> None:
    """`docker compose up -d` brings stack healthy; /health returns 200."""
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        shutil.copy(REPO_ROOT / ".env.example", env_file)

    subprocess.run(
        ["docker", "compose", "up", "-d", "--wait"],
        cwd=REPO_ROOT,
        check=True,
        timeout=180,
    )

    try:
        # Health must respond within 60s
        deadline = time.time() + 60
        last_err: Exception | None = None
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(
                    "http://localhost:5001/health", timeout=3
                ) as resp:
                    body = resp.read().decode().strip()
                    assert resp.status == 200
                    assert body == "OK"
                    return
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2)
        pytest.fail(f"/health did not return 200/OK within 60s: {last_err!r}")
    finally:
        subprocess.run(
            ["docker", "compose", "down"], cwd=REPO_ROOT, check=False, timeout=60
        )
