"""Round-trip: write a row -> backup -> wipe -> restore -> verify row still present."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(not shutil.which("docker"), reason="docker not available")
def test_backup_restore_round_trip(tmp_path: Path) -> None:
    # Bring stack up
    if not (REPO / ".env").exists():
        shutil.copy(REPO / ".env.example", REPO / ".env")
    subprocess.run(["docker", "compose", "up", "-d", "--wait"], cwd=REPO, check=True, timeout=180)
    try:
        # 1. Write a marker
        subprocess.run(
            [
                "docker",
                "compose",
                "exec",
                "-T",
                "postgres",
                "psql",
                "-U",
                "llmops",
                "-d",
                "llmops",
                "-c",
                "DROP TABLE IF EXISTS backup_marker;"
                " CREATE TABLE backup_marker (note text);"
                " INSERT INTO backup_marker VALUES ('round-trip');",
            ],
            cwd=REPO,
            check=True,
            timeout=30,
        )

        # 2. Backup
        out = tmp_path / "bk"
        subprocess.run(["bash", "scripts/backup.sh", str(out)], cwd=REPO, check=True, timeout=120)
        assert (out / "llmops.sql.gz").exists()
        assert (out / "artifacts.tar.gz").exists()

        # 3. Wipe marker
        subprocess.run(
            [
                "docker",
                "compose",
                "exec",
                "-T",
                "postgres",
                "psql",
                "-U",
                "llmops",
                "-d",
                "llmops",
                "-c",
                "DROP TABLE backup_marker;",
            ],
            cwd=REPO,
            check=True,
            timeout=30,
        )

        # 4. Restore
        subprocess.run(["bash", "scripts/restore.sh", str(out)], cwd=REPO, check=True, timeout=120)

        # 5. Verify marker present again
        r = subprocess.run(
            [
                "docker",
                "compose",
                "exec",
                "-T",
                "postgres",
                "psql",
                "-U",
                "llmops",
                "-d",
                "llmops",
                "-tA",
                "-c",
                "SELECT note FROM backup_marker;",
            ],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert "round-trip" in r.stdout
    finally:
        subprocess.run(["docker", "compose", "down"], cwd=REPO, check=False, timeout=60)
