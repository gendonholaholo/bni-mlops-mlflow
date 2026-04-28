#!/usr/bin/env bash
# scripts/backup.sh — dump postgres + tar artifacts volume
set -euo pipefail

OUT_DIR="${1:-./backups/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT_DIR"

source .env

# docker-compose prefixes volume names with the project name (defaults to dir basename).
# COMPOSE_PROJECT_NAME env var overrides; fall back to the dir basename.
ARTIFACTS_VOL="${COMPOSE_PROJECT_NAME:-$(basename "$PWD")}_llmops_artifacts"

echo "→ pg_dump"
docker compose exec -T postgres pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  | gzip > "$OUT_DIR/llmops.sql.gz"

echo "→ tar artifacts volume ($ARTIFACTS_VOL)"
# Resolve to absolute path so the docker -v mount works whether OUT_DIR is
# relative (default) or absolute (e.g. pytest tmp_path).
ABS_OUT="$(cd "$OUT_DIR" && pwd)"
docker run --rm \
  -v "$ARTIFACTS_VOL":/data:ro \
  -v "$ABS_OUT":/backup \
  alpine \
  tar czf /backup/artifacts.tar.gz -C /data .

echo "[ok] backup written to $OUT_DIR"
