#!/usr/bin/env bash
# scripts/restore.sh — restore from a backup directory produced by backup.sh
set -euo pipefail

IN_DIR="${1:?usage: restore.sh <backup-dir>}"
test -f "$IN_DIR/llmops.sql.gz" || { echo "missing $IN_DIR/llmops.sql.gz"; exit 1; }
test -f "$IN_DIR/artifacts.tar.gz" || { echo "missing $IN_DIR/artifacts.tar.gz"; exit 1; }

source .env

# docker-compose prefixes volume names with the project name (defaults to dir basename).
ARTIFACTS_VOL="${COMPOSE_PROJECT_NAME:-$(basename "$PWD")}_llmops_artifacts"

echo "→ stop mlflow (postgres stays up)"
docker compose stop mlflow

echo "→ drop and recreate database"
docker compose exec -T postgres psql -U "$POSTGRES_USER" -d postgres \
  -c "DROP DATABASE IF EXISTS $POSTGRES_DB;"
docker compose exec -T postgres psql -U "$POSTGRES_USER" -d postgres \
  -c "CREATE DATABASE $POSTGRES_DB;"

echo "→ restore postgres"
gunzip -c "$IN_DIR/llmops.sql.gz" | \
  docker compose exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"

echo "→ restore artifacts volume ($ARTIFACTS_VOL)"
# Resolve to absolute path so the docker -v mount works whether IN_DIR is
# relative (default) or absolute (e.g. pytest tmp_path).
ABS_IN="$(cd "$IN_DIR" && pwd)"
docker run --rm \
  -v "$ARTIFACTS_VOL":/data \
  -v "$ABS_IN":/backup:ro \
  alpine \
  sh -c "rm -rf /data/* && tar xzf /backup/artifacts.tar.gz -C /data"

echo "→ restart mlflow"
docker compose start mlflow

echo "[ok] restore complete"
