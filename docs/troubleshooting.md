# Troubleshooting

## "MLFLOW_TRACKING_URI is required"
Your env doesn't have it. `export MLFLOW_TRACKING_URI=http://localhost:5001` (or your server URL).

## docker compose unhealthy
`docker compose logs mlflow` and `docker compose logs postgres`. Common causes:
- `.env` not present → `cp .env.example .env`.
- Port 5001 already in use → `lsof -i:5001`.
- Postgres data corrupted → `docker compose down -v` (warning: wipes everything).

## "Prompt not found" when calling load_prompt
- Check alias exists: `uv run llmops list-prompts`.
- Confirm `MLFLOW_TRACKING_URI` matches the server you registered against.

## CI register-prompts step fails with "ban TID253"
You added `import mlflow` outside `_mlflow_adapter.py`. Move the import or call into the adapter.

## uv sync fails on Python 3.14.4
A transitive dep doesn't have a cp314 wheel yet. See spec Section 9 mitigation steps.

## Backup / Restore
`bash scripts/backup.sh ./backups/<dir>` and `bash scripts/restore.sh ./backups/<dir>`.

## Logs
- MLflow: `docker compose logs -f mlflow`
- Postgres: `docker compose logs -f postgres`
