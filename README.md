# bni-llmops

LLM Ops SDK for BNI — framework-agnostic tracing + prompt registry on top of MLflow.

## What it does

- **Tracing**: every LLM call, agent step, orchestrator hand-off captured with input, output, latency, exceptions, and parent-child span structure. View in MLflow UI.
- **Prompt registry**: prompts stored as YAML in Git → registered automatically to MLflow on merge → `staging`/`production` aliases for promotion.
- **Standalone**: one `docker compose up` brings up MLflow + Postgres. SDK is `import llmops` and goes.

## Quickstart (5 minutes)

```bash
# 1. Bring up the tracking stack
git clone <this-repo>
cd ml-ops-test
cp .env.example .env
docker compose up -d --wait

# 2. From your project (separate terminal):
uv add git+ssh://git@github.com/<org>/bni-llmops@v0.1.0
export MLFLOW_TRACKING_URI=http://localhost:5001
```

```python
# 3. Use it:
import llmops

@llmops.trace_agent("agent_tujuan")
def run_tujuan(pain_point: str) -> str:
    prompt = llmops.load_prompt("agent_tujuan@production")
    # ... call your LLM with prompt.format(...)
    return "result"

run_tujuan("user can't find report X")
```

Open `http://localhost:5001` and you'll see the trace with span hierarchy.

## Recipes

- `docs/recipes/openai.md` — OpenAI SDK
- `docs/recipes/langchain.md` — LangChain (uses `mlflow.langchain.autolog()`)
- `docs/recipes/custom.md` — bring your own orchestration

## Architecture

See `docs/architecture.md` for the high-level diagram.

## Spec

Full design: `docs/superpowers/specs/2026-04-28-bni-llmops-design.md`.

## Operations

- Backup: `bash scripts/backup.sh ./backups/$(date -u +%Y%m%dT%H%M%SZ)`
- Restore: `bash scripts/restore.sh <backup-dir>`
- Doctor: `uv run llmops doctor`
- Logs: `docker compose logs -f mlflow`

## License

Internal (BNI).
