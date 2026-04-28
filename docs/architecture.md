# Architecture

```mermaid
flowchart LR
  Caller["Colleague's agentic system\n(orchestrator + child agents + RAG)"]
  SDK["bni-llmops SDK\n• trace_agent\n• load_prompt / register_prompt\n• set_alias"]
  MLflow["MLflow Tracking Server\n:5001 (UI + REST)\n--serve-artifacts"]
  PG[(Postgres 18)]
  Vol[(Local artifact volume)]
  Caller --> SDK
  SDK -- "HTTP (MLflow REST)" --> MLflow
  MLflow --> PG
  MLflow --> Vol

  Git["prompts/*.yaml\n(SSoT in Git)"]
  GHA["GitHub Actions\n(merge to gos-dev)"]
  Git --> GHA
  GHA -- "register_prompt + set_alias" --> MLflow
```

## Coupling rule

Only `src/llmops/_mlflow_adapter.py` imports `mlflow`. All other SDK modules go through the adapter. Enforced via ruff TID253 in CI.

## Why this shape

- **Single source of truth for MLflow coupling** — when MLflow's API changes (or we swap providers), we change one file.
- **Easy testability** — every other module can be unit-tested with the adapter mocked; no MLflow live calls in unit tests.
- **Easy framework extensibility** — the adapter pattern lets us add a Langfuse/Phoenix backend later without rewriting the SDK.
