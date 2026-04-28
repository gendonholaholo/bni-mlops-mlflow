# bni-llmops

LLM tracing + prompt registry SDK on top of MLflow. Drop-in for any Python agentic codebase.

## Install

```bash
uv add "git+ssh://git@github.com/gendonholaholo/bni-mlops-mlflow.git@v0.1.0"
export MLFLOW_TRACKING_URI=http://localhost:5001
```

## Use

```python
import llmops

@llmops.trace_agent("agent_tujuan")
def run(question: str) -> str:
    prompt = llmops.load_prompt("agent_demo@production")
    return your_llm(prompt.format(question=question))

run("user can't find report X")
```

Trace appears at `$MLFLOW_TRACKING_URI` within seconds.

## API

| Symbol | Purpose |
|---|---|
| `@llmops.trace_agent(name)` / `with llmops.trace_agent(name):` | wrap a function or block in a span |
| `llmops.load_prompt("name@alias")` | load by alias (or `"name/version"`) — returns object with `.template` and `.format(**vars)` |
| `llmops.register_prompt(name, template, ...)` | register a new version (idempotent) |
| `llmops.set_alias(name, alias, version, from_alias=...)` | move alias; writes audit tags |
| `llmops.LLMOpsError` (and subclasses) | catch SDK-level errors |

## Environment

| Var | Purpose |
|---|---|
| `MLFLOW_TRACKING_URI` | **required** — your MLflow server URL |
| `LLMOPS_EXPERIMENT_NAME` | optional — default `bni-agentic-prd` |
| `LLMOPS_DISABLE_TRACING` | optional — `true` makes every `trace_agent` a no-op |

## Self-host the tracking stack

If no MLflow server is running yet:

```bash
git clone git@github.com:gendonholaholo/bni-mlops-mlflow.git
cd bni-mlops-mlflow
cp .env.example .env
docker compose up -d --wait
```

UI: <http://localhost:5001>.

---

More: `docs/recipes/{openai,langchain,custom}.md` · `docs/architecture.md` · `docs/troubleshooting.md`
