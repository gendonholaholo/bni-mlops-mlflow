# Recipe: OpenAI SDK

This recipe uses `mlflow.openai.autolog()` to capture every OpenAI call automatically, plus `llmops.trace_agent` to add agent-level structure.

## Setup

```bash
uv add openai
export OPENAI_API_KEY=sk-...
export MLFLOW_TRACKING_URI=http://localhost:5001
```

## Code

```python
import llmops
import mlflow.openai
import openai

mlflow.openai.autolog()  # capture every chat.completions.create call

client = openai.OpenAI()

@llmops.trace_agent("agent_tujuan")
def identify_objective(pain_point: str) -> str:
    prompt = llmops.load_prompt("agent_tujuan@production")
    text = prompt.format(pain_point=pain_point)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": text}],
    )
    return resp.choices[0].message.content
```

In MLflow UI you'll see:
- `agent_tujuan` outer span (your orchestrator step)
- inside: `chat.completions.create` LLM span (auto-logged) with model, tokens, latency

## Notes

- `autolog()` is idempotent — calling it twice is safe.
- It doesn't override your `trace_agent` boundaries; both layers nest cleanly.
