# Recipe: Custom Orchestration

If you don't use OpenAI's SDK or LangChain (or you do, but want explicit control), use `llmops.trace_agent` directly to draw your span boundaries.

## Setup

```bash
export MLFLOW_TRACKING_URI=http://localhost:5001
# Plus whatever LLM SDK / HTTP client you're using
```

## Pattern: nested agents

```python
import llmops

@llmops.trace_agent("orchestrator")
def run_pipeline(pain_point: str) -> str:
    objective = identify_objective(pain_point)
    plan = plan_release(objective)
    return plan

@llmops.trace_agent("agent_tujuan")
def identify_objective(pain_point: str) -> str:
    prompt = llmops.load_prompt("agent_tujuan@production")
    return call_my_llm(prompt.format(pain_point=pain_point))

@llmops.trace_agent("agent_rilis")
def plan_release(objective: str) -> str:
    prompt = llmops.load_prompt("agent_rilis@production")
    return call_my_llm(prompt.format(objective=objective))

def call_my_llm(text: str) -> str:
    # your HTTP call, your custom client, anything
    ...
```

In MLflow UI you'll see a single trace rooted at `orchestrator`, with `agent_tujuan` and `agent_rilis` as siblings underneath. Every span carries its inputs, outputs, latency, and (if it raised) the exception status.

## Pattern: context manager (when decorator doesn't fit)

```python
import llmops

def step_with_branching(x):
    with llmops.trace_agent("agent_x", input_size=len(x)):
        if some_condition(x):
            with llmops.trace_agent("agent_x.branch_a"):
                return handle_a(x)
        else:
            with llmops.trace_agent("agent_x.branch_b"):
                return handle_b(x)
```

## Notes

- Exceptions inside a `with trace_agent(...)` block are captured with `status=ERROR` on the span, then re-raised — your error handling is undisturbed.
- Set `LLMOPS_DISABLE_TRACING=true` to make every `trace_agent` a no-op (useful for local dev without a tracking server).
- Loaded prompts are auto-recorded; the outermost `trace_agent` flushes a `llmops.prompt_versions` tag with the full `name -> version` map.
