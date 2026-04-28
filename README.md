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

## Observability primitives

### Group traces by session and user

```python
with llmops.trace_agent("chat_turn"):
    llmops.set_trace_metadata(session_id="conv-001", user_id="alice")
    reply = your_llm(...)
```

Traces with the same `session_id` group under one row in the MLflow Sessions tab and are queryable via `mlflow.search_traces(filter_string='metadata."mlflow.trace.session" = \'conv-001\'')`.

### Tag traces for filtering

```python
with llmops.trace_agent("agent_rilis"):
    llmops.set_trace_tags(env="prod", team="prd-pipeline", retry_count=3)
```

Values are coerced to `str` at the SDK boundary. Filter via `tags."env" = 'prod'`.

### Log hyperparameters at runtime

```python
with llmops.trace_agent("chatbot"):
    llmops.log_hyperparams(model="qwen2.5:7b", temperature=0.7, top_k=40)
    ollama.chat(...)
```

Values land on the active span as `llmops.hp.<key>` attributes — visible in the trace detail view.

### Type your spans

```python
from llmops import SpanType

@llmops.trace_agent("retriever_step", span_type=SpanType.RETRIEVER)
def fetch_docs(q: str) -> list[str]: ...
```

Constants mirror MLflow 3.11.x's `SpanType` enum (15 values: `LLM`, `AGENT`, `CHAT_MODEL`, `RETRIEVER`, `TOOL`, `CHAIN`, `EMBEDDING`, `RERANKER`, `PARSER`, `MEMORY`, `WORKFLOW`, `TASK`, `GUARDRAIL`, `EVALUATOR`, `UNKNOWN`).

### Capture outputs

```python
# As decorator — return value is auto-captured
@llmops.trace_agent("agent_tujuan")
def run(q: str) -> dict: return {"answer": ...}

# As context manager — call set_outputs explicitly
with llmops.trace_agent("agent_rilis") as t:
    result = your_llm(...)
    t.set_outputs(result)
```

### Version generation hyperparameters with the prompt

```python
llmops.register_prompt(
    name="agent_demo",
    template="answer the user: {{question}}",
    model_config={"model": "qwen2.5:7b", "temperature": 0.7, "num_ctx": 4096},
)
```

`model_config` lands on the prompt version itself, so changes to generation params bump the prompt version alongside the template.

### Autolog third-party SDKs

```python
llmops.autolog("openai")        # mlflow.openai.autolog
llmops.autolog("anthropic")     # mlflow.anthropic.autolog
llmops.autolog("langchain", log_input_examples=True)
```

Passes through to `mlflow.<provider>.autolog(**kwargs)` for all 17 supported providers (`llmops.SUPPORTED_PROVIDERS`).

## API

| Symbol | Purpose |
|---|---|
| `@llmops.trace_agent(name, span_type=None, **attrs)` / `with llmops.trace_agent(...) as t:` | wrap a function or block in a span; `t.set_outputs(value)` records the output |
| `llmops.SpanType.{AGENT,LLM,RETRIEVER,...}` | canonical span-type string constants |
| `llmops.set_trace_metadata(session_id=, user_id=)` | populate `TraceInfo.request_metadata` for Sessions tab grouping |
| `llmops.set_trace_tags(**tags)` | populate `TraceInfo.tags` for `search_traces` filtering |
| `llmops.log_hyperparams(**hp)` | record runtime hyperparams as `llmops.hp.*` span attributes |
| `llmops.load_prompt("name@alias")` | load by alias (or `"name/version"`) — returns object with `.template` and `.format(**vars)` |
| `llmops.register_prompt(name, template, model_config=None, ...)` | register a new version (idempotent); `model_config` versions generation hyperparams |
| `llmops.set_alias(name, alias, version, from_alias=...)` | move alias; writes audit tags |
| `llmops.autolog(provider, **kwargs)` | passthrough to `mlflow.<provider>.autolog(**kwargs)` |
| `llmops.SUPPORTED_PROVIDERS` | frozenset of 17 supported autolog provider names |
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

## Roadmap

### Shipped — Tier 1 (current release)

Foundation for production observability + prompt versioning. Resolves #1, #2, #3.

- `trace_agent` context manager + decorator with adapter init guarantee
- `SpanType` constants (15 values) for typed spans
- `set_trace_metadata(session_id=, user_id=)` — populates `TraceInfo.request_metadata` for the MLflow Sessions tab and `search_traces` filtering
- `set_trace_tags(**tags)` — populates `TraceInfo.tags` with str-coerced values
- `log_hyperparams(**hp)` — runtime hyperparams as `llmops.hp.*` span attributes
- `set_outputs(value)` / decorator auto-capture
- `register_prompt(model_config=...)` — generation hyperparams versioned with the template
- `autolog(provider, **kwargs)` — passthrough for 17 providers

### Tier 2 — next (driver: agentic PRD pipeline)

Priority-ordered. Each item is a thin wrapper over `mlflow.genai.*`; preserves the SDK's banned-module-rule architecture.

| Priority | Feature | Why |
|---|---|---|
| 1 | `response_format` field in prompt YAML | PRD pipeline needs structured output enforcement (fixed sections: Tujuan / Rilis / Fitur / User Flow / Analytics). MLflow `register_prompt(response_format=...)` accepts Pydantic class or JSON schema. |
| 2 | Chat-format prompts (`template:` accepts `list[dict]`) | PRD agents and most production chatbots use `[{role, content}, ...]` not single template strings. Already supported by MLflow natively. |
| 3 | `llmops.evaluate(...)` thin wrapper | PRD regression testing using built-in scorers (Correctness, Guidelines, RetrievalRelevance) and custom judges via `make_judge`. |
| 4 | `llmops.log_feedback(trace_id, value, rationale=)` | Human-review loop for the PRD generation pipeline. |

Possible add-on (deferred unless performance demands it): parallel sub-agent context propagation via `mlflow.tracing.context` — only relevant if PRD sub-agents fan out via threads/asyncio (current span stack is `threading.local`).

### Tier 3 — intentionally out-of-scope

These MLflow features are useful but heavy and largely duplicate `mlflow.genai.*` directly. Recommendation: consumers use raw `mlflow` for these rather than expanding the SDK surface.

- Datasets API (`mlflow.genai.datasets`)
- `optimize_prompt` (DSPy-based prompt optimization)
- `ConversationSimulator`
- Labeling sessions

---

More: `docs/recipes/{openai,langchain,custom}.md` · `docs/architecture.md` · `docs/troubleshooting.md`
