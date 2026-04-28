# Recipe: LangChain

This recipe uses `mlflow.langchain.autolog()` to capture every LangChain run automatically, plus `llmops.trace_agent` to add agent-level structure on top.

## Setup

```bash
uv add langchain langchain-openai
export OPENAI_API_KEY=sk-...
export MLFLOW_TRACKING_URI=http://localhost:5001
```

## Code

```python
import llmops
import mlflow.langchain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

mlflow.langchain.autolog()  # capture every LangChain invoke / stream / batch

@llmops.trace_agent("agent_tujuan")
def identify_objective(pain_point: str) -> str:
    prompt_record = llmops.load_prompt("agent_tujuan@production")

    # Build an LCEL chain — autolog will capture each step
    template = ChatPromptTemplate.from_template(prompt_record.template)
    chain = template | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()

    return chain.invoke({"pain_point": pain_point})
```

In MLflow UI you'll see:
- `agent_tujuan` outer span (your orchestrator step)
- inside: `RunnableSequence` span (auto-logged) with the prompt template, model call, parser steps as nested spans

## Notes

- `mlflow.langchain.autolog()` is idempotent.
- The autolog spans nest cleanly under `trace_agent` — your agent boundary is the outer scope; LangChain internals are the children.
- For chains that fan out (e.g., `RunnableParallel`), each branch shows up as a sibling span with its own latency.
