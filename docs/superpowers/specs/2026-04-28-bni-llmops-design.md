# BNI LLM Ops — Design Specification

| Field | Value |
|---|---|
| **Date** | 2026-04-28 |
| **Author** | Gos Shafadonia |
| **Status** | Draft (pending review) |
| **Scope version** | v1.0 |
| **Branch** | `feat/llmops-foundation` (from `gos-dev`) |

---

## 1. Context

A colleague at BNI is building an **agentic LLM system** that generates Product Requirement Documents (PRDs). The system has an orchestrator agent that delegates to specialized child agents (Tujuan, Rilis, Fitur, etc.), with RAG over PDF/Excel/Word pain-point documents. The orchestrator synthesizes child outputs and post-processes into a final PRD.

This specification covers a **separate, standalone LLM Ops service** that the colleague consumes to instrument the agentic system. It is **not** the agentic system itself.

**Stakeholders:**
- **Primary user**: the colleague (single Python developer).
- **Secondary observers**: PM and team leads (read-only via MLflow UI).
- **Future users**: any other BNI engineer building LLM applications (must remain framework-agnostic).

---

## 2. Goals & Non-Goals

### 2.1 Goals (v1)

1. **Trace every agent step** — orchestrator, child agents, LLM calls — with input, output, latency, errors, parent-child span structure.
2. **Version every prompt** with Git as source of truth, MLflow as runtime registry, alias-based environment promotion (`staging` → `production`).
3. **Run as a standalone Docker service** with one-command spin-up.
4. **Onboarding in ≤30 minutes** for any colleague new to the codebase.
5. **Framework-agnostic SDK** that works with raw OpenAI SDK, Anthropic SDK, LangChain, LlamaIndex, or custom orchestration — no lock-in.
6. **GitHub Actions CI** that gates prompt promotion lifecycle.

### 2.2 Non-Goals (deferred to v2 with explicit triggers)

| Item | Why deferred | Trigger to add |
|---|---|---|
| LLM-as-judge evaluation harness | Needs real trace samples to curate golden datasets; building eval in vacuum produces noise | ≥50 real trace runs collected from colleague's runtime |
| Authentication (SSO / token) | Localhost / Docker-network deployment is sufficient for single team | Deployed to a shared server |
| MinIO / S3 artifact store | Local volume + `--serve-artifacts` proxy adequate for ≤5 users / ≤50GB | Team grows beyond that, or artifacts exceed disk |
| Gradio UI | SDK already covers "act"; MLflow UI covers "observe"; no overlap (per tool-boundaries principle) | Explicit non-technical operator request |
| MLflow Deployments Server (gateway) | Not needed for tracking + prompts; unnecessary complexity | ≥2 LLM provider abstractions to manage |
| Cost tracking | Meaningful only when paired with eval | Bundled with v2 eval rollout |
| Drift / regression detection | Requires baseline metrics from eval | After eval has been running ≥2 weeks |
| PyPI publish | Internal use only | Open-source decision |

---

## 3. High-Level Architecture

```
                      ┌──────────────────────────────────────┐
                      │   Colleague's agentic system (Python) │
                      │   (orchestrator + child agents + RAG) │
                      └──────────────┬───────────────────────┘
                                     │ uses
                                     ▼
                      ┌──────────────────────────────────────┐
                      │   bni-llmops SDK (this project)       │
                      │   • trace_agent (decorator + ctxmgr)  │
                      │   • load_prompt / register_prompt     │
                      │   • set_alias                         │
                      └──────────────┬───────────────────────┘
                                     │ HTTP (MLflow REST)
                                     ▼
        ┌────────────────────────────────────────────────────┐
        │   Docker Compose stack                              │
        │   ┌────────────────────────┐  ┌──────────────────┐ │
        │   │ MLflow Tracking Server │──│  Postgres        │ │
        │   │ :5001 (UI + REST)      │  │  (backend store) │ │
        │   │ --serve-artifacts      │  └──────────────────┘ │
        │   └────────┬───────────────┘                       │
        │            │ writes                                 │
        │            ▼                                        │
        │   ┌────────────────────────┐                       │
        │   │ Local volume           │                       │
        │   │ (artifact store)       │                       │
        │   └────────────────────────┘                       │
        └────────────────────────────────────────────────────┘

        ┌─── Git repo (this project) ───────────────────────┐
        │   prompts/*.yaml  ── source of truth for prompts  │
        │           │                                        │
        │           ▼ on push to gos-dev                     │
        │   GitHub Actions ── register_prompt + set_alias   │
        │                     into MLflow Tracking Server    │
        └────────────────────────────────────────────────────┘
```

**Coupling rule (enforced in CI):** Only `src/llmops/_mlflow_adapter.py` may `import mlflow`. All other modules use the adapter. This guarantees a single point of change when MLflow API evolves.

---

## 4. Components

### 4.1 SDK (`bni-llmops` Python package)

**Public API (via `import llmops`):**

```python
# Tracing — decorator and context manager (same callable)
@llmops.trace_agent("agent_tujuan")
def run_agent_tujuan(pain_point: str) -> str: ...

with llmops.trace_agent("synthesis_step"):
    ...

# Prompts
prompt = llmops.load_prompt("agent_tujuan@production")
text = prompt.format(pain_point="...", context="...")

llmops.register_prompt(
    name="agent_tujuan",
    template="...",
    commit_message="git SHA",
    tags={...},
)
llmops.set_alias("agent_tujuan", alias="staging", version=3)
```

**Idempotency contract (SDK-level, not CLI-level):** `register_prompt` is idempotent. If the `(name, template)` pair matches the latest existing version, no new version is created and the existing version is returned. This contract lives in `prompts.py` — not in the CLI. The CLI `register-prompts` command iterates over YAML files and calls `register_prompt` for each; idempotency is enforced once, at the SDK layer, so direct SDK callers and CLI users get identical behavior.

**Initialization model:** **Implicit only.** No `init()` function. Configuration via environment variables, read lazily on first SDK call. Selected because it minimizes integration friction.

**Required env vars (validated at first call, fail-loud if missing):**
- `MLFLOW_TRACKING_URI` (e.g., `http://localhost:5001`)

**Optional env vars (with documented defaults):**
- `LLMOPS_EXPERIMENT_NAME` (default: `bni-agentic-prd`)
- `LLMOPS_DISABLE_TRACING` (default: `false`; if `true`, all trace calls become no-ops — useful for local dev without server)

**Error handling — bifurcated:**
- **Config errors** (missing env, unreachable server at first call, prompt alias not found): `LLMOpsConfigError` raised, agent crashes early in dev cycle.
- **Runtime trace errors** (network hiccup during span flush): logged at `WARNING` level with span ID + cause; agent continues, return value preserved. LLM Ops must never become a liability that crashes user-facing flows.

**Internal layout (Approach 2 — layered with thin adapter):**

```
src/llmops/
├── __init__.py            # Public API surface (re-exports)
├── _config.py             # Frozen Config dataclass, env-loaded
├── _mlflow_adapter.py     # SOLE module that imports mlflow
├── tracing.py             # trace_agent (uses adapter)
├── prompts.py             # load_prompt, register_prompt, set_alias (uses adapter)
├── cli.py                 # `llmops` command via typer
└── exceptions.py          # LLMOpsError hierarchy
```

**Coupling enforcement (CI):** ruff `flake8-tidy-imports` rule **TID253** (`banned-module-level-imports`) forbids `mlflow` imports anywhere in `src/llmops/` except `_mlflow_adapter.py`. Configuration in `pyproject.toml`:

```toml
[tool.ruff.lint.flake8-tidy-imports]
banned-module-level-imports = ["mlflow"]

[tool.ruff.lint.per-file-ignores]
"src/llmops/_mlflow_adapter.py" = ["TID253"]
```

`ruff check` enforces this on every PR; a violation is a CI hard fail (no override).

**MLflow API namespace note (validated against latest docs):** MLflow's prompt registry has slight namespace asymmetry — `mlflow.genai.register_prompt`, `mlflow.genai.load_prompt`, but `mlflow.set_prompt_alias` (no `genai` namespace). The adapter abstracts this; SDK callers see only `llmops.register_prompt` / `llmops.load_prompt` / `llmops.set_alias`.

### 4.2 MLflow Tracking Stack (Docker Compose)

**Two services:**

| Service | Image | Purpose |
|---|---|---|
| `mlflow` | `ghcr.io/mlflow/mlflow:v3.11.1` (pinned) | Tracking server, UI, prompt registry, artifact proxy |
| `postgres` | `postgres:18-alpine` (pinned major) | Backend store for runs, experiments, prompts, registered models |

**Volumes (named, persistent across `docker compose down/up`):**
- `llmops_pgdata` → postgres `/var/lib/postgresql/data`
- `llmops_artifacts` → mlflow artifact root

**Ports:**
- MLflow UI / REST: `5001` (host) → `5000` (container). Note: host port 5000 is reserved by macOS Control Center.
- Postgres: NOT exposed to host (only on Docker network).

**MLflow server invocation:**
```
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri postgresql://llmops:${POSTGRES_PASSWORD}@postgres:5432/llmops \
  --artifacts-destination /mlartifacts \
  --serve-artifacts
```

**Configuration:** all secrets / connection params via `.env` (gitignored). `.env.example` checked in with documented placeholders.

**Healthcheck:** HTTP GET `http://mlflow:5000/health` returns 200 within 60s of `docker compose up`.

### 4.3 Prompts-as-Code

**Source of truth:** `prompts/*.yaml` in this repo.

**Schema:** see Appendix B. Validation via pydantic at CI time and at SDK `register_prompt` time.

**Lifecycle:**
1. Author edits `prompts/<agent>.yaml` on a feature branch off `gos-dev`.
2. PR opens → CI validates schema; lints; runs unit tests.
3. Merge to `gos-dev` → CI workflow `register-prompts.yml` triggers:
   - Diffs `prompts/` against previous commit.
   - For each changed prompt: `register_prompt(name, template, commit_message=<sha>, tags=...)` → produces a new MLflow version.
   - `set_alias(name, "staging", new_version)`.
   - Posts a comment on the merge commit with version table.
4. Manual trigger of `promote.yml` (workflow_dispatch) to promote `staging → production` for a specific prompt. The CI logs:
   - Updates alias.
   - Adds audit tags to the prompt version (see Section 5.3).

**Aliases used:** `staging`, `production`. (No `dev`/`canary` v1.)

### 4.4 GitHub Actions CI Workflows

| Workflow | Trigger | Purpose | Target time |
|---|---|---|---|
| `ci.yml` | `pull_request: branches: [gos-dev]` | ruff check + ruff format check + pytest + prompt YAML schema validate + `uv sync --python 3.14.4` | ≤ 3 min |
| `register-prompts.yml` | `push: branches: [gos-dev]` | Diff prompts/, register changed versions, set `staging` alias, comment summary | ≤ 5 min |
| `promote.yml` | `workflow_dispatch` (inputs: `prompt_name`, `from_alias`, `to_alias`) | Shells out to `uv run llmops promote ...` — single code path | ≤ 1 min |

**Single audit-tag writer:** the `promote.yml` workflow does NOT contain its own MLflow client logic. It runs `uv run llmops promote <name> <from> <to>`, which calls `prompts.set_alias()` which writes the audit tags (Section 5.3). The CLI command and the workflow are NOT two parallel implementations — they share `src/llmops/cli.py` → `prompts.set_alias()`. This eliminates risk of audit-tag drift.

**Runner setup:** uses `astral-sh/setup-uv@v8` + `actions/setup-python@v6` with `python-version: "3.14.4"`. `setup-uv` provides `enable-cache: true` for built-in dependency caching (no separate `actions/cache` step needed).

**Secrets needed:**
- `MLFLOW_TRACKING_URI` — internal MLflow URL accessible from CI runner.
- (No API keys for v1 since no eval.)

### 4.5 CLI Tool (`llmops`)

Run via `uv run llmops <command>`.

| Command | Args | Purpose |
|---|---|---|
| `register-prompts` | `<dir>` (default `./prompts`) | Bulk-register all prompts in directory; idempotent (same template → no new version) |
| `promote` | `<prompt_name> <from_alias> <to_alias>` | Migrate alias; writes audit tags |
| `list-prompts` | (none) | Print table of registered prompts + versions + aliases |
| `doctor` | (none) | Validate env vars, MLflow reachability, postgres connectivity |
| `--help` / `--version` | — | Standard |

**Exit codes:** `0` success, `1` runtime error, `2` invalid arguments.

---

## 5. Conventions

### 5.1 Naming

| Item | Convention | Example |
|---|---|---|
| MLflow experiment | Single experiment | `bni-agentic-prd` |
| MLflow run name | Caller-provided session UUID | `7a3e...d20b` |
| Span name | Agent name (lowercase, snake_case) | `agent_tujuan` |
| Prompt name | snake_case, must equal YAML filename stem | `agent_tujuan` ↔ `prompts/agent_tujuan.yaml` |
| Branch | `feat/<topic>` from `gos-dev` | `feat/llmops-foundation` |

### 5.2 Run-level tags (auto-set by SDK)

| Tag key | Source | Purpose |
|---|---|---|
| `llmops.sdk_version` | `bni-llmops` package version | Identify which SDK build produced trace |
| `llmops.git_sha` | env var `GIT_SHA` if set | Link trace to caller's deployed commit |
| `llmops.session_id` | provided by caller or UUID4 auto | Group multi-agent run together |
| `llmops.prompt_versions` | JSON-serialized dict `{name: version}`, accumulated via thread-local across `load_prompt` calls inside a run, written once on outermost trace context exit | Know exactly which prompt versions ran |

**`llmops.prompt_versions` mechanics (locked):** Run-level tags in MLflow are flat string key/value pairs. To capture multiple prompt versions used during a single run, the SDK keeps a thread-local `dict[str, int]` that accumulates `(name, version)` entries on each `load_prompt(name@alias)` call. When the **outermost** `trace_agent` context exits, the SDK serializes the dict to a JSON string and writes it as the run tag `llmops.prompt_versions`. This guarantees one consistent value per run, regardless of how many prompts were loaded. Nested `trace_agent` contexts share the same thread-local dict and do NOT write the tag (only the outermost does).

### 5.3 Promotion audit tags (auto-set on prompt versions during `promote`)

| Tag key | Source |
|---|---|
| `promoted_from_alias` | CI input |
| `promoted_to_alias` | CI input |
| `promoted_at` | ISO 8601 timestamp |
| `promoted_by` | GitHub `actor` |
| `promoted_git_sha` | `GITHUB_SHA` |

### 5.4 Branching

- All feature branches: `feat/<topic>` off `gos-dev`.
- PR target: `gos-dev`.
- `main` is updated by promoting `gos-dev` → `main` (separate workflow, out of scope here).
- `register-prompts` workflow triggers on `push` to `gos-dev` only, not `main`.

---

## 6. Tech Decisions Locked

| Decision | Value | Rationale |
|---|---|---|
| Python version (SDK + dev) | `3.14.4` (`requires-python = ">=3.14,<3.15"`) | Matches colleague's project version |
| Ruff `target-version` | `"py314"` | Match Python version |
| MLflow client lib | `mlflow >= 3.11, < 4` | Latest stable; pin major to avoid breaking changes |
| MLflow server image | `ghcr.io/mlflow/mlflow:v3.11.1` (pinned patch) | Matches client major; upgrade is an explicit PR |
| Postgres image | `postgres:18-alpine` | Latest stable major (Postgres 18) |
| Package manager | `uv` | Per project standard |
| Lint + format | `ruff check` + `ruff format` | Per project standard |
| CI platform | GitHub Actions | Per locked decision |
| Repo layout | Mono-repo (SDK + compose + prompts + CI) | KISS for single team |
| SDK distribution | Install via `uv add git+ssh://...@<tag>` | No PyPI v1 |
| Versioning | SemVer (`0.1.0` → `1.0.0` when stable) | Conventional |
| MLflow UI port | `5001` | macOS reserves 5000 |
| Authentication | None (localhost / Docker network only) | Single-team v1 |

---

## 7. Statement of Work (SoW)

### D1. Python SDK `bni-llmops`
Public API per Section 4.1. Framework-agnostic. Type-hinted. Test-covered ≥80%.

### D2. MLflow Tracking Stack
`docker-compose.yml`, `.env.example`, persistent volumes, healthcheck, `--serve-artifacts` proxy. One-command spin-up.

### D3. Prompts-as-Code
`prompts/*.yaml` directory, pydantic schema (Appendix B), example prompt for demo.

### D4. CLI Tool
`llmops register-prompts | promote | list-prompts | doctor` per Section 4.5.

### D5. GitHub Actions Workflows
Three workflows per Section 4.4 with all triggers wired to `gos-dev`.

### D6. Documentation
- `README.md` — 5-minute quickstart (verified end-to-end by reproduction)
- `docs/recipes/openai.md` — integration with raw OpenAI SDK
- `docs/recipes/langchain.md` — integration with `mlflow.langchain.autolog()`
- `docs/recipes/custom.md` — manual `@llmops.trace_agent` for arbitrary orchestration
- `docs/architecture.md` — diagram + component map
- `docs/troubleshooting.md`

### D7. Test Suite
- Unit tests (`tests/unit/`) — mock the MLflow adapter
- Integration tests (`tests/integration/`) — ephemeral MLflow + postgres in CI service container
- Compose smoke test — `docker compose up && curl /health`

### D8. Operational Scripts
- `scripts/backup.sh` — `pg_dump` + tar artifacts volume
- `scripts/restore.sh` — restore from backup
- Documented in `docs/troubleshooting.md`.

---

## 8. Definition of Done (DoD)

### 8.1 SDK
- [ ] `uv add git+ssh://...@v0.1.0` installs cleanly on Python 3.14.4
- [ ] `uv sync --python 3.14.4` passes in CI without dependency resolution error
- [ ] Quickstart 5-line snippet from README works end-to-end; trace appears in MLflow UI ≤ 10s
- [ ] `trace_agent` captures: input args, return value, latency, exception with traceback, nested span parent-child links
- [ ] `load_prompt("name@alias")` returns object with `.format(**vars)`; raises `LLMOpsPromptNotFoundError` with clear message if alias missing
- [ ] All public functions have type hints, docstrings, and at least one minimal example in docstring
- [ ] Test coverage ≥80% on `src/llmops/`
- [ ] `ruff check` + `ruff format --check` zero issues
- [ ] CI rule enforced: only `_mlflow_adapter.py` may `import mlflow` (verified in `ci.yml`)

### 8.2 Tracking Stack
- [ ] `docker compose up -d` reaches healthy state ≤ 60s
- [ ] MLflow UI accessible at `http://localhost:5001`
- [ ] Postgres data persists across `docker compose down && docker compose up`
- [ ] Artifacts persist; trace replay after restart works
- [ ] `.env.example` documents every variable with inline comment

### 8.3 Prompts-as-Code
- [ ] `prompts/agent_demo.yaml` example exists and validates
- [ ] Schema rejects invalid YAML with clear error (e.g., variable in `{{ }}` not declared in `variables`)
- [ ] File name `<x>.yaml` must match field `name: x`

### 8.4 CI
- [ ] `ci.yml` PR run completes ≤ 3 min
- [ ] `register-prompts.yml` on merge to `gos-dev` registers new versions and sets `staging` alias
- [ ] `promote.yml` workflow_dispatch logs alias change with audit tags (Section 5.3)

### 8.5 CLI
- [ ] Every command has `--help`
- [ ] `llmops doctor` validates env vars + MLflow reachability + postgres ping; reports each step pass/fail with reason

### 8.6 Documentation
- [ ] Fresh clone → follow README → first trace visible ≤ 10 minutes (verified by independent attempt)
- [ ] All 3 recipes have working code samples
- [ ] Architecture diagram is accurate (matches Section 3)

### 8.7 Operational
- [ ] `scripts/backup.sh` produces a restorable artifact (verified by `restore.sh` round-trip on test data)
- [ ] Postgres + artifact backup procedure documented

### 8.8 End-to-End Acceptance
1. Fresh clone of repo
2. `cp .env.example .env` and edit
3. `docker compose up -d`
4. (In a separate Python project) `uv add git+ssh://...@v0.1.0`
5. `import llmops` → use `@llmops.trace_agent` and `llmops.load_prompt`
6. Open `http://localhost:5001` → see trace with nested spans
7. Edit a prompt YAML, push PR → CI registers a new version on merge
8. `uv run llmops promote agent_tujuan staging production` → audit tags written
9. **Total wall-clock onboarding ≤ 30 minutes** for a colleague new to the repo.

**Note on step 7:** the PR → merge cycle requires a reviewer. The 30-minute budget covers steps 1–6 (technical setup) and step 8 (CLI promote, ~10 seconds). Step 7's review wait time is human-loop and excluded from the technical onboarding budget. If validating end-to-end without waiting on review, step 7 can be substituted with a direct push to `gos-dev` in a sandbox repo.

---

## 9. Risks & Implementation-Time Validations

| Risk | Mitigation |
|---|---|
| MLflow 3.11.1 declares `requires_python = ">=3.10"` (no upper bound) but classifier list only mentions Python 3.10 — Python 3.14 is not explicitly tagged. Transitive deps may also lag on cp314 wheels | DoD step `uv sync --python 3.14.4` MUST pass before merge. If a dep lacks a 3.14 wheel, evaluate: (a) bump the dep to a newer release with cp314 wheels, (b) bump MLflow to a newer minor that lists 3.14 explicitly in classifiers, (c) temporary fallback to `requires-python = ">=3.13,<3.15"` (rekan can still run on 3.14). |
| MLflow client/server protocol drift | Server image and client lib pinned to same major (`3.x`). Upgrade in lockstep via PR. |
| Prompt YAML schema evolves | Schema version field reserved (`schema_version: 1`); future migrations explicit. |
| `--serve-artifacts` proxy bandwidth bottleneck | Acceptable for single-team v1; documented out-of-scope. Trigger to MinIO is artifact volume >50 GB or team >5 users. |
| Empty prompt directory triggers no-op CI | `register-prompts.yml` exits 0 with informative log when no diff. |

---

## 10. Future Work (v2 Outlook)

When triggers in Section 2.2 fire, v2 adds:

- **Eval module** — new file `src/llmops/eval.py` using same `_mlflow_adapter`. New CLI command `llmops eval <dataset>`. Golden dataset format (`eval/datasets/*.jsonl`) curated from real traces. Configurable judge model URI (default OpenAI gpt-4o-mini, override Azure/Anthropic/Ollama via env).
- **Eval-on-PR CI gate** — new workflow that runs eval against `staging` alias on every prompt-changing PR; comments scores; blocks merge if metrics regress.
- **Auth** — reverse proxy with bearer token or OIDC.
- **MinIO** — drop-in artifact store swap via env var; no SDK changes.

No spec rewrite needed — all v2 additions are additive on this design.

---

## Appendix A: Repository Layout

```
bni-llmops/
├── pyproject.toml              # Python 3.14, mlflow>=3.11,<4, ruff target py314
├── uv.lock
├── .python-version             # 3.14.4
├── .env.example
├── .gitignore                  # ignores .env, .venv, __pycache__, *.egg-info
├── docker-compose.yml          # mlflow + postgres
├── docker/
│   └── mlflow/
│       └── Dockerfile          # FROM ghcr.io/mlflow/mlflow:v3.11.1 (custom psycopg deps if needed)
├── src/
│   └── llmops/
│       ├── __init__.py         # public API surface (re-exports)
│       ├── _config.py          # frozen Config dataclass, env-loaded
│       ├── _mlflow_adapter.py  # SOLE module importing mlflow
│       ├── tracing.py          # trace_agent (decorator + context mgr)
│       ├── prompts.py          # load_prompt, register_prompt, set_alias
│       ├── cli.py              # llmops typer app
│       └── exceptions.py       # LLMOpsError hierarchy
├── prompts/                    # SSoT prompt templates
│   ├── _schema.py              # pydantic model + validator entrypoint
│   └── agent_demo.yaml         # example
├── tests/
│   ├── unit/
│   │   ├── test_tracing.py
│   │   ├── test_prompts.py
│   │   ├── test_config.py
│   │   └── test_cli.py
│   └── integration/
│       └── test_e2e.py         # spins ephemeral MLflow + postgres
├── docs/
│   ├── architecture.md
│   ├── troubleshooting.md
│   ├── recipes/
│   │   ├── openai.md
│   │   ├── langchain.md
│   │   └── custom.md
│   └── superpowers/
│       └── specs/
│           └── 2026-04-28-bni-llmops-design.md  # THIS DOC
├── scripts/
│   ├── backup.sh               # pg_dump + tar artifacts
│   └── restore.sh              # restore from backup
├── .github/
│   └── workflows/
│       ├── ci.yml              # PR checks
│       ├── register-prompts.yml # on push to gos-dev
│       └── promote.yml         # workflow_dispatch
└── README.md                   # 5-minute quickstart
```

---

## Appendix B: Prompt YAML Schema (pydantic)

```yaml
# prompts/agent_tujuan.yaml
schema_version: 1                       # required, int, currently 1

name: agent_tujuan                      # required, must match filename stem
                                        # regex: ^[a-z][a-z0-9_]*$

description: |                          # required, ≥10 chars
  Identifies the high-level objective ("Tujuan") of a product
  from a given pain-point context. Returns structured JSON.

template: |                             # required, Jinja-style {{ var }}
  You are a product analyst. Given the following pain point and context,
  identify the underlying objective.

  Context:
  {{ context }}

  Pain point:
  {{ pain_point }}

  Return JSON: { "tujuan": "...", "rationale": "..." }

variables:                              # required list of strings
  - context                             # every {{ x }} in template MUST appear here
  - pain_point                          # every entry MUST appear in template

tags:                                   # optional dict[str, str]
  domain: prd
  agent_type: child
  owner: <colleague_handle>
```

**Template semantics (locked):** Templates use **variable-substitution-only** semantics with `{{ var }}` syntax — Jinja-flavored placeholder form, but **no control flow, no filters, no conditionals, no loops**. This matches MLflow Prompt Registry's `Prompt.format(**vars)` runtime behavior. Implementation can use a simple regex extractor + `str.replace` or MLflow's built-in formatter; either is acceptable as long as no Jinja-only feature is used.

**Validation rules (pydantic + custom validator):**
1. `schema_version == 1`.
2. `name` matches `^[a-z][a-z0-9_]*$`.
3. File name without extension equals `name`.
4. Every `{{ identifier }}` in `template` appears in `variables`.
5. Every entry in `variables` appears at least once as `{{ entry }}` in `template`. (Safe under variable-substitution-only semantics — no `{% if %}` blocks where a variable might be conditionally referenced.)
6. `description` length ≥10.
7. `tags` keys/values are strings; no nested structures.

Validation runs:
- In CI (`ci.yml`) on every PR.
- In SDK (`register_prompt`) before any MLflow call.
- In CLI (`llmops register-prompts`) before any MLflow call.

---

## Appendix C: Memory & Principle Compliance Checklist

This design adheres to user-locked principles:

- [x] **Validate against latest official docs** — all tech stack validated against primary sources (PyPI, ghcr.io, hub.docker.com, github.com/<org>/releases, official docs sites). Context7 was found stale on MLflow latest version (showed v3.1.4 when actual latest is v3.11.1) and on `setup-uv` / `setup-python` action versions; primary sources used as authoritative
- [x] **Decisions binding upfront** — every option explicitly IN or OUT (Section 2.2 + 6 + 9)
- [x] **Tool boundaries clear** — SDK = act, MLflow UI = observe, no overlap
- [x] **MLflow aliases not stages** — all promotion uses `set_prompt_alias`
- [x] **Explainability first** — every component explainable in 1-2 plain sentences
- [x] **Use existing libraries** — uv/ruff/MLflow/pydantic/typer; no custom reinvention
- [x] **uv for Python** — pyproject.toml + uv.lock; no pip
- [x] **ruff for lint+format** — config in pyproject.toml
- [x] **Branch from gos-dev** — `feat/llmops-foundation` off `gos-dev`; CI triggers on `gos-dev`
- [x] **Tools ready for anyone** — framework-agnostic SDK; minimal env-only setup; recipes for common stacks

---

**End of Specification.**
