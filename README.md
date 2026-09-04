<div align="center">

# 💼 Career Copilot Agent

### RAG-Grounded Gap Analysis & Self-Correcting Cover Letters — an Agentic LLM System

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Agent%20Orchestration-1C3C3C)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?logo=openai&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18.3-61DAFB?logo=react&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-Multi--stage%20build-2496ED?logo=docker&logoColor=white)

[![CI](https://github.com/momo840505/career-copilot-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/momo840505/career-copilot-agent/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Open%20App-2EA44F?logo=render&logoColor=white)](https://career-copilot-agent.onrender.com)
![Status](https://img.shields.io/badge/Status-Live-2EA44F)

A multi-agent job-application assistant that reads a job description, checks it against
my own resume + portfolio (via RAG), tells me honestly where I match and where I don't,
and drafts a grounded, citation-checked cover letter — with a human approval step before
anything goes out.

[Overview](#-overview) •
[Live Demo](#-live-demo) •
[Architecture](#️-architecture) •
[Setup](#️-setup) •
[Evals](#-evals) •
[API and MCP Server](#-api-and-mcp-server) •
[Deployment](#-deployment) •
[Observability](#-observability) •
[Engineering Notes](#-engineering-notes)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Why This Exists](#-why-this-exists)
- [Live Demo](#-live-demo)
- [Architecture](#️-architecture)
- [Build Roadmap](#️-build-roadmap)
- [Setup](#️-setup)
- [Evals](#-evals)
- [API and MCP Server](#-api-and-mcp-server)
- [Deployment](#-deployment)
- [Observability](#-observability)
- [Tech Stack](#-tech-stack)
- [Engineering Notes](#-engineering-notes)
- [Skills Demonstrated](#-skills-demonstrated)

---

# 🚀 Overview

This is the 6th project in my data science / AI engineering portfolio. Where the other five
(`retail-demand-forecasting`, `cyber-risk-intelligence-lakehouse`, `gamewise-ai`,
`smart-hydro-alert`, `flight-reliability-platform`) cover classic ML, data engineering, NLP
retrieval, and IoT streaming, this one is deliberately an **agentic LLM system**: multi-step
planning, structured-output validation, self-correction loops, human-in-the-loop, evals for
generative output (not just accuracy metrics), an MCP server, a full React frontend, and a
live Docker deployment — the pieces the other five don't touch.

---

# 🧠 Why This Exists

Skills gap it closes relative to the rest of the portfolio:

| Already demonstrated elsewhere | New in this project |
|---|---|
| ML pipelines, time-series forecasting | Multi-step **agent orchestration** (LangGraph state machine) |
| Semantic search (GameWise AI) | **RAG with enforced citations** + groundedness checking |
| FastAPI services, Docker, AWS deploys | **Structured output validation** (Pydantic + retry-on-failure) |
| Dashboards or CI for data pipelines | **Evals for generative output** (golden set, LLM-as-judge, CI gate) |
| — | **Human-in-the-loop** approval checkpoint |
| — | **MCP server** exposing the agent's tools to any MCP client |
| — | **Production observability** (structured logs, `/metrics`) on a live deployment |

---

# 🌐 Live Demo

### [🧭 Open Career Copilot Agent](https://career-copilot-agent.onrender.com)

Paste in a real job description and watch the full pipeline run against my actual resume
and project write-ups — gap analysis, then a grounded cover letter with a critic verdict.

> **Free-tier hosting note:** the Render instance spins down after 15 minutes of
> inactivity. The first request after a while can take 30-60 seconds to wake back up —
> if the first load times out, just retry once the instance is warm. Access is gated
> behind a shared access code (keeps my OpenAI bill from being open to the entire
> internet) — reach out if you'd like a demo login.

### Job description input

![JD input form, with an example JD loaded and a live character counter](docs/images/jd_form.png)

### Gap analysis — fit score, matched / partial / missing, with evidence citations

![Gap analysis result showing a 60% fit score ring, matched/partial/missing counts, and cited evidence per requirement](docs/images/gap_analysis_result.png)

### Cover letter — self-correction loop passed on the first attempt

![Cover letter result showing "Critic: passed, 0 revisions" and inline evidence citations](docs/images/cover_letter_result.png)

### History — every run is saved per access-code client

![History tab listing a cover letter run and a gap analysis run with timestamps](docs/images/history.png)

---

# 🏗️ Architecture

### System overview (frontend, API, deployment)

```mermaid
flowchart TB
    subgraph Client["Browser"]
        UI["React SPA"]
    end

    subgraph Server["Render - single Docker container"]
        Auth["Access-code gate"]
        API["FastAPI"]
        Graph["LangGraph pipeline"]
        Chroma[("ChromaDB")]
        SQLite[("SQLite")]
        Obs["observability.py"]
    end

    OpenAI[["OpenAI API"]]

    UI -->|HTTPS| Auth --> API
    API --> Graph
    Graph --> Chroma
    Graph --> OpenAI
    API --> SQLite
    API -.-> Obs
```

`Auth` checks `X-Access-Code` on every route but `/health` and `/metrics`;
`X-Client-Id` is additionally required on every route except those two AND
`/auth/verify`, which only needs a valid access code to answer (there's no
per-client history to scope at that point). `API` is FastAPI serving both the
built React frontend and the JSON routes from the same origin. `Graph` is the
LangGraph pipeline below, which calls `Chroma` for retrieval and `OpenAI` for
every LLM step. `SQLite` holds per-client history; `Obs` (dotted line —
logging only, not a request path) is `observability.py`'s structured logs +
`/metrics`.

### Agent pipeline (inside the LangGraph node)

```mermaid
flowchart TB
    A["Job description"] --> B["parse_jd<br/>structured output"]
    B --> C["retrieve_evidence<br/>RAG over resume <br/> + 5 projects"]
    C --> D["gap_analysis<br/>matched / partial / missing"]
    D --> E["draft_writer<br/>cites chunk ids"]
    E --> F{"critic<br/>grounded? on-tone?"}
    F -- fails --> E
    F -- passes --> G["human_review<br/>pause for approval"]
    G --> H["final cover letter <br/> talking points"]
```

---

# 🛣️ Build Roadmap

- [x] Phase 0 — repo scaffold, environment
- [x] Phase 1 — RAG corpus (resume + 5 project write-ups) + retrieval
- [x] Phase 2 — `parse_jd` node with validated structured output
- [x] Phase 3 — `retrieve_evidence` + `gap_analysis` nodes
- [x] Phase 4a — `draft_writer` + `critic` nodes (manually chained, self-correction loop by hand)
- [x] Phase 4b — wired into an actual LangGraph `StateGraph` (conditional loop + `interrupt()` human-in-the-loop)
- [x] Phase 5 — eval harness (golden JD set, objective metrics + groundedness judge) + GitHub Actions CI
- [x] Phase 6 — FastAPI service + MCP server
- [x] Phase 7 — Docker + cloud deploy + observability + final write-up
  - [x] 7a — SQLite history + shared access-code gate
  - [x] 7b — React frontend SPA
  - [x] 7c — Docker packaging (multi-stage, frontend + backend in one image)
  - [x] 7d — Render deployment (live)
  - [x] 7e — Observability (structured logging + `/metrics`)
  - [x] 7f — README polish, screenshots, final architecture diagram

---

# ⚙️ Setup

```powershell
cd career-copilot-agent
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
copy .env.example .env
# then edit .env and paste your real OPENAI_API_KEY

# build the RAG index (needs your API key, costs a fraction of a cent)
python scripts/build_index.py

# smoke test: ask it something
python scripts/query_demo.py "does she have SQL and cloud deployment experience?"

# run the tests that don't need an API key
pytest -m "not requires_api"

# Phase 2: try the JD parser
python scripts/parse_jd_demo.py

# Phase 3: run the full parse -> retrieve -> gap-analysis chain
python scripts/pipeline_demo.py

# Phase 4a: draft_writer <-> critic self-correction loop (manual, pre-LangGraph)
python scripts/draft_and_critique_demo.py

# Phase 4b: the same pipeline as an actual LangGraph StateGraph, with a real
# human-in-the-loop pause (interrupt() / Command(resume=...)) before a letter
# is considered final
python scripts/graph_demo.py

# Phase 5: run the golden JD eval set (objective metrics + groundedness judge),
# needs an API key and costs a bit more (multiple JDs x multiple judge votes)
python scripts/run_evals.py

# Phase 6: run the pipeline as an HTTP service (FastAPI). Docs/try-it-out UI at
# http://127.0.0.1:8000/docs once it's running.
python scripts/run_api.py

# Phase 6: run the pipeline as an MCP server, so any MCP-aware client (Claude
# Desktop, an IDE, another agent) can call search_evidence / analyze_job_description /
# draft_cover_letter directly. This is the one to use day to day -- runs directly in
# your already-activated venv, no extra tooling needed:
mcp run src/career_copilot/mcp_server.py --transport streamable-http

# Optional: `mcp dev` opens an interactive inspector in the browser instead of just
# starting the server -- but unlike `mcp run` above, it launches the server through
# `uv run` (an isolated, reproducible throwaway environment, by the Inspector's own
# design), so it needs the `uv` tool installed separately first, and that throwaway
# environment won't have this project's other dependencies (langgraph, chromadb, ...)
# unless told to include them. Not verified end-to-end here -- `mcp run` above is the
# confirmed, day-to-day path; treat this one as a starting point to debug from if
# you want the interactive inspector specifically:
pip install uv
mcp dev src/career_copilot/mcp_server.py --with-editable .

# Phase 7c/7d: build and run the same Docker image the live deployment uses
docker build -t career-copilot .
docker run --rm -p 8000:8000 --env-file .env career-copilot
```

> Tip: if your project folder path contains Chinese characters (e.g. under `Desktop\作品集\`),
> keep the **virtual environment itself** on an ASCII-only path if you hit any install errors —
> e.g. `python -m venv C:\venvs\career-copilot` and point your editor's interpreter at that,
> while the source code stays in your normal folder. Most tools handle Unicode paths fine now,
> but this is the first thing to try if `pip install` or `chromadb` complain.

---

# 🧪 Evals

`src/career_copilot/data/golden_jds/` holds 3 fixed job descriptions chosen to stress
different parts of the system: a strong-fit data-analyst role (expect mostly `matched`),
a deliberately weak-fit senior DevOps role (expect mostly `missing` — the honesty check),
and a mixed-fit role similar to the real JDs used during development. `scripts/run_evals.py`
runs each through the full pipeline (via `graph/pipeline.py`, the non-interactive version
of the graph — no human in the loop, so it can run unattended in CI) and reports two kinds
of check, deliberately kept separate:

- **Hard gates** (`career_copilot/eval/metrics.py`) — objective, code-based, deterministic:
  citation validity, no undisclaimed mention of a missing skill, and whether the
  writer/critic loop converged within `MAX_REVISIONS`. A failing run fails the build.
- **Groundedness judge** (`career_copilot/eval/groundedness_judge.py`) — an LLM-as-judge
  score, asked `n_votes` times per draft rather than once, reporting the median *and* the
  spread across votes. This is where the "LLM-as-judge noise" limitation flagged back in
  Phase 4 gets addressed rather than just noted — informational only, not a hard gate,
  because a single noisy number shouldn't block a merge.

`.github/workflows/ci.yml` runs `pytest -m "not requires_api"` on every push/PR (free, no
secret needed), and a separate `eval-gate` job runs the real golden-set eval — gated to
pushes to `main`, manual dispatch, or a weekly schedule, so it never burns API cost on
every commit or runs (uselessly, without a secret) on an external PR.

To let `eval-gate` actually run in your own fork/repo, set, under **Settings → Secrets
and variables → Actions**: `OPENAI_API_KEY` as a repository **Secret** (required), and
optionally `OPENAI_CHAT_MODEL` / `OPENAI_CRITIC_MODEL` as repository **Variables** if you
want CI to use the same model choices as your local `.env` — `ci.yml` falls back to
`gpt-4o-mini` / `gpt-4o` respectively if you don't set them, matching `.env.example`'s
own recommendation. See the matching engineering note below for why that fallback
exists — it's not a hypothetical.

## ⚠️ Known Limitations, and Why This Is by Design

No LLM-based system — this one included — can guarantee it never makes a questionable
call on a brand-new input; a "100% success rate" claim for a system like this would be a
red flag, not a selling point. What this project actually promises, and what the eval
harness exists to check honestly rather than paper over, is narrower and more defensible:
**nothing wrong ever reaches a human unflagged.** Every structured-output call goes
through `invoke_structured`'s validate-and-retry loop; if every attempt still fails, the
pipeline stops cleanly with a `StructuredOutputError` and a clear message — it never
silently ships a fabricated citation, an item double-classified in a gap report, or an
overclaimed skill. `gap_analysis`'s retry budget was widened from 2 to 4 once the golden
set showed its failure mode scales with how many requirements a JD packs in (more
requirements = more independent chances for a same-report conflict); that's a bounded,
evidence-driven adjustment, deliberately not an open-ended chase to prompt-engineer away
every edge case a future JD might introduce, which has no actual finish line. See the
"Phase 5" entries in the engineering notes below for the specific failures this surfaced
and how each was reasoned through.

In practice, on the 3-JD golden set: the widened `gap_analysis` budget is not
theoretical — one confirmation run needed all 5 attempts (4 identical "STUCK" repeats
before finally breaking out) to converge on a JD that would have failed outright under
the old budget of 3. The densest JD (~10 requirements, thin evidence coverage) still
occasionally exhausts `draft_writer`'s retry budget and fails — cleanly, with a clear
`StructuredOutputError` and no bad citation ever shipped. That's the accepted, expected
shape of "safe failure" this section describes, not a bug still being chased.

---

# 🔌 API and MCP Server

Two thin entry points over the same pipeline everything else in this project calls —
neither duplicates any pipeline logic, they just expose it differently.

**FastAPI service** (`src/career_copilot/api/app.py`, launched via
`python scripts/run_api.py`): `GET /health` (no LLM call — confirms the service is up
and an API key is configured), `GET /metrics` (see [Observability](#-observability)),
`POST /gap-analysis` (parse -> retrieve -> gap analysis), and `POST /draft` /
`POST /draft/{thread_id}/decision` (the full pipeline, draft/critic loop included, over
the same compiled `build_graph.py` StateGraph `graph_demo.py` drives interactively — so
the API now has the same human-in-the-loop pause: `POST /draft` always comes back with
`status: "pending_review"` plus a `thread_id`, never a finished letter by itself; the
caller reviews `critic_passed` / `critic_issues` and sends
`POST /draft/{thread_id}/decision` with `{"action": "approve"}` to finalize — the only
path that persists a history record — or `{"action": "revise", "feedback": "..."}` to
send it back through `draft_writer` for another attempt, which pauses at
`pending_review` again). The checkpointer behind this is in-memory per process, so a
paused draft doesn't survive a redeploy — fine for a single-instance demo. A
`StructuredOutputError` — every repair attempt in `invoke_structured`'s retry loop
exhausted — is mapped to `502 Bad Gateway`, not `400/422`: the request itself was fine,
an upstream dependency (the LLM) is what failed to deliver. Interactive docs at `/docs`
once it's running.

**MCP server** (`src/career_copilot/mcp_server.py`): exposes `search_evidence`,
`analyze_job_description`, and `draft_cover_letter` as MCP tools, so any MCP-aware
client (Claude Desktop, an IDE, another agent) can call this pipeline directly with no
HTTP client of its own. Run it with `mcp run src/career_copilot/mcp_server.py
--transport streamable-http` — confirmed live, runs directly in the same venv
everything else in this project uses, no extra tooling required. `mcp dev` (the
interactive browser inspector) is a separate, optional path with its own `uv`
dependency — see the setup section above and the engineering note below. Built against
the `mcp[cli]` **v2.x** API (`from mcp.server import MCPServer`); see the other
engineering note below for why that's pinned explicitly rather than left as
`mcp>=1.0.0`.

---

# 🐳 Deployment

The live demo above runs from a single Docker image that packages both halves of the
project — no separate frontend host, no separate API host.

**Multi-stage `Dockerfile`:** stage 1 (`node:22-slim`) runs `npm install && npm run
build` on `frontend/`, producing static assets; stage 2 (`python:3.11-slim`) installs
the backend, then `COPY --from=frontend-build` pulls in the built `dist/` folder. FastAPI
mounts it as a `StaticFiles(..., html=True)` route at `"/"` — registered *after* every
API route, since Starlette matches routes in registration order and a mount at `"/"`
would otherwise shadow everything (see `api/app.py`'s comment on this). The result: one
container, one port, same-origin frontend and API — no CORS complexity in production.

**`docker/entrypoint.sh`:** Render's free tier gives the container an ephemeral
filesystem, so the Chroma RAG index has to be rebuilt on every boot — the entrypoint does
that automatically, but skips it if an index is already present on disk (e.g. a local
`docker run` with a mounted volume), so it doesn't re-pay OpenAI embedding costs on every
restart for nothing.

**`render.yaml`:** a committed Render Blueprint — `runtime: docker`, points straight at
this repo's `Dockerfile`, `healthCheckPath: /health`. `OPENAI_API_KEY` and `ACCESS_CODE`
are marked `sync: false` so Render prompts for them in its own dashboard instead of ever
living in this (public) repo. This file is both how the live deployment is actually
configured and a checked-in record of it — worth more in a portfolio than a screenshot of
a dashboard nobody else can see.

**Free-tier reality, not glossed over:** Render's free plan spins the instance down after
15 minutes of no traffic and takes 30-60 seconds to wake it back up on the next request —
see the note in [Live Demo](#-live-demo) above. This is a deliberate, disclosed trade-off
for a $0/month personal-project deployment, not a hidden limitation.

---

# 📈 Observability

Two small additions, both in `src/career_copilot/observability.py`, wired into the
FastAPI app (`api/app.py`) and the shared `invoke_structured` retry loop
(`graph/structured.py`) that every LLM node goes through:

**Structured logging.** Every HTTP request (a middleware wraps the whole app) and
every LLM call (parse_jd / gap_analysis / draft_writer / critic, each labeled by
`node_name`) logs one line with method/path/status/duration or node/attempt/duration —
plus, critically, whether that call hit the STUCK-retry path described in the Phase 5
engineering note below. `LOG_FORMAT=json` (the Docker image's default — see
`docker/entrypoint.sh`) switches this to one parseable JSON object per line, which is
what actually matters on Render: its Logs tab just tails stdout, so JSON-per-line is
the difference between "grep-able" and "a wall of free text." Local dev keeps a plain
human-readable format by default.

**`GET /metrics`.** A process-wide, thread-safe, **in-memory** counter registry — no
database, no Prometheus, no external service. That's a deliberate match to the
deployment, not a shortcut: Render's free tier runs exactly one instance with an
ephemeral filesystem, so anything that tried to persist metrics across restarts would
be solving a problem this deployment doesn't actually have. It answers the two
questions worth asking first when something looks wrong — is the API taking traffic
and erroring (per-route count / avg latency / status-code breakdown), and is the LLM
pipeline healthy or burning retries (per-node call / retry / **STUCK** / failure
counts). Deliberately left open (no access-code gate, like `/health`): it exposes only
aggregate counts and latencies — no JD text, no letters, nothing tied to a specific
`client_id` — so gating it would only make it harder to check the service's health
without protecting anything actually sensitive. Try it live:
**[career-copilot-agent.onrender.com/metrics](https://career-copilot-agent.onrender.com/metrics)**

---

# 🧰 Tech Stack

## Agent orchestration and LLM engineering

- LangGraph (`StateGraph`, conditional edges, `interrupt()` human-in-the-loop)
- LangChain (`with_structured_output`, chat model abstraction)
- OpenAI API (chat completions + embeddings)
- Pydantic (structured-output schemas, validate-and-retry)

## RAG and retrieval

- ChromaDB (persistent vector store)
- Chunked resume + 5 project write-ups as the retrieval corpus
- Enforced citation checking (no chunk_id, no claim)

## API and frontend

- FastAPI (JSON API + same-origin static frontend host)
- React 18 + Vite (SPA, hand-rolled inline SVG icons, no icon-library dependency)
- MCP (`mcp[cli]` v2.x) server exposing the same pipeline as agent tools
- SQLite (per-client history)

## DevOps and deployment

- Docker (multi-stage: Node build stage + Python runtime stage)
- Render (Blueprint deploy, `render.yaml`)
- GitHub Actions (`ci.yml`: unit tests every push, golden-set eval-gate on `main`)

## Testing and quality

- pytest, with a fake-LLM harness for retry-loop logic (no API key needed)
- A 3-JD golden eval set with both hard, deterministic gates and an LLM-as-judge score

---

# 📝 Engineering Notes

Keeping a running log here — this is exactly the kind of "I broke it, here's how I found
it and fixed it" material that makes for a strong interview story (see the target-leakage
catch on `cyber-risk-intelligence-lakehouse` for the same idea applied to an ML pipeline).

- **Phase 3 — self-contradictory gap classification.** On the very first real JD run, the
  model classified the same requirement (e.g. "技術文件撰寫") as **both** `matched` and
  `missing` in the same report. Nothing in the schema forbade it — Pydantic validators only
  see one field/item at a time, not the relationship between three separate lists. Fixed by
  adding `check_no_duplicate_requirements()`, a whole-report check wired into the same
  `invoke_structured(..., validate=...)` retry loop already used for the hallucinated-citation
  check. Caught by actually running the pipeline against a real JD, not by unit tests alone —
  the unit tests were added *after*, to lock the fix in (`tests/test_gap_duplicate_check.py`).

- **Phase 4 — LLM-as-judge noise, and a stateless writer.** Running
  `draft_and_critique_demo.py` against a real JD surfaced two separate reliability problems in
  the self-correction loop: (1) the critic's verdict on the *same* sentence was inconsistent
  across separate calls (flagged, then not flagged, then flagged again) — a known
  characteristic of LLM-as-judge systems, not something code alone fixes (Phase 5's eval
  harness exists partly to quantify and manage this); (2) the writer only ever saw the *latest*
  round of critic feedback, so a problem fixed in attempt 1 (generic filler) could silently
  reappear in attempt 3 once the critic's most recent verdict happened not to repeat it — the
  writer had no memory of its own revision history. Fixed #2 directly: `draft_writer` now takes
  the full accumulated feedback history across all attempts, not just the newest verdict. #1 is
  mitigated, not solved — a separately-configurable `OPENAI_CRITIC_MODEL` (a stronger judge
  tends to be less noisy) and, longer-term, majority-vote / multi-judge calibration in Phase 5.

- **Phase 4 — the retry loop never showed the model its own previous answer.** A deeper bug
  than the two above: `invoke_structured`'s repair loop rebuilt every retry from the *original*
  prompt plus an abstract error description — the model never saw the JSON it had actually
  produced. So a "fix this" retry wasn't a targeted edit, it was a blind re-roll with a hint,
  and the same class of mistake (a duplicated requirement) could resurface on a *different* item
  each attempt. Fixed by echoing the model's previous (schema-valid but rule-violating) output
  back as an `AIMessage` before the error feedback, so retries become genuine edits. Verified
  with a scripted fake-LLM test (no API key needed) that the retry message sequence actually
  contains the prior output before shipping it — this affects all four nodes, since they all
  share `invoke_structured`.

- **Phase 4 — don't make the model do lookups it can be handed pre-joined.** Even with the
  above fixed, the critic still flagged 100% of claims as "ungrounded" across three straight
  attempts, unrelated to how much the draft actually improved. Root cause: the critic was shown
  two separately-shaped lists — claims with `chunk_id` references, and a flat evidence list
  keyed by `chunk_id` — and asked to cross-reference them itself. It effectively gave up and
  defaulted to "flag everything." Fixed by doing the join in code: each claim is now shown with
  its own cited evidence text directly beneath it, so the model only has to make one judgment
  (does *this* text support *this* claim?) instead of a lookup plus a judgment. General lesson:
  when an LLM step behaves suspiciously uniformly regardless of input, suspect the *input
  format* before assuming the model or the prompt wording is at fault.

- **Phase 4 — a correct catch that still couldn't be fixed, because the flag carried no
  reason.** With the previous bug fixed, the critic started giving genuinely accurate feedback
  — e.g. correctly catching that "I have utilized Excel extensively... in various projects"
  overclaims evidence that only lists Excel as one tool among several. But `draft_writer`
  couldn't act on it: the same overclaim (barely reworded) reappeared nearly verbatim across
  all 3 attempts. Root cause: `CriticVerdict.ungrounded_claims` was `list[str]` — just the
  flagged sentence's own text, echoing back what the writer already wrote, with no explanation
  of *why* it was rejected. The writer could only guess at what to change. Fixed by restructuring
  it into a list of `UngroundedClaim{claim_text, reason}` (the same `(item, note)` shape already
  used by `GapItem` in `gap_analysis`), and updating the critic's prompt to require a specific,
  actionable reason per flag. General lesson, paired with the one above: getting an LLM step's
  *judgment* right isn't enough if the *shape of the feedback it returns* can't actually drive a
  fix downstream — the consumer of a verdict matters as much as the verdict itself.

- **Phase 4 — reworded, not fixed: the writer kept swapping one unsupported qualifier for
  another.** With reasons now attached, the critic's feedback on the same Excel claim was
  specific and correct across all 3 attempts (its only evidence, `skills::chunk2`, is a bare tool
  name in a list — no detail on how much or how well it was used). But the writer's 3 rewrites
  were "used it extensively for data manipulation and visualization" → "used it as part of my
  data analysis work" → "allows me to handle various data tasks effectively": three different
  unsupported qualifiers, never zero. The writer had no concept of "this evidence is too thin to
  support ANY elaboration — the fix is to say less, not say it differently." Fixed by adding an
  explicit rule to `draft_writer`'s prompt: match claim strength to evidence specificity — a bare
  mention (a name in a list, no elaboration) earns only a bare claim ("I have used Excel"), never
  a strength word ("extensively", "proficient", "effectively") unless a *different* cited chunk
  actually describes the scope or outcome of that use. General lesson: when a self-correction
  loop keeps "fixing" the same flaw with a same-strength synonym, the model isn't missing the
  *feedback* — it's missing the *concept* that the right move is subtraction, not rephrasing.
  With this fix, the loop converged in 2 attempts on the same real JD that previously never
  converged in 3 — confirmed against live output, not just the unit tests.

- **Phase 4b — the `interrupt()` re-run gotcha.** Wiring `human_review` into the actual
  `StateGraph` surfaced a non-obvious LangGraph behavior worth flagging explicitly rather than
  discovering it the hard way: when a graph resumes after `interrupt()`, the node function that
  called it **reruns from the top** — every line before the `interrupt()` call executes again,
  not just the lines after it. For `human_review` that's harmless (it only reads existing state
  before pausing), but it's exactly the kind of thing that would silently double-fire a real side
  effect — an email send, a paid API call, a DB write — placed before `interrupt()` in a node
  that later gets resumed. Documented directly in `_node_human_review`'s docstring as a rule: any
  side effect in an interruptible node belongs strictly *after* the `interrupt()` call.

- **Phase 4b — the checkpointer was silently pickling our own classes.** The first live run of
  `graph_demo.py` printed six `"Deserializing unregistered type ... this will be blocked in a
  future version"` warnings — one per custom class stored in `AgentState`
  (`JDRequirements`, `RetrievedChunk`, `EvidenceBundle`, `GapReport`, `CoverLetterDraft`,
  `CriticVerdict`). Looked it up rather than ignoring a "just a warning": LangGraph's checkpoint
  format can reconstruct arbitrary Python objects from msgpack unless the allowed types are
  restricted, which is exactly the shape of a real, disclosed vulnerability
  ([GHSA-g48c-2wqr-h844](https://github.com/langchain-ai/langgraph/security/advisories/GHSA-g48c-2wqr-h844))
  — a compromised checkpoint store becomes a code-execution vector, not just a data leak. Our
  own state only ever holds our own trusted classes, so the actual risk here is low, but the
  fix is one line either way: construct the checkpointer with an explicit
  `allowed_msgpack_modules` allow-list (`JsonPlusSerializer(allowed_msgpack_modules=[...])`
  passed into `InMemorySaver(serde=...)`) instead of leaving it on the permissive default. General
  lesson: a "this will be blocked in a future version" log line is the library telling you a
  default is about to change under you — worth 10 minutes to fix now, on my own schedule,
  instead of on a future `pip install --upgrade` day. Confirmed fixed on the next live run — the
  warnings are gone.

- **Phase 4b — the writer invented a placeholder chunk_id, and the safety net actually caught
  it.** Same live run surfaced a genuinely new failure: for the "missing" requirement
  (企業流程自動化), `draft_writer` tried to write an honest bridging sentence ("I have
  streamlined workflows in my previous roles...") but had no real evidence chunk that actually
  supports that specific sentence — so instead of dropping the claim, it invented a fake
  `chunk_id` (first an empty string, then `"_"`) to satisfy the schema's "every claim needs a
  citation" rule. `check_claims_cite_real_evidence` — the same outside-the-schema validator from
  Phase 4a that catches hallucinated citations — caught both attempts immediately and fed the
  error back; the model got it right on the 3rd try (a real `work_experience` chunk instead).
  **The end-to-end system did exactly what it's supposed to do here** — a bad citation never
  reached the human — but it burned 2 of 3 retries doing it, which is a real cost (latency, token
  spend, and it eats into the retry budget `gap_analysis`/`critic` might also need). Root cause:
  the prompt's rule for "missing" items said a "strong partial" could be "honestly bridged," but
  didn't say what to cite when bridging, and didn't forbid inventing a placeholder outright.
  Tightened both: a `claims` entry now must cite a real chunk for the *related* skill it's
  bridging FROM, never the missing skill itself, and never a placeholder — and if no real chunk
  supports a sentence, it doesn't become a `claims` entry at all. General lesson, and arguably the
  most reassuring one in this log: layered defenses are meant to be redundant. The prompt should
  ideally have prevented this on the first try, and now prompts closer to that — but the
  validate-and-retry safety net built in Phase 2 is what actually made the failure invisible to
  the end user while that prompt gap still existed. That's the point of defense in depth.

- **Phase 4b — the stronger-worded rule only halved the problem; removing the input killed it.**
  Re-ran the same JD after the rule tightened above: it helped (1 failed attempt instead of 2,
  and the model chose to drop the unsupportable sentence entirely rather than invent a citation
  for it), but it still reached for a placeholder `"_"` on its first try, both times. A prompt
  telling a model "don't do X" is a probabilistic nudge, not a guarantee — no amount of stronger
  wording gets that to zero, because the model was still being shown the thing that tempted it:
  the "missing" item's name, and gap_analysis's `suggested_talking_points` (which in earlier runs
  had already been observed bridging language for non-partial items too — see above). The actual
  fix: stop showing `draft_writer` the "missing" list and `suggested_talking_points` at all.
  `_format_gap_report` now returns only "matched" and "partial" items, each with its own real
  evidence — there is structurally nothing left to invent a citation for, because there's nothing
  shown that doesn't already have one. Nothing about honesty is lost: the full gap report
  (missing items included) still reaches the human reviewer through `human_review`'s payload; the
  letter's job was never to enumerate every gap, only to make the honest case for what's there.
  General lesson, and the cleanest version of one that keeps recurring in this log: when a model
  keeps misusing a piece of input despite being told not to, the more reliable fix is usually to
  stop handing it that input, not to word the instruction more forcefully. Confirmed clean on the
  next live run — no more placeholder-citation retries.

- **Phase 5 — closing the loop on "LLM-as-judge noise" from Phase 4.** Back in Phase 4, the
  critic's inconsistent verdicts on the same sentence were noted as a known limitation, "not
  something code alone fixes... Phase 5's eval harness exists partly to quantify and manage
  this." `groundedness_judge.py` is that follow-through: instead of trusting a single LLM-as-judge
  call, it asks the same question `n_votes` times and reports the *spread* across votes alongside
  the median score — a tight cluster means the judge's opinion is probably real, a wide spread
  means today's single score would have been noise. Deliberately kept separate from the objective,
  code-based checks in `metrics.py` (citation validity, missing-skill leaks, convergence), which
  ARE treated as hard CI gates precisely because they're deterministic — the groundedness score
  stays informational, never a hard gate, because a metric this noisy shouldn't be trusted to
  single-handedly fail a build.

- **Phase 5 — the eval set immediately did its job: it broke on a JD I'd never manually tried.**
  Every prior live bug in this log was found against the same one or two JDs I kept re-pasting by
  hand. The very first `run_evals.py` run, against 3 JDs specifically chosen to be *different*
  from those, hit two hard failures: `gap_analysis` classified "樞紐分析" into two buckets on
  **all 3/3 attempts, with the byte-identical error every time**; separately, `draft_writer`
  invented the placeholder chunk_id `"_"` on **all 3/3 attempts**, also byte-identical — the same
  symptom the Phase 4b fixes above were supposed to have closed, on a JD that happened to hit a
  different "partial" item with thin evidence. This is exactly the point of a golden set: a
  single hand-picked JD tests one path through the system, repeatedly; a small deliberately-varied
  set finds the paths that path never touched.

- **Phase 5 — the deeper bug under both failures: retries were stuck, not just wrong.** Both
  failures above share a root cause that's more fundamental than either individual prompt issue:
  `parse_jd`/`gap_analysis`/`critic` all run at `temperature=0.0` (see `llm.py`) for
  reproducibility, but `invoke_structured`'s repair message is itself deterministic — same error
  text in, same wording out. Feed a temperature-0 model the identical follow-up prompt twice and
  it can regenerate the identical (wrong) answer, which is exactly what happened: not 3 different
  failed attempts at a fix, but the same non-fix repeated 3 times, burning the entire retry budget
  for zero benefit. (`draft_writer` runs at 0.3, not 0.0, and still repeated identically — low
  temperature narrows the odds without needing to hit zero.) Fixed in `invoke_structured` itself,
  so every node gets it at once: each attempt's error is compared to the previous attempt's: on a
  byte-identical repeat, the next attempt's prompt gets an explicit "you're stuck, make a
  genuinely different decision" note, AND that attempt runs at a bumped temperature (0.7) via
  `llm.model_copy(update={"temperature": 0.7})` — deliberately *not* `llm.bind(temperature=...)`,
  which chains through a `RunnableBinding` wrapper with real, documented bugs interacting with
  `with_structured_output` (langchain-ai/langchain#23167, #35320) that could silently drop the
  override; `model_copy` sidesteps that whole risk by working on a plain field, not the Runnable
  composition machinery. Locked in with `tests/test_invoke_structured_retry.py` — a fake, scripted
  LLM asserts the exact temperature sequence across attempts, both for the stuck case (escalates)
  and the normal case (never does), without needing a real API key. General lesson: a retry loop
  that always resends "the same kind of nudge" isn't actually retrying — it's repeating; a retry
  loop needs the ability to recognize when IT'S the one stuck, not just the model.

- **Phase 5 — the test written to lock in the fix above had its own bug.** The very first run of
  `test_invoke_structured_retry.py` failed 2/5 tests — not because the fix was wrong (the captured
  log lines showed `invoke_structured` correctly detecting "STUCK — repeats previous error" right
  where expected), but because the fake `_FakeLLM.model_copy()` used in the test hard-coded
  `return _FakeLLM(...)` instead of `return type(self)(...)`. Real Pydantic's `model_copy()`
  preserves the concrete subclass of whatever it's copying; this fake didn't, so the escalated
  attempt's `TrackingLLM` subclass silently downgraded to a plain untracked `_FakeLLM` on copy, and
  the test's own instrumentation missed calls it should have seen — an under-verified assumption
  in a fake standing in for real behavior, not a flaw in the thing being tested. Also caught, while
  fixing that: the test's assumption that `with_structured_output` gets rebuilt every attempt was
  wrong too — the real code reuses one pre-built client for every non-escalated attempt and only
  rebuilds it for an escalated one, so the correct expected call sequence is `[0.0, 0.7]` (build
  once, then once more on escalation), not one entry per attempt. Both fixed directly in the fake
  and the assertions. Small, on-brand lesson to end on: a test that fails doesn't automatically
  mean the code under test is wrong — verify which side of the assertion the bug is actually on
  before "fixing" the wrong one.

- **Phase 5 — one requirement stayed stuck even at the bumped temperature: noise wasn't the whole
  story.** Re-ran the golden set after the stuck-retry fix: JD03's placeholder-citation failure was
  gone (self-corrected in 1 attempt this time, where it had previously exhausted all 3), but JD01
  hit a NEW instance of the duplicate-bucket bug — "樞紐分析" (pivot tables) — and this time even
  the temperature-0.7 escalated attempt reproduced the identical error a 3rd time. That's a useful
  negative result: it means this particular confusion isn't pure sampling noise that a bit of
  temperature can shake loose — the model has a stable (if wrong) reason for wanting to put a
  narrow sub-skill in two buckets, and no amount of re-rolling fixes a stable reasoning error.
  Root cause: the JD asked for "Excel 樞紐分析" (Excel pivot tables) as its own requirement, but
  the evidence only supports general "Excel" experience without naming pivot tables specifically —
  exactly the "sub-skill nested under a broader matched skill" case, and `gap_analysis`'s prompt
  had no tie-breaking rule for it (the same category of gap `draft_writer` had for claim strength
  back in Phase 4a, just never carried over to this node). Fixed with an explicit, ordered
  tie-breaker: a sub-skill with no evidence of its own but real evidence for its broader
  tool/domain is "partial" — never "matched" (that overclaims the specific technique) and never
  "missing" (that discards real, relevant signal). General lesson to close Phase 5 on: the
  stuck-retry fix and this one are complementary, not redundant — one recovers from a random slip,
  the other prevents a systematic misjudgment; a system needs both, and neither substitutes for
  the other.

- **Phase 5 — where this line of debugging stops, and why stopping here is the right call, not a
  shortcut.** One more golden-set run after the tie-breaker fix: JD01 now converged cleanly (2
  revisions, all hard gates pass), but JD03 — the densest JD, ~10 parsed requirements — hit yet
  another duplicate-bucket conflict, this time on 3-4 items at once, and exhausted its retries.
  Diagnosed the actual pattern behind all of Phase 5's failures: they cluster on `gap_analysis`
  specifically, and specifically scale with how many requirements a JD has — more requirements
  means more independent chances for at least one classification conflict, and clearing several
  conflicts in the same report is a harder combinatorial problem than clearing one. The fix here
  is deliberately NOT another prompt patch chasing this run's specific offending items (that's
  the pattern this whole section has been following, and every fix so far has found a new edge
  case rather than reaching zero) — it's widening `gap_analysis`'s own `invoke_structured` retry
  budget from 2 to 4, sized to the actual problem (more surface area needs more attempts), while
  every other node keeps the default. This is where Phase 5 stops, on purpose: chasing full
  determinism on every possible JD has no finish line for an LLM-based system, and isn't actually
  the right goal — see "Known Limitations" above for what the actual goal is instead, and why a
  system that sometimes needs a retry (and says so clearly when it does) is a stronger, more
  honest result than a claim of zero failures ever would be.

- **Phase 6 — an unpinned dependency almost repeated the exact mistake the LangGraph checkpoint
  advisory had already taught this project to avoid.** `requirements.txt` had `mcp>=1.0.0` sitting
  in it since the very first scaffold, unpinned "because it hadn't been needed yet." Before
  writing `mcp_server.py` against it, I checked the actual current API rather than trusting a
  remembered tutorial — and the SDK had made a breaking change between v1.x
  (`from mcp.server.fastmcp import FastMCP`) and v2.x (`from mcp.server import MCPServer`, the
  class itself renamed), released alongside the "2026-07-28 MCP specification". An unpinned
  `pip install -r requirements.txt` run today would have silently picked up v2, and code written
  against a v1.x tutorial (the only kind that existed when this project's requirements file was
  first scaffolded) would have failed at import time with no clue why beyond an `ImportError`.
  This is the same category of bug the LangGraph checkpoint serializer advisory caught back in
  Phase 4b, just at dependency-resolution time instead of runtime: an unpinned or under-specified
  dependency isn't a convenience, it's a silently deferred break waiting for whoever installs
  next. Fixed by pinning `mcp[cli]>=2,<3` with an explanatory comment, and writing `mcp_server.py`
  against the verified current v2 README rather than assumption.

- **Phase 6 — `scripts/run_api.py` deliberately breaks from every other script's import
  pattern.** Every other script in `scripts/` starts with `sys.path.insert(0, ...)` to make
  `career_copilot` importable without an editable install. Tracing through how `uvicorn --reload`
  actually works before writing this script caught why that pattern would silently fail here
  specifically: `--reload` runs the real server in a **separate subprocess** that re-imports the
  string app target (`"career_copilot.api.app:app"`) fresh — a `sys.path.insert()` done in the
  parent process that launches uvicorn simply isn't visible to that subprocess, so the reloader
  would hit `ModuleNotFoundError` the instant a file changed and it respawned. `run_api.py`
  relies on the project's already-established `pip install -e .` editable install instead, which
  makes `career_copilot` importable from any process, no path patching required. A one-line
  difference from the other scripts, caught by reasoning through the tool's actual process model
  before running it rather than after debugging a reload-only failure live.

- **Phase 6 — my own em dash broke `pip install -r requirements.txt` on Windows.** The very
  first live run of the Phase 6 changes failed immediately, before any package even resolved:
  `UnicodeDecodeError: 'cp950' codec can't decode byte 0xe2 in position 759`. `pip`'s requirements-file
  parser has no way to know a `.txt` file's encoding (no BOM, no `# -*- coding -*-` cookie — that's
  a Python source-file convention, not a `pip` one) and falls back to the OS's locale-preferred
  encoding when it can't detect one; on Windows with a Traditional Chinese system locale, that's
  `cp950` (Big5), not UTF-8. The explanatory comment I'd written for the `mcp[cli]` pin used a
  literal em dash (`—`, `U+2014`) — 3 UTF-8 bytes, byte 0 of which (`0xe2`) is not a valid `cp950`
  lead byte on its own, hence the crash at that exact byte offset. Every other file in this repo
  (`pyproject.toml`, `.env.example`, all `.py` sources) is pure ASCII or, for `.py` files, read
  under Python's own UTF-8-by-default source encoding — `requirements.txt` was the one place an
  em dash could actually reach a codec that isn't UTF-8-aware. Confirmed by scanning every
  pip/build-adjacent config file in the repo for bytes above 0x7F: `requirements.txt` was the only
  offender, and the one non-ASCII byte sequence in it was exactly this em dash. Fixed by replacing
  it with a plain hyphen — general lesson: a comment that reads fine in an editor can still be a
  landmine for whichever tool reads the file next with a different, non-UTF-8 fallback encoding;
  `requirements.txt`/`.cfg`/`.ini`-style config files (anything without an explicit encoding
  declaration mechanism) are worth keeping strictly ASCII, not just "looks like plain text."

- **Phase 6 — `mcp dev` and `mcp run` are not the same execution model, and only one of them was
  actually needed here.** First live attempt at the MCP server used `mcp dev
  src/career_copilot/mcp_server.py`, which opens an interactive browser Inspector — it opened fine,
  but the server itself showed **Failed / Connection closed**, with the Inspector's own "Servers"
  panel revealing the actual command it was trying to run: `uv run --with mcp==2.1.1 mcp run
  src/career_copilot/mcp_server.py`. Running that exact command by hand (bypassing the Inspector's
  UI, which was also garbling the console output — a separate, likely display-layer encoding issue
  on top of the real one) gave a plain, readable answer: `uv` — a separate Python packaging tool —
  wasn't installed at all. `mcp dev` launches the Inspector's target server through `uv run` by
  design, to test against a clean, reproducible throwaway environment rather than whatever's
  already active in the caller's shell; it needs `uv` installed as a prerequisite, and that
  throwaway environment starts with only `mcp` in it, not this project's other dependencies. `mcp
  run` — the command actually documented for real client use, not just interactive debugging — has
  no such requirement: it ran directly in the already-fully-set-up project venv on the first try,
  no `uv` needed. Since `mcp run` is what a real MCP client (Claude Desktop, an IDE, another agent)
  actually uses, and it was confirmed working end-to-end, this is where Phase 6's MCP verification
  stopped — `mcp dev`/Inspector is documented as an optional, not-fully-verified extra for anyone
  who specifically wants the interactive debugging UI, rather than chased to full certainty for a
  path that isn't actually load-bearing. Same "verify the thing that matters, don't chase every
  side path to zero" judgment call as the retry-budget decision back in Phase 5.

- **Phase 6 — the FastAPI service, live-tested end to end, is where all of Phase 4/5's fixes paid
  off at once.** A real ~9-requirement "Software Engineer" JD (SQL, Git, Docker, RESTful APIs,
  Java, Spring Framework, Kubernetes, AWS, Microservices) run through both `/gap-analysis` and
  `/draft` via the `/docs` UI came back clean on the first live request: every citation traced to a
  real chunk_id, every "partial" gap (Java, Spring Framework, Kubernetes, Microservices) was
  honestly bridged rather than overclaimed or invented, `critic_passed: true` after exactly one
  revision. A separate request on what looks like the same JD did hit a `{'Docker': 2}`
  duplicate-bucket conflict on its first `invoke_structured` attempt — and self-corrected within
  the same request, returning `200 OK` with no trace of the conflict in the response. That's not a
  contradiction with the "known limitations" section above; it's the expected shape of the exact
  thing that section describes: the same underlying model behavior sometimes needs a retry and
  sometimes doesn't, and the system's job was never to eliminate that variance, only to make sure a
  client calling this API only ever sees a fully-resolved report or a clean `502`, never a
  half-fixed one. Confirmed live, on real (not scripted) input, that both outcomes hold.

- **Phase 6 — going live on GitHub Actions surfaced a config-drift bug the local dev loop could
  never have caught.** First real `eval-gate` run (after finally getting `OPENAI_API_KEY` into the
  repo's Actions secrets — an earlier attempt failed because the secret had been named `API_KEY`,
  not `OPENAI_API_KEY`, a plain typo caught immediately from the same `RuntimeError` message
  `get_collection` already raises for a missing key) got past that, actually called the real API —
  and then failed `critic_converged` on **all 3** golden JDs at once, every one stuck at
  `revision_count == max_revisions + 1` (fully exhausted, never satisfied). Uniform failure across
  every JD, on a gate that had converged locally before, pointed away from "the model just had a
  bad run" and toward a systematic difference between the two environments. It was: local `.env`
  sets `OPENAI_CRITIC_MODEL=gpt-4o` (`.env.example` says why — plain `gpt-4o-mini` "can be
  over-strict" as a critic), but the CI workflow only ever passed through `OPENAI_API_KEY`, so
  `eval-gate` was silently running the critic on the stricter default the whole time. Not a bug in
  the pipeline itself — every hard gate did exactly its job, correctly failing a run whose critic
  genuinely never passed — but a bug in what CI was actually testing: it wasn't exercising the same
  configuration this project is actually meant to run with. Fixed two ways at once: `ci.yml` now
  passes `OPENAI_CHAT_MODEL`/`OPENAI_CRITIC_MODEL` through as repository Variables, AND falls back
  to the `.env.example`-recommended values (`gpt-4o-mini` / `gpt-4o`) via `${{ vars.X || 'default'
  }}` if those Variables are never set — so a forgotten config step degrades to the *documented*
  default behavior, not a silently stricter one nobody chose. General lesson, and a fitting one to
  end this project's CI setup on: `.env` being gitignored (correctly, since it holds the API key)
  means every *other* variable in it needs a deliberate, explicit decision about whether CI should
  mirror it — "add the secret" is not the same task as "make CI match local," and this project
  found out the difference the hard way, on the very first real run.

- **Phase 7 — two "fixed" failures came back on `eval-gate`, and this time the fix went one layer
  deeper.** First `eval-gate` run after pushing the Phase 7 frontend/Docker/Render work failed two
  golden JDs — and both failures were symptoms this log already claimed were closed. (1)
  `01_strong_fit_data_analyst.txt`: `gap_analysis` classified a requirement into two buckets, the
  exact Phase 3 bug `check_no_duplicate_requirements` was built to catch — except this time the
  retry loop never converged. Attempts 2-4 were byte-identical STUCK repeats (Phase 5's bumped-
  temperature escalation kicked in as designed), but attempt 5, now genuinely different, just moved
  the same conflict onto a *different* requirement (`資料倉儲設計經驗`) instead of resolving it. All
  5 retries spent, still duplicated — Phase 5's fix (break the determinism) worked exactly as
  intended and it still wasn't enough, because breaking a stuck loop guarantees a *different*
  answer, not a *correct* one. (2) `03_mixed_fit_ops_automation.txt`: `draft_writer` invented the
  placeholder chunk_id `"_"` again — the Phase 4b/5 failure that `_format_gap_report` was supposed
  to have structurally prevented by only ever showing the writer matched/partial items, each with
  its own real evidence. It still had a hole: nothing actually *required* a matched/partial item to
  carry real evidence in the first place — `check_matched_and_partial_have_evidence` didn't exist
  yet, so a `partial` item with an empty `evidence_chunk_ids` list (a valid gap_analysis output the
  schema never forbade) sailed straight through and handed `draft_writer` exactly the "nothing to
  cite" situation Phase 4b's fix assumed could no longer happen.
  Both fixes this time are structural rather than another round of stronger wording or more
  retries: `gap_analysis.py`'s duplicate-bucket check was rewritten from raise-and-retry into
  `dedupe_requirements`, which resolves the conflict deterministically (`partial` > `matched` >
  `missing` — see its docstring for why that order is the honest one) instead of spending more LLM
  calls hoping one attempt lands cleanly; and the new `check_matched_and_partial_have_evidence`
  closes the actual hole Phase 4b's fix depended on without enforcing. Also fixed one level further
  upstream: `parse_jd`'s prompt was silently splitting an "X 或 Y" JD bullet (e.g. "系統測試或自動化
  測試經驗") into two separate near-duplicate requirements, which is exactly the kind of pair
  `gap_analysis` tends to get confused between — told it to keep interchangeable alternatives as one
  requirement instead. General lesson, and maybe the most important one in this whole log: a fix
  that "confirmed clean on the next live run" was tested against the JD that broke it, not against
  the space of JDs that could. A golden set that runs on every push is what turned "seems fixed" into
  "still broken, on a case I hadn't tried" — automatically, unattended, weeks after the original fix
  — instead of leaving it to be rediscovered by a real user.

- **Phase 7d — a "Not Found" on a fully successful deploy was a stale browser tab, not a broken
  mount.** After the first Render Blueprint deploy went "Live", the actual URL showed a plain-text
  `Not Found` page. Two candidate explanations existed at that point: the `StaticFiles(..., html=
  True)` mount in `api/app.py` genuinely wasn't finding `index.html` (it would return exactly this
  plain-text 404 — distinct from FastAPI's own JSON `{"detail": ...}` 404 for an unmatched route,
  which is what made this diagnosable at all), or something else entirely. Reading the actual
  Render build log settled it before guessing further: the frontend build stage completed
  (`dist/index.html` produced and copied into the image), and the runtime log already showed `GET /
  HTTP/1.1" 200 OK` with `==> Your service is live 🎉` — the deploy had, in fact, fully succeeded.
  The `Not Found` screenshot was a stale/cached response from before that deploy finished. A hard
  refresh confirmed it: the live site loaded correctly, and a real gap-analysis run against the
  deployed backend returned a full result. General lesson: when a symptom and a log disagree, trust
  the log — a build/runtime log is ground truth for what the server actually did; a browser tab can
  be showing you the past.

- **Phase 7e — the observability layer had to be verified two different ways, because neither
  alone was authoritative.** `observability.py` is pure standard library (no LangChain, no
  FastAPI), so its JSON-log formatting and metrics-aggregation logic could be genuinely executed
  and asserted against in the sandbox this was built in — real code, real output, not a read-through.
  But the FastAPI wiring around it (the request-logging middleware, `GET /metrics`) couldn't be:
  that sandbox has no network access to install `fastapi`, the same restriction that blocked a real
  `docker build` back in Phase 7c. The honest resolution wasn't to skip verification, or to claim
  a level of confidence the setup didn't support — it was to verify what could actually run there,
  document what couldn't, and let the two things that *are* authoritative confirm the rest once
  pushed: the real `eval-gate` CI (full dependencies, runs the whole test suite for real) and the
  live Render deployment itself. Both did: CI passed, and `GET /metrics` on the live URL came back
  showing real accumulated route counts within 90 seconds of the new container's boot — the new
  code wasn't just present in the image, it was already doing its job in production. General
  lesson: "I can't fully verify this here" is a fine thing to say out loud, as long as it's paired
  with a concrete plan for what *will* verify it, and that plan gets followed through rather than
  left as a promise.

- **Phase 8 — `check_matched_and_partial_have_evidence` itself turned out to have the same flaw
  it was built to fix in others: raise-and-retry that never converges.** Live `eval-gate` failed
  `03_mixed_fit_ops_automation.txt` again, but differently this time — not `draft_writer` inventing
  a placeholder chunk_id, but `gap_analysis` itself exhausting all 5 `invoke_structured` attempts,
  every single one rejected by this exact check for the exact same three uncited requirements
  (`RPA`, `企業流程自動化`, `AI工具應用` — buzzwords with real semantic pull but nothing retrieved to
  back them). The STUCK-retry escalation (bumped temperature, explicit "make a different decision"
  note) fired as designed and still didn't help: the model kept re-asserting "partial"/"matched"
  with an empty `evidence_chunk_ids` list rather than either of the two fixes the error message
  spelled out. This is the identical shape of the Phase 7 duplicate-bucket failure that motivated
  `dedupe_requirements` in the first place — a model repeating the same classification decision
  isn't a case that needs a clearer prompt or another retry, it needs the retry loop taken out of
  the loop entirely. So `check_matched_and_partial_have_evidence` was rewritten the same way:
  `reclassify_uncited_as_missing` now moves an uncited matched/partial item straight into `missing`
  deterministically, no LLM call and no invented citation, exactly mirroring the reasoning already
  proven out by `dedupe_requirements`. General lesson: a validator that only ever raises is really
  betting the model can fix what it just got wrong, on demand, forever — that bet doesn't always
  pay off, and the fallback the error message already recommends is often safe to just apply
  directly instead of asking for it.

---

# 🎯 Skills Demonstrated

This project demonstrates practical experience with:

### Agentic AI and LLM engineering

- Multi-step agent orchestration with LangGraph (`StateGraph`, conditional edges)
- Structured-output validation and self-healing retry loops (Pydantic + LangChain)
- Self-correction loops (writer ⇄ critic) with accumulated feedback history
- Human-in-the-loop approval via `interrupt()` / `Command(resume=...)`
- Prompt engineering informed by root-cause debugging, not trial and error

### RAG and retrieval

- Vector-store-backed retrieval (ChromaDB) over a real, chunked document corpus
- Anti-hallucination citation enforcement (every claim traces to a real chunk_id)
- Evidence-formatting design that shapes what a downstream LLM step can and can't do

### Evals for generative AI

- A golden-JD eval set covering strong-fit, weak-fit, and mixed-fit cases on purpose
- Objective, deterministic hard gates alongside a spread-aware LLM-as-judge score
- A CI gate that has caught real, previously-"fixed" regressions automatically

### Software engineering and APIs

- FastAPI service design (error-code mapping, dependency injection, same-origin static hosting)
- An MCP server exposing the same pipeline as agent-callable tools
- React 18 + Vite SPA, hand-rolled UI (no icon-library dependency)
- SQLite-backed per-client history with a shared access-code auth gate

### DevOps, deployment, and observability

- Multi-stage Docker builds (Node build stage + Python runtime stage, one final image)
- Blueprint-based cloud deployment (Render, `render.yaml`, committed config)
- Structured JSON logging and an in-process `/metrics` endpoint sized to the actual deployment
- Root-causing real production symptoms (a stale "Not Found," CI regressions) from logs, not guesses
