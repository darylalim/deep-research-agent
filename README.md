# Deep Research Agent

A **deep research agent** built with [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview)
(LangChain 1.0 + LangGraph), served as a [Streamlit](https://streamlit.io) app. Ask it a
research question; it plans the work, delegates focused web searches to a subagent,
synthesizes a cited answer, keeps durable findings across sessions, and asks for your
approval before writing files.

## What's wired up

| Capability | How | Where |
| --- | --- | --- |
| **Planning** | `write_todos`, via langchain's `TodoListMiddleware` | — |
| **Web search** | Tavily (`tavily_search`) | `deep_research/tools.py` |
| **Subagent orchestration** | A `researcher` subagent, delegated to via the `task` tool | `deep_research/subagents.py` |
| **Persistent memory** | `SqliteStore` behind a `/memories/` route (cross-session) | `deep_research/agent.py` |
| **Durable thread state + interrupts** | `SqliteSaver` checkpointer (survives restarts) | `deep_research/agent.py` |
| **Human-in-the-loop** | `interrupt_on` gates `write_file` / `edit_file` / `delete` / `execute` | `deep_research/agent.py` + `turns.py` |
| **The app** | Streamlit chat with a live work log and in-page approvals | `streamlit_app.py` + `deep_research/webui.py` |
| **Observability** | LangSmith tracing via env vars | `.env` |

### The persistence model (two layers)

Deep Agents separates two kinds of state, and this project uses a disk-backed
option for each so **everything survives a restart** with no database server:

- **Checkpointer (`SqliteSaver`)** — the conversation, todo list, and any
  *pending* approval for a given `thread_id`. Stored in
  `.deep_research/checkpoints.sqlite`.
- **Store (`SqliteStore`)** — long-term memory shared across every thread.
  A `CompositeBackend` routes only the `/memories/` path prefix here; all other
  agent files stay in the ephemeral (but checkpointed) per-thread state. Stored
  in `.deep_research/memories.sqlite`.

So a fact the agent writes to `/memories/topic.md` in one session is readable in
the next; a scratch draft it writes to `/report.md` lives only in that thread.

## Setup

Requires **Python ≥ 3.11** and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies into a local venv
uv sync

# 2. Provide credentials
cp .env.example .env
# then edit .env and fill in ANTHROPIC_API_KEY and TAVILY_API_KEY
# (LangSmith keys are optional but recommended)
```

Keys you need:
- `ANTHROPIC_API_KEY` — Claude model access ([console.anthropic.com](https://console.anthropic.com))
- `TAVILY_API_KEY` — web search, free tier available ([app.tavily.com](https://app.tavily.com))

## Run

```bash
uv run streamlit run streamlit_app.py   # http://localhost:8501
```

Ask a question in the chat box. While the agent works you watch its plan, each
delegated sub-question, and every search query appear live; when it finishes, the cited
answer lands in the transcript with that work log collapsed above it.

When the agent wants to write a file, the turn pauses and an approval card shows the
**full** proposed content. You pick **Approve**, **Edit** (prefilled with the real
arguments, so narrowing a path is an edit rather than a retype), **Reject** (with an
optional reason for the agent), or **Respond** (answer on the tool's behalf). Nothing is
preselected, and unparsable edit JSON is never treated as approval. **Abandon this
turn** is always available as a way out.

The sidebar holds:

- **Thread** — conversations are checkpointed per thread and survive a restart, so
  reopening the app resumes the `main` thread exactly where you left off, including a
  pending approval.
- **Export transcript** — every question in the thread with its cited answer, as a
  markdown download.
- **Durable memory** — everything under `/memories/`, read straight from the Store.

It binds to `localhost` only (`.streamlit/config.toml`). The page has no authentication
and its visitor can approve file writes and spend your API keys, so putting it on a
network means putting real auth in front of it first.

## How it fits together

```
create_deep_agent(
    model         = ChatAnthropic("claude-opus-5-5")   # no temperature — Opus 5.5 rejects it
    tools         = [tavily_search]                    # orchestrator can search directly
    subagents     = [researcher]                       # …or delegate breadth via `task`
    backend       = CompositeBackend(
                        default = StateBackend,         # ephemeral, per-thread (checkpointed)
                        routes  = {"/memories/": StoreBackend},  # durable, cross-session
                    )
    interrupt_on  = {write_file, edit_file, delete, execute}  # human approval (needs a checkpointer)
    checkpointer  = SqliteSaver(...)                    # durable thread state + interrupts
    store         = SqliteStore(...)                    # durable long-term memory
)
```

The page drives the human-in-the-loop protocol: it streams the turn to exhaustion, and if
anything paused on a gated tool it shows an approval card for each pending action,
collects one decision per action, and resumes. Resuming can hit the next gated tool, so
it loops until the turn finishes. Because Streamlit reruns the script on every
interaction, that loop is unrolled across reruns via `st.session_state` rather than
written as a `while` loop.

A turn can carry **more than one** interrupt — the orchestrator dispatches each
`task` call as its own concurrent graph task and every subagent inherits
`interrupt_on`, so two `researcher`s fanned out in one turn can each raise their
own. The resume value is therefore a mapping of interrupt id → that interrupt's
decisions:

```python
Command(resume={interrupt_id: {"decisions": [...]}, ...})
```

A flat `Command(resume={"decisions": [...]})` makes LangGraph raise `RuntimeError:
When there are multiple pending interrupts, you must specify the interrupt id when
resuming`. The mapping form is also correct for the single-interrupt case, so there
is one code path.

The options it offers aren't fixed — the interrupt carries a per-tool
`allowed_decisions`, and the middleware rejects anything outside it, so the controls
are built from that (`approve` / `edit` / `reject` / `respond`, minus whatever the tool
forbids).

## Project layout

```
deep_research/
├── config.py       # env loading, model, state paths, key checks
├── tools.py        # Tavily web-search tool
├── subagents.py    # the `researcher` subagent
├── agent.py        # build_agent() assembles the agent; open_agent() adds disk persistence
├── turns.py        # what a turn did and what the user may see: feed, approvals, answer
└── webui.py        # Streamlit rendering + approval widgets (reuses turns.py's rules)
streamlit_app.py    # `streamlit run streamlit_app.py` — the page and its rerun state machine
.streamlit/         # theme (light AND dark, so the mode stays the reader's choice)
evals/              # LangSmith evaluations of the agent's actual behaviour
```

## Extending it

- **Add a tool** — build it in `tools.py`, then add it to the orchestrator's
  `tools=[...]` in `agent.py` (or to a subagent's `tools` in `subagents.py`).
- **Add a subagent** — return another `SubAgent` dict from `subagents.py` and
  include it in `subagents=[...]`. Give it its own tools and system prompt.
- **Gate more tools** — add tool names to `GATED_TOOLS` in `agent.py`. Use an
  `InterruptOnConfig` value (e.g. `{"allowed_decisions": ["approve", "reject"]}`)
  to restrict the available decisions per tool.
- **Go to production memory** — swap `SqliteStore` for `PostgresStore` (and
  `SqliteSaver` for a Postgres checkpointer) in `agent.py`.
- **Change what the app shows** — `deep_research/webui.py` for rendering,
  `streamlit_app.py` for the page's sequence. A new kind of feed line means a new
  `FeedEvent` kind emitted by `turns.ActivityFeed` *plus* a branch in
  `webui.render_event`, which is an if/elif chain that draws nothing for a kind it
  doesn't know — so `turns.FEED_KINDS` is the list it is checked against, and
  `tests/test_webui.py` goes red if the two disagree.

## License

[MIT](LICENSE) — the same license as the upstream stack this builds on
(`deepagents`, `langchain`, `langgraph`). Use, fork, and vendor the wiring
patterns freely.
