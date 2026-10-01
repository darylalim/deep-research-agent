# Deep Research Agent

A research agent built on [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview)
(LangChain 1.0 + LangGraph) and served as a [Streamlit](https://streamlit.io) app.

Ask it a question and it will:

1. Plan the work as a todo list.
2. Check its long-term memory for earlier findings.
3. Search the web itself, or hand sub-questions to parallel `researcher` subagents.
4. Write a report with inline citations.
5. Ask for your approval before saving anything to memory.

## Setup

Requires **Python 3.11+** and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # install dependencies into ./.venv
cp .env.example .env    # then fill in your keys
```

| Key | Required | Purpose |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | Yes | Claude model access ([console.anthropic.com](https://console.anthropic.com)) |
| `TAVILY_API_KEY` | Yes | Web search, free tier available ([app.tavily.com](https://app.tavily.com)) |
| `LANGSMITH_API_KEY` | No | Tracing, and required to run `evals/` ([smith.langchain.com](https://smith.langchain.com)) |

## Run

```bash
uv run streamlit run streamlit_app.py   # http://localhost:8501
```

- **Live work log.** As the agent works you see its plan, each delegated sub-question, and
  every search query. The finished answer appears with the log collapsed above it.
- **Approvals.** When the agent wants to write, edit or delete a file, the turn pauses and
  shows the full proposed change. Choose **Approve**, **Edit**, **Reject** (with an optional
  reason) or **Respond**. Nothing is preselected, and invalid edit JSON is never treated as
  approval. **Abandon this turn** is always available.
- **Sidebar.** Switch threads, export the thread as Markdown, and browse durable memory.

Threads and pending approvals are saved to disk, so restarting the app resumes where you
left off.

> [!WARNING]
> The app binds to `localhost` only (`.streamlit/config.toml`). It has no authentication,
> and anyone who can reach it can spend your API keys and approve file writes. Put real
> auth in front of it before exposing it on a network.

## How it works

```python
create_deep_agent(
    model        = ChatAnthropic("claude-opus-5-5"),      # no sampling params: Opus 5 rejects them
    tools        = [tavily_search],                       # quick lookups by the orchestrator
    subagents    = [researcher],                          # breadth, delegated via `task`
    middleware   = [TodoListMiddleware()],                # provides `write_todos`
    backend      = CompositeBackend(
                       default = StateBackend(),          # per-thread scratch space
                       routes  = {"/memories/": StoreBackend(...)},  # shared across threads
                   ),
    interrupt_on = GATED_TOOLS,                           # human approval; needs a checkpointer
    checkpointer = SqliteSaver(...),                      # .deep_research/checkpoints.sqlite
    store        = SqliteStore(...),                      # .deep_research/memories.sqlite
)
```

State is kept in two SQLite files, so no database server is needed:

| Layer | Holds | Scope |
| --- | --- | --- |
| **Checkpointer** | Conversation, todo list, pending approvals | One thread |
| **Store** | Files under `/memories/` | Every thread |

A note written to `/memories/topic.md` is readable in later sessions. A file written
anywhere else lives only in its thread.

The page drives the approval loop: it streams a turn to the end, shows a form for every
pending action, then resumes with `Command(resume={interrupt_id: {"decisions": [...]}})`.
Resuming can hit another gated tool, so this repeats until the turn finishes. Parallel
researchers can each pause in the same turn, which is why decisions are keyed by interrupt
id.

## Project layout

```
deep_research/
├── config.py       # env loading, model, state paths, key checks
├── tools.py        # Tavily web-search tool
├── subagents.py    # the `researcher` subagent
├── agent.py        # assembles the agent and its persistence
├── turns.py        # what a turn did: activity feed, approvals, answer, stop reasons
└── webui.py        # Streamlit rendering and approval widgets
streamlit_app.py    # the page and its rerun state machine
evals/              # LangSmith evaluations of the agent's behaviour
tests/              # offline test suite
.streamlit/         # server binding and light/dark themes
```

## Development

```bash
uv run pytest                  # offline tests, no keys or network
uv run pytest -m live          # tests against the real APIs
uv run ruff check && uv run ruff format
uv run ty check                # type check
uv run python -m evals --upload           # sync the eval dataset to LangSmith
uv run python -m evals --run --limit 1    # run the agent on one example (costs tokens)
```

See [`CLAUDE.md`](CLAUDE.md) for design decisions and the reasoning behind them.

## Extending it

| To | Do this |
| --- | --- |
| **Add a tool** | Build it in `tools.py`, then add it to `tools=[...]` in `agent.py` or to a subagent in `subagents.py`. |
| **Add a subagent** | Return another `SubAgent` dict from `subagents.py` and add it to `subagents=[...]`. |
| **Gate a tool** | Add its name to `GATED_TOOLS` in `agent.py`. Use an `InterruptOnConfig` to limit the allowed decisions. |
| **Use Postgres** | Swap `SqliteStore` and `SqliteSaver` for their Postgres versions in `agent.py`. |
| **Add a feed line** | Add a kind to `turns.FEED_KINDS`, emit it from `turns.ActivityFeed`, and render it in `webui.render_event`. |

## License

[MIT](LICENSE)
