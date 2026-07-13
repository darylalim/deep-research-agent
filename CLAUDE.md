# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A **deep research agent** — a thin, opinionated assembly layer over the
[`deepagents`](https://docs.langchain.com/oss/python/deepagents/overview) library
(currently v0.6.x) on LangChain 1.0 + LangGraph. Two files carry the weight:
`agent.py` (~130 lines) — *how* `create_deep_agent()` is wired — and `cli.py`
(~320 lines, the largest module here) — the human-in-the-loop interrupt/resume
protocol that wiring implies. `config.py`, `tools.py`, and `subagents.py` are
genuinely small support modules.

## Commands

```bash
uv sync                          # install deps + the project itself (editable) into ./.venv
uv run python -m deep_research   # run the interactive REPL (the only entry point)
uv run pytest                    # offline test suite (no keys/network needed)
uv run pytest -m live            # opt-in tests that hit real Anthropic/Tavily APIs
uv run python -m evals --upload  # create/sync the LangSmith eval dataset (free)
uv run python -m evals --run --limit 1   # score the agent (~100k tokens PER example)
uv run ruff check                # lint  (add --fix to autofix)
uv run ruff format               # format
uv run ty check                  # type check (Astral's ty)
```

- **Use the Astral skills.** When working with Python here, invoke the relevant
  `/astral:<skill>` — `/astral:uv`, `/astral:ty`, `/astral:ruff` — so that
  dependency management, type checking, and lint/format follow current best
  practices rather than remembered defaults.
- **A PostToolUse hook already runs ruff and pytest for you.** `.claude/settings.json`
  wires `.claude/hooks/post-edit.sh` to every `Edit`/`Write` of a `.py` file — as
  `bash <path>`, deliberately, so a lost exec bit can't silently disable the gate;
  wire any new hook the same way. It formats and autofixes with ruff, reports
  whatever ruff *can't* autofix, and — for edits under `deep_research/` or `tests/` —
  runs the offline suite (~1s, no keys, no network). A non-zero exit blocks with the
  failure in stderr. Both steps live in one script, in that order, deliberately:
  `ruff check --fix` rewrites the file, so a pytest run racing it in a parallel hook
  could read a half-rewritten tree. The same settings file `deny`s reads *and* edits
  of `.env` and `.deep_research/**` (live agent state: checkpoints, memories,
  *pending approvals*) — both gitignored, so git cannot undo damage to them — and
  `deny`s *edits* of `uv.lock`, which **is** committed: regenerate it with `uv lock`,
  never hand-edit it.
- **Tests** live in `tests/` (pytest). The offline suite is deliberately narrow —
  it targets the branching logic in `cli.py` and the load-bearing wiring
  invariants (the `open_agent()` assembly smoke test, the `GATED_TOOLS` safety
  gate, the Opus 4.8 no-sampling invariant, the `/memories/` route and its store
  namespace, and deepagents 0.7.0 backend readiness), not the agent's LLM output —
  that half lives in `evals/` (see *Evaluating it*, below). Tests
  use *real* langchain/langgraph types so fakes match runtime shapes. The `live`
  marker is registered and **deselected by default** (`addopts = -m 'not live'`);
  those need real keys. New behavior should come with a test in the matching
  file; verify a safety/invariant test actually bites by breaking the source and
  watching it go red.
- **The project is installable** (`[build-system]` = hatchling; the import package
  `deep_research` differs from the distribution name `deep-research-agent`, wired
  via `[tool.hatch.build.targets.wheel]`). So `uv sync` installs it editable and
  `import deep_research` works without a `pythonpath` shim.
- **CI** (`.github/workflows/ci.yml`): a `lint` job (ruff + `ruff format --check` +
  ty) and a `test` matrix over Python 3.11–3.13, all via `uv`. The offline suite
  needs no secrets. Keep it green.
- **No console-script entry point** — the app is invoked only as a module
  (`python -m deep_research` → `__main__.py` → `cli.main`).
- **`ruff` selects more than the defaults** (`[tool.ruff.lint]` in `pyproject.toml`):
  `E,F,I,UP,B,SIM,RUF,BLE`, with `E501` ignored because the formatter owns line
  length. `BLE` is load-bearing — it is what makes the `# noqa: BLE001` on the
  broad `except` in `cli.py::main` an *enforced* suppression; under ruff's
  defaults that rule is off, so the directive would be dead (`RUF100` catches
  exactly this). `target-version` is inferred from `requires-python`, so the
  3.11 floor governs `UP` fixes without a separate setting.
- **`ty`** (Astral's type checker) is a pinned `dev` dependency; run it with
  `uv run ty check`. Some source files carry inline `# ty: ignore[...]` comments
  — see the deliberate false-positive suppression in `config.py::build_model`.
- Requires `.env` with `ANTHROPIC_API_KEY` and `TAVILY_API_KEY` (copy from
  `.env.example`). `config.missing_keys()` hard-exits the CLI if either is unset.

## Architecture: the three things that span multiple files

Editing any one of these means understanding the others.

### 1. Two-layer persistence, both disk-backed (`agent.py` + `config.py`)

Deep Agents separates two kinds of state; this project gives each a *different*
sqlite file so everything survives a restart with no DB server:

- **Checkpointer** (`SqliteSaver` → `.deep_research/checkpoints.sqlite`) — per-`thread_id`
  conversation, todo list, and any *pending HITL interrupt*.
- **Store** (`SqliteStore` → `.deep_research/memories.sqlite`) — long-term memory
  shared across *all* threads.

The bridge between them is `build_backend()` in `agent.py`: a `CompositeBackend`
whose `default=StateBackend()` (ephemeral, per-thread, but checkpointed) and whose
`routes={"/memories/": StoreBackend(namespace=lambda _runtime: MEMORY_NAMESPACE)}`
sends only that path prefix to the durable Store. **Consequence:** a file the agent
writes to `/memories/foo.md` is readable in the next session; anything it writes
elsewhere lives only in that thread. The system prompt in `agent.py` encodes this
convention — it now tells the agent to write to `/memories/` *or nothing*, since the
report itself goes to the user (see *Evaluating it*) and every write costs a human
approval — so changing the route or the prompt must stay in sync.

`MEMORY_NAMESPACE = ("filesystem",)` is the Store namespace those files are filed
under, and its **value must never change**. It is exactly what deepagents' legacy
auto-detection resolves to for this app (its `assistant_id` branch is a LangGraph
Platform concept a local CLI never sets), so it is the key every note already in
`memories.sqlite` lives under — change it and the user's durable memory is orphaned,
silently. Passing it *explicitly* is separately required: a `StoreBackend` with no
`namespace` is itself deprecated for removal in 0.7.0.

The backend is passed to `create_deep_agent()` as an **instance**
(`backend=build_backend()`). deepagents also accepts a
`Callable[[Runtime], BackendProtocol]` factory there, and this project used to — but
the factory form, along with `StateBackend(runtime)` / `StoreBackend(runtime)` and a
`StoreBackend` with no explicit `namespace`, is deprecated for **removal in
deepagents 0.7.0** (the backends resolve the runtime themselves now). Hence the
`deepagents>=0.6.12,<0.7` cap in `pyproject.toml`: it is capped at the *minor*
because deepagents is 0.x, so that is where it breaks.

**Three tests in `test_agent_wiring.py` keep this 0.7.0-ready**, because the
deprecations fire at *different times* and no single test sees them all:
`test_backend_construction_is_free_of_deprecation_warnings` (the `runtime` form —
warns at construction), `test_memory_namespace_is_explicit_and_unchanged` (the
missing-`namespace` form — warns only when a store op actually *resolves* the
namespace, so the construction test cannot see it), and
`test_open_agent_passes_a_backend_instance_not_a_factory` (the factory form — it
asserts on what `create_deep_agent()` actually receives). None of these is reachable
from the assembly smoke test: a backend is not exercised until the first filesystem
tool call at *invoke* time.

### 2. The `interrupt_on` ↔ `checkpointer` dependency (`agent.py`)

`GATED_TOOLS` (`write_file`, `edit_file`, `execute`) is passed as `interrupt_on`,
which pauses the graph for human approval. **This REQUIRES a checkpointer** — the
pending interrupt is persisted there. The two are wired together in the same
`create_deep_agent()` call; don't add interrupts without a checkpointer, and note
that gating a *new* tool is just adding its name to `GATED_TOOLS` (a `True` expands
to all four decisions — `approve`, `edit`, `reject`, `respond`; an
`InterruptOnConfig` narrows them).

### 3. The HITL resume loop is split across `agent.py` and `cli.py`

`agent.py` declares *which* tools interrupt. `cli.py` implements the protocol that
drives them:

- `agent.invoke(...)` returns a state containing `__interrupt__` when a gated tool
  is proposed.
- The middleware bundles all of *one agent's* pending tool calls into a single
  interrupt whose value carries **two parallel lists**: `action_requests` (what the
  agent wants to do — name/args/description, and *not* what may be decided about it)
  and `review_configs` (per-tool `allowed_decisions`, keyed by `action_name`).
  `_collect_decisions` returns `dict[interrupt_id, list[decision]]` — one decision
  per `action_request`, in order, *within each interrupt* — and looks the permitted
  decisions up **by name**, not by position.
- **A turn can hold more than one interrupt.** The orchestrator dispatches each
  `task` call as its own concurrent graph task, and every subagent inherits
  `interrupt_on` — so two `researcher`s fanned out in one turn (which `SYSTEM_PROMPT`
  explicitly encourages) each raise their own. Resume therefore takes a **mapping of
  interrupt id → that interrupt's resume value**:
  `Command(resume={interrupt_id: {"decisions": [...]}, ...})`. A flat
  `Command(resume={"decisions": [...]})` makes LangGraph raise `RuntimeError: When
  there are multiple pending interrupts, you must specify the interrupt id when
  resuming`. The mapping is also correct for the ordinary single-interrupt case, so
  there is one code path — keep it that way.
- Resuming can hit the *next* gated tool, so `cli.py` **loops**
  `while result.get("__interrupt__")` until the turn finishes.

**The menu is not hardcoded, and must not be.** The middleware raises `ValueError`
if a decision's type is outside that tool's `allowed_decisions`, and `main`'s broad
`except` would swallow it into a one-line `! error:` — losing the whole turn. So
`_prompt_decision` builds its options from the `ReviewConfig` it is handed. This is
invisible today only because every `GATED_TOOLS` value is `True` (all four
decisions); the moment one becomes an `InterruptOnConfig` that drops `edit`, a
hardcoded `[e]dit` option would break the turn.

If you change what's gated, or how decisions are shaped, both `_collect_decisions`
/ `_prompt_decision` in `cli.py` and `GATED_TOOLS` in `agent.py` are in scope.

### Lifecycle: `open_agent()` is a context manager for a reason

The compiled agent holds open sqlite connections (checkpointer + store), so it's
only valid inside the `with open_agent() as agent:` block. Run the entire session
inside it (as `cli.main` does); don't return the agent out of the block.

## Orchestrator ↔ subagent model

- The **orchestrator** (`agent.py`, `SYSTEM_PROMPT`) plans with the built-in
  `write_todos`, can call `tavily_search` directly for quick lookups, and delegates
  breadth to the `researcher` subagent via the built-in `task` tool.
- The **`researcher` subagent** (`subagents.py`) runs with **isolated context** —
  it can't see the main conversation and is **one-shot/stateless** per `task` call,
  so its prompt must be self-contained. It exists to keep the orchestrator's context
  lean by absorbing many searches and returning one cited summary.
- `write_todos` and `task` are **built into Deep Agents** — they're not defined in
  this repo. Only `tavily_search` (`tools.py`) is a custom tool here.

## Model constraint (real gotcha)

The default model is `claude-opus-4-8`, built in `config.py::build_model()` **with
no `temperature`/`top_p`/`top_k`** — Opus 4.8 returns a 400 if any sampling param
is sent. `ChatAnthropic` omits unset params, so leave them unset. Override the
model via `DEEP_RESEARCH_MODEL`; only widen sampling params if the target model
accepts them. Subagents inherit this model — the `researcher` dict in `subagents.py`
has no `model` key — so the rule covers them too.

`max_tokens` defaults to **16000** (`DEEP_RESEARCH_MAX_TOKENS`), and that is not
arbitrary: the CLI calls `agent.invoke()` (non-streaming), and 16k keeps a response
comfortably under the Anthropic SDK's HTTP timeout while still leaving room for a
synthesized report. Raising it materially means switching to streaming.

## Prompt caching is already on — don't wire it again

`create_deep_agent()` appends an `AnthropicPromptCachingMiddleware` itself, to the
orchestrator *and* to every subagent (`deepagents/graph.py`, `_append_prompt_caching_middleware`).
It sets three breakpoints — last system-prompt block, last tool definition (which
covers the whole contiguous tool set), and a top-level `cache_control` in
`model_settings` that auto-caches the growing message tail. So the system prompt,
the tool schemas, and the conversation history are all cached already. Adding
another caching middleware via `middleware=[...]` would give you *two* of them
fighting over the same breakpoints; don't.

**This is easy to get wrong, so check before you "fix" it.** Nothing in this repo
mentions caching, and the middleware hooks `wrap_model_call` — which is *not* a
graph node, so it never appears in `agent.nodes`. Reading `agent.py` and `config.py`
gives no hint it exists, and the natural (wrong) conclusion is that every call
re-sends the prefix at full price. It doesn't. More generally: `create_deep_agent()`
does substantially more than its arguments suggest, so "this repo doesn't configure
X" is not evidence that X is off — grep the installed package before acting.

The default TTL is **5m**. That covers a whole turn (its many model calls fire back
to back) but expires while a human reads a report or sits on an approval, so a slow
turn re-writes its prefix. Leave it: the 1h TTL doubles the write multiplier
(2x vs 1.25x) to buy back only the idle gap.

`test_live.py::test_prompt_caching_actually_serves_the_prefix_from_cache` asserts the
cache is genuinely *read* (`cache_read` > 0), not merely that the middleware is
present — a prefix under Anthropic's 4096-token minimum would honor `cache_control`
and still cache nothing. Measured, the system+tools prefix is **~11.9k tokens**, so
it clears that bar with room to spare. The cache is keyed on the **prefix, not the
thread**: a brand-new `thread_id` reads a prefix an earlier thread (or an earlier
*process*) warmed. That is why the test runs two one-turn threads rather than two
turns on one thread — an opening turn can't hit the sharp edge where invoking fresh
input on a thread with a *pending HITL interrupt* resumes the model node on a message
list ending in an assistant message, which Opus 4.8 rejects as prefill (400). `cli.py`
never trips this because it loops on `__interrupt__` and resumes with `Command(resume=...)`
instead of sending a new turn.

## Evaluating it (`evals/`) — the half pytest deliberately skips

`pytest` grades the *wiring*; `evals/` grades the *agent*. It runs the real thing
against a LangSmith dataset and scores the trajectory (the workflow `SYSTEM_PROMPT`
promises) and the prose (citations, responsiveness). Traces are already on — LangChain
auto-instruments from `LANGSMITH_TRACING`/`LANGSMITH_API_KEY`, which `config.py`'s
import-time `load_dotenv()` puts in the environment. No code wires it.

Four things here were **measured**, not inferred, and each one breaks a harness that
assumes otherwise:

- **The trajectory is not in the returned state.** `invoke()` on a two-part question
  returns messages containing `['ls', 'task', 'task', 'write_file']` and **zero**
  `tavily_search` calls — every search happens inside a `researcher`, whose context is
  isolated. The searches exist only in the stream (`subgraphs=True`, collect
  `ToolMessage.name` across all namespaces; a subagent's namespace looks like
  `('tools:<uuid>',)`) or in the trace. Measured: 7 searches, all in subagents, none
  visible to the orchestrator. An evaluator reading `result["messages"]` is blind to
  the agent's entire research activity.
- **Every full run interrupts**, because `write_file` is gated and step 5 of the prompt
  says to persist findings. An unattended `invoke()` returns `__interrupt__` and no
  answer, scoring 0 for the wrong reason. `harness._approve_all` mirrors `cli.py`'s
  id-keyed resume.
- **Grade the orchestrator against `orchestrator_trajectory`, never `trajectory`.**
  deepagents gives *every* declarative subagent its own `TodoListMiddleware` and
  `FilesystemMiddleware` (`graph.py:643-651`), so the `researcher` really can call
  `write_todos`, `ls` and `write_file` — whatever `subagents.py` lists in its `tools` —
  and its tool messages stream out *before* the parent's `task` result. Flatten the two
  and a researcher tidying up after itself scores the **orchestrator** a pass on plan /
  check-memory / persist, including on the very `write_todos` defect this eval exists to
  watch. It would read as "the prompt fix worked" when it had not. Only
  `searched_the_web` counts the whole tree, deliberately.
- **A unique `thread_id` is not isolation.** `/memories/` is routed to the Store, which
  is shared across *every* thread — so example 2 reads what example 1 wrote and skips
  researching. `harness._reset_state()` drops both databases between examples, which is
  exactly why `evaluate(..., max_concurrency=1)` is mandatory: `config.py` freezes
  `STATE_DIR` at import, so all examples share one directory and would delete each
  other's database mid-run. Parallelism here needs a *process* per example.
- **The evals OWN their state dir — they never inherit one.** This is a three-layer
  invariant and each layer exists because the one above it is not enough:
  `__main__.py` **overrides** `DEEP_RESEARCH_STATE_DIR` with a fresh temp dir rather
  than `setdefault`-ing it (`setdefault` no-ops against the value `load_dotenv()` just
  read from the user's `.env` — and `DEEP_RESEARCH_STATE_DIR` is the documented way to
  *relocate the agent's real state*, so the eval would have dropped the user's live
  memories); `_reset_state()` unlinks the two databases **by name** and never
  `rmtree`s the directory (a blank `DEEP_RESEARCH_STATE_DIR=` resolves to the repo
  root, and `Path("").resolve()` + `rmtree` deletes the working tree); and
  `ensure_isolated_state_dir` refuses any path that **is or contains** the live
  `.deep_research/`. If you touch any of these, keep the others.
- **The judge is Haiku, not the app's Opus.** Grading is cheap classification, and Opus
  4.8 400s on `temperature` — a judge wants `temperature=0`. Do not reuse
  `config.build_model()` for it.

The dataset (`evals/dataset.py`) is deliberately **reference-free**: examples carry a
structural expectation (`min_delegations`) rather than a hand-written gold answer, because
inventing facts about live services and grading against them is worse than not grading.
Add references by curating them from real traces, per the `langsmith-dataset` skill.

`evals` is listed in `[tool.hatch.build.targets.wheel]` not because it belongs in the
wheel — nothing imports it at runtime — but because the editable install is what puts it
on `sys.path` for `python -m evals` and `tests/test_evals.py`.

### What the evals found, and what came of it

**The user was being shown uncited claims — fixed.** `cli.py` printed only the *last*
assistant message, but the agent composes its cited report in the message that also
proposes `write_file` and then signs off after the tool returns. Measured: 33 source URLs
in the turn, **zero** in what the user saw, and a closing line pointing at a "summary
above" that had never been printed. `cli.render_turn()` now renders the whole turn.
`evals/harness.py` imports that exact function — the eval must grade the bytes the REPL
prints, or the citation metric is fiction. Two consequences: `/report.md` is gone from
`SYSTEM_PROMPT` (it existed to hold a report the user couldn't see, and cost a second
approval), and the prompt now tells the agent not to narrate, since its words all arrive
at once.

**`write_todos` compliance is a threshold effect, not a blanket failure.**
`TodoListMiddleware` injects *its own* system prompt that says, four different ways, to
skip the todo list for anything under three steps ("it is better to just do the task
directly and NOT call this tool at all"). `SYSTEM_PROMPT` step 1 now explicitly names and
overrides that guidance. Measured after the change: **5/5 across the dataset's five
questions, but only 2/5 when the same *marginal* question (a two-part comparison) is run
five times.** The agent is not ignoring step 1 — it is judging whether the question earns
a todo list, and a simple two-parter sits right on its threshold, so it flips. Harder
questions plan every time. Treat a `plans_with_todos` failure on an easy question as
signal about the *question*, not proof of a regression — and never "fix" it by weakening
the evaluator. A deterministic fix would
mean monkeypatching `deepagents.graph.TodoListMiddleware` to replace its prompt (the class
takes `system_prompt=`, but deepagents constructs it internally as `TodoListMiddleware()`,
and passing your own via `middleware=[...]` registers a *second* `write_todos` tool —
there is no dedupe). That coupling was judged not worth it; the eval watches it instead.

**`claims_are_cited` scores a proportion, and must not be turned back into a bool.**
"Is *every* claim cited?" is a conjunction over every claim in the report: at 95%
per-claim compliance a 30-claim report passes 0.95³⁰ ≈ 21% of the time, at 90% it passes
4%. The boolean therefore read 0 on anything long enough to be worth writing, and scored
"missing one citation" identically to "cited nothing" — no gradient, so it could never
show that a fix had worked. It measured 0 on 4 of 5 examples while `response_cites_sources`
passed 5 of 5. As a proportion the same five answers score 0%, 23%, 36%, 71%, 100%, which
localises the real gap: the agent *does* put URLs in the answer, but leaves dates, versions
and benchmark figures floating free of them. `_citation_score` carries the argument, and
`test_citation_score_is_a_proportion_not_a_conjunction` pins it.

This is the `create_deep_agent()` lesson again, and it is the second time it has bitten in
this repo: **the harness injects prompts and middleware you never wrote.** "The repo
doesn't configure X" is not evidence that X is unconfigured — grep the installed package.

## Extending it (where things go)

- **New tool** → build in `tools.py`, then add to `tools=[...]` in `agent.py`
  (orchestrator) or a subagent's `tools` in `subagents.py`.
- **New subagent** → return another `SubAgent` dict from `subagents.py`, add to
  `subagents=[...]` in `agent.py`.
- **Gate a tool** → add its name to `GATED_TOOLS` in `agent.py`.
- **Production persistence** → swap `SqliteStore`/`SqliteSaver` for the Postgres
  equivalents in `open_agent()`; the `CompositeBackend` routing is unchanged.
- **Env overrides** (all read in `config.py`): `DEEP_RESEARCH_MODEL`,
  `DEEP_RESEARCH_MAX_TOKENS`, `DEEP_RESEARCH_STATE_DIR`. All three are resolved into
  module-level constants at *import* time, so they must be set before
  `deep_research.config` is first imported — a `monkeypatch.setenv` in a test body is
  too late, which is why `tests/conftest.py` sets them as top-level code.
