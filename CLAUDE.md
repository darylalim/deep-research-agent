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
uv run python -m deep_research   # run the interactive REPL (the CLI front door)
uv run --group serve langgraph dev  # serve the SAME agent over HTTP for Studio / deep-agents-ui
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
  failure in stderr. **Two consequences for how you edit here:** `ruff check --fix`
  deletes an import whose first *use* lands in a *later* edit, so add an import and its
  first use in the SAME edit (splitting them silently drops the import); and since
  pytest reruns on every `.py` edit, sequence a multi-edit refactor so each step leaves
  the suite green (e.g. rewrite the last user of a symbol before deleting the symbol),
  or the hook blocks mid-refactor. Both steps live in one script, in that order,
  deliberately:
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
  namespace, deepagents 0.7.0 backend readiness, and the `langgraph dev`
  served-graph assembly — it must build with **no** local checkpointer/store yet
  keep the same gate), not the agent's LLM output —
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
that gating a *new* tool is adding its name to `GATED_TOOLS` (a `True` expands
to all four decisions — `approve`, `edit`, `reject`, `respond`; an
`InterruptOnConfig` narrows them).

**But a name in `GATED_TOOLS` that the backend never exposes is a no-op, not a safety
net** — the tool has to reach the model before an interrupt can fire on it, and
`execute` is exactly that case. `FilesystemMiddleware.wrap_model_call` strips `execute`
from `request.tools` on *every* model call unless the backend satisfies
`SandboxBackendProtocol`, and for a `CompositeBackend` that is decided by its
`.default` — ours is `StateBackend`. So the entry is **latent**: kept because it costs
nothing and is what stands between a future sandbox backend and unreviewed shell, but
it has never fired, and `SYSTEM_PROMPT` no longer promises a pause for it.
`test_execute_is_latent_because_the_backend_cannot_run_it` pins the reason and goes red
the day the backend changes — which is when the prompt needs its sentence back.

**Do not "fix" this by wiring `LocalShellBackend`.** It is one kwarg away and it is
`subprocess.run(shell=True)` on the host; its own docstring says it has no sandboxing,
isolation, or security restrictions. A keystroke at an approval prompt is not a security
boundary against a tool that can read `.env`. A research agent needs arithmetic, not a
shell — add a pure-Python tool if that gap ever bites.

The gate's *end-to-end* enforcement lives in the evals, not here:
`mutations_require_approval` (`evals/evaluators.py`) compares every mutation the agent
proposed against every tool that actually interrupted. `test_agent_wiring.py` only proves
the dict *says* `True`; measured, flipping `GATED_TOOLS["write_file"]` to `False` left
all six other code metrics and both judges green.

### 3. The HITL resume loop is split across `agent.py` and `cli.py`

`agent.py` declares *which* tools interrupt. `cli.py` implements the protocol that
drives them:

- **The CLI streams** (`stream_mode="updates"`, `subgraphs=True`) — same call the eval
  harness has always made — so an interrupt arrives as an `{"__interrupt__": (...)}`
  *chunk*, not as a key on a result. (`invoke()`'s `result["__interrupt__"]` was itself
  only a post-drain aggregate LangGraph assembled internally.)
- **Drain the stream, THEN prompt, then restream.** Not a style choice. An interrupt chunk
  does not end the stream: LangGraph does not treat a `GraphInterrupt` as a failure, so
  sibling tasks in the same superstep run on and a *second* researcher's interrupt arrives
  after the first. And the graph executes *inside* the generator — blocking on `input()`
  mid-iteration freezes the Pregel loop, and starting the resume stream tears the old
  generator down, cancelling a still-running researcher whose interrupt was never emitted
  and throwing away searches you already paid for. `_stream_turn` drains; `main` decides on
  the whole set; the loop restreams with `Command(resume=...)`.
- **The same interrupt is emitted TWICE, and `_collect_decisions` must dedupe by id.**
  With `subgraphs=True`, an interrupt raised inside a subagent is emitted at the subagent's
  namespace *and* again, bubbled, at the root — same `Interrupt.id`. Prompting per
  occurrence asks the human to approve one researcher's `write_file` twice and (since the
  resume mapping is keyed by id) silently keeps only the second answer. The old blocking
  `invoke()` never saw this because it streams with `subgraphs=False`, so the child emitted
  nothing — the duplication appears the moment you stream.

  **`evals/harness.TurnRecorder` must dedupe it too, and for a worse reason.** `_approve_all`
  is immune by accident (a dict keyed by id), but `TurnRecorder.gated` appended one entry per
  *occurrence* — and because `mutations_require_approval` compares **multisets**, that surplus
  entry silently absorbs a genuinely *ungated* mutation of the same name. Measured: two
  proposed `write_file`s, one real interrupt emitted twice, a file written unreviewed —
  **score 1**. The safety metric, defeated by the thing it was written to catch. Both sides of
  the comparison have to count real events, not emissions.

- **Dedupe on TOOL-CALL ids, never on `BaseMessage.id`.** Both `cli.ActivityFeed` and
  `harness.TurnRecorder` learned this the hard way. A resumed superstep re-emits the *cached
  writes* of siblings that already succeeded (`_reapply_writes_to_succeeded_nodes`), and those
  arrive as **fresh** `ToolMessage` objects — measured: `id=None` on the first pass, a
  brand-new uuid on the resume. A message-id seen-set matches neither, so every replayed
  result gets through: the feed reprinted a finished researcher's line once per approval
  round, and `TurnRecorder` counted one delegation as two, which would let
  `delegates_breadth` pass an example demanding more than actually happened. A tool call
  executes exactly once; its id is the honest key.

- **A rejection reaches the stream as `status="error"`.** `HumanInTheLoopMiddleware` answers a
  rejected call with a synthetic `ToolMessage` whose status is `"error"` and whose content is
  *the human's own reason*, if they gave one. So nothing downstream can tell a rejection from
  a crash — the feed printed `! write_file failed: too risky` at the person who had just typed
  `r`. `main` passes the decisions it collected to `ActivityFeed.note_declined`; a feed that
  guesses from the message content cannot be right.

- **A thread rewrite is not new activity.** `PatchToolCallsMiddleware.before_agent` answers
  dangling tool calls by returning `{"messages": [RemoveMessage(REMOVE_ALL_MESSAGES), *the
  entire thread]}` — and it fires on exactly the turn *after* one abandoned at an approval
  prompt. Anything that renders stream messages must skip an update carrying a `RemoveMessage`,
  or that turn opens by replaying the previous turn's whole feed.
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
  `while pending := _stream_turn(...)` until the turn finishes.

**The feed shows actions; the answer comes from state.** `ActivityFeed` renders the tool
activity on the stream — the plan, each delegated sub-question, every search query — and
prints **no prose**, because the stream carries the *researchers'* assistant messages and
the user must never see one. The printed answer is `render_turn(agent.get_state(config).values)`,
which is the same call `evals/harness.py` makes to build the `response` it grades. Build the
printed answer from stream chunks instead and the citation metrics become fiction — and a
subagent's internal prose gets shown as the agent's answer. Same rule for `/export`
(`render_thread`), which reads the checkpoint for exactly this reason even though the
streaming loop has the chunks in hand.

**The feed's plan and `ls` lines are orchestrator-only, and that is the same rule the evals
enforce.** Every declarative subagent gets its own `TodoListMiddleware` *and*
`FilesystemMiddleware`, so a `researcher` really does call `write_todos` and `ls` — and
those chunks stream out under its namespace. Render them namespace-blind and you print a
researcher's private checklist as the agent's plan (appearing to supersede the plan the user
was just shown), and print `⌕ /memories/` on a turn where the orchestrator never looked —
hiding the very direct-path defect this file tells you to keep watching. It is the
`orchestrator_trajectory` vs `trajectory` distinction, in the display layer. Also note the
`ls` body is **not** newline-separated: deepagents builds it as `str(paths)`, a Python list
repr (`"[]"`), so counting lines reports "1 file(s)" for an empty store, forever.

A subagent's stream namespace (`('tools:<pregel-task-uuid>',)`) **cannot be bound** back to
the `task` tool-call that spawned it — the uuid is not the tool-call id, and correlating
them would mean assuming dispatch order matches emission order under concurrency. So the
feed doesn't pretend to: dispatch and completion lines carry the real sub-question
(recovered via `tool_call_id`), and each search line carries its query. The feed's seen-set
is keyed on **call ids**, not message ids — a re-streamed superstep re-emits cached writes
as fresh `ToolMessage` objects, and `BaseMessage.id` is optional, so message-id dedupe lets
the duplicate through (observed: a researcher's completion line printed twice, once per
approval round).

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

### The second front door: serving over HTTP (`graph.py` + `langgraph.json`)

`open_agent()` is the CLI's way in; `graph.py` is the other one — the module
`langgraph dev` (and `langgraph up` / LangGraph Platform, and the `deep-agents-ui`
web app) loads. Both delegate to one shared builder, `agent.build_agent(*,
checkpointer=None, store=None)`, which owns the entire assembly — model, tools,
subagent, system prompt, HITL gate, `/memories/` routing, name — so the two front
doors genuinely **cannot** drift: add a tool or subagent in `agent.py` and both gain
it in one edit. The single deliberate difference is persistence: `graph.py` calls
`build_agent()` with **no `checkpointer=` and no `store=`**. On the server the topology inverts — the
*server* owns persistence and injects both at runtime — so a compiled-in
checkpointer is redundant at best and overridden at worst. Omitting it is also what
keeps `interrupt_on` legal here: `create_deep_agent(interrupt_on=...)` compiles with
no checkpointer (the requirement is enforced at *invoke* time, which the server
satisfies), so the HITL gate fires server-side exactly as for the CLI —
`deep-agents-ui`'s tool-approval UI renders the same `action_requests` /
`review_configs` payload `cli.py::_collect_decisions` parses. Verified live: the
served graph's node list contains `HumanInTheLoopMiddleware.after_model`.

`build_backend()` is reused **unchanged**, and that is the load-bearing part.
`StoreBackend` resolves the store from the runtime via `get_store()` (deepagents
`backends/store.py`), not from a constructor arg — so `/memories/` transparently
reads the *server's* store instead of the CLI's `SqliteStore`: same route, same
`MEMORY_NAMESPACE`, different concrete backend. **Consequence:** notes the CLI wrote
to `.deep_research/memories.sqlite` are *not* visible to a served instance; it is a
different physical store under the same namespace key.

**`langgraph.json` must reference the graph as a MODULE path, never a file path:**
`"deep_research.graph:graph"`, not `"./deep_research/graph.py:graph"`. The loader
(`langgraph_api/graph.py`) branches on one character — a value containing `/` is
imported by file path (`spec_from_file_location` → a standalone module with no
package parent → `from .agent import …` dies with "attempted relative import with no
known parent package"); no `/` means a dotted-module import
(`importlib.import_module` → proper package → relative imports work). The project is
installed editable, so the dotted form resolves and lets `graph.py` keep the same
relative imports as every other module. Measured, not guessed: the file-path form
was tried and crashed the server on startup.
(The offline suite only pins *assembly*, never serving — so verify a `graph.py` /
`langgraph.json` change live: `uv run --group serve langgraph dev`, then curl
`/assistants/search`, expecting graph_id `research`.)

And it points at the module-level *compiled* `graph`, not the `build_graph`
**factory** (`deep_research.graph:build_graph`): `langgraph_api` re-invokes a graph
factory on **every run** (`invoke_factory`), which would rebuild the model, tools,
and subagent per request; a compiled graph is built once at import and reused. The
cost is that importing `graph.py` constructs the agent — which is why the tests
import it *inside* the test body, so a build break fails those tests rather than
erroring the whole file at collection.

`langgraph dev` needs the `inmem` extra (`langgraph-cli[inmem]`), kept in its own
**`serve` dependency-group** — out of the default set, so the lint/test CI jobs'
bare `uv sync` stays lean — and run with `uv run --group serve langgraph dev`. It
writes throwaway pickled state to `.langgraph_api/` (gitignored). Three tests in
`test_agent_wiring.py` pin the wiring: `test_served_graph_assembles_offline` (it
compiles), `test_served_graph_delegates_to_the_shared_builder` (it routes through
`build_agent`, not a re-inlined `create_deep_agent`), and
`test_shared_builder_gates_and_routes_without_persistence` (no checkpointer/store,
yet the SAME `GATED_TOOLS` / `SYSTEM_PROMPT` / `backend` **objects** *and* a
non-empty tool/subagent set — identity checks, so a divergent second assembly goes
red).

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

`max_tokens` defaults to **32000** (`DEEP_RESEARCH_MAX_TOKENS`), and what makes that
safe is **`streaming=True` on the model**, not anything about the CLI. The two are
orthogonal and it is easy to conflate them:

- `streaming=True` flips the *model's own HTTP request* to SSE (`_should_stream()` →
  `_stream()` → `generate_from_stream()`), while still handing the graph one complete
  `AIMessage`. Nothing downstream — LangGraph, deepagents, the HITL middleware, the eval
  harness — can tell the difference.
- The **graph's** `stream_mode` does *not* affect the wire format. The agent's model node
  calls `model_.invoke()` unconditionally, so `cli.py` streaming with
  `stream_mode="updates"` buys a live activity feed and **zero** `max_tokens` headroom.

This corrects a premise that was wrong here for a long time. The old note claimed 16k kept
responses "comfortably under the Anthropic SDK's HTTP timeout". **There is no such
timeout**: langchain passes `default_request_timeout=None` straight into
`anthropic.Client(timeout=None)`, and the httpx client ends up with `Timeout(timeout=None)`
— measured. That also *disarms* the SDK's own guard, which only fires when the client still
carries the SDK default timeout. So a non-streaming request over the guard's threshold
(`3600 * max_tokens / 128_000 > 600`, i.e. **max_tokens > 21_333**) would not raise — it
would hang the REPL indefinitely, which is strictly worse than the failure the 16k pin was
imagined to prevent.

**Raise `max_tokens` and set `streaming=True` together, or neither.** Verified that
streaming adds only `stream: true` to the payload, so the Opus 4.8 no-sampling-params rule
is untouched. Two tests in `test_config.py` pin this: `_should_stream()` must be True, and
the request payload must carry no sampling param. Note `streaming=False` passed
*explicitly* would hard-disable streaming via `_streaming_disabled()`, so don't "be safe"
by spelling out the default.

**Prompt caching under streaming is measured, not assumed.** Anthropic reports usage in the
`message_delta` when streaming, so `cache_read` arrives by a different route — and
`test_live.py::test_prompt_caching_actually_serves_the_prefix_from_cache` (which asserts
`cache_read > 0` after two real turns) **passes with `streaming=True`**. Run it after any
change to `build_model()`: a silently-cold cache roughly doubles the input cost of every
turn with no visible symptom. That test calls `agent.invoke()`, not the streaming CLI, which
is also the cleanest demonstration that the model's wire format is independent of the graph's
`stream_mode` — and that tool calls reassemble correctly from partial JSON deltas.

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

**`write_todos` compliance is ~80%, and the number needed 15 runs to find.**
`TodoListMiddleware` injects *its own* system prompt that says, four different ways, to
skip the todo list for anything under three steps ("it is better to just do the task
directly and NOT call this tool at all"). `SYSTEM_PROMPT` step 1 now explicitly names and
overrides that guidance. Measured after the change: 5/5 across the dataset, then **2/5 and
10/10 on two separate repeat-studies of the *same* question** — identical prompt (verified
from the traces) and clean cold starts (verified: one human message per run, so no state
leaked). Pooled, 12/15 ≈ **80%**. Both samples are individually unremarkable at p≈0.8; the
apparent contradiction is what small n looks like. Do not tune the prompt off a handful of
runs, and never "fix" a `plans_with_todos` failure by weakening the evaluator.

**The SAME mechanism bit a second time, in `BASE_AGENT_PROMPT`, and nobody had swept for
it.** `create_deep_agent()` **appends** `BASE_AGENT_PROMPT` *after* your system prompt
(`graph.py`: `final_system_prompt = system_prompt + "\n\n" + base_prompt`), so it also wins
on recency. It opens with "The user can see your responses and tool outputs **in real
time**" and closes with a `## Progress Updates` section: "For longer tasks, provide brief
progress updates at reasonable intervals". No harness profile suppresses it — the registered
ones are `claude-opus-4-7` / `sonnet-4-6` / `haiku-4-5`, so `claude-opus-4-8` gets an empty
profile.

Measured on a real REPL run: the agent printed **three paragraphs of stale narration** above
its answer ("Let me first check memory", "Memory is empty", "Both subagents returned solid,
cited findings"), one of which restated a `⌕ /memories/ · empty` line the user had already
watched scroll past. `SYSTEM_PROMPT`'s closing section now names and overrides it, and
`test_system_prompt_overrides_the_injected_narration_guidance` asserts *both* halves — so it
tells you which one broke if deepagents ever drops the section (our override becomes dead
weight) or someone trims ours (the narration returns).

**Note the trap in the reasoning, not just the bug.** When the streaming feed was being
planned, this fix was deferred on the argument that "streaming makes the narration guidance
*correct*, so the override shouldn't land". That was wrong: what streams is the **tool
feed**; the prose still arrives in one block from `render_turn` at the end. The rule is the
one this file keeps relearning — **measure it, don't reason about it**. Prompt-injection by
the harness is now 3 for 3 at beating an explicit instruction here (prompt caching,
`write_todos`, narration); assume there is a fourth and grep before you argue.

**A tempting hypothesis that the data killed.** In the 5-run study, every run that planned
failed to persist and vice versa — a perfect anti-correlation suggesting the prompt's five
steps compete for one budget. At n=10 it evaporated: **9 of 10 runs did both.** Had we
"fixed" the prompt architecture off n=5, we would have redesigned around noise. (Note the
n=10 study cannot formally *test* the association — planning hit 10/10, so the 2×2 has an
empty row and no power. The refutation comes from the nine counterexamples, not from the
p-value, which is vacuous.) A deterministic fix would
mean monkeypatching `deepagents.graph.TodoListMiddleware` to replace its prompt (the class
takes `system_prompt=`, but deepagents constructs it internally as `TodoListMiddleware()`,
and passing your own via `middleware=[...]` registers a *second* `write_todos` tool —
there is no dedupe). That coupling was judged not worth it; the eval watches it instead.

**`claims_are_cited` took two fixes, and both are the kind that get undone.**

*It scores a proportion, and must not become a bool again.* "Is *every* claim cited?" is a
conjunction over every claim in the report: at 95% per-claim compliance a 30-claim report
passes 0.95³⁰ ≈ 21% of the time, at 90% it passes 4%. The boolean read 0 on anything long
enough to be worth writing, and scored "missing one citation" identically to "cited
nothing" — no gradient, so it could never show a fix had worked. `_citation_score` carries
the argument; `test_citation_score_is_a_proportion_not_a_conjunction` pins it.

*It must justify each verdict, or it invents them.* Asked merely to list uncited claims, it
flagged figures whose citation sat at the end of their own bullet. Adjudicated claim by
claim against the text, **it was wrong on 9 of 12** — it does not spontaneously honour
inherited attribution (a source closing a bullet, paragraph, or the line under a table
covers what it closes). The `_UncitedClaim` schema now forces it to quote the nearest
citation and say why that source fails, so it has to look before it accuses. That single
change moved the same five stored answers from 0/23/36/71/100% to **25/83/87/90/100%**
(mean 46% → 77%) — and 77% matches what independent adjudicators found by hand. **Do not
simplify `uncited_claims` back to a list of strings.**

The corrected numbers also relocated the defect. It was never "the agent doesn't cite":
every *delegated* answer scored 83-100%, because `RESEARCHER_PROMPT` mandates inline
citations and anything a subagent touches comes back sourced. It was the orchestrator's own
`tavily_search` path — the one answer it researched itself scored **25%**. `SYSTEM_PROMPT`
step 4 now says citation applies to what you gathered yourself, not only to what a subagent
handed you. Measured A/B on the same dataset with the same judge: **77% → 98% mean
coverage**, and that direct-search answer went 25% → 100%.

**The orchestrator's own `tavily_search` path is a standing blind spot — check any new
workflow rule against it.** Two of the five `SYSTEM_PROMPT` steps turned out to apply only
when the agent delegated. Citation was one (25% on the answer it researched itself, vs
83-100% when a subagent did). Checking `/memories/` was the other, and it was total:
measured **0 of 5** direct lookups ran `ls` first — the trajectory was a bare
`['tavily_search']` every time. The mechanism is structural, not random: `RESEARCHER_PROMPT`
independently disciplines everything a subagent touches, so the delegated path gets a second
enforcement the direct path never sees, and a rule stated once in `SYSTEM_PROMPT` quietly
covers only half the agent's behaviour. Naming the direct path explicitly in step 2 took it
to **5 of 5**. When you add a rule the orchestrator must follow, say whether it holds for a
quick lookup it runs itself — the answer is almost always yes, and it will not infer it.

**`answers_the_question` needed the identical treatment, so treat this as the pattern, not
an anecdote.** As a bare bool it failed an answer carrying complete per-tier RPM/ITPM/OTPM
tables, because that answer *opened* with a caveat that exact figures move and the reader's
console is authoritative — the judge read the hedge and stopped. Given the same
evidence-forced, proportional schema (`_UnansweredPart`: quote the passage that comes
nearest, say why it falls short) it scores the same text 2/2. **An LLM judge's false
positives are an output-schema problem, not a prompt-wording one.** Both judges had already
been *told* the right rule in prose and ignored it; making the evidence a required field is
what fixed them. Three separate judge miscalibrations in this file's history mistook a judge
defect for an agent defect — check the instrument before you tune the agent.

This is the `create_deep_agent()` lesson again, and it is the second time it has bitten in
this repo: **the harness injects prompts and middleware you never wrote.** "The repo
doesn't configure X" is not evidence that X is unconfigured — grep the installed package.

## Extending it (where things go)

- **New tool** → build in `tools.py`, then add to `tools=[...]` in `agent.py`
  (orchestrator) or a subagent's `tools` in `subagents.py`.
- **New subagent** → return another `SubAgent` dict from `subagents.py`, add to
  `subagents=[...]` in `agent.py`.
- **Gate a tool** → add its name to `GATED_TOOLS` in `agent.py`. Check the model is
  actually *offered* that tool, though — gating a name the backend never exposes is a
  no-op (see `execute`, above).
- **Production persistence** → swap `SqliteStore`/`SqliteSaver` for the Postgres
  equivalents in `open_agent()`; the `CompositeBackend` routing is unchanged.
- **Serve it over HTTP / drive it from a web UI** → `uv run --group serve langgraph
  dev` loads `deep_research.graph:graph` (see *The second front door*, above). Point
  `deep-agents-ui` or LangGraph Studio at `http://127.0.0.1:2024`, assistant id
  `research`. `graph.py` delegates to `agent.build_agent()` with no checkpointer/store
  — the server provides them, so don't add them back; keep `langgraph.json`'s graph
  value a module path (not a file path) pointing at the compiled `graph` (not the
  `build_graph` factory).
- **Env overrides** (all read in `config.py`): `DEEP_RESEARCH_MODEL`,
  `DEEP_RESEARCH_MAX_TOKENS`, `DEEP_RESEARCH_STATE_DIR`. All three are resolved into
  module-level constants at *import* time, so they must be set before
  `deep_research.config` is first imported — a `monkeypatch.setenv` in a test body is
  too late, which is why `tests/conftest.py` sets them as top-level code.
