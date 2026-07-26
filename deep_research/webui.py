"""Streamlit front door: display and the approval protocol, nothing else.

The third way into the same agent, after `cli.py` (the terminal REPL) and `graph.py`
(the `langgraph dev` / Studio / deep-agents-ui server). Like both of those, this adds
nothing to what the agent *is* — `agent.build_agent()` stays the single source of that,
so a tool or subagent added there shows up here for free.

What this module adds is a second *renderer*, and every rule about what a user may be
shown is imported from `cli.py` rather than restated:

- `ActivityFeed` decides what happened; `StreamlitFeed` only re-skins it via `_emit`.
- `thread_sections` slices the thread into turns; the page turns slices into bubbles.
- `pending_reviews` / `allowed_decisions_by_tool` parse an interrupt; the widgets here
  present it.
- `export_markdown` builds the transcript the download button hands over, byte for byte
  the same document `/export` writes.

The one rule worth repeating, because it is the easiest one to lose when a stream is
right there in your hand: **the answer comes from the checkpoint, never from the
stream.** The stream carries the *researchers'* assistant messages, which the user must
never see, and `evals/harness.py` grades exactly what `render_turn` / `thread_sections`
produce. Building chat bubbles out of stream chunks would leak subagent-internal prose
into the UI and make the citation metrics fiction.

Some names imported below are underscore-private to `cli.py`. That is deliberate: they
are private to the *package*, not to the module, and the alternative — a second
implementation of "how a proposed action is described to a reviewer" — is the exact
duplication this file exists to avoid.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from typing import Any

import streamlit as st

from .agent import MEMORY_NAMESPACE, MEMORY_ROUTE, open_agent
from .cli import (
    DECISION_KEYS,
    DEFAULT_ALLOWED_DECISIONS,
    ActivityFeed,
    FeedEvent,
    _one_line,
    _reviewer_note,
    _short,
    allowed_decisions_by_tool,
    pending_reviews,
)

# A long argument value gets a scrolling code block past this many lines rather than
# growing the page without bound.
_SCROLL_AFTER_LINES = 20
_SCROLL_HEIGHT_PX = 420

# Thread ids with a turn currently running, PROCESS-wide. Module-level mutable state is
# normally a bug in a Streamlit app — `st.session_state` is the per-user store and this
# is shared across every session — but shared is exactly the point here: the thing being
# guarded (one LangGraph thread's checkpoint) is shared too. See `claim_thread`.
_ACTIVE_THREADS: set[str] = set()
_ACTIVE_THREADS_LOCK = threading.Lock()


@contextmanager
def claim_thread(thread_id: str) -> Iterator[bool]:
    """Take exclusive process-wide use of `thread_id`; yields False if already taken.

    **Per-session state cannot do this job, and assuming it could was a real bug.** The
    page disables its own input while a turn runs, but `st.session_state` is per browser
    session, so two tabs — both defaulting to thread `main` — could each start a turn and
    run two concurrent `agent.stream()` calls against one checkpoint. That is a LangGraph
    write race, not a sqlite one, so the backends' own locks do not cover it.

    Deliberately non-blocking: the caller is told "no" and says so, rather than waiting.
    A blocking lock would park a Streamlit script thread for the several minutes a
    research turn takes, which reads to the second user as a hung page.

    **Scope is the streaming window, not the whole turn.** The claim is released while a
    turn sits at an approval prompt, because holding it across an indefinite human wait
    would wedge the thread for the process's lifetime if that person walked away. A
    second session arriving during that window sees the same pending interrupt anyway,
    via `recover_pending`, which is the sane outcome rather than a race.

    Note this guards a *process*. Running the REPL and the browser against the same
    thread at the same time is still on the operator — as is the fact that
    `SqliteStore.from_conn_string` (unlike `SqliteSaver`) sets no `journal_mode=WAL`, so
    simultaneous writers on `memories.sqlite` can raise `database is locked`.
    """
    with _ACTIVE_THREADS_LOCK:
        claimed = thread_id not in _ACTIVE_THREADS
        if claimed:
            _ACTIVE_THREADS.add(thread_id)
    try:
        yield claimed
    finally:
        if claimed:
            with _ACTIVE_THREADS_LOCK:
                _ACTIVE_THREADS.discard(thread_id)


@st.cache_resource(show_spinner="Opening the research agent…")
def open_cached_agent() -> tuple[Any, ExitStack]:
    """The compiled agent, built once per server process and shared by every session.

    **The `ExitStack` is returned because it must not be garbage collected**, not
    because a caller wants it. `open_agent()` is a `@contextmanager` holding two live
    sqlite connections (`SqliteSaver`, `SqliteStore`) inside a `with closing(...)`; the
    stack owns the only reference to that generator. Drop the stack and the generator is
    finalized, `closing()` runs, and the agent is left holding two closed connections —
    a failure that would surface much later, as an unrelated-looking sqlite error on some
    subsequent turn. Caching the pair keeps the connections open for the process
    lifetime, which is exactly the lifetime a server wants.

    Both sqlite backends are opened `check_same_thread=False` and guard their cursors
    with a `threading.Lock`, so sharing one agent across Streamlit's per-session script
    threads is safe. Concurrent turns on the SAME `thread_id` are not safe, and the
    page's own disabled input does not prevent them — `claim_thread` does.
    """
    stack = ExitStack()
    agent = stack.enter_context(open_agent())
    return agent, stack


def should_render_question(
    asking: str | None, sections: Sequence[tuple[str, str]]
) -> bool:
    """Whether the in-flight question still needs drawing above the live feed.

    False once the checkpoint carries it. The human message is written on the graph's
    first superstep, so it is there for every rerun after the first, and drawing it
    unconditionally would show it twice for the whole length of an approval.

    **Substring, and stripped on both sides** — equality against the raw `st.chat_input`
    value missed two real cases, each double-drawing the question:

    - `thread_sections` strips every message, so a prompt with a trailing newline never
      matched its checkpointed form.
    - `thread_sections` joins consecutive same-speaker messages with `\\n\\n`. When the
      previous turn produced no assistant prose — exactly what a classifier refusal looks
      like, and a case this app already handles elsewhere — the old and new questions
      merge into ONE human section equal to neither.

    Extracted from the page so it can be tested; the failure it guards is invisible in a
    screenshot until an approval sits on screen with the question drawn twice.
    """
    if not asking:
        return False
    asked = asking.strip()
    return not any(kind == "human" and asked in text for kind, text in sections)


def recover_pending(
    state: Any, *, skip: frozenset[str] | set[str] = frozenset()
) -> list[Any]:
    """Interrupts the graph is still waiting on, read from the CHECKPOINT.

    Without this, a pending approval lives only in `st.session_state`, and a refresh, a
    new tab, or a session expiry strands it: the graph still considers the turn paused,
    but the page shows no form and re-enables the chat input. The next question then
    sends fresh input to a thread with a pending interrupt — the prefill 400 CLAUDE.md
    documents, which `cli.py` never trips only because it never leaves the process.

    It is also what makes the README's "a thread you start in one continues in the other"
    true for a thread the REPL left paused at an approval.

    **`skip` is what stops that recovery from being a trap, and it is not optional.**
    Clearing `st.session_state.pending` does NOT resume the graph — the interrupt is
    still in the checkpoint — so a page that abandons a turn and then recovers blindly
    reads the very same interrupt straight back on the next pass. Measured, before this
    argument existed: "Abandon this turn" left `pending == [Interrupt(id='i1', …)]` with
    the approval form redrawn directly under its own "Turn abandoned" notice, and an
    interrupt carrying no `action_requests` drove an unbounded full-rerun loop — 1019
    reruns in 6 seconds, each paying for `agent.get_state`. The session had no way out,
    in the one control whose entire job is being the way out.

    Passing the ids the session has deliberately given up on is therefore how the page
    matches the REPL. `cli.main` abandons a turn by letting its broad `except` drop it
    and reading the next question, leaving the graph paused; the dangling tool calls are
    answered on the following turn by `PatchToolCallsMiddleware.before_agent`, which is
    also what keeps that next question clear of the prefill 400. Skipping the interrupt
    rather than resuming it reproduces exactly that, and it is the only option that works
    for the case the escape hatch exists for: a tool whose `allowed_decisions` this UI
    cannot render may not permit `reject` either, so "resume with a rejection" would
    raise `ValueError` inside the middleware on precisely the stuck turn it was meant
    to free.

    `StateSnapshot.interrupts` is already the flattened
    `[i for task in tasks_with_writes for i in task.interrupts]` (langgraph
    `pregel/main.py`), so there is no need to walk `.tasks` here.
    """
    return [
        interrupt
        for interrupt in (getattr(state, "interrupts", ()) or ())
        if getattr(interrupt, "id", None) not in skip
    ]


def render_event(event: FeedEvent) -> None:
    """Draw one `FeedEvent` into whatever container is currently active.

    Ambient by design — it takes no container argument — which is what lets the same
    function serve the live stream (inside a `st.status`) and the replay of a finished
    turn (inside an `st.expander`) with no stale references carried across reruns.
    """
    kind = event.kind
    if kind == "plan":
        count = len(event.items)
        st.markdown(
            f":material/checklist: **Plan** · {count} step{'s' if count != 1 else ''}"
        )
        st.markdown("\n".join(f"{n}. {item}" for n, item in enumerate(event.items, 1)))
    elif kind == "delegate":
        # Truncated, though the browser could wrap it. A `task` description is the whole
        # self-contained prompt the orchestrator wrote — measured at ~13 rendered lines
        # each on a real two-researcher turn, which buried the searches around it and
        # made the live feed unscannable. More generous than the terminal's 90 because
        # there is room for two lines here, but not unbounded.
        st.markdown(
            f":material/person_search: **researcher** · {_one_line(event.text, 220)}"
        )
    elif kind == "search":
        # A subagent's searches are dimmed rather than indented: markdown collapses
        # leading whitespace, so the terminal feed's indent has no equivalent here.
        query = _one_line(event.text, 140)
        st.markdown(
            f':material/search: "{query}"'
            if event.is_orchestrator
            else f':material/search: :gray["{query}"]'
        )
    elif kind == "read":
        st.markdown(f":material/description: reading `{event.text}`")
    elif kind == "listed":
        st.markdown(f":material/folder_open: `{event.text}` · {event.detail}")
    elif kind == "done":
        st.markdown(
            f":material/task_alt: **researcher** · {event.text}"
            if event.text
            else ":material/task_alt: **researcher** finished"
        )
    elif kind == "rejected":
        st.markdown(f":material/block: `{event.text}` — rejected, as you asked")
    elif kind == "failed":
        st.markdown(
            f":material/error: :red[`{event.text}` failed] — "
            f"{_one_line(event.detail, 160)}"
        )
    elif kind == "refusal":
        st.markdown(f":material/warning: :orange[researcher · {event.text}]")


class StreamlitFeed(ActivityFeed):
    """`ActivityFeed` with the terminal swapped for the current Streamlit container.

    Only `_emit` changes, so every rule in `absorb` is inherited rather than
    reimplemented: dedupe on tool-call ids (not message ids — a replayed superstep
    re-emits cached writes as fresh `ToolMessage`s with new uuids), skip updates
    carrying a `RemoveMessage`, orchestrator-only plan and `ls` lines, and never a word
    of a researcher's prose. See `FeedEvent` for why that reuse is not optional.

    It also **keeps the events**, which the terminal never had to. A Streamlit rerun
    re-executes the script from scratch and discards everything previously drawn, but a
    turn that pauses for approval spans several reruns — so the feed has to be able to
    redraw itself. This instance lives in `st.session_state` for the whole turn, which is
    also what carries the seen-set across resume rounds and stops an approval from
    replaying lines the user already watched appear.
    """

    def __init__(self) -> None:
        super().__init__()
        self.events: list[FeedEvent] = []

    def _emit(self, event: FeedEvent) -> None:
        self.events.append(event)
        render_event(event)

    def replay(self) -> None:
        """Redraw every event so far, for the reruns an approval costs."""
        for event in self.events:
            render_event(event)


def render_action(request: dict[str, Any]) -> None:
    """Show a proposed action so a human can actually *read* it before allowing it.

    This is the whole point of the gate. `write_file` interrupts because someone should
    see the content before it lands in durable, gitignored `/memories/` that git cannot
    restore, and a reviewer who cannot read the diff approves it unread.

    Three deliberate differences from `cli._render_action`:

    - **Nothing is elided.** The terminal caps the preview at `PREVIEW_LINES` because a
      scrollback buffer is a poor place to dump a report; a browser has no such excuse,
      and `st.code` scrolls. An elided review is a review that gets rubber-stamped.
    - **`language=None`**, i.e. plain monospace. Guessing a highlighter from the tool or
      the file extension would reintroduce exactly the per-tool special-casing the CLI
      avoids, and would need updating every time `GATED_TOOLS` grows.
    - **Every value goes through `st.code`, including short scalars.** Not a style
      choice — arguments are written by the MODEL, and the earlier version rendered
      scalars with `st.markdown`, so a `file_path` of `[safe](http://elsewhere)` drew a
      link instead of the literal string that would be passed to the tool. Formatted
      markdown is precisely wrong on the one screen whose entire job is showing a
      reviewer, verbatim, what is about to happen.

      Sending everything down one path also deletes this function's copy of the CLI's
      long-vs-scalar classification (the `\\n`-or-over-120-chars test and its
      `splitlines() or [""]`). That duplication was the same drift `pending_reviews` and
      `allowed_decisions_by_tool` were extracted to prevent, and it sat in the app's only
      security-relevant screen. There is nothing left here to drift.
    """
    name = request.get("name", "<tool>")
    args = request.get("args", {})

    st.markdown(f"**:material/pause_circle: Approval required — `{name}`**")
    if note := _reviewer_note(request):
        st.info(note)

    if not isinstance(args, dict):  # not a shape the middleware produces, but cheap
        st.code(_short(args), language=None)
        return

    for key, value in args.items():
        text = value if isinstance(value, str) else _short(value, 200)
        lines = text.splitlines() or [""]
        count = f" · {len(lines)} lines" if len(lines) > 1 else ""
        st.caption(f"{key}{count}")
        st.code(
            text,
            language=None,
            wrap_lines=True,
            height=(
                _SCROLL_HEIGHT_PX if len(lines) > _SCROLL_AFTER_LINES else "content"
            ),
        )


def decision_controls(
    key: str, request: dict[str, Any], allowed: Sequence[str] | None
) -> dict[str, Any] | None:
    """Render the controls for one proposed action; return its decision, or `None`.

    `None` means "not decidable yet" — no choice made, an empty `respond`, or unparsable
    edit arguments — and the caller keeps submit disabled until every action yields one.

    Only the decisions `allowed` permits for *this* tool are offered, which is
    load-bearing rather than cosmetic: the middleware raises `ValueError` on a decision
    outside a tool's `allowed_decisions`, and that exception would surface as a dead turn.
    Every value in `GATED_TOOLS` is `True` today (all four decisions), but narrowing them
    with an `InterruptOnConfig` is supported and documented, so the menu is built from
    what the interrupt actually carries. `DECISION_KEYS` supplies the ordering so the CLI
    and the UI offer the same choices in the same sequence.

    **There is no default selection**, which is where this deliberately diverges from the
    REPL. The CLI defaults to approve because bare Enter has to mean something; a UI has
    no such affordance, so requiring an explicit click costs the reviewer nothing and
    removes the one path by which a gate degrades into a rubber stamp.
    """
    permitted = set(DEFAULT_ALLOWED_DECISIONS if allowed is None else allowed)
    options = [decision for decision in DECISION_KEYS if decision in permitted]
    if not options:
        # Gated with a decision set this UI cannot produce. Guessing would just raise
        # inside the graph, so say so instead.
        st.error(
            f"`{request.get('name')}` allows no decision this app can send "
            f"(tool allows: {sorted(permitted) or 'nothing'})."
        )
        return None

    choice = st.segmented_control(
        "Decision",
        options,
        format_func=str.capitalize,
        key=f"{key}:type",
        label_visibility="collapsed",
    )

    if choice is None:
        st.caption("Choose a decision to continue.")
        return None
    if choice == "approve":
        return {"type": "approve"}
    if choice == "reject":
        reason = st.text_input(
            "Reason for the agent (optional)", key=f"{key}:reason"
        ).strip()
        return {"type": "reject", **({"message": reason} if reason else {})}
    if choice == "respond":
        # The human answers *on behalf of* the tool; the tool never runs, so an empty
        # message would hand the model an empty tool result.
        message = st.text_area(
            "Reply to the agent on the tool's behalf", key=f"{key}:respond"
        ).strip()
        if not message:
            st.caption("A response needs a message.")
            return None
        return {"type": "respond", "message": message}

    # edit — prefilled with the real arguments, so narrowing a path or trimming a file is
    # an edit rather than a retype. Unparsable JSON returns None and NEVER falls back to
    # approving the original arguments: someone chose `edit` because the write looked
    # wrong, and a typo is not consent. This is the only security boundary the app has.
    raw = st.text_area(
        "Replacement arguments (JSON)",
        value=json.dumps(request.get("args", {}), indent=2, ensure_ascii=False),
        key=f"{key}:edit",
        height=_SCROLL_HEIGHT_PX,
    )
    try:
        edited = json.loads(raw)
    except json.JSONDecodeError as exc:
        st.error(f"Not valid JSON ({exc.msg}) — nothing will be approved.")
        return None
    if not isinstance(edited, dict):
        # `"/memories/y.md"` and `[1, 2]` are valid JSON but not valid *args*. The
        # middleware does not validate, so this would sail through into a ToolCall with
        # non-dict args and only blow up at tool execution.
        st.error('Arguments must be a JSON object, e.g. {"file_path": "..."}.')
        return None
    return {
        "type": "edit",
        "edited_action": {"name": request.get("name"), "args": edited},
    }


def reviewable_actions(pending: list[Any]) -> int:
    """How many distinct pending actions a human could actually decide on.

    Distinct because the same interrupt is emitted twice under `subgraphs=True`; see
    `pending_reviews`. Zero means the turn paused on something with no `action_requests`
    at all, which resuming cannot fix — the page abandons the turn rather than looping.
    """
    return sum(
        len(value.get("action_requests", [])) for _, value in pending_reviews(pending)
    )


def approval_form(pending: list[Any]) -> dict[str, list[dict[str, Any]]] | None:
    """Render every pending action; return `{interrupt_id: [decision, ...]}` on submit.

    `None` until the reviewer has produced a valid decision for *every* pending action
    and pressed the button. Partial state is never sent: a resume carrying fewer
    decisions than the interrupt has action requests is not a smaller approval, it is an
    undefined one.

    Keyed by interrupt id and deliberately not flattened, because **a turn can hold more
    than one interrupt**. The orchestrator dispatches each `task` call as its own
    concurrent graph task and every subagent inherits `interrupt_on`, so two
    `researcher`s fanned out in one turn each raise their own. LangGraph rejects a flat
    resume in that case with `RuntimeError: When there are multiple pending interrupts,
    you must specify the interrupt id when resuming`. The mapping form is also correct
    for the ordinary single-interrupt case, so there is one code path — the same rule
    `cli._collect_decisions` follows.

    Widget keys are `f"{interrupt_id}:{index}"`, which stays stable across the reruns
    that picking a decision costs. Anything derived from position in `pending` would not
    be: the same interrupt appears twice in that list, and nothing guarantees the order
    chunks arrived in.
    """
    by_interrupt: dict[str, list[dict[str, Any]]] = {}
    complete = True
    for interrupt_id, value in pending_reviews(pending):
        allowed_by_tool = allowed_decisions_by_tool(value)
        decisions: list[dict[str, Any]] = []
        for index, request in enumerate(value.get("action_requests", [])):
            with st.container(border=True):
                render_action(request)
                decision = decision_controls(
                    f"{interrupt_id}:{index}",
                    request,
                    allowed_by_tool.get(request.get("name")),
                )
            if decision is None:
                complete = False
            else:
                decisions.append(decision)
        if decisions:
            by_interrupt[interrupt_id] = decisions

    ready = complete and bool(by_interrupt)
    submitted = st.button(
        "Send decisions",
        type="primary",
        icon=":material/send:",
        disabled=not ready,
    )
    return by_interrupt if submitted and ready else None


def memory_files(agent: Any) -> list[tuple[str, str]]:
    """Everything under `/memories/`, as sorted `(path, content)` pairs.

    Read straight from the Store rather than through the agent, because this is a
    read-only view for the human and routing it through the model would cost a turn and
    an approval. `MEMORY_NAMESPACE` is imported, not spelled out — its value is the key
    every note already in `memories.sqlite` lives under, and a second literal is a second
    thing to get wrong.

    **The paths are reconstructed, not read verbatim.** `CompositeBackend` strips the
    route prefix before delegating, so a note the agent wrote to
    `/memories/pricing.md` is stored under the key `/pricing.md` — and a browser that
    displayed the raw key would show the user a path that does not exist, in an app whose
    whole subject is where findings are kept. `MEMORY_ROUTE` puts it back, and comes from
    the same constant the route itself is built from so the two cannot drift.

    **Paginated, because `store.search` takes a `limit` and silently returns no more
    than that.** This used to pass `limit=100` and stop, so the 101st note simply
    vanished from the one view whose subject is where durable findings are kept — with
    nothing distinguishing "never saved" from "not shown". deepagents' own
    `StoreBackend.ls` loops the same way (`backends/store.py`).

    Handles both stored shapes: deepagents writes `content` as a plain string, and as a
    `list[str]` under its legacy `file_format="v1"`.
    """
    store = getattr(agent, "store", None)
    if store is None:
        return []
    files: list[tuple[str, str]] = []
    page = 100
    offset = 0
    while True:
        items = store.search(MEMORY_NAMESPACE, limit=page, offset=offset)
        for item in items:
            value = getattr(item, "value", None)
            content = value.get("content", "") if isinstance(value, dict) else ""
            if isinstance(content, list):
                content = "\n".join(str(line) for line in content)
            key = str(getattr(item, "key", "?")).lstrip("/")
            files.append((MEMORY_ROUTE + key, str(content)))
        if len(items) < page:
            return sorted(files)
        offset += page


@st.cache_data(ttl=60, max_entries=1, show_spinner=False)
def cached_memory_files(_agent: Any) -> list[tuple[str, str]]:
    """`memory_files`, but not re-read from sqlite on every single rerun.

    The sidebar expander that shows these runs its body whether or not it is open, and a
    turn reruns the page on every approval click and widget change — so the uncached
    version pulled every note body out of sqlite each time, to render a panel nobody had
    opened.

    `_agent` is underscore-prefixed so Streamlit excludes it from the cache key (it is
    not hashable and never varies within a process); the key is therefore constant and
    `max_entries=1` is the whole cache. The 60s TTL bounds staleness on its own, and
    `refresh_memory_files()` drops it immediately after a turn that may have written.
    """
    return memory_files(_agent)


def refresh_memory_files() -> None:
    """Drop the memory cache, so a just-approved write shows up at once."""
    cached_memory_files.clear()


@st.fragment
def memory_browser(agent: Any, *, busy: bool) -> None:
    """The `/memories/` browser — a fragment for what it stops the page from doing.

    Every widget interaction in a Streamlit app reruns the whole script, which for this
    page means re-reading the checkpoint, rebuilding the export document, and redrawing
    every chat bubble in the thread. Picking a different note out of the sidebar depends
    on none of that; as a fragment, that click redraws this expander alone.

    **The point is that the page does not repaint, not that it is faster.** Measured on
    a 40-turn thread, a whole full rerun is order 7 ms of server work (`get_state`
    0.44 ms) — so read this as keeping the rest of the page *stable*, and do not
    reconstruct a cost argument for it. See CLAUDE.md for the numbers.

    `busy` is still honoured, and the reason is worth stating because the fragment
    makes it look redundant. A fragment-scoped rerun cannot tear down `agent.stream`
    the way `RerunException` from a full rerun does — but a fragment is *also*
    re-executed on every full rerun, and leaving a live control on screen during a turn
    is precisely the shape of bug this page has already paid for once. The control stays
    disabled.

    **Deliberately not `st.expander(..., on_change="rerun")`**, which is the reference's
    preferred fix for a collapsed expander still computing its body. It renders the
    selectbox only while the panel is open, and two tests reach for
    `sidebar.selectbox[0]` to prove `busy` reaches this widget at all
    (`test_input_is_disabled_while_an_approval_is_pending` and its idle positive
    control) — tests CLAUDE.md records as having once passed for the wrong reason, so
    weakening them is not a trade worth making.

    **Those tests are the WHOLE of the argument, and the cost sentence this docstring
    used to end on was wrong.** It claimed `cached_memory_files` reduces the
    closed-panel cost to a dict lookup. That is true of the sqlite read and false of
    what crosses the wire: the cache elides the read, not the elements. `st.code`
    below produces the selected note's full body on every rerun, and Streamlit sends a
    closed expander's contents to the frontend anyway — measured, a collapsed expander
    produced 15 of 15 elements where the lazy form produced none. The magnitude here
    is unmeasured, so if anyone ever times it and it is not small, the tests are the
    thing to renegotiate; do not reconstruct a cost defence in their place. This is
    the same error as the fragments' original speed rationale, one layer down.

    Note the page reaches the OPPOSITE conclusion for the transcript's work-log
    expanders, and the difference is this decorator, not the cost. A rerun fired from
    inside a fragment is fragment-scoped, so the lazy form could not tear down
    `agent.stream` from here; in the page body it could, and `st.expander` accepts no
    `disabled` to close that window. Safety decides it there; tests decide it here.

    Targets `st.sidebar` explicitly rather than inheriting whatever container happens to
    be ambient — the same rule `approval_panel` follows by creating its own
    `st.chat_message`. A fragment may write into a container made outside it, but only
    one already written to during a full run, and depending on the caller's `with` block
    for placement means moving that call silently relocates the panel. Call order within
    the sidebar still decides where it lands; nothing else does.
    """
    with st.sidebar.expander("Durable memory", icon=":material/database:"):
        try:
            # Cached: an expander runs its body whether or not it is open, so the
            # uncached read pulled every note out of sqlite on each rerun to fill a
            # panel nobody had opened.
            saved = cached_memory_files(agent)
        except Exception as exc:  # noqa: BLE001 — a sidebar read must not kill the page
            st.caption(f"Could not read memory: {exc}")
            saved = []
        if saved:
            # Keyed so the selection survives the options list changing under it — a
            # turn that writes a note calls `refresh_memory_files()`, and an unkeyed
            # widget's identity includes its options, so it would reset on that rerun.
            chosen = st.selectbox(
                "File",
                [path for path, _ in saved],
                label_visibility="collapsed",
                disabled=busy,
                key="memory_file",
            )
            st.code(dict(saved)[chosen], language=None, height=240, wrap_lines=True)
        else:
            st.caption(
                "Nothing saved yet. Findings the agent writes to `/memories/` persist "
                "across every thread and session."
            )
