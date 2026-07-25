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
from collections.abc import Sequence
from contextlib import ExitStack
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

# Long argument values get a scrolling code block past this many lines rather than
# growing the page without bound.
_SCROLL_AFTER_LINES = 20
_SCROLL_HEIGHT_PX = 420


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
    threads is safe. Two *concurrent turns on the same `thread_id`* are still a bad idea
    — that is a LangGraph checkpoint race, not a sqlite one — so the page disables input
    while a turn is in flight.
    """
    stack = ExitStack()
    agent = stack.enter_context(open_agent())
    return agent, stack


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

    Two deliberate differences from `cli._render_action`, both in the same direction:

    - **Nothing is elided.** The terminal caps the preview at `PREVIEW_LINES` because a
      scrollback buffer is a poor place to dump a report; a browser has no such excuse,
      and `st.code` scrolls. An elided review is a review that gets rubber-stamped.
    - **`language=None`**, i.e. plain monospace. Guessing a highlighter from the tool or
      the file extension would reintroduce exactly the per-tool special-casing the CLI
      avoids, and would need updating every time `GATED_TOOLS` grows.
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
        # A long or multi-line string is the thing the reviewer is here to read; give it
        # its own block. Everything else is a scalar — a path, a flag — and reads inline.
        if isinstance(value, str) and ("\n" in value or len(value) > 120):
            lines = value.splitlines() or [""]
            st.caption(f"{key} · {len(lines)} line{'s' if len(lines) != 1 else ''}")
            st.code(
                value,
                language=None,
                wrap_lines=True,
                height=(
                    _SCROLL_HEIGHT_PX if len(lines) > _SCROLL_AFTER_LINES else "content"
                ),
            )
        else:
            st.markdown(f"`{key}` · {_short(value, 200)}")


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

    Handles both stored shapes: deepagents writes `content` as a plain string, and as a
    `list[str]` under its legacy `file_format="v1"`.
    """
    store = getattr(agent, "store", None)
    if store is None:
        return []
    files: list[tuple[str, str]] = []
    for item in store.search(MEMORY_NAMESPACE, limit=100):
        value = getattr(item, "value", None)
        content = value.get("content", "") if isinstance(value, dict) else ""
        if isinstance(content, list):
            content = "\n".join(str(line) for line in content)
        key = str(getattr(item, "key", "?")).lstrip("/")
        files.append((MEMORY_ROUTE + key, str(content)))
    return sorted(files)
