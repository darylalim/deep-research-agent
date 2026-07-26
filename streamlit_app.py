"""Browser front door for the deep research agent — `streamlit run streamlit_app.py`.

A page script, deliberately: it is the sequence of things that appear on screen, and
every reusable decision it makes lives in `deep_research/webui.py` (which in turn
imports the display rules from `deep_research/cli.py`, rather than restating them).

**The turn is a state machine spread across reruns, and that is the whole design.**
Streamlit re-executes this file top to bottom on every interaction, but a research turn
can pause mid-flight for human approval — so the turn cannot live in one pass. Four
keys in `st.session_state` carry it:

    payload   the next thing to send the graph: a new question, or a `Command(resume=…)`
    question  what the human asked, until the checkpoint has it too
    feed      the `StreamlitFeed` for the turn in flight — events AND the seen-set
    pending   interrupts drained from the stream, waiting on a human

The loop those four implement is exactly `cli.main`'s `while pending := _stream_turn(…)`,
unrolled across reruns instead of iterations. That matters, because two of the rules it
encodes are not obvious and were paid for once already:

- **Drain the stream, THEN ask.** An interrupt chunk does not end the stream; sibling
  tasks in the same superstep keep running and a second researcher's interrupt arrives
  after the first. The graph also executes *inside* the generator, so pausing mid-
  iteration would freeze the Pregel loop, and abandoning it would cancel a still-running
  researcher whose searches you already paid for. `_stream_turn` exhausts the stream;
  only then does this page render an approval form.
- **The answer comes from the checkpoint, never from the stream.** The transcript below
  is built by `thread_sections(agent.get_state(...).values)` — the same slicing
  `/export` and `evals/harness.py` use. The stream carries the *researchers'* prose,
  which the user must never see.

**Script order is load-bearing too, in a smaller way.** The one genuinely slow thing on
an idle pass is `agent.get_state(config)`, which deserializes the thread's entire
message list; `export_markdown` and the transcript loop are both built from it. So the
page's own chrome — title, caption, chat input — is written *before* that read rather
than after it. Two things follow: the frame paints while the checkpoint loads instead of
behind it, and submitting a question returns from `st.chat_input` and reruns without
ever paying for the read whose results that pass would have thrown away.
"""

from __future__ import annotations

from datetime import UTC, datetime

import streamlit as st
from langgraph.types import Command

from deep_research import webui
from deep_research.cli import (
    _declined_tools,
    _stream_turn,
    _turn_refusal,
    export_markdown,
    thread_sections,
)
from deep_research.config import MEMORY_DB, MODEL_NAME, missing_keys

st.set_page_config(
    page_title="Deep research agent",
    page_icon=":material/travel_explore:",
    layout="centered",
)

# --- credentials -------------------------------------------------------------------
# Same gate as `cli.main`, which hard-exits rather than failing later inside a turn.
if missing := missing_keys():
    st.title("Deep research agent")
    st.error("Missing required environment variables.")
    for _key, _why in missing.items():
        st.markdown(f"- `{_key}` — {_why}")
    st.caption("Copy `.env.example` to `.env`, fill these in, then restart the app.")
    st.stop()

agent, _stack = webui.open_cached_agent()

st.session_state.setdefault("thread_id", "main")
st.session_state.setdefault("payload", None)
st.session_state.setdefault("question", None)
st.session_state.setdefault("feed", None)
st.session_state.setdefault("pending", [])
# Per-turn adornments, keyed by the answer's index in `thread_sections`. Both are
# display-only and rebuilt per thread — the durable record is the checkpoint.
st.session_state.setdefault("work_logs", {})
st.session_state.setdefault("refusals", {})
st.session_state.setdefault("notice", None)

busy = st.session_state.payload is not None or bool(st.session_state.pending)

# --- header and input ---------------------------------------------------------------
# ABOVE the checkpoint read on purpose (see the module docstring). Everything here is
# cheap and depends on nothing but session state, so it paints while `get_state` runs
# rather than behind it — and a submitted question reruns from here, skipping a read of
# the whole message list that this pass would only have discarded. `st.chat_input` is
# pinned to the bottom of the page by Streamlit, so moving it up the script does not
# move it up the screen.
st.title("Deep research agent")
st.caption(
    "Plans the work, delegates web searches to a subagent, and synthesizes a cited "
    "answer. Writing a file pauses for your approval."
)

prompt = st.chat_input("Ask a research question", disabled=busy, submit_mode="disable")
if prompt:
    st.session_state.update(
        question=prompt,
        payload={"messages": [{"role": "user", "content": prompt}]},
        feed=webui.StreamlitFeed(),
        notice=None,
    )
    st.rerun()

# --- sidebar: thread ----------------------------------------------------------------
with st.sidebar:
    st.subheader("Session", divider="gray")
    thread_id = st.text_input(
        "Thread",
        value=st.session_state.thread_id,
        disabled=busy,
        help=(
            "Conversations are checkpointed per thread and survive a restart. "
            "Memory under /memories/ is shared across every thread."
        ),
    ).strip()

if not thread_id:
    # The widget has no `key`, so its emptied value persists across reruns while
    # `st.session_state.thread_id` keeps the old name — a blank box and an app quietly
    # still on the previous thread, with nothing to notice it by. Say so rather than
    # picking a thread on the user's behalf.
    st.sidebar.warning(
        f"A thread needs a name — still on `{st.session_state.thread_id}`."
    )
elif thread_id != st.session_state.thread_id:
    # Work logs and refusal notes are indexed against the *previous* thread's
    # transcript, so they are meaningless here. The answers themselves are in the
    # checkpoint and reappear on their own.
    st.session_state.update(thread_id=thread_id, work_logs={}, refusals={}, notice=None)
    st.rerun()

config = {"configurable": {"thread_id": st.session_state.thread_id}}
# ONE read, and the only slow thing on an idle pass — which is why the page's chrome is
# written above it. `get_state` deserializes the thread's whole message list, the
# largest object in the app, and this used to be called twice more at the end of every
# turn.
state = agent.get_state(config)
values = state.values
sections = thread_sections(values)

# An approval that outlived the browser session. `pending` otherwise lives only in
# session state, so a refresh or a new tab strands the turn: the graph still considers
# it paused, but the page would show no form and re-enable the chat input, and the next
# question would send fresh input to a thread with a pending interrupt (the prefill 400
# CLAUDE.md documents). Recovered only when this session is not mid-turn — during a turn
# the live stream is the authority.
_idle = not st.session_state.pending and st.session_state.payload is None
if _idle and (recovered := webui.recover_pending(state)):
    st.session_state.pending = recovered
    # The feed for that turn died with the old session; a fresh one still carries
    # `note_declined` and collects whatever the resumed stream emits.
    if st.session_state.feed is None:
        st.session_state.feed = webui.StreamlitFeed()
    # Rerun rather than carrying on: `busy` was computed above, before this recovery, so
    # the sidebar and chat input on THIS pass still think the session is idle.
    st.rerun()

# --- sidebar: export and durable memory ---------------------------------------------
with st.sidebar:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    document = export_markdown(values, st.session_state.thread_id, stamp)
    # DISABLED WHILE BUSY, and not for tidiness. Streamlit interrupts a running script at
    # its next `st.*` call, and the feed calls `st.markdown` on every line — so a click
    # here mid-turn raises `RerunException` inside `_stream_turn`, tearing down the
    # `agent.stream` generator and cancelling researchers whose searches are already paid
    # for. (`RerunException` derives from BaseException, so the `except Exception` around
    # the stream cannot catch it; the payload is consumed before streaming for that
    # reason.) Only the chat input and thread field were guarded before.
    st.download_button(
        "Export transcript",
        data=document,
        file_name=f"research-{st.session_state.thread_id}-{stamp}.md",
        mime="text/markdown",
        icon=":material/download:",
        disabled=busy or not document,
        width="stretch",
        help="Every question in this thread with its cited answer.",
    )

    # A fragment, so browsing saved notes redraws that expander instead of rerunning
    # the whole page — including the `get_state` read above and every chat bubble
    # below. See `webui.memory_browser` for why it still honours `busy`.
    webui.memory_browser(agent, busy=busy)

    st.caption(f"Model · `{MODEL_NAME}`")
    st.caption(f"Memory · `{MEMORY_DB.name}`")

# --- transcript, from the checkpoint -------------------------------------------------
for _index, (_kind, _text) in enumerate(sections):
    with st.chat_message("user" if _kind == "human" else "assistant"):
        if _events := st.session_state.work_logs.get(_index):
            with st.expander("Work log", icon=":material/manage_search:"):
                for _event in _events:
                    webui.render_event(_event)
        if _note := st.session_state.refusals.get(_index):
            st.warning(f"{_note} — rephrasing or narrowing it usually helps.")
        st.markdown(_text)

# The question of a turn already in flight, drawn only until the checkpoint carries it.
# The comparison is subtler than it looks — see `webui.should_render_question`.
if webui.should_render_question(st.session_state.question, sections):
    with st.chat_message("user"):
        st.markdown(st.session_state.question)

if st.session_state.notice:
    st.info(st.session_state.notice)


# --- approvals ------------------------------------------------------------------------
@st.fragment
def approval_panel() -> None:
    """The approval screen, scoped so deciding does not rerun the page.

    **A fragment because of what the controls cost, not what they do.** Choosing a
    decision, typing a rejection reason, or editing the JSON arguments each triggered a
    full rerun: the `get_state` read above, `export_markdown` over the whole thread, and
    every chat bubble redrawn — to change one widget, on the screen where the reviewer
    most needs to concentrate. Scoped here, those interactions redraw this panel alone.

    **`st.form` is the obvious alternative and it is wrong.** A form suppresses reruns
    until submit, and `webui.decision_controls` *depends* on them: the reason box, the
    prefilled JSON editor, and the "not valid JSON" error only exist because picking a
    decision reruns and re-renders. Batching the inputs would delete the validation that
    `webui.py` calls the only security boundary the app has.

    Both exits call a bare `st.rerun()`, which is app-scoped even from inside a fragment
    — required, because each one hands control back to the page's turn loop, and only a
    full rerun re-enters the streaming block below with the new `payload`.

    The `st.chat_message` wrapper is created *inside* the fragment rather than around the
    call. A fragment may write into a container made outside it, but only one that was
    already written to during a full run — owning the container removes that condition
    entirely.
    """
    with st.chat_message("assistant"):
        # The work this turn already did, above the thing it is asking permission for.
        # Not decoration: an approval is a rerun, and a rerun discards everything the
        # `st.status` box drew — so without this replay the reviewer decides whether to
        # allow a `write_file` having just lost sight of every search that produced it.
        # Collapsed, because the proposed action is what they are here to read.
        if st.session_state.feed and st.session_state.feed.events:
            with st.expander("Work log", icon=":material/manage_search:"):
                st.session_state.feed.replay()
        st.markdown("**The agent is waiting on you.**")
        if decisions := webui.approval_form(st.session_state.pending):
            # A rejected call reaches the stream as a ToolMessage with `status="error"`
            # carrying the human's own reason, so nothing downstream can tell a rejection
            # from a crash. Tell the feed, or it reports an honoured decision as a bug.
            st.session_state.feed.note_declined(
                _declined_tools(st.session_state.pending, decisions)
            )
            st.session_state.update(
                pending=[],
                payload=Command(
                    resume={
                        interrupt_id: {"decisions": chosen}
                        for interrupt_id, chosen in decisions.items()
                    }
                ),
            )
            st.rerun()

        # An escape hatch, and a required one. `approval_form` keeps submit disabled
        # until every action has a valid decision — so a tool gated with an
        # `InterruptOnConfig` this UI cannot render a control for leaves the button
        # permanently disabled, while `busy` has already disabled the chat input and the
        # thread field and `st.stop()` below ends the page. The session would have no
        # control left that could move it forward. `cli._prompt_decision` raises for the
        # same input and `cli.main`'s broad `except` abandons the turn; this is the
        # browser's equivalent, and it works for any stuck approval, not just that one.
        if st.button("Abandon this turn", icon=":material/close:"):
            st.session_state.update(
                pending=[],
                payload=None,
                question=None,
                feed=None,
                notice=(
                    "Turn abandoned. The pending action was not taken; ask again to "
                    "start a fresh turn."
                ),
            )
            st.rerun()


if st.session_state.pending:
    if not webui.reviewable_actions(st.session_state.pending):
        # Paused on something with no action requests. Resuming would just re-interrupt,
        # so say so and drop the turn rather than looping. Same call `cli.main` makes.
        st.session_state.update(
            pending=[],
            payload=None,
            question=None,
            feed=None,
            notice="The turn paused with no reviewable action, so it was abandoned.",
        )
        st.rerun()

    approval_panel()
    st.stop()

# --- run or resume the turn ------------------------------------------------------------
if st.session_state.payload is not None:
    feed = st.session_state.feed
    # CONSUMED BEFORE STREAMING, not after. Streamlit aborts a running script at its next
    # `st.*` call by raising `RerunException` — which derives from BaseException, so the
    # `except Exception` below does not catch it and cannot clear the payload on the way
    # out. Leaving it set meant the next rerun re-entered here with the SAME user
    # message: the question appended to the thread twice, `thread_sections` merging the
    # pair into one doubled bubble, and every search paid for again. Taking it now makes
    # an interrupted turn simply end — the same outcome as Ctrl-C in the REPL.
    payload = st.session_state.payload
    st.session_state.payload = None
    with (
        st.chat_message("assistant"),
        st.status("Researching…", expanded=True) as status,
        webui.claim_thread(st.session_state.thread_id) as claimed,
    ):
        if not claimed:
            # Another browser session is already running this thread. Per-session `busy`
            # cannot see that; two tabs both default to thread `main`.
            status.update(label="Thread busy", state="error")
            st.session_state.update(
                question=None,
                feed=None,
                notice=(
                    f"Thread `{st.session_state.thread_id}` already has a turn running "
                    "in another session. Wait for it, or switch threads."
                ),
            )
            st.rerun()
        feed.replay()  # everything this turn already did, before the approval
        try:
            pending = _stream_turn(agent, payload, config, feed)
        except Exception as exc:  # noqa: BLE001 — surface any runtime error to the user
            # Whatever prose the turn already produced is in the checkpoint and reappears
            # in the transcript on the next run: the agent composes its cited report in
            # the same message that proposes `write_file`, so an abandoned turn is
            # routinely one that had already done every search.
            status.update(label="Turn failed", state="error")
            st.session_state.update(question=None, feed=None, notice=f"Error: {exc}")
            st.rerun()

        if pending:
            status.update(
                label="Waiting for your approval", state="complete", expanded=False
            )
            st.session_state.pending = pending
            st.rerun()

        status.update(label="Research complete", state="complete", expanded=False)

    # Attach this turn's work log and any refusal note to the answer it produced. A
    # classifier refusal is a 200 with empty content — no exception, no prose — so a
    # turn CAN finish with no `ai` section at all, and the note then has nothing to hang
    # on. It becomes a page-level notice instead of vanishing, which is the whole point
    # of detecting it: silence reads as a bug in this app rather than a decision by the
    # model, and gives the user no reason to think rephrasing would help.
    settled_values = agent.get_state(config).values  # one read, feeding both
    settled = thread_sections(settled_values)
    refusal = _turn_refusal(settled_values)
    if settled and settled[-1][0] == "ai":
        st.session_state.work_logs[len(settled) - 1] = tuple(feed.events)
        if refusal:
            st.session_state.refusals[len(settled) - 1] = refusal
        notice = None
    else:
        notice = refusal or "The agent finished this turn without saying anything."

    # The turn may have written a note; drop the cached listing so the sidebar shows it.
    webui.refresh_memory_files()
    st.session_state.update(question=None, feed=None, notice=notice)
    st.rerun()
