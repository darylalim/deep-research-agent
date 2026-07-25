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

if thread_id and thread_id != st.session_state.thread_id:
    # Work logs and refusal notes are indexed against the *previous* thread's
    # transcript, so they are meaningless here. The answers themselves are in the
    # checkpoint and reappear on their own.
    st.session_state.update(thread_id=thread_id, work_logs={}, refusals={}, notice=None)
    st.rerun()

config = {"configurable": {"thread_id": st.session_state.thread_id}}
values = agent.get_state(config).values
sections = thread_sections(values)

# --- sidebar: export and durable memory ---------------------------------------------
with st.sidebar:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    document = export_markdown(values, st.session_state.thread_id, stamp)
    st.download_button(
        "Export transcript",
        data=document,
        file_name=f"research-{st.session_state.thread_id}-{stamp}.md",
        mime="text/markdown",
        icon=":material/download:",
        disabled=not document,
        width="stretch",
        help="Every question in this thread with its cited answer.",
    )

    with st.expander("Durable memory", icon=":material/database:"):
        try:
            saved = webui.memory_files(agent)
        except Exception as exc:  # noqa: BLE001 — a sidebar read must not kill the page
            st.caption(f"Could not read memory: {exc}")
            saved = []
        if saved:
            chosen = st.selectbox(
                "File", [path for path, _ in saved], label_visibility="collapsed"
            )
            st.code(dict(saved)[chosen], language=None, height=240, wrap_lines=True)
        else:
            st.caption(
                "Nothing saved yet. Findings the agent writes to `/memories/` persist "
                "across every thread and session."
            )
    st.caption(f"Model · `{MODEL_NAME}`")
    st.caption(f"Memory · `{MEMORY_DB.name}`")

# --- header and input ---------------------------------------------------------------
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

# The question of a turn already in flight. Rendered only when the checkpoint does not
# already carry it: the human message is written on the graph's first superstep, so it
# is present for every rerun after the first, and drawing it unconditionally would show
# it twice for the whole length of an approval.
if (asking := st.session_state.question) and not any(
    kind == "human" and text == asking for kind, text in sections
):
    with st.chat_message("user"):
        st.markdown(asking)

if st.session_state.notice:
    st.info(st.session_state.notice)

# --- approvals ------------------------------------------------------------------------
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
    st.stop()

# --- run or resume the turn ------------------------------------------------------------
if st.session_state.payload is not None:
    feed = st.session_state.feed
    with (
        st.chat_message("assistant"),
        st.status("Researching…", expanded=True) as status,
    ):
        feed.replay()  # everything this turn already did, before the approval
        try:
            pending = _stream_turn(agent, st.session_state.payload, config, feed)
        except Exception as exc:  # noqa: BLE001 — surface any runtime error to the user
            # Whatever prose the turn already produced is in the checkpoint and reappears
            # in the transcript on the next run: the agent composes its cited report in
            # the same message that proposes `write_file`, so an abandoned turn is
            # routinely one that had already done every search.
            status.update(label="Turn failed", state="error")
            st.session_state.update(
                payload=None, question=None, feed=None, notice=f"Error: {exc}"
            )
            st.rerun()

        if pending:
            status.update(
                label="Waiting for your approval", state="complete", expanded=False
            )
            st.session_state.update(pending=pending, payload=None)
            st.rerun()

        status.update(label="Research complete", state="complete", expanded=False)

    # Attach this turn's work log and any refusal note to the answer it produced. A
    # classifier refusal is a 200 with empty content — no exception, no prose — so a
    # turn CAN finish with no `ai` section at all, and the note then has nothing to hang
    # on. It becomes a page-level notice instead of vanishing, which is the whole point
    # of detecting it: silence reads as a bug in this app rather than a decision by the
    # model, and gives the user no reason to think rephrasing would help.
    settled = thread_sections(agent.get_state(config).values)
    refusal = _turn_refusal(agent.get_state(config).values)
    if settled and settled[-1][0] == "ai":
        st.session_state.work_logs[len(settled) - 1] = tuple(feed.events)
        if refusal:
            st.session_state.refusals[len(settled) - 1] = refusal
        notice = None
    else:
        notice = refusal or "The agent finished this turn without saying anything."

    st.session_state.update(payload=None, question=None, feed=None, notice=notice)
    st.rerun()
