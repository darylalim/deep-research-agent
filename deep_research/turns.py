"""What a research turn did, and what the user may be shown of it.

Everything here is presentation-free: it reads the stream and the checkpoint and returns
data. `webui.py` draws it and `streamlit_app.py` sequences it; `evals/harness.py` grades
exactly what `render_turn` produces. Keeping these rules in one place is the point —
they are subtle, each was paid for once already, and a second copy is a second place
for them to drift:

- `ActivityFeed` turns stream chunks into `FeedEvent`s: dedupe on tool-call ids, skip
  thread rewrites, orchestrator-only plan and `ls`, never a word of a researcher's prose.
- `_stream_turn` drains a stream to exhaustion and returns what it paused on.
- `pending_reviews` / `allowed_decisions_by_tool` / `_declined_tools` parse an approval
  interrupt; `webui.approval_form` presents it.
- `render_turn` / `thread_sections` / `export_markdown` read the answer back from the
  CHECKPOINT, never from the stream.
- `_stop_note` / `_turn_stop` name a turn the API ended with no prose.

Which decisions are on offer for an approval is not fixed: each interrupt carries a
`ReviewConfig` per tool saying what that tool permits, and the middleware raises
`ValueError` on anything outside it. So the decision controls are built from that
config rather than hardcoding approve/edit/reject.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Any

from .config import MODEL_NAME


@dataclass(frozen=True)
class FeedEvent:
    """One thing the agent did, as data rather than as a printed line.

    `ActivityFeed` decides *what happened* — the subtle half, and the one this
    module's docstrings spend pages on: dedupe on tool-call ids, skip thread
    rewrites, orchestrator-only plan and `ls`. `webui.render_event` decides how it
    *looks*. Keeping the two apart is what lets the decisions be tested as data,
    with no renderer in the way, and lets `webui.StreamlitFeed` add drawing by
    overriding `_emit` alone.

    That separation is not a nicety. This absorb-and-dedupe logic already exists
    twice (here and in `evals/harness.TurnRecorder`), and the SAME call-id dedupe bug
    was found and fixed in both, separately. A third hand-written copy in the
    renderer would be a third place for it to come back — so there isn't one.

    `items` is a tuple, not a list, so the whole event stays hashable and frozen:
    the web UI keeps a per-turn list of these in `st.session_state` and re-renders
    it on every rerun, which is only safe while an event cannot be mutated after
    the fact.
    """

    kind: str
    text: str = ""
    detail: str = ""
    items: tuple[str, ...] = ()
    is_orchestrator: bool = True


# What deepagents' `_format_file_paths` (`middleware/filesystem.py`) returns for an EMPTY
# listing: a bare sentinel, not the `"[]"` repr that 0.6 produced. Hardcoded rather than
# imported — it is one private function's return literal, and guessing wrong degrades a
# single feed line to `?` rather than breaking anything. `test_the_empty_ls_sentinel_still_
# matches_what_deepagents_returns` calls the real function and goes red if it ever changes.
_LS_EMPTY = "No files found"

# Every `kind` the feed can emit — the contract between `ActivityFeed` and its renderer.
# `webui.render_event` is an if/elif chain that silently draws NOTHING for a kind it does
# not recognize, so adding a kind here without a branch there would blank that line with
# no error anywhere. `test_webui.py::test_every_feed_kind_is_rendered` asserts every kind
# draws something, and `test_feed_kinds_lists_every_kind_the_renderer_actually_handles`
# asserts the reverse, which is what turns that silence into a red test.
FEED_KINDS: tuple[str, ...] = (
    "plan",
    "delegate",
    "search",
    "read",
    "listed",
    "done",
    "rejected",
    "failed",
    "refusal",
)


def _text_of(message: Any) -> str:
    """Extract plain text from a message whose content may be a string or a
    list of content blocks."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def _this_turn(messages: list[Any]) -> list[Any]:
    """Everything after the last human message — one turn's worth of the thread.

    Factored out so `render_turn` and `_turn_stop` slice the thread the *same* way.
    They must: the stop note and the answer are printed side by side, and a note
    scoped to a different span than the prose it annotates would report last turn's
    refusal above this turn's answer.
    """
    start = 0
    for index, message in enumerate(messages):
        if getattr(message, "type", None) == "human":
            start = index + 1
    return list(messages[start:])


@dataclass(frozen=True)
class StopNote:
    """Why a turn produced no prose, and what the user can actually do about it.

    Two fields rather than one sentence because the **remedy differs per stop reason**
    and used to be hardcoded at the print sites (`streamlit_app`, and a terminal REPL
    this project once had).
    That was survivable while `refusal` was the only silent stop and stopped being so
    the moment a second one existed: telling someone to rephrase a question that was
    never the problem is worse than saying nothing, because it sends them to fix the
    one thing that is already fine.
    """

    reason: str
    remedy: str


# Every stop reason that ends a generation with **HTTP 200, no exception, and no prose**.
# Each bills, raises nothing, and lands in the checkpoint as an assistant message that
# `render_turn` renders as `''` — which used to be reported as `(the agent said nothing)`,
# describing the symptom while hiding the cause and reading as a bug in this code rather
# than as something the API told us.
#
# **This table is a snapshot of a Literal that GROWS — re-read
# `anthropic/types/stop_reason.py` on an SDK bump.** `model_context_window_exceeded` is
# the proof and the reason this is a dict instead of an `== "refusal"`: it arrived in
# anthropic 0.120 (absent at 0.116), `langchain_anthropic` contains no reference to it at
# all, so it reaches `response_metadata["stop_reason"]` raw and, until it was listed here,
# fell straight through to `(the agent said nothing)` — the exact misattribution this
# whole mechanism exists to prevent, reopened by a dependency upgrade.
# `test_cli_parsing.py::TestStopReasonsAreAccountedFor` compares this table against the
# SDK's own enum and goes red when the next one lands, because prose telling a future
# reader to re-check a file is not a check.
_SILENT_STOPS: dict[str, StopNote] = {
    "refusal": StopNote(
        "the model declined this request",
        "rephrasing or narrowing it usually helps",
    ),
    "model_context_window_exceeded": StopNote(
        "this turn exceeded the model's context window",
        # Emphatically NOT "rephrase it". The question was fine; the *thread* is what
        # grew. Nothing in this project or in deepagents' default middleware stack
        # summarizes or trims a thread, and threads here are checkpointed and resumed
        # indefinitely by design — so on a long-lived thread this is a matter of time
        # rather than a matter of the question, and only a fresh thread clears it.
        "starting a fresh thread resets it",
    ),
}


def _stop_note(message: Any) -> StopNote | None:
    """A note if the API reported *why* this message carries no prose, else None.

    **Branch on `stop_reason`, never on `stop_details`.** `langchain_anthropic` copies
    `stop_reason` into `response_metadata` and drops `stop_details` entirely — grep it:
    `chat_models.py` names `stop_reason` three times and `stop_details` not once. So the
    refusal *category* is not available at this layer and must not be invented; the
    branch below is defensive, for the day langchain starts passing the field through.
    Saying "the model declined" is honest; naming a category we never received is not.

    That defensive branch reads **`category`**, which is the field the SDK actually
    defines — `anthropic/types/refusal_stop_details.py`: `RefusalStopDetails` is
    `{type: "refusal", category: <a Literal of policy names>|None, explanation: str|None}`.
    Getting that key wrong is invisible precisely *because* the branch is dead, so read it
    off the installed SDK rather than guessing from the surrounding key names. The **key**
    is the stable part; the member list is not, and it is deliberately not written out
    here — `general_harms` was added in SDK 0.120 and a copy of the enum in a docstring
    is a second thing to keep true. `TestStopReasonsAreAccountedFor` pins it against the
    SDK instead. Reading the value generically rather than matching on members is what
    makes a newly-added category print instead of vanish.

    `explanation` stays unused on purpose: the SDK documents it as "not guaranteed to be
    stable", and an unstable string is not something to put in front of a user as the
    reason their question was refused.
    """
    metadata = getattr(message, "response_metadata", None)
    if not isinstance(metadata, dict):
        return None
    reason = metadata.get("stop_reason")
    # `.get` on a non-str would raise for an unhashable value, and `stop_reason` is
    # whatever landed in a dict we did not build.
    note = _SILENT_STOPS.get(reason) if isinstance(reason, str) else None
    if note is None:
        return None
    details = metadata.get("stop_details")
    # BOTH shapes, attribute first. A wrong assumption about the *type* here is exactly as
    # invisible as the wrong *key* this branch already shipped once, and for the same
    # reason — the branch is dead, so nothing can tell a right guess from a wrong one until
    # the field arrives. The two langchain paths genuinely differ: `_format_output`
    # (non-streaming) builds `response_metadata` from `data.model_dump()`, so dicts; the
    # `message_delta` handler — the one this app always takes, since `streaming=True` —
    # assembles it field by field off the raw event and has to call `.model_dump()`
    # explicitly for `container` and `context_management`, which is proof that what sits
    # there is pydantic models. `stop_details` added to that block without one would be a
    # `RefusalStopDetails` instance, and a dict-only read would drop the category without
    # a sound. `getattr` on a dict returns None, so the fallback still covers the other.
    category = getattr(details, "category", None) or (
        details.get("category") if isinstance(details, dict) else None
    )
    if isinstance(category, str) and category:
        return StopNote(f"{note.reason} ({category})", note.remedy)
    return note


def _turn_stop(result: dict[str, Any]) -> StopNote | None:
    """The stop note for this turn, if any assistant message this turn carried one.

    The first one, not all of them: a turn that refuses twice refused for one reason,
    and two identical lines above the answer is noise. Read from the checkpoint rather
    than the stream, for the same reason `render_turn` is — the page has the final
    state in hand there, and a mid-stream refusal that the orchestrator then recovered
    from would still be recorded in it.
    """
    for message in _this_turn(result.get("messages", [])):
        if getattr(message, "type", None) == "ai" and (note := _stop_note(message)):
            return note
    return None


def render_turn(result: dict[str, Any]) -> str:
    """Everything the agent said this turn, in order — not only its last message.

    Printing just the final assistant message silently loses the answer. The agent
    composes its cited report in the *same* message that proposes `write_file`, and
    then signs off once the tool returns — so `messages[-1]` is the sign-off. Measured
    on a real run: 33 source URLs in the turn, **zero** in the last message, and a
    closing line pointing at a "summary above" the user had never been shown.

    Only this turn: everything after the last human message, so a long thread does not
    reprint its history.

    `evals/harness.py` imports this, deliberately — the eval that grades whether the
    user was shown any sources must grade exactly what the user is shown, or the two
    drift and the metric becomes fiction. For a finished turn this equals the last `ai`
    section of `thread_sections`, which is what the page draws;
    `test_render_turn_is_the_answer_the_page_draws` pins that.
    """
    texts = [
        text
        for message in _this_turn(result.get("messages", []))
        if getattr(message, "type", None) == "ai"
        and (text := _text_of(message).strip())
    ]
    # Assistant prose, or nothing. There is deliberately no fallback to "whatever ended
    # the turn" — that used to be `_text_of(messages[-1])`, and it was harmless only
    # while this function was called exclusively on *completed* turns. It isn't: a turn
    # abandoned at an approval prompt or by an API error is still rendered, and there the
    # last message is routinely something that must never be shown as the agent's words:
    #   - the user's OWN question, echoed back as the agent's answer, when the turn was
    #     abandoned before the agent said anything;
    #   - a raw `tavily_search` ToolMessage — multiple KB of serialized result dicts —
    #     when the turn was abandoned mid-search.
    # Both would also reach `evals/harness.py`, which renders `response` with this exact
    # function and hands it to the judges: they would grade the question, or a JSON blob,
    # as the agent's answer.
    return "\n\n".join(texts)


def thread_sections(state: dict[str, Any]) -> list[tuple[str, str]]:
    """The thread as ordered `(speaker, text)` sections, speaker being `human` or `ai`.

    The grouping half of `render_thread`, split out because the page needs the same
    sections as *chat bubbles* rather than as markdown headings. One definition of
    "how a thread divides into turns", two renderers — the same split as `FeedEvent`,
    and for the same reason: the rules below are load-bearing and were paid for once
    already.

    Assistant **prose only**, and deliberately not the last message. The agent composes
    its cited report in the same message that proposes `write_file` and then signs off
    once the tool returns, so `messages[-1]` is "findings saved, see the summary above"
    with none of the 33 source URLs the turn actually produced — the exact regression
    this repo has already paid for once. Tool payloads are skipped for the same reason
    `ActivityFeed` never prints them.

    Consecutive messages from one speaker are ONE section. That report-then-sign-off
    pair is two `ai` messages saying one thing; splitting them would render one answer
    as two.
    """
    sections: list[tuple[str, list[str]]] = []
    for message in state.get("messages", []):
        kind = getattr(message, "type", None)
        if kind not in ("human", "ai"):
            continue
        if not (text := _text_of(message).strip()):
            continue  # e.g. an assistant message that only carried a tool call
        if sections and sections[-1][0] == kind:
            sections[-1][1].append(text)
        else:
            sections.append((kind, [text]))
    return [(kind, "\n\n".join(texts)) for kind, texts in sections]


def render_thread(state: dict[str, Any]) -> str:
    """The whole conversation as markdown — every question with its cited answer.

    `render_turn` with the slice removed. `AgentState.messages` uses `add_messages`, so
    the checkpointed list *is* the whole thread; nothing needs to walk
    `get_state_history`.

    Same discipline as `render_turn`, and for the same measured reason: assistant prose
    only. Not the last message (a sign-off — "findings saved, see the summary above" —
    with none of the 33 source URLs the turn actually produced, which is the exact
    regression this repo has already paid for once), and not tool payloads. Every claim
    the user might rely on a week later lives in the prose, because `SYSTEM_PROMPT` step
    4 requires the citations inline there.

    Include the question. A cited report with no question is unusable later, and the
    question is right there.

    One dependency worth naming, because it is invisible from this repo's source: this is
    complete only because deepagents' summarization middleware — which
    `create_deep_agent()` appends without being asked — is deliberately *non-mutating*.
    It records eviction in a private field and leaves `state["messages"]` intact,
    explicitly so that replay and evals still work. LangChain's own
    `SummarizationMiddleware` instead rewrites the list with
    `RemoveMessage(id=REMOVE_ALL_MESSAGES)`. Wire that one via `middleware=[...]` and
    every long thread's export silently truncates, with no error and no failing test.
    """
    return "\n\n".join(
        f"## {'you' if kind == 'human' else 'agent'}\n\n{text}"
        for kind, text in thread_sections(state)
    )


def export_markdown(state: dict[str, Any], thread_id: str, stamp: str) -> str:
    """The thread as a self-contained markdown document, or `""` if there is none yet.

    Behind the page's "Export transcript" download button. `stamp` is passed in
    rather than read here: the caller also puts it in the filename, and a header and
    filename disagreeing about when a report was taken is the kind of small lie that
    makes an archive useless.

    "Exported", not "answered" — the messages carry no timestamps, and the only
    per-turn clock lives in the checkpointer's snapshots. A date we did not measure is
    a date we invented.
    """
    text = render_thread(state)
    if not text:
        return ""
    header = (
        f"# Deep research — thread `{thread_id}`\n\n"
        f"*Model: `{MODEL_NAME}`. Exported {stamp}.*\n"
    )
    return f"{header}\n{text}\n"


def _short(value: Any, limit: int = 300) -> str:
    """Compactly render tool args for display."""
    try:
        text = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(value)
    return text if len(text) <= limit else text[:limit] + " …"


# What the middleware itself assumes for a tool gated with a bare `True`, in the order
# the decision controls offer them. The set is the fallback for an interrupt that
# carries no matching `ReviewConfig`; the order is used for every interrupt, so a
# narrowed tool's options still appear in the same sequence.
DEFAULT_ALLOWED_DECISIONS = ("approve", "edit", "reject", "respond")

# The middleware's own default `description_prefix`. See `_reviewer_note`.
DEFAULT_DESCRIPTION_PREFIX = "Tool execution requires approval"


def _reviewer_note(request: dict[str, Any]) -> str | None:
    """The part of an action's `description` a human actually needs.

    The middleware builds the default description as
    `f"{prefix}\\n\\nTool: {name}\\nArgs: {args}"` (langchain's
    `human_in_the_loop.py`) — i.e. the tool name we already show as a header, and
    the raw `args` **dict repr** we can render far better ourselves. Showing it
    verbatim is what put an escaped-newline Python dict in front of the reviewer.

    So strip that boilerplate and keep only what is left. Usually nothing — but a
    tool gated with an `InterruptOnConfig` may carry a real, human-written
    `description` (a string, or one built by a callable), and that is worth showing.
    Same principle as the menu: honor what the interrupt hands us rather than
    assuming the default shape.
    """
    description = request.get("description")
    if not isinstance(description, str):
        return None
    boilerplate = f"Tool: {request.get('name')}\nArgs: {request.get('args', {})}"
    note = description.replace(boilerplate, "").replace(DEFAULT_DESCRIPTION_PREFIX, "")
    return note.strip() or None


def pending_reviews(interrupts: list[Any]) -> list[tuple[str, dict[str, Any]]]:
    """Every DISTINCT pending interrupt, as ordered `(interrupt_id, HITLRequest)` pairs.

    **The same interrupt is emitted TWICE.** With `subgraphs=True`, an interrupt raised
    inside a subagent is emitted at the subagent's namespace *and* again, bubbled, at the
    root — same `Interrupt.id`, two chunks. Honouring both would ask the human to approve
    one researcher's `write_file` twice and, since every resume mapping is keyed by id,
    silently keep only the second answer. Approval fatigue is exactly how a gate stops
    being a gate.

    Deduping lives here, in one place, because three callers need it and each one that
    rolls its own is a chance to get it wrong: `webui.approval_form` (the invariant is
    "one set of controls per pending action"), `webui.reviewable_actions`, and
    `_declined_tools`. `evals/harness._approve_all` is immune only by
    accident — it writes into a dict keyed by id without asking anyone anything, so a
    duplicate is idempotent — and `harness.TurnRecorder` was NOT immune, which cost this
    project a silently defeated safety metric.

    An interrupt with no id, or whose value is not the mapping the middleware documents,
    is dropped rather than guessed at.
    """
    seen: set[str] = set()
    reviews: list[tuple[str, dict[str, Any]]] = []
    for interrupt in interrupts:
        interrupt_id = getattr(interrupt, "id", None)
        value = getattr(interrupt, "value", None)
        if interrupt_id is None or interrupt_id in seen or not isinstance(value, dict):
            continue
        seen.add(interrupt_id)
        reviews.append((interrupt_id, value))
    return reviews


def allowed_decisions_by_tool(value: dict[str, Any]) -> dict[str, list[str]]:
    """Which decisions each tool in this interrupt permits, keyed by tool name.

    Looked up by **name** rather than by position. The middleware happens to append
    `action_requests` and `review_configs` in lockstep today, but it documents the latter
    as the policy "for all possible actions" — so a name lookup stays correct if it is
    ever deduplicated, and a positional one would silently offer the wrong menu.
    """
    return {
        config["action_name"]: config["allowed_decisions"]
        for config in value.get("review_configs", [])
        if config.get("action_name") and config.get("allowed_decisions")
    }


class ActivityFeed:
    """Records what the agent is doing, as it does it, as a list of `FeedEvent`s.

    The turn used to be a black box: minutes of nothing, then a wall of text. This
    reads the tool activity arriving on
    `agent.stream(..., stream_mode="updates", subgraphs=True)`. It draws nothing
    itself — `webui.StreamlitFeed` overrides `_emit` to draw each event as it lands.

    Three things it must get right, each of which is a bug waiting to happen:

    **It records actions, never prose.** The stream carries the *researchers'*
    assistant messages too, and the user must never see one — they are a subagent's
    internal working, and `evals/harness.py` refuses to build its graded `response`
    from the stream for exactly this reason. The answer comes from the final
    checkpoint, so the transcript, the exported file, and the eval's `response` stay
    the same bytes. Same rule as `harness.TurnRecorder`: actions only.

    **It records each event once, keyed on the TOOL CALL id.** On resume,
    `HumanInTheLoopMiddleware.after_model` re-emits the AIMessage that proposed the gated
    call, and the re-streamed superstep re-emits the *cached writes* of the siblings that
    already succeeded (`_reapply_writes_to_succeeded_nodes`) — so without deduping, an
    approval replays lines the user just watched scroll past.

    The key is the call id (`tool_call["id"]`, and `tool_call_id` on the result) rather
    than the *message* id, because a tool call executes exactly once while
    `BaseMessage.id` is optional and unstable: a replayed `ToolMessage` arrives with
    `id=None` on the first pass and a **fresh uuid** on the resume, so a message-id
    seen-set matches neither and lets every duplicate through. The same defect was found
    (and fixed) in `harness.TurnRecorder`, where it was double-counting delegations.

    **It does not pretend to know which researcher is which.** A subagent's namespace is
    `('tools:<pregel-task-uuid>',)`, and that uuid is not the `task` tool-call id — so
    binding a search back to the sub-question that spawned it would mean assuming
    dispatch order matches first-emission order under concurrency. It doesn't have to:
    the *dispatch* and *completion* lines carry the real description (recovered by
    `tool_call_id`), and each search line carries its actual query. That is the
    information worth having, and all of it is true.
    """

    def __init__(self) -> None:
        # Every event this turn, in order. Kept rather than discarded because a turn
        # that pauses for approval spans several Streamlit reruns, and each rerun has to
        # redraw what the user already watched appear (`webui.StreamlitFeed.replay`).
        self.events: list[FeedEvent] = []
        self._printed: set[str] = set()  # event keys already recorded
        self._task_descriptions: dict[str, str] = {}  # task tool_call id -> description
        self._ls_paths: dict[str, str] = {}  # ls tool_call id -> the path it listed
        self._declined: set[str] = set()  # tool names the human rejected this turn

    def note_declined(self, names: set[str]) -> None:
        """Tell the feed which tools the human just rejected.

        Needed because a rejection is indistinguishable from a crash by the time it
        reaches the stream: `HumanInTheLoopMiddleware` answers a rejected call with a
        synthetic `ToolMessage` carrying **`status="error"`** — and if the human supplied
        a reason, that reason *becomes* the content. So the feed would show
        `write_file failed: too risky` to the person who just clicked Reject, reporting
        their own honoured decision as a bug in the agent.

        Name-level, not call-level, because an `ActionRequest` carries no tool-call id.
        The imprecision only bites if one turn both rejects a `write_file` and has a
        *different* `write_file` genuinely fail — in which case the failure is reported as
        a rejection. Rare, and it errs toward the truthful half.
        """
        self._declined |= names

    def absorb(self, namespace: tuple[str, ...], chunk: Any) -> list[Any]:
        """Fold in one `(namespace, update)` chunk; return any interrupts it carried.

        Deliberately the same shape as `harness.TurnRecorder.absorb` — that one has been
        run against the live agent, and divergence between the two is how the app and
        the eval start disagreeing about what happened.
        """
        if not isinstance(chunk, dict):
            return []

        interrupts: list[Any] = []
        is_orchestrator = not namespace
        for node, update in chunk.items():
            if node == "__interrupt__":
                interrupts.extend(update)
                continue
            if not isinstance(update, dict):
                continue
            messages = update.get("messages", []) or []
            if any(getattr(m, "type", None) == "remove" for m in messages):
                # A THREAD REWRITE, not new activity — skip the whole update.
                # `PatchToolCallsMiddleware.before_agent` answers dangling tool calls by
                # returning `{"messages": [RemoveMessage(REMOVE_ALL_MESSAGES), *the entire
                # thread]}`. It fires on exactly the turn *after* one you abandoned at an
                # approval prompt — so without this guard, that turn opens by replaying
                # the previous turn's whole feed: its plan, its delegations, every search.
                # (The feed is per-turn, so its seen-set has never heard of those calls.)
                #
                # Keyed on the RemoveMessage rather than the node name, so any middleware
                # that rewrites the message list wholesale is covered, not just this one.
                continue
            if (todos := update.get("todos")) and is_orchestrator:
                # ORCHESTRATOR ONLY. Through deepagents 0.6.x every declarative subagent
                # got its own `TodoListMiddleware`, so a `researcher` really could call
                # `write_todos` — and its list streamed out under `('tools:<uuid>',)`.
                # Rendering that would print a researcher's private checklist as the
                # agent's plan, appearing to supersede the plan the user was just shown.
                #
                # 0.7.x drops that middleware from the subagent stack, so today the guard
                # is defence in depth rather than a live fix — `build_agent` passes
                # `TodoListMiddleware` for the ORCHESTRATOR only, and nothing propagates it
                # down. Keep it anyway: a `todos` update under a subagent namespace is not
                # this agent's plan whatever put it there, and the sibling `ls` guard
                # below IS still live, since subagents keep their `FilesystemMiddleware`.
                # This is the same
                # orchestrator/subagent conflation `evals/harness.py` keeps apart with
                # `orchestrator_trajectory` vs `trajectory`; the display layer has to make
                # the same distinction, for the same reason.
                self._render_plan(todos)
            for message in messages:
                self._render_message(namespace, message)
        return interrupts

    def _once(self, key: str) -> bool:
        """True the first time this event is seen, False every time after."""
        if key in self._printed:
            return False
        self._printed.add(key)
        return True

    def _emit(self, event: FeedEvent) -> None:
        """Record one event — the ONLY method a renderer overrides.

        Everything above this line decided *whether* an event happened and what it
        says. `webui.StreamlitFeed` extends this to draw the event too, and inherits
        every rule in `absorb`, which is the point (see `FeedEvent`).
        """
        self.events.append(event)

    def _render_plan(self, todos: list[Any]) -> None:
        # `write_todos` returns a Command that updates the `todos` channel, so the whole
        # list arrives in the chunk — no need to parse the tool call.
        items = [t.get("content", "?") for t in todos if isinstance(t, dict)]
        # Keyed on the contents: a replayed superstep re-emits the identical plan (noise),
        # but a plan the agent genuinely revised is a different list, and worth showing.
        if not items or not self._once(f"plan:{items}"):
            return
        self._emit(FeedEvent("plan", items=tuple(items)))

    def _render_message(self, namespace: tuple[str, ...], message: Any) -> None:
        is_orchestrator = not namespace
        kind = getattr(message, "type", None)

        if kind == "ai":
            if not is_orchestrator:
                self._render_stop(namespace, message)
            for call in getattr(message, "tool_calls", None) or []:
                self._render_call(call, is_orchestrator)
        elif kind == "tool":
            self._render_result(message, is_orchestrator)

    def _render_stop(self, namespace: tuple[str, ...], message: Any) -> None:
        """Say so when a *researcher* is stopped for a reason the API reported.

        Both entries in `_SILENT_STOPS`, not just refusals — a researcher that overran
        the context window is invisible in exactly the same way and for exactly the same
        reason, so a check that knew only about classifiers would have to be written
        twice.

        SUBAGENT ONLY — the caller enforces that. An orchestrator stop is reported by
        the page, from the checkpoint (`_turn_stop`), exactly where the missing answer
        would have been; recording it here as well would say it twice.

        A researcher's stop is otherwise completely INVISIBLE. It ends that subagent's
        turn with empty content, so the `task` result comes back thin and the
        orchestrator synthesizes around the hole. The user watches a delegation get
        dispatched, watches it complete, and reads a thinner answer than they asked for,
        with nothing anywhere saying why.

        Keyed on the NAMESPACE — coarser than this class's call-id rule, deliberately. A
        stop carries no tool call, and `BaseMessage.id` is precisely the unreliable
        key the class docstring warns about (`None` on the first pass, a fresh uuid on
        the resume), so it is the only stable key available. The cost is that two
        distinct stops inside one researcher collapse to one line — the same
        deliberate imprecision as `note_declined` being name-level rather than
        call-level, and it errs toward under-reporting a repeat, never toward inventing
        an event.

        Emitted under the `"refusal"` feed kind, which is now broader than it reads. The
        kind is an internal selector for the renderer and never reaches the user — who
        sees `note.reason` — and renaming it is atomic across `FEED_KINDS` and
        `webui.render_event` (test_webui asserts set equality in BOTH directions), so it
        cannot be sequenced past the pytest hook. Only the `reason` distinguishes the two
        stops here; only the page's turn-level note needs the remedy.
        """
        note = _stop_note(message)
        if note and self._once(f"stop:{namespace}"):
            self._emit(FeedEvent("refusal", text=note.reason))

    def _render_call(self, call: dict[str, Any], is_orchestrator: bool) -> None:
        name = call.get("name")
        args = call.get("args") or {}
        if not self._once(f"call:{call.get('id')}"):
            return

        if name == "task":
            # `TaskToolSchema` guarantees `description` — it is the self-contained prompt
            # the orchestrator wrote, and it becomes the researcher's only message.
            description = args.get("description", "?")
            self._task_descriptions[call.get("id", "")] = description
            self._emit(FeedEvent("delegate", text=description))
        elif name == "tavily_search":
            # Announced at CALL time, not on the result: the query is the informative
            # part and this keeps the feed live. It also means never touching the
            # ToolMessage body, which for a search is multiple KB of serialized results.
            self._emit(
                FeedEvent(
                    "search",
                    text=args.get("query", "?"),
                    is_orchestrator=is_orchestrator,
                )
            )
        elif name == "read_file" and is_orchestrator:
            self._emit(FeedEvent("read", text=args.get("file_path", "?")))
        elif name == "ls" and is_orchestrator:
            # Remember what it listed, so the result line can name it. `ls` takes an
            # arbitrary `path`; hardcoding "/memories/" would be a guess, and it is the
            # ONE line of the feed the user reads to check the agent obeyed SYSTEM_PROMPT
            # step 2 — a line that lies about that is worse than no line.
            self._ls_paths[call.get("id", "")] = args.get("path", "?")

    def _render_result(self, message: Any, is_orchestrator: bool) -> None:
        name = getattr(message, "name", None)
        call_id = getattr(message, "tool_call_id", "")
        if not self._once(f"result:{call_id}"):
            return

        # A failed tool is the one result worth surfacing — Tavily raises rather than
        # returning an empty list, so a fruitless search arrives as an error, and a
        # silent feed would make it look like the search simply never happened.
        if getattr(message, "status", None) == "error":
            if name in self._declined:
                # Not a failure — the human rejected it, and the middleware reports that
                # to the model as an `status="error"` ToolMessage. See `note_declined`.
                self._emit(FeedEvent("rejected", text=str(name)))
            else:
                self._emit(
                    FeedEvent("failed", text=str(name), detail=_text_of(message))
                )
            return

        if name == "ls" and is_orchestrator:
            # ORCHESTRATOR ONLY — a `researcher` has its own `FilesystemMiddleware` and
            # can call `ls` on its own state-backed filesystem. Rendering that would tell
            # the user durable memory was consulted on a turn where the orchestrator never
            # looked, hiding the exact "the direct path skips /memories/" defect CLAUDE.md
            # says to keep watching.
            #
            # And the body is NOT newline-separated entries. `_format_file_paths` renders
            # a non-empty listing as `str(paths)` — a Python list repr on ONE line,
            # `"['/memories/a.md']"` — and an empty one as the bare `_LS_EMPTY` sentinel.
            # Counting lines is wrong for both: it reported "1 file(s)" for an EMPTY store,
            # every time. So match the sentinel, else parse the repr, and if it is neither
            # say `?` rather than inventing a number.
            #
            # Handling BOTH empty shapes is the point. `"[]"` was 0.6's and the sentinel is
            # 0.7's, and for a while only the repr was handled — so an empty `/memories/`
            # rendered `?`, which reads as "the feed could not tell" when the truth was
            # "the orchestrator looked and found nothing". This line is the direct-path
            # signal the evals watch; one that cannot say `empty` is barely worth printing.
            path = self._ls_paths.get(call_id, "/memories/")
            body = _text_of(message).strip()
            if body == _LS_EMPTY:
                count = "empty"
            else:
                try:
                    entries = ast.literal_eval(body)
                except (ValueError, SyntaxError):
                    entries = None
                if isinstance(entries, list):
                    count = f"{len(entries)} file(s)" if entries else "empty"
                else:
                    count = "?"
            self._emit(FeedEvent("listed", text=path, detail=count))
        elif name == "task":
            # The one honest way to name a researcher: recover the sub-question from the
            # `task` call this result answers. Its stream namespace cannot be bound back
            # to that call without assuming dispatch order matches emission order under
            # concurrency, so we don't pretend to.
            description = self._task_descriptions.get(call_id)
            self._emit(FeedEvent("done", text=description or ""))


def _one_line(text: Any, limit: int) -> str:
    """Collapse a value to a single, bounded line — a feed line must not swell into a
    researcher's whole prompt."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _stream_turn(
    agent: Any, payload: Any, config: dict[str, Any], feed: ActivityFeed
) -> list[Any]:
    """Run one stream to exhaustion, feeding the feed; return the pending interrupts.

    **Drain, THEN ask.** Not a style choice — an interrupt chunk does not end the
    stream. LangGraph does not treat a `GraphInterrupt` as a failure, so sibling tasks in
    the same superstep keep running and a *second* researcher's interrupt arrives after
    the first. Worse, the graph executes inside this generator: pausing for a human
    mid-iteration freezes the Pregel loop, and starting the resume stream would tear the
    old generator down — cancelling a still-running researcher whose interrupt was never
    emitted, and throwing away searches you already paid for.

    So the loop is: exhaust the stream, collect everything pending, decide on the whole
    set, restream with `Command(resume=...)`. That is the shape `evals/harness.py` has
    been running against the live agent all along.
    """
    pending: list[Any] = []
    for namespace, chunk in agent.stream(
        payload, config=config, stream_mode="updates", subgraphs=True
    ):
        pending.extend(feed.absorb(namespace, chunk))
    return pending


def _declined_tools(
    interrupts: list[Any], by_interrupt: dict[str, list[dict[str, Any]]]
) -> set[str]:
    """The tool names the human just rejected.

    `webui.approval_form` returns one decision per `action_request`, in order, within
    each interrupt — so zipping the two back together recovers which *tool* each decision
    was about. `pending_reviews` supplies the same deduplication it does, for the same
    reason: a subagent's interrupt arrives twice.
    """
    declined: set[str] = set()
    for interrupt_id, value in pending_reviews(interrupts):
        requests = value.get("action_requests", [])
        decisions = by_interrupt.get(interrupt_id, [])
        for request, decision in zip(requests, decisions, strict=False):
            if decision.get("type") == "reject":
                declined.add(request.get("name", "?"))
    return declined
