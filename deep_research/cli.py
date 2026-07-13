"""Interactive command-line chat for the deep research agent.

Runs a REPL against a single persistent thread. When the agent proposes a gated
action (writing a file, running a command) the turn pauses and this CLI collects
one decision per pending action, then resumes the run.

Which decisions are on offer is not fixed: each interrupt carries a `ReviewConfig`
per tool saying what that tool permits, and the middleware raises `ValueError` on
anything outside it. `_prompt_decision` therefore builds its menu from that config
rather than hardcoding approve/edit/reject.
"""

from __future__ import annotations

import ast
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langgraph.types import Command

from .agent import open_agent
from .config import MEMORY_DB, MODEL_NAME, missing_keys

BANNER = f"""\
╭──────────────────────────────────────────────────────────────╮
│  Deep Research Agent                                          │
│  model: {MODEL_NAME:<52}│
│  Ask a research question. The agent plans, delegates web      │
│  searches to a subagent, synthesizes a cited answer, and      │
│  remembers durable findings across sessions.                  │
│                                                               │
│  Commands:  /help  /thread <id>  /export [path]  /exit        │
╰──────────────────────────────────────────────────────────────╯"""

HELP = f"""\
Commands:
  /help            show this help
  /thread <id>     switch to a different conversation thread (default: "main")
  /export [path]   write this thread — every question and its cited answer — to
                   a markdown file (default: ./research-<thread>-<utc>.md)
  /exit, /quit     leave

Notes:
  • The agent's work is shown live as it happens: its plan, each sub-question it
    delegates, and every search it runs.
  • Conversation, todos, and pending approvals persist across restarts
    (checkpointed to .deep_research/checkpoints.sqlite).
  • Durable findings the agent saves under /memories/ persist across every
    thread and session ({MEMORY_DB.name}).
  • Writing a file pauses for your approval — you see the full contents first,
    and can approve, edit, reject, or answer the agent on the tool's behalf."""


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
    user was shown any sources must grade exactly what the CLI prints, or the two drift
    and the metric becomes fiction.
    """
    messages = result.get("messages", [])
    start = 0
    for index, message in enumerate(messages):
        if getattr(message, "type", None) == "human":
            start = index + 1

    texts = [
        text
        for message in messages[start:]
        if getattr(message, "type", None) == "ai"
        and (text := _text_of(message).strip())
    ]
    # Assistant prose, or nothing. There is deliberately no fallback to "whatever ended
    # the turn" — that used to be `_text_of(messages[-1])`, and it was harmless only
    # while this function was called exclusively on *completed* turns. It isn't:
    # `_print_unfinished_turn` calls it on turns abandoned at an approval prompt or by
    # an API error, and there the last message is routinely something that must never be
    # printed as the agent's words:
    #   - the user's OWN question, echoed back under an `agent >` header, when the turn
    #     was abandoned before the agent said anything;
    #   - a raw `tavily_search` ToolMessage — multiple KB of serialized result dicts —
    #     when the turn was abandoned mid-search.
    # Both would also reach `evals/harness.py`, which renders `response` with this exact
    # function and hands it to the judges: they would grade the question, or a JSON blob,
    # as the agent's answer.
    return "\n\n".join(texts)


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
    sections: list[tuple[str, list[str]]] = []
    for message in state.get("messages", []):
        kind = getattr(message, "type", None)
        if kind not in ("human", "ai"):
            continue
        if not (text := _text_of(message).strip()):
            continue  # e.g. an assistant message that only carried a tool call
        # Consecutive messages from the same speaker are ONE section. The agent's turn is
        # routinely two messages — the cited report (sent alongside the `write_file` call)
        # and then a sign-off once the tool returns — and splitting them into two `##
        # agent` headings would make one answer look like two.
        if sections and sections[-1][0] == kind:
            sections[-1][1].append(text)
        else:
            sections.append((kind, [text]))
    return "\n\n".join(
        f"## {'you' if kind == 'human' else 'agent'}\n\n" + "\n\n".join(texts)
        for kind, texts in sections
    )


def _short(value: Any, limit: int = 300) -> str:
    """Compactly render tool args for display."""
    try:
        text = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(value)
    return text if len(text) <= limit else text[:limit] + " …"


# The decision types the middleware understands, in the order we offer them, each
# mapped to the key that selects it and how it renders in the menu. `respond` has
# no free letter left, hence `re[s]pond`.
DECISION_KEYS: dict[str, tuple[str, str]] = {
    "approve": ("a", "[a]pprove"),
    "edit": ("e", "[e]dit"),
    "reject": ("r", "[r]eject"),
    "respond": ("s", "re[s]pond"),
}

# What the middleware itself assumes for a tool gated with a bare `True`. Used
# only as a fallback for an interrupt that carries no matching `ReviewConfig`.
DEFAULT_ALLOWED_DECISIONS = ("approve", "edit", "reject", "respond")

# The middleware's own default `description_prefix`. See `_reviewer_note`.
DEFAULT_DESCRIPTION_PREFIX = "Tool execution requires approval"

# How much of a long string argument (a file body, a shell command) to show before
# eliding. The gate exists so a human reads the content before it lands in durable,
# gitignored `/memories/` that git cannot restore, so this has to be generous enough
# for a real note — an elided review is a review that gets rubber-stamped.
PREVIEW_LINES = 40


def _reviewer_note(request: dict[str, Any]) -> str | None:
    """The part of an action's `description` a human actually needs.

    The middleware builds the default description as
    `f"{prefix}\\n\\nTool: {name}\\nArgs: {args}"` (langchain's
    `human_in_the_loop.py`) — i.e. the tool name we already print as a header, and
    the raw `args` **dict repr** we can render far better ourselves. Printing it
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


def _render_action(request: dict[str, Any]) -> None:
    """Print a proposed action so a human can *read* it.

    This is the whole point of the gate. `write_file` interrupts because a human
    should see the content before it is written — but a markdown report reaches us
    inside `args["content"]` as one string, and both of the ways this used to be
    displayed (the middleware's dict-repr `description`, then a 300-char JSON clip
    of the same dict) render it as a single unreadable line of escaped `\\n`, twice.
    A reviewer who cannot read the diff approves it unread, and the gate becomes
    theater.

    Rendered per-argument instead, and generically — `execute`'s `command` and any
    future gated tool's long string argument get the same treatment as `content`,
    with no per-tool special-casing to keep in sync with `GATED_TOOLS`.
    """
    name = request.get("name", "<tool>")
    args = request.get("args", {})

    print(f"\n  ⏸  Approval required — {name}")
    if note := _reviewer_note(request):
        print(f"     {note}")

    if not isinstance(args, dict):  # not a shape the middleware produces, but cheap
        print(f"     args: {_short(args)}")
        return

    for key, value in args.items():
        # A long or multi-line string is the thing the human is here to read: give
        # it real newlines and its own block. Everything else is a scalar — a path,
        # a flag — and reads fine inline.
        if isinstance(value, str) and ("\n" in value or len(value) > 120):
            lines = value.splitlines() or [""]
            count = f" ({len(lines)} lines)" if len(lines) > 1 else ""
            print(f"     {key}:{count}")
            for line in lines[:PREVIEW_LINES]:
                print(f"     │ {line}")
            if len(lines) > PREVIEW_LINES:
                print(f"     │ … {len(lines) - PREVIEW_LINES} more lines")
        else:
            print(f"     {key}: {_short(value, 200)}")


def _prompt_decision(
    request: dict[str, Any], allowed_decisions: Sequence[str] | None = None
) -> dict[str, Any]:
    """Ask the human to decide on one proposed action.

    Only the decisions `allowed_decisions` permits for *this* tool are offered.
    That restriction is load-bearing, not cosmetic: the middleware raises
    `ValueError` on a decision type outside the tool's `allowed_decisions`, and
    `main`'s broad `except` would swallow it into a one-line error, losing the
    turn. Every value in `GATED_TOOLS` is currently `True` (which permits all
    four), but an `InterruptOnConfig` narrowing them is a supported, documented
    thing to do — so the CLI has to honor whatever it is handed.

    **Approval is only ever returned for an affirmative act** — `a`, the empty
    default, or a deliberately blank edit. Never as a fallback from a failure to
    parse what the human typed: a mistyped edit means they wanted to *change* the
    args, so approving the original ones is the one outcome they certainly did not
    ask for.
    """
    name = request.get("name", "<tool>")

    # `None` means "no ReviewConfig came with this request" → assume the default.
    # An *empty* list is different: it means nothing is permitted. Don't conflate.
    permitted = set(
        DEFAULT_ALLOWED_DECISIONS if allowed_decisions is None else allowed_decisions
    )
    allowed = [d for d in DECISION_KEYS if d in permitted]
    if not allowed:
        # The tool is gated with a decision set this CLI cannot produce. Guessing
        # would just raise inside the graph, so fail loudly with the real reason.
        raise ValueError(
            f"no supported decision for '{name}' "
            f"(tool allows: {sorted(permitted) or 'nothing'})"
        )

    by_key = {DECISION_KEYS[d][0]: d for d in allowed}
    menu = " / ".join(DECISION_KEYS[d][1] for d in allowed)
    # Approving is the default only when it is actually on offer.
    default = "approve" if "approve" in permitted else None
    prompt = f"     {menu}{' (default a)' if default else ''} > "

    _render_action(request)

    while True:
        choice = input(prompt).strip().lower()
        if not choice and default:
            decision = default
        elif choice in by_key:
            decision = by_key[choice]
        elif choice in permitted and choice in DECISION_KEYS:
            decision = choice  # the full word, e.g. "approve"
        else:
            print(f"     ? choose {', '.join(DECISION_KEYS[d][0] for d in allowed)}.")
            continue

        if decision == "approve":
            return {"type": "approve"}
        if decision == "reject":
            reason = input("     reason for the agent (optional) > ").strip()
            return {"type": "reject", **({"message": reason} if reason else {})}
        if decision == "respond":
            # The human answers *on behalf of* the tool; the tool never runs, so
            # an empty message would hand the model an empty tool result.
            message = input("     reply to the agent on the tool's behalf > ").strip()
            if not message:
                print("     ? a response needs a message.")
                continue
            return {"type": "respond", "message": message}

        # edit — a *deliberately* blank line means "never mind, take it as-is", and
        # that shortcut is only legal when approve is permitted.
        can_fall_back = "approve" in permitted
        hint = "blank = approve as-is" if can_fall_back else "required"
        print(f"     enter replacement args as JSON ({hint}):")
        raw = input("     > ").strip()
        if not raw:
            if can_fall_back:
                return {"type": "approve"}
            print(
                "     ? this tool does not allow approving unchanged — edit or reject."
            )
            continue
        try:
            new_args = json.loads(raw)
        except json.JSONDecodeError as exc:
            # NEVER fall back to approve here. This used to return `approve` with the
            # ORIGINAL, unedited args whenever the tool permitted approving — so a
            # reviewer who chose `edit` precisely because the write looked wrong, and
            # then fat-fingered the JSON, silently approved the very write they were
            # trying to narrow. A typo is not consent, and this is the only security
            # boundary the app has. Re-prompt; `a` is right there if they mean it.
            print(f"     ! not valid JSON ({exc.msg}) — nothing approved. Try again.")
            continue
        if not isinstance(new_args, dict):
            # `"/memories/y.md"` and `[1, 2]` are valid JSON but not valid *args*.
            # The middleware doesn't validate, so this would sail through into a
            # ToolCall with non-dict args and only blow up at tool execution.
            print('     ! args must be a JSON object, e.g. {"file_path": "..."}.')
            continue
        return {"type": "edit", "edited_action": {"name": name, "args": new_args}}


def _collect_decisions(interrupts: list[Any]) -> dict[str, list[dict[str, Any]]]:
    """Collect decisions for every pending action, grouped by interrupt id.

    The human-in-the-loop middleware bundles all of *one agent's* pending tool
    calls into a single interrupt whose value is a HITLRequest with two parallel
    lists: `action_requests` (what the agent wants to do) and `review_configs`
    (which decisions are legal for each, keyed by `action_name`). That interrupt's
    resume value is `{"decisions": [...]}`, one decision per *request*, in order.

    But a turn can carry **more than one** interrupt. The orchestrator dispatches
    each `task` call as its own concurrent graph task, and every subagent inherits
    `interrupt_on` — so two `researcher` subagents fanned out in one turn (which
    `SYSTEM_PROMPT` explicitly encourages) can each raise their own interrupt. This
    is why the result is keyed by `interrupt.id` and not flattened: LangGraph
    raises `RuntimeError("When there are multiple pending interrupts, you must
    specify the interrupt id when resuming")` unless the resume value is a mapping
    of interrupt id → that interrupt's resume value. The mapping form is also
    correct for the ordinary single-interrupt case, so there is one code path.

    `review_configs` is looked up by name rather than by position: the middleware
    happens to append the two lists in lockstep today, but it documents the field
    as the policy "for all possible actions", so a name lookup stays correct if
    it is ever deduplicated.
    """
    by_interrupt: dict[str, list[dict[str, Any]]] = {}
    for interrupt in interrupts:
        value = getattr(interrupt, "value", interrupt)
        interrupt_id = getattr(interrupt, "id", None)
        if not isinstance(value, dict) or interrupt_id is None:
            continue
        if interrupt_id in by_interrupt:
            # THE SAME INTERRUPT, TWICE. With `subgraphs=True` an interrupt raised inside
            # a subagent is emitted at the subagent's namespace AND again, bubbled, at
            # the root — same `Interrupt.id`, two chunks. Prompting per occurrence would
            # ask the human to approve one researcher's `write_file` twice, and (since
            # this dict is keyed by id) silently keep only the second answer. Approval
            # fatigue is exactly how a gate stops being a gate.
            #
            # Deduped HERE, in the function that does the prompting, rather than in the
            # caller — the invariant is "one prompt per pending action", and it should
            # hold however this is called. `evals/harness._approve_all` is immune to the
            # same duplication only by accident: it writes into a dict keyed by id
            # without asking anyone anything, so a duplicate is idempotent.
            continue
        allowed_by_tool = {
            config["action_name"]: config["allowed_decisions"]
            for config in value.get("review_configs", [])
            if config.get("action_name") and config.get("allowed_decisions")
        }
        decisions = [
            _prompt_decision(request, allowed_by_tool.get(request.get("name")))
            for request in value.get("action_requests", [])
        ]
        if decisions:
            by_interrupt[interrupt_id] = decisions
    return by_interrupt


class ActivityFeed:
    """Prints what the agent is doing, as it does it.

    The turn used to be a black box: one `… working …` line, then minutes of nothing,
    then a wall of text. This renders the tool activity arriving on
    `agent.stream(..., stream_mode="updates", subgraphs=True)`.

    Three things it must get right, each of which is a bug waiting to happen:

    **It prints actions, never prose.** The stream carries the *researchers'* assistant
    messages too, and the user must never see one — they are a subagent's internal
    working, and `evals/harness.py` refuses to build its graded `response` from the
    stream for exactly this reason. The answer comes from `render_turn` on the final
    checkpoint, so the terminal, the exported file, and the eval's `response` stay the
    same bytes. Same rule as `harness.TurnRecorder`: actions only.

    **It prints each event once, keyed on the TOOL CALL id.** On resume,
    `HumanInTheLoopMiddleware.after_model` re-emits the AIMessage that proposed the gated
    call, and the re-streamed superstep re-emits the *cached writes* of tasks that
    already finished — so without deduping, every approval replays lines the user just
    watched scroll past. The key is the call id (`tool_call["id"]`, and `tool_call_id` on
    the result) rather than the *message* id, because a tool call executes exactly once,
    while `BaseMessage.id` is optional — a re-emitted `ToolMessage` need not carry one,
    and keying on it silently lets the duplicate through. (Observed: the completion line
    for a finished researcher printed twice, once per approval round.)

    **It does not pretend to know which researcher is which.** A subagent's namespace is
    `('tools:<pregel-task-uuid>',)`, and that uuid is not the `task` tool-call id — so
    binding a search back to the sub-question that spawned it would mean assuming
    dispatch order matches first-emission order under concurrency. It doesn't have to:
    the *dispatch* and *completion* lines carry the real description (recovered by
    `tool_call_id`), and each search line carries its actual query. That is the
    information worth having, and all of it is true.
    """

    def __init__(self) -> None:
        self._printed: set[str] = set()  # event keys already on screen
        self._task_descriptions: dict[str, str] = {}  # task tool_call id -> description
        self._ls_paths: dict[str, str] = {}  # ls tool_call id -> the path it listed

    def absorb(self, namespace: tuple[str, ...], chunk: Any) -> list[Any]:
        """Fold in one `(namespace, update)` chunk; return any interrupts it carried.

        Deliberately the same shape as `harness.TurnRecorder.absorb` — that one has been
        run against the live agent, and divergence between the two is how the REPL and
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
            if (todos := update.get("todos")) and is_orchestrator:
                # ORCHESTRATOR ONLY. deepagents gives every declarative subagent its own
                # `TodoListMiddleware`, so a `researcher` really can call `write_todos` —
                # and its list streams out under `('tools:<uuid>',)`. Rendering that would
                # print a researcher's private checklist as the agent's plan, appearing to
                # supersede the plan the user was just shown. This is the same
                # orchestrator/subagent conflation `evals/harness.py` keeps apart with
                # `orchestrator_trajectory` vs `trajectory`; the display layer has to make
                # the same distinction, for the same reason.
                self._render_plan(todos)
            for message in update.get("messages", []) or []:
                self._render_message(namespace, message)
        return interrupts

    def _once(self, key: str) -> bool:
        """True the first time this event is seen, False every time after."""
        if key in self._printed:
            return False
        self._printed.add(key)
        return True

    def _render_plan(self, todos: list[Any]) -> None:
        # `write_todos` returns a Command that updates the `todos` channel, so the whole
        # list arrives in the chunk — no need to parse the tool call.
        items = [t.get("content", "?") for t in todos if isinstance(t, dict)]
        # Keyed on the contents: a replayed superstep re-emits the identical plan (noise),
        # but a plan the agent genuinely revised is a different list, and worth showing.
        if not items or not self._once(f"plan:{items}"):
            return
        print(f"\n  ✎ plan · {len(items)} item{'s' if len(items) != 1 else ''}")
        for index, item in enumerate(items, 1):
            print(f"      {index}. {item}")

    def _render_message(self, namespace: tuple[str, ...], message: Any) -> None:
        is_orchestrator = not namespace
        kind = getattr(message, "type", None)

        if kind == "ai":
            for call in getattr(message, "tool_calls", None) or []:
                self._render_call(call, is_orchestrator)
        elif kind == "tool":
            self._render_result(message, is_orchestrator)

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
            print(f"  → researcher · {_one_line(description, 90)}")
        elif name == "tavily_search":
            # Announced at CALL time, not on the result: the query is the informative
            # part and this keeps the feed live. It also means never touching the
            # ToolMessage body, which for a search is multiple KB of serialized results.
            indent = "  " if is_orchestrator else "      "
            print(f'{indent}⌕ "{_one_line(args.get("query", "?"), 80)}"')
        elif name == "read_file" and is_orchestrator:
            print(f"  ▸ reading {args.get('file_path', '?')}")
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
            print(f"  ! {name} failed: {_one_line(_text_of(message), 100)}")
            return

        if name == "ls" and is_orchestrator:
            # ORCHESTRATOR ONLY — a `researcher` has its own `FilesystemMiddleware` and
            # can call `ls` on its own state-backed filesystem. Rendering that would tell
            # the user durable memory was consulted on a turn where the orchestrator never
            # looked, hiding the exact "the direct path skips /memories/" defect CLAUDE.md
            # says to keep watching.
            #
            # And the body is NOT newline-separated entries: deepagents builds it as
            # `str(paths)` — a Python list repr, `"[]"` or `"['/memories/a.md']"`. Counting
            # lines therefore reported "1 file(s)" for an EMPTY store, every time, and the
            # "empty" branch was unreachable. Parse the repr, and if it is not one, say so
            # rather than inventing a number.
            path = self._ls_paths.get(call_id, "/memories/")
            try:
                entries = ast.literal_eval(_text_of(message).strip())
            except (ValueError, SyntaxError):
                entries = None
            if isinstance(entries, list):
                count = f"{len(entries)} file(s)" if entries else "empty"
            else:
                count = "?"
            print(f"  ⌕ {path} · {count}")
        elif name == "task":
            # The one honest way to name a researcher: recover the sub-question from the
            # `task` call this result answers. Its stream namespace cannot be bound back
            # to that call without assuming dispatch order matches emission order under
            # concurrency, so we don't pretend to.
            description = self._task_descriptions.get(call_id)
            print(
                f"  ✓ researcher · {_one_line(description, 90)}"
                if description
                else "  ✓ researcher"
            )


def _one_line(text: Any, limit: int) -> str:
    """Collapse a value to a single, bounded line — feed lines must not wrap or wrap
    the terminal in a researcher's whole prompt."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _stream_turn(
    agent: Any, payload: Any, config: dict[str, Any], feed: ActivityFeed
) -> list[Any]:
    """Run one stream to exhaustion, printing the feed; return the pending interrupts.

    **Drain, THEN prompt.** Not a style choice — an interrupt chunk does not end the
    stream. LangGraph does not treat a `GraphInterrupt` as a failure, so sibling tasks in
    the same superstep keep running and a *second* researcher's interrupt arrives after
    the first. Worse, the graph executes inside this generator: blocking on `input()`
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


def _print_unfinished_turn(agent: Any, config: dict[str, Any]) -> None:
    """Print whatever the agent already said, when a turn ends early.

    Both `except` arms below used to `continue` straight back to the input prompt,
    skipping the `render_turn` at the bottom of the loop — and with it, the answer.
    That is not a hypothetical loss. The agent composes its cited report in the *same*
    assistant message that proposes the `write_file` (the reason `render_turn` exists
    at all), so the turn a human is most likely to Ctrl-C — the one sitting at an
    approval prompt — is reliably the one that has already done every search and
    written the entire report. Minutes of work and dozens of sources, discarded on a
    keystroke, while the prose sat in the checkpoint the whole time. Read it back.

    Best-effort by construction: we are already on an error path, and a failure to
    read the checkpoint must never replace the error the user actually needs to see.

    The abandoned tool call left dangling by this is not a problem for the *next* turn:
    deepagents puts `PatchToolCallsMiddleware` at the graph entry, which answers any
    dangling tool call with a synthetic "cancelled" ToolMessage before the model runs.
    """
    try:
        text = render_turn(agent.get_state(config).values)
    except Exception:  # noqa: BLE001 — salvage must not mask the failure that got us here
        return
    if text:
        print(f"\nagent (unfinished turn) > {text}")


def _export(agent: Any, config: dict[str, Any], thread_id: str, target: str) -> None:
    """Write the thread to a markdown file the user can actually keep.

    Plain `Path.write_text`, deliberately — NOT the agent's own `write_file` tool. That
    route is not merely heavier, it is incoherent: `HumanInTheLoopMiddleware` interrupts
    on the tool calls of the *model's* last message, so there is no way to invoke a gated
    tool without a model turn. Exporting through the agent would mean an Opus call, and
    an approval prompt asking the human to approve the thing the human just typed — with
    the model free to rename, reword, or decline it. And a `/memories/` path is not a
    file at all: it is a row in `memories.sqlite`, which is precisely why `SYSTEM_PROMPT`
    step 5 says "/memories/ or nothing" and stopped asking the agent to write reports.
    `/export` gives the user the artifact that prompt deliberately stopped producing, at
    zero tokens and zero approvals — so `SYSTEM_PROMPT` needs no changes and must not be
    told about it.

    Sourced from the checkpoint, never from the stream, even though the streaming loop
    has the chunks in hand. The stream carries the researchers' own prose, which the user
    never saw and whose citations the eval deliberately refuses to credit; taking the
    convenient path would put subagent-internal text into the user's file and drift the
    export away from both the terminal output and the eval's graded `response`.
    """
    text = render_thread(agent.get_state(config).values)
    if not text:
        print("! nothing to export — this thread has no answers yet.")
        return

    named = bool(target)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    # `expanduser`, because `/export ~/report.md` is the obvious thing to type and no
    # shell expanded it for us — the path arrived as the literal string `~/report.md`.
    # Without this it fails with a bare ENOENT (or, if a stray `~` directory exists in
    # cwd, silently succeeds into `./~/report.md`), losing a report that took minutes.
    path = (
        Path(target).expanduser() if named else Path(f"research-{thread_id}-{stamp}.md")
    )
    if path.exists() and not named:
        print(f"! {path} already exists — pass an explicit path to overwrite.")
        return

    header = (
        f"# Deep research — thread `{thread_id}`\n\n"
        f"*Model: `{MODEL_NAME}`. Exported {stamp}.*\n"
        # "Exported", not "answered": the messages carry no timestamps, and the only
        # per-turn clock lives in the checkpointer's snapshots. A date we did not
        # measure is a date we invented.
    )
    try:
        # Explicit encoding, always: the default is locale-dependent, and a real report
        # is full of em-dashes. Failing on the one machine the user cannot debug is not
        # a hypothetical. Caught here, not by main's turn-scoped `except`.
        path.write_text(f"{header}\n{text}\n", encoding="utf-8")
    except OSError as exc:
        print(f"! export failed: {exc}")
        return
    print(f"(exported to {path.resolve()})")


def main() -> None:
    missing = missing_keys()
    if missing:
        print("Missing required environment variables:\n")
        for key, why in missing.items():
            print(f"  - {key}\n      {why}")
        print("\nCopy .env.example to .env, fill these in, then re-run.")
        sys.exit(1)

    thread_id = "main"
    print(BANNER)

    with open_agent() as agent:
        while True:
            try:
                user_input = input("\nyou > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye.")
                return

            if not user_input:
                continue
            if user_input in ("/exit", "/quit"):
                print("bye.")
                return
            if user_input == "/help":
                print(HELP)
                continue
            if user_input.startswith("/thread"):
                parts = user_input.split(maxsplit=1)
                if len(parts) == 2 and parts[1].strip():
                    thread_id = parts[1].strip()
                    print(f"(switched to thread '{thread_id}')")
                else:
                    print(f"(current thread: '{thread_id}')")
                continue
            if user_input.startswith("/export"):
                parts = user_input.split(maxsplit=1)
                _export(
                    agent,
                    {"configurable": {"thread_id": thread_id}},
                    thread_id,
                    parts[1].strip() if len(parts) == 2 else "",
                )
                continue

            config = {"configurable": {"thread_id": thread_id}}
            feed = ActivityFeed()
            payload: Any = {"messages": [{"role": "user", "content": user_input}]}
            try:
                # Drain the stream, decide on everything it paused for, restream. There
                # is no `__interrupt__` key to loop on any more: under `stream_mode=
                # "updates"` interrupts only ever arrive as chunks, and `invoke()`'s
                # `result["__interrupt__"]` was itself just a post-drain aggregate that
                # LangGraph assembled internally. Same loop as `evals/harness.py`.
                while pending := _stream_turn(agent, payload, config, feed):
                    by_interrupt = _collect_decisions(pending)
                    if not by_interrupt:
                        # Nothing reviewable — resuming would just re-interrupt.
                        print("\n! paused with no reviewable action; abandoning turn.")
                        break
                    # Keyed by interrupt id: a turn can hold several interrupts at once
                    # (concurrent researchers), and LangGraph rejects a resume that
                    # doesn't say which interrupt each value belongs to.
                    payload = Command(
                        resume={
                            interrupt_id: {"decisions": decisions}
                            for interrupt_id, decisions in by_interrupt.items()
                        }
                    )
            except KeyboardInterrupt:
                # Ctrl-C is a BaseException (not Exception), so it must be caught
                # separately — otherwise it escapes mid-turn as a raw traceback
                # instead of returning to the prompt like Ctrl-C does at input().
                print("\n(interrupted — back to prompt)")
                _print_unfinished_turn(agent, config)
                continue
            except Exception as exc:  # noqa: BLE001 — surface any runtime error to the user
                print(f"\n! error: {exc}")
                _print_unfinished_turn(agent, config)
                continue

            # From the checkpoint, NOT from the stream — even though the stream just went
            # past us and it would be easy. The stream carries the researchers' own
            # assistant messages, and the user must never be shown one; `evals/harness.py`
            # renders its graded `response` with this same call for exactly that reason,
            # so building the printed answer any other way makes the citation metrics
            # fiction. The feed shows *actions*; the answer comes from state.
            answer = render_turn(agent.get_state(config).values)
            print(f"\nagent > {answer}" if answer else "\n(the agent said nothing)")


if __name__ == "__main__":
    main()
