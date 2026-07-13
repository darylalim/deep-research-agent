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

import json
import sys
from collections.abc import Sequence
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
│  Commands:  /help   /thread <id>   /exit                      │
╰──────────────────────────────────────────────────────────────╯"""

HELP = f"""\
Commands:
  /help          show this help
  /thread <id>   switch to a different conversation thread (default: "main")
  /exit, /quit   leave

Notes:
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

            config = {"configurable": {"thread_id": thread_id}}
            print("… working (planning, searching, synthesizing)…")
            try:
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": user_input}]},
                    config=config,
                )
                # Resuming may hit the next gated tool, so loop until no interrupt.
                while result.get("__interrupt__"):
                    by_interrupt = _collect_decisions(result["__interrupt__"])
                    if not by_interrupt:
                        # Nothing reviewable — resuming would just re-interrupt.
                        print("\n! paused with no reviewable action; abandoning turn.")
                        break
                    # Keyed by interrupt id: a turn can hold several interrupts at
                    # once (concurrent subagents), and LangGraph rejects a resume
                    # that doesn't say which interrupt each value belongs to.
                    result = agent.invoke(
                        Command(
                            resume={
                                interrupt_id: {"decisions": decisions}
                                for interrupt_id, decisions in by_interrupt.items()
                            }
                        ),
                        config=config,
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

            # `render_turn` is prose-or-nothing now, so an empty string is possible —
            # a turn abandoned at the "nothing reviewable" break above, say. Say so
            # rather than printing a bare `agent > ` and looking broken.
            answer = render_turn(result)
            print(f"\nagent > {answer}" if answer else "\n(the agent said nothing)")


if __name__ == "__main__":
    main()
