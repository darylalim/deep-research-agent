"""Interactive command-line chat for the deep research agent.

Runs a REPL against a single persistent thread. When the agent proposes a gated
action (writing a file, running a command) the turn pauses and this CLI collects
an approve / edit / reject decision, then resumes the run.
"""

from __future__ import annotations

import json
import sys
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

HELP = """\
Commands:
  /help          show this help
  /thread <id>   switch to a different conversation thread (default: "main")
  /exit, /quit   leave

Notes:
  • Conversation, todos, and pending approvals persist across restarts
    (checkpointed to .deep_research/checkpoints.sqlite).
  • Durable findings the agent saves under /memories/ persist across every
    thread and session ({memory_db}).
  • Writing files or running commands pauses for your approval.""".format(
    memory_db=MEMORY_DB.name
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


def _last_ai_text(result: dict[str, Any]) -> str:
    """Return the text of the final assistant message in a result state."""
    messages = result.get("messages", [])
    for message in reversed(messages):
        if getattr(message, "type", None) == "ai":
            return _text_of(message).strip()
    return _text_of(messages[-1]).strip() if messages else ""


def _short(value: Any, limit: int = 300) -> str:
    """Compactly render tool args for display."""
    try:
        text = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(value)
    return text if len(text) <= limit else text[:limit] + " …"


def _prompt_decision(request: dict[str, Any]) -> dict[str, Any]:
    """Ask the human to approve / edit / reject one proposed action."""
    name = request.get("name", "<tool>")
    args = request.get("args", {})
    description = request.get("description")

    print(f"\n  ⏸  Approval required — {name}")
    if description:
        print(f"     {description}")
    print(f"     args: {_short(args)}")

    while True:
        choice = (
            input("     [a]pprove / [e]dit / [r]eject (default a) > ").strip().lower()
        )
        if choice in ("", "a", "approve"):
            return {"type": "approve"}
        if choice in ("r", "reject"):
            reason = input("     reason for the agent (optional) > ").strip()
            return {"type": "reject", **({"message": reason} if reason else {})}
        if choice in ("e", "edit"):
            print("     enter replacement args as JSON (blank = approve as-is):")
            raw = input("     > ").strip()
            if not raw:
                return {"type": "approve"}
            try:
                new_args = json.loads(raw)
            except json.JSONDecodeError:
                print("     ! not valid JSON — approving with the original args.")
                return {"type": "approve"}
            return {"type": "edit", "edited_action": {"name": name, "args": new_args}}
        print("     ? choose a, e, or r.")


def _collect_decisions(interrupts: list[Any]) -> list[dict[str, Any]]:
    """Build one decision per pending action across all active interrupts.

    The human-in-the-loop middleware bundles every pending tool call for a turn
    into a single interrupt whose value is a HITLRequest with `action_requests`;
    the resume payload is `{"decisions": [...]}` with one decision per request,
    in order.
    """
    decisions: list[dict[str, Any]] = []
    for interrupt in interrupts:
        value = getattr(interrupt, "value", interrupt)
        action_requests = (
            value.get("action_requests", []) if isinstance(value, dict) else []
        )
        for request in action_requests:
            decisions.append(_prompt_decision(request))
    return decisions


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
                    decisions = _collect_decisions(result["__interrupt__"])
                    result = agent.invoke(
                        Command(resume={"decisions": decisions}), config=config
                    )
            except KeyboardInterrupt:
                # Ctrl-C is a BaseException (not Exception), so it must be caught
                # separately — otherwise it escapes mid-turn as a raw traceback
                # instead of returning to the prompt like Ctrl-C does at input().
                print("\n(interrupted — back to prompt)")
                continue
            except Exception as exc:  # noqa: BLE001 — surface any runtime error to the user
                print(f"\n! error: {exc}")
                continue

            print(f"\nagent > {_last_ai_text(result)}")


if __name__ == "__main__":
    main()
