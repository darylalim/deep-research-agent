"""Configuration: environment loading, model construction, and state paths.

Everything the rest of the package needs to know about *where* things live and
*how* to build the model is centralized here.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

# Load `.env` (if present) into os.environ before anything reads a key.
load_dotenv()

# --- Model -----------------------------------------------------------------
# Default to Claude Opus 5, the current most-capable Opus-tier model.
# IMPORTANT: Opus 5 rejects `temperature` / `top_p` / `top_k` with a 400 — the same
# rule that held on Opus 4.8, and the reason no sampling parameter is set anywhere
# in this file. `ChatAnthropic` omits those params when they are left unset.
MODEL_NAME = os.environ.get("DEEP_RESEARCH_MODEL", "claude-opus-5")
# 64k, not 32k, because on Opus 5 `max_tokens` now has to cover the model's THINKING
# as well as its answer — see the `thinking` note in `build_model`. `streaming=True`
# is what makes a ceiling this high safe.
MAX_TOKENS = int(os.environ.get("DEEP_RESEARCH_MAX_TOKENS", "64000"))


def build_model() -> ChatAnthropic:
    """Construct the Claude chat model used by the orchestrator and subagents.

    **`streaming=True` is what holds the `max_tokens` ceiling open, and it is not
    about the CLI.** It flips the *model's own HTTP request* to SSE
    (`_should_stream()` → `_stream()` → `generate_from_stream()`), while still handing
    the graph one complete `AIMessage` — nothing downstream, in LangGraph, deepagents,
    the HITL middleware, or the eval harness, can tell the difference. It is
    independent of whatever `stream_mode` a caller passes to `agent.stream()`: the
    agent's model node calls `model_.invoke()` unconditionally, so graph-level
    streaming does *not* make the request stream.

    This corrects a premise that was wrong here for a long time. The old comment said
    16k kept responses "comfortably under the SDK's HTTP timeout". There is no such
    timeout: langchain passes `default_request_timeout=None` straight into
    `anthropic.Client(timeout=None)`, and the httpx client ends up with
    `Timeout(timeout=None)` — measured. Which also disarms the SDK's own guard, since
    that only fires when the client still has the SDK default timeout. So a
    non-streaming request over the guard's threshold
    (`3600 * max_tokens / 128_000 > 600`, i.e. **max_tokens > 21_333**) would not raise
    — it would hang the REPL indefinitely, which is strictly worse than the failure the
    16k pin was imagined to prevent. Raise `max_tokens` and set `streaming=True`
    together, or neither.

    Sampling params stay unset: Opus 5 returns a 400 on `temperature`/`top_p`/`top_k`,
    and `ChatAnthropic` omits what is unset. Verified that streaming adds only
    `stream: true` to the payload, so it does not disturb that invariant — nor prompt
    caching, which reports `cache_read` in the `message_delta` either way.

    **`thinking` is set explicitly, and `display="summarized"` is load-bearing — it is
    not a cosmetic choice.** Opus 5 *inverted* the old default: omitting the parameter
    meant no thinking on Opus 4.8/4.7 and now runs adaptive thinking. So thinking is on
    either way; what we cannot accept is Opus 5's default `display="omitted"`.

    Measured, against the real API: under `display="omitted"` the thinking block comes
    back carrying only `{"signature": ..., "type": "thinking"}` — **no `thinking` text
    field at all** — and `langchain_anthropic` therefore has nothing to send back when
    that message is replayed. The next request dies with

        400 messages.1.content.1.thinking.thinking: Field required

    This is not a multi-turn-only edge case, which is what makes it fatal here: *every
    tool call replays the assistant message that requested it* alongside the tool result,
    so a single research turn re-sends its own thinking block the moment `tavily_search`
    returns. With `display="summarized"` the block carries real text, survives the round
    trip, and the replay succeeds — verified end to end.

    `display` controls visibility only: thinking happens and is billed identically either
    way, so this costs nothing. It also never reaches the user — `cli._text_of` collects
    bare strings and `{"type": "text"}` blocks, and a thinking block is neither.

    Two more consequences, neither optional:

    - **`MAX_TOKENS` had to go up.** `max_tokens` caps thinking *plus* the answer, so the
      32k that comfortably held a cited report now has to hold the reasoning too. A tight
      ceiling truncates mid-report with `stop_reason="max_tokens"` and no exception.
    - **Do NOT "save tokens" by passing `thinking={"type": "disabled"}`.** On Opus 5 that
      has a documented failure mode aimed squarely at this app: the model sometimes writes
      a tool call into its *visible text* instead of emitting a `tool_use` block. The turn
      succeeds, nothing raises, and the call simply never runs — worst on tool-heavy search
      workloads, which is the whole agent. (It can also leak `<thinking>` tags into the
      answer.) Lower `output_config.effort` is the supported cost lever; disabling is not.
      Disabling is additionally a 400 above `high` effort.

    The cost of being explicit: `thinking` in this shape is a 400 on pre-4.6 models, which
    expect the removed `budget_tokens` form. Pointing `DEEP_RESEARCH_MODEL` at one of those
    means dropping this argument too. That is the right trade — a broken default model is
    worse than a constrained override.
    `test_config.py::test_thinking_is_adaptive_and_summarized_never_disabled` pins all of it.

    This call used to carry three `ty: ignore`s for a false positive — ty built the
    signature from the Pydantic *aliases* (`model_name`, `max_tokens_to_sample`) and
    did not model `populate_by_name`, so `model=` and `max_tokens=` read as unknown
    arguments and `model_name` as a missing one. **ty 0.0.63 fixed it**; measured on
    the identical `langchain-anthropic` build, 0.0.58 reports all three and 0.0.63 is
    clean. The directives are gone because a dead one is itself a `ty check` failure
    (`unused-ignore-comment`), the same way `RUF100` treats a dead `noqa`. That is why
    the dev floor is `ty>=0.0.63` and not the older pin — with an earlier ty these
    three lines are hard errors again.
    """
    return ChatAnthropic(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        streaming=True,
        # `display="summarized"` is REQUIRED, not cosmetic: Opus 5's default
        # ("omitted") returns thinking blocks with no `thinking` text, which then
        # cannot be replayed — and every tool result replays the assistant message
        # that requested it. See the docstring above for the measured 400.
        thinking={"type": "adaptive", "display": "summarized"},
    )


# --- Local state (gitignored, survives restarts) ---------------------------
STATE_DIR = Path(os.environ.get("DEEP_RESEARCH_STATE_DIR", ".deep_research")).resolve()
# Thread state (conversation + todos + pending HITL interrupts). Backing store
# for the checkpointer — this is what lets an interrupted approval survive a
# process restart.
CHECKPOINT_DB = STATE_DIR / "checkpoints.sqlite"
# Long-term memory shared across every thread/session — the `/memories/` route.
MEMORY_DB = STATE_DIR / "memories.sqlite"


def ensure_state_dir() -> None:
    """Create the local state directory if it does not exist yet."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)


# --- Required credentials --------------------------------------------------
REQUIRED_KEYS: dict[str, str] = {
    "ANTHROPIC_API_KEY": "Claude model access — https://console.anthropic.com",
    "TAVILY_API_KEY": "web search — https://app.tavily.com (free tier available)",
}


def missing_keys() -> dict[str, str]:
    """Return the subset of required keys that are not set in the environment."""
    return {key: why for key, why in REQUIRED_KEYS.items() if not os.environ.get(key)}
