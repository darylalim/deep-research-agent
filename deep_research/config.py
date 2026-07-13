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
# Default to Claude Opus 4.8, the current most-capable Opus-tier model.
# IMPORTANT: Opus 4.8 rejects `temperature` / `top_p` / `top_k` with a 400.
# `ChatAnthropic` omits those params when they are left unset, so we
# deliberately construct the model without any sampling parameters.
MODEL_NAME = os.environ.get("DEEP_RESEARCH_MODEL", "claude-opus-4-8")
# 32k, and it is `streaming=True` below that makes that safe — see `build_model`.
MAX_TOKENS = int(os.environ.get("DEEP_RESEARCH_MAX_TOKENS", "32000"))


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

    Sampling params stay unset: Opus 4.8 returns a 400 on `temperature`/`top_p`/`top_k`,
    and `ChatAnthropic` omits what is unset. Verified that streaming adds only
    `stream: true` to the payload, so it does not disturb that invariant — nor prompt
    caching, which reports `cache_read` in the `message_delta` either way.

    The `ty: ignore` silences a false positive: ty builds the signature from the
    Pydantic *aliases* (`model_name`, `max_tokens_to_sample`) and doesn't model
    `populate_by_name`.
    """
    return ChatAnthropic(  # ty: ignore[missing-argument]
        model=MODEL_NAME,  # ty: ignore[unknown-argument]
        max_tokens=MAX_TOKENS,  # ty: ignore[unknown-argument]
        streaming=True,
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
