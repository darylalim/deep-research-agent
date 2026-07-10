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
# 16k keeps non-streaming responses comfortably under the SDK's HTTP timeout
# while leaving ample room for a synthesized report.
MAX_TOKENS = int(os.environ.get("DEEP_RESEARCH_MAX_TOKENS", "16000"))


def build_model() -> ChatAnthropic:
    """Construct the Claude chat model used by the orchestrator and subagents."""
    # `model=` / `max_tokens=` are the idiomatic, documented kwargs and work at
    # runtime (the fields allow population by name). The `ty: ignore` silences a
    # false positive: ty builds the signature from the Pydantic *aliases*
    # (`model_name`, `max_tokens_to_sample`) and doesn't model `populate_by_name`.
    return ChatAnthropic(model=MODEL_NAME, max_tokens=MAX_TOKENS)  # ty: ignore[unknown-argument, missing-argument]


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
