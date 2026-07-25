"""Shared test setup.

Sets dummy credentials and an isolated state directory *before* anything imports
`deep_research.config` — that module resolves `STATE_DIR` and calls
`load_dotenv()` at import time, so these values must be in place first. This is
top-level code (not a fixture) on purpose: pytest imports `conftest.py` before
the sibling test modules, whereas a fixture would run too late for a test
module's own `from deep_research... import ...` line.

Order matters: load the real `.env` first so a developer's actual keys populate
`os.environ` for the opt-in `live` suite (whose import path never imports
`config`, so nothing else would load `.env`). Then `setdefault` fills in
deterministic dummies for anything still unset — the offline suite needs no real
keys, and CI has no `.env`.
"""

from __future__ import annotations

import os
import tempfile

from dotenv import load_dotenv

# Real .env first (for the opt-in `live` suite), then dummies for anything unset.
load_dotenv()
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-dummy")
os.environ.setdefault("TAVILY_API_KEY", "tvly-test-dummy")
# A throwaway dir for the checkpointer/store sqlite files the smoke test creates.
os.environ["DEEP_RESEARCH_STATE_DIR"] = tempfile.mkdtemp(prefix="deep_research_tests_")
# Unset, NOT pinned: `test_max_tokens_leaves_room_for_thinking_and_the_answer` exists to
# assert the shipped DEFAULT is high enough to hold Opus 5's thinking plus a long cited
# report. Pinning a value here would make it assert what this file just set — vacuous.
# Popping makes it exercise the real `os.environ.get(..., "64000")` fallback in
# `config.py`.
#
# It has to be popped rather than merely left alone because `load_dotenv()` ran above (it
# must: the opt-in `live` suite needs the developer's real keys). So a developer who caps
# spend with `DEEP_RESEARCH_MAX_TOKENS=32000` in their `.env` — a documented, supported
# override — would otherwise fail an offline test about an invariant they never touched.
# And because `.claude/hooks/post-edit.sh` runs this suite on every `.py` edit, that
# failure would block all further editing until they found and unset it.
os.environ.pop("DEEP_RESEARCH_MAX_TOKENS", None)
