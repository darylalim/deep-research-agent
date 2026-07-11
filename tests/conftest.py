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
