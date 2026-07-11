"""Opt-in tests that exercise real external APIs (Anthropic / Tavily).

These are DESELECTED by default (`addopts = -m "not live"` in pyproject.toml)
because they need real keys + network, cost money, and — for full agent runs —
are non-deterministic. Run them deliberately, with real keys in the environment:

    uv run pytest -m live

This module intentionally contains only a minimal smoke check; it is not part of
the offline suite's guarantees.
"""

from __future__ import annotations

import pytest


@pytest.mark.live
def test_tavily_search_returns_a_result() -> None:
    from deep_research.tools import build_web_search

    tool = build_web_search(max_results=1)
    result = tool.invoke({"query": "capital of France"})
    # `assert result` alone is vacuous: TavilySearch swallows request errors into
    # a truthy `{"error": ...}` dict and (with handle_tool_error) turns empty
    # results into a truthy error *string*. Assert the real success shape so an
    # auth/network failure or an empty result set actually fails the test.
    assert isinstance(result, dict), (
        f"expected a dict payload, got {type(result)}: {result!r}"
    )
    assert "error" not in result, f"search errored: {result.get('error')!r}"
    assert result.get("results"), f"no results in payload: {result!r}"
