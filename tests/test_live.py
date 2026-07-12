"""Opt-in tests that exercise real external APIs (Anthropic / Tavily).

These are DESELECTED by default (`addopts = -m "not live"` in pyproject.toml)
because they need real keys + network, cost money, and — for full agent runs —
are non-deterministic. Run them deliberately, with real keys in the environment:

    uv run pytest -m live

These checks are deliberately narrow — a Tavily smoke test, and the one invariant
that can *only* be observed against the real API (that the prompt cache is being
read). Neither is part of the offline suite's guarantees.
"""

from __future__ import annotations

from typing import Any

import pytest


@pytest.mark.live
def test_prompt_caching_actually_serves_the_prefix_from_cache() -> None:
    """The system+tools prefix is really served from Anthropic's prompt cache.

    Nothing in this repo switches caching on: `create_deep_agent()` appends an
    `AnthropicPromptCachingMiddleware` itself, for the orchestrator *and* every
    subagent. That is exactly why it deserves a test — it is invisible from here.
    The middleware hooks `wrap_model_call`, so it is not a graph node and does
    not appear in `agent.nodes`; `agent.py` gives no hint it exists. If a future
    deepagents drops the default, no other test would go red and the input bill
    would just silently multiply (cache reads bill at ~0.1x of base input).

    Asserting the middleware is *present* would be the weaker test — it would
    still pass if the prefix fell under Anthropic's minimum cacheable size (4096
    tokens on Opus 4.8), where `cache_control` is honored but nothing is cached.
    So assert on the token accounting instead.

    Two *separate* threads, one turn each — not two turns on one thread. The
    cache is keyed on the prompt prefix, not the thread, so the second thread
    reads the system+tools prefix the first one wrote. Both invokes are therefore
    a thread's opening turn, which keeps the test off two sharp edges: it never
    has to resume a gated tool call, and it never re-enters an interrupted thread
    with fresh input (which resumes the model node on a message list ending in an
    assistant message — a 400 on Opus 4.8, which rejects prefill).
    """
    from langchain_core.messages import AIMessage

    from deep_research.agent import open_agent

    # Cheap prompt: this test is about token accounting, not research behavior.
    # It may still choose to call a tool — that's fine, an interrupt just ends the
    # turn early and the model call we need has already happened by then.
    prompt = "Reply with exactly: OK. Do not call any tools and do not write any files."
    turn = {"messages": [{"role": "user", "content": prompt}]}

    def cache_reads(state: dict[str, Any]) -> int:
        return sum(
            (message.usage_metadata or {})
            .get("input_token_details", {})
            .get("cache_read", 0)
            for message in state["messages"]
            if isinstance(message, AIMessage)
        )

    with open_agent() as agent:
        # First thread warms the prefix; second reads it back.
        agent.invoke(turn, config={"configurable": {"thread_id": "cache-probe-warm"}})
        result = agent.invoke(
            turn, config={"configurable": {"thread_id": "cache-probe-read"}}
        )

    assert cache_reads(result) > 0, (
        "the prompt prefix was rebuilt from scratch instead of being read from cache. "
        "Either deepagents' AnthropicPromptCachingMiddleware is no longer applied, or "
        "the system+tools prefix fell under Anthropic's 4096-token cacheable minimum. "
        "Every turn is now paying full input price for the prefix."
    )


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
