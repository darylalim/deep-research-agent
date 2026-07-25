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
    still pass if the prefix fell under Anthropic's minimum cacheable size (512
    tokens on Opus 5), where `cache_control` is honored but nothing is cached.
    So assert on the token accounting instead. That minimum is per-model and is
    NOT monotonic across generations — 512 on Opus 5, 1024 on Opus 4.8, 4096 on
    Opus 4.6 — so a `DEEP_RESEARCH_MODEL` override can move the bar under the
    prefix without anything else changing. The measured ~11.9k system+tools
    prefix clears all three with room to spare.

    Two *separate* threads, one turn each — not two turns on one thread. The
    cache is keyed on the prompt prefix, not the thread, so the second thread
    reads the system+tools prefix the first one wrote. Both invokes are therefore
    a thread's opening turn, which keeps the test off two sharp edges: it never
    has to resume a gated tool call, and it never re-enters an interrupted thread
    with fresh input (which resumes the model node on a message list ending in an
    assistant message — a 400 on Opus 5, which still rejects prefill).

    This is also the test to run after a MODEL change, not just after a
    `build_model()` change: it is the only check here that exercises the real
    wire format, so it is what catches a new model rejecting the request shape.
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
        "the system+tools prefix fell under this model's cacheable minimum (512 tokens "
        "on Opus 5; higher on older models). "
        "Every turn is now paying full input price for the prefix."
    )


@pytest.mark.live
def test_a_thinking_turn_survives_a_tool_result_round_trip() -> None:
    """An assistant message carrying a thinking block can be replayed to the API.

    This is the regression that the Opus 5 upgrade nearly shipped. Opus 5 runs
    adaptive thinking, and under its DEFAULT `display="omitted"` the thinking
    block comes back as `{"signature", "type"}` with no `thinking` text — so
    `langchain_anthropic` cannot send it back, and the replay dies with
    `400 messages.N.content.M.thinking.thinking: Field required`.

    It is not a multi-turn-only bug, which is what makes it worth a live test:
    **every tool result replays the assistant message that requested the call**,
    so a research turn trips it the moment `tavily_search` returns — the agent
    would fail on its first delegation. `config.build_model()` therefore pins
    `display="summarized"`, and `test_config.py` asserts that payload shape.

    That offline test stops anyone editing the config. THIS one catches the other
    direction: `langchain_anthropic` learning to round-trip omitted-display blocks
    (making the pin removable), or regressing so that even summarized ones break.
    Neither is visible without the real API.
    """
    from langchain_core.messages import HumanMessage, ToolMessage
    from langchain_core.tools import tool

    from deep_research.config import build_model

    @tool
    def lookup(topic: str) -> str:
        """Look up factual data about a topic."""
        return "Rate A: 120/min. Rate B: 2400/hour. Tier cap: 50000/day."

    question = (
        "Use the lookup tool, then reason carefully: which of rate A or rate B is "
        "higher per day, and does either exceed the tier cap? Show the arithmetic."
    )
    model = build_model().bind_tools([lookup])
    answer = model.invoke([HumanMessage(question)])

    if not answer.tool_calls:
        pytest.skip("model answered without the tool; nothing to round-trip")
    thinking = [
        block
        for block in answer.content
        if isinstance(block, dict) and block.get("type") == "thinking"
    ]
    if not thinking:
        # Adaptive thinking is the model's call, so this is a skip, not a failure —
        # the assertion below is only meaningful when a thinking block exists.
        pytest.skip("adaptive thinking declined to think on this turn")

    assert thinking[0].get("thinking"), (
        "thinking block came back with no text — `display` is not 'summarized'. "
        "Replaying this message will 400 with `thinking.thinking: Field required`, "
        "which breaks every turn that calls a tool."
    )

    # Answer EVERY tool call: the API rejects a replay that leaves one unanswered,
    # and the model routinely emits several in one message.
    replay = [
        HumanMessage(question),
        answer,
        *(
            ToolMessage(content=lookup.invoke(call["args"]), tool_call_id=call["id"])
            for call in answer.tool_calls
        ),
    ]
    # The assertion IS that this does not raise a 400.
    assert model.invoke(replay) is not None


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
