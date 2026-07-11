"""Tests for the assembly in `agent.py`.

The keystone is `test_open_agent_assembles_offline`: because this project is a
thin configuration layer over `deepagents`/`langchain`, the dominant breakage
risk is a dependency upgrade changing a kwarg or backend contract. This test
exercises the entire wiring (model + tools + subagent + CompositeBackend routes
+ interrupt_on/checkpointer + store) with no network and no real API keys, so it
catches that class of failure cheaply.
"""

from __future__ import annotations

from deep_research.agent import GATED_TOOLS, open_agent


def test_mutating_and_shell_tools_are_gated() -> None:
    # Safety property: writing files and running shell commands must require
    # human approval. LangChain only gates a tool when its value is `True` or an
    # InterruptOnConfig with a truthy `allowed_decisions` (human_in_the_loop.py:
    # 252-260); a value of `False` — or a config missing `allowed_decisions` —
    # silently un-gates while the key stays present. Assert the value, not just
    # the key, so a value flip can't defeat approval unnoticed.
    for tool_name in ("write_file", "edit_file", "execute"):
        assert tool_name in GATED_TOOLS, f"{tool_name} is not gated"
        config = GATED_TOOLS[tool_name]
        gated = config is True or (
            isinstance(config, dict) and config.get("allowed_decisions")
        )
        assert gated, f"{tool_name} is present but its value does not enable gating"


def test_open_agent_assembles_offline() -> None:
    with open_agent() as agent:
        assert agent is not None
        # It's a compiled LangGraph — it must expose the invoke entry point the
        # CLI drives.
        assert hasattr(agent, "invoke")
