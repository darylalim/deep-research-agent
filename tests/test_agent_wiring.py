"""Tests for the assembly in `agent.py`.

The keystone is `test_open_agent_assembles_offline`: because this project is a
thin configuration layer over `deepagents`/`langchain`, the dominant breakage
risk is a dependency upgrade changing a kwarg or backend contract. This test
exercises the entire wiring (model + tools + subagent + CompositeBackend routes
+ interrupt_on/checkpointer + store) with no network and no real API keys, so it
catches that class of failure cheaply.
"""

from __future__ import annotations

import warnings
from typing import Any
from unittest import mock

from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

from deep_research import agent as agent_module
from deep_research.agent import (
    GATED_TOOLS,
    MEMORY_NAMESPACE,
    build_backend,
    open_agent,
)


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


def test_only_memories_is_routed_to_the_durable_store() -> None:
    # The two-layer persistence contract: `/memories/` (and nothing else) reaches
    # the cross-session Store; everything else stays thread-scoped. The route key
    # is also hardcoded *in prose* in agent.py's SYSTEM_PROMPT, so a change here
    # that isn't mirrored there makes the agent write "durable" notes that vanish.
    backend = build_backend()
    assert list(backend.routes) == ["/memories/"]
    assert isinstance(backend.routes["/memories/"], StoreBackend)
    assert isinstance(backend.default, StateBackend)


def test_backend_construction_is_free_of_deprecation_warnings() -> None:
    # deepagents 0.7.0 REMOVES `StateBackend(runtime)` / `StoreBackend(runtime)`.
    # That warning fires at CONSTRUCTION, so promoting it to an error here is what
    # keeps the repo 0.7.0-ready — nothing else would catch it, since a backend
    # isn't exercised until the first filesystem tool call at invoke time.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        build_backend()


def test_memory_namespace_is_explicit_and_unchanged() -> None:
    # The OTHER 0.7.0 removal: a `StoreBackend` with no explicit `namespace` is
    # deprecated too — but it only warns when a store operation actually resolves
    # the namespace, so the construction-time test above cannot see it. Resolve it
    # here and promote the warning to an error.
    #
    # The asserted value is not cosmetic. `("filesystem",)` is exactly what
    # deepagents' legacy auto-detection returns for this app today, and it is the
    # key every note already in `memories.sqlite` is stored under. Changing it
    # orphans the user's durable memory, silently and unrecoverably.
    store_backend = build_backend().routes["/memories/"]
    assert isinstance(store_backend, StoreBackend)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        # Private, but it is the only place the namespace deprecation surfaces —
        # and the namespace is durable data, so it is worth reaching for.
        namespace = store_backend._get_namespace()
    assert namespace == MEMORY_NAMESPACE == ("filesystem",)


def test_open_agent_passes_a_backend_instance_not_a_factory() -> None:
    # 0.7.0 also removes the callable-factory form of `backend=`. Asserting that
    # `build_backend()` returns a non-callable is vacuous — a CompositeBackend
    # never is. What actually matters is the object handed to `create_deep_agent`,
    # so capture that instead: a plain function passed here still type-checks and
    # still passes every other test, but is exactly the deprecated form.
    captured: dict[str, Any] = {}
    real_create_deep_agent = agent_module.create_deep_agent

    def spy(*args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return real_create_deep_agent(*args, **kwargs)

    with mock.patch.object(agent_module, "create_deep_agent", spy), open_agent():
        pass

    assert "backend" in captured
    assert not callable(captured["backend"]), (
        "backend= must be an instance, not a factory"
    )
    assert isinstance(captured["backend"], CompositeBackend)


def test_open_agent_assembles_offline() -> None:
    with open_agent() as agent:
        assert agent is not None
        # It's a compiled LangGraph — it must expose the invoke entry point the
        # CLI drives.
        assert hasattr(agent, "invoke")
