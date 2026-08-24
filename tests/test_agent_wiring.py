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
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest import mock

from deepagents import graph as graph_module
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents.middleware.filesystem import supports_execution

from deep_research import agent as agent_module
from deep_research.agent import (
    GATED_TOOLS,
    MEMORY_NAMESPACE,
    SYSTEM_PROMPT,
    build_backend,
    open_agent,
)


@contextmanager
def _capture_calls(module: Any, attr: str) -> Iterator[list[dict[str, Any]]]:
    """Replace `module.attr` with a pass-through spy; yield each call's kwargs.

    The assembly tests all need the same thing — swap a constructor
    (`create_deep_agent`, `build_agent`) for a spy that records how it was called and
    still delegates to the real one, so the wiring is exercised for real — so the
    scaffolding lives here once instead of being re-inlined per test.
    """
    calls: list[dict[str, Any]] = []
    real = getattr(module, attr)

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return real(*args, **kwargs)

    with mock.patch.object(module, attr, spy):
        yield calls


def test_mutating_and_shell_tools_are_gated() -> None:
    # Safety property: writing files and running shell commands must require
    # human approval. LangChain only gates a tool when its value is `True` or an
    # InterruptOnConfig with a truthy `allowed_decisions` (human_in_the_loop.py:
    # 252-260); a value of `False` — or a config missing `allowed_decisions` —
    # silently un-gates while the key stays present. Assert the value, not just
    # the key, so a value flip can't defeat approval unnoticed.
    #
    # Note what this does NOT prove: that the gate can ever *fire*. An entry here is
    # inert unless the tool actually reaches the model, which for `execute` it does
    # not — see the next test. This one is the real gate for `write_file`/`edit_file`
    # and a statement of intent for `execute`.
    for tool_name in ("write_file", "edit_file", "delete", "execute"):
        assert tool_name in GATED_TOOLS, f"{tool_name} is not gated"
        config = GATED_TOOLS[tool_name]
        gated = config is True or (
            isinstance(config, dict) and config.get("allowed_decisions")
        )
        assert gated, f"{tool_name} is present but its value does not enable gating"


# Every tool the model may be offered that CANNOT change the world. Written down rather
# than defaulted so that a tool arriving from a dependency upgrade has to be classified by
# a person: unknown means "fail", never "assume harmless". Same shape, and the same
# reasoning, as `_SILENT_STOPS` vs `NOT_SILENT` in `test_cli_parsing.py`.
READ_ONLY_TOOLS = frozenset(
    {
        "ls",
        "read_file",
        "glob",
        "grep",
        "write_todos",  # rewrites a state channel, not the filesystem
        "task",  # delegates; the subagent's OWN mutations interrupt on their own
        "tavily_search",
    }
)


def test_every_mutating_tool_the_model_is_offered_is_gated() -> None:
    """Derive the safety check from the BUILT AGENT, never from `GATED_TOOLS` itself.

    This is the test whose absence let the deepagents 0.6.12 -> 0.7.8 upgrade through.
    0.7.x added a `delete` tool to `FilesystemMiddleware`, `GATED_TOOLS` did not list it,
    and all 202 tests stayed green — because every one of them asked what the dict *says*
    rather than what the model is *handed*. The agent could have deleted a note under
    `/memories/`, the one place its writes are durable, with no approval prompt.

    `test_mutating_and_shell_tools_are_gated` is the complement and both are needed: it
    asserts the three names we care about are gated *properly* (a `False` value silently
    un-gates while the key stays), and this one asserts there is no FOURTH name nobody
    noticed. Neither implies the other.
    """
    with open_agent() as agent:
        offered = set(agent.nodes["tools"].bound.tools_by_name)

    unclassified = offered - READ_ONLY_TOOLS - set(GATED_TOOLS)
    assert not unclassified, (
        f"the agent is offered {sorted(unclassified)}, which is neither gated nor listed "
        "as read-only. If it can change the world (write, edit, delete, execute), add it "
        "to GATED_TOOLS; if it cannot, add it to READ_ONLY_TOOLS. Do not leave it "
        "unclassified — that is how `delete` arrived unguarded in deepagents 0.7.x."
    )
    # The other direction, so the allowlist cannot quietly rot into a list of names that
    # no longer exist while still granting blanket permission to whatever replaced them.
    assert offered >= READ_ONLY_TOOLS, (
        f"READ_ONLY_TOOLS names tools the agent no longer has: "
        f"{sorted(READ_ONLY_TOOLS - offered)}"
    )


def test_write_todos_is_offered_because_the_prompt_mandates_it() -> None:
    """SYSTEM_PROMPT step 1 orders a tool that deepagents stopped providing.

    Through 0.6.x `TodoListMiddleware` was in deepagents' base stack. 0.7.0 removed it, so
    `write_todos` vanished from the agent while step 1 went on demanding it — an
    instruction to call a tool the model is never offered. `build_agent` now passes the
    middleware explicitly, which is only correct on 0.7+: on 0.6.x it would have
    registered a SECOND `write_todos`, since nothing dedupes them.

    Nothing else catches this. The prompt is a string, the tool list comes from a
    dependency, and no test crossed the two — the failure would have surfaced as the
    `plans_with_todos` eval quietly scoring zero and the feed's plan line going blank.
    """
    with open_agent() as agent:
        assert "write_todos" in agent.nodes["tools"].bound.tools_by_name
    assert "write_todos" in SYSTEM_PROMPT


def test_execute_is_latent_because_the_backend_cannot_run_it() -> None:
    # Why `GATED_TOOLS["execute"]` is a no-op today, pinned so nobody has to
    # rediscover it: `FilesystemMiddleware.wrap_model_call` filters `execute` out of
    # `request.tools` on EVERY model call unless the backend supports execution, and
    # for a `CompositeBackend` that is decided by its `.default` — ours is a
    # `StateBackend`. The model is never offered the tool, so the interrupt cannot
    # fire, so SYSTEM_PROMPT must not promise a pause for it (it no longer does).
    #
    # This goes red the day someone gives the backend a sandbox. That is exactly when
    # it should: at that moment `execute` becomes real, the latent gate starts firing,
    # and the prompt needs its sentence back. Read it as a tripwire, not a wish.
    assert not supports_execution(build_backend())


def test_nothing_is_appended_to_our_system_prompt() -> None:
    """The model receives SYSTEM_PROMPT and nothing else — asserted on the real string.

    Through 0.6.x, `create_deep_agent()` APPENDED `BASE_AGENT_PROMPT` after our prompt,
    where it won on recency: it opened with "the user can see your responses and tool
    outputs in real time" and closed with a "## Progress Updates" section asking for
    "brief progress updates at reasonable intervals". Both claims were false here — the
    user watches a live feed of TOOL activity, while the agent's prose does not stream at
    all — and the measured cost was three paragraphs of stale narration printed above the
    answer. `SYSTEM_PROMPT` carried a section naming and overriding it, and the version of
    this test that guarded it asserted BOTH halves precisely so it would say which one had
    moved.

    deepagents 0.7.0 is what moved: it no longer authors a base prompt at all
    (`BASE_AGENT_PROMPT` survives only as a deprecated module attribute, removal slated
    for 0.9.0), so the override became the dead weight its own failure message predicted
    and was deleted with this rewrite. What replaces it is stricter than either half was:
    the exact string handed to `create_agent` must BE `SYSTEM_PROMPT`.

    Read it in both directions. It goes red if deepagents (or a harness profile — an
    unregistered model gets an empty one, and `claude-opus-5` is unregistered today)
    starts contributing prompt text again, which is when a narration override would be
    needed back. And it goes red if anything in this repo starts appending to the prompt
    at assembly time rather than editing the constant, which is where a prompt change
    would otherwise become invisible to every other test in this file.
    """
    with _capture_calls(graph_module, "create_agent") as calls, open_agent():
        pass

    assert calls, "create_agent was never called"
    assembled = calls[0]["system_prompt"]
    assert assembled == SYSTEM_PROMPT, (
        "something now contributes system-prompt text beyond SYSTEM_PROMPT. If it is a "
        "revived base/profile prompt, check whether it tells the model to narrate "
        "progress — that guidance is false for this app and needs overriding again.\n"
        f"extra: {assembled.replace(SYSTEM_PROMPT, '<SYSTEM_PROMPT>')!r}"
    )


def test_the_agent_is_still_told_not_to_narrate() -> None:
    """The instruction outlived the prompt it used to argue with, and must.

    Deleting the override above is not the same as deciding narration is fine. Prose
    still arrives in one block from `render_turn` when the turn ends, so "let me first
    check memory" is read next to the feed line that already showed the result. Nothing
    injects the contrary guidance today, but nothing rules the behavior out either.
    """
    assert "Do NOT narrate" in SYSTEM_PROMPT


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
    with _capture_calls(agent_module, "create_deep_agent") as calls, open_agent():
        pass

    assert len(calls) == 1
    assert "backend" in calls[0]
    assert not callable(calls[0]["backend"]), (
        "backend= must be an instance, not a factory"
    )
    assert isinstance(calls[0]["backend"], CompositeBackend)


def test_open_agent_assembles_offline() -> None:
    with open_agent() as agent:
        assert agent is not None
        # It's a compiled LangGraph — it must expose the invoke entry point the
        # CLI drives.
        assert hasattr(agent, "invoke")


def test_served_graph_assembles_offline() -> None:
    # The `langgraph dev` / Platform front door (`deep_research/graph.py`), served to
    # deep-agents-ui / LangGraph Studio. `graph.py` builds its module-level `graph`
    # (the compiled object `langgraph.json` loads) at import, so assert on THAT rather
    # than rebuilding it. It must be a compiled LangGraph the SDK can drive, and must
    # compile *despite* `interrupt_on` being set with no checkpointer (enforced at
    # invoke time, which the server satisfies).
    #
    # Imported inside the test, not at module top: `graph.py` constructs a full agent
    # at import, so keeping that import here means a build break fails only this test
    # rather than erroring the whole file at collection.
    from deep_research.graph import graph

    assert graph is not None
    assert hasattr(graph, "invoke")


def test_shared_builder_gates_and_routes_without_persistence() -> None:
    # `build_agent` is what BOTH front doors call, so pin its invariants once, in the
    # served configuration (no checkpointer/store): do NOT pass a checkpointer or
    # store (the server owns persistence and injects its own), yet keep the gate, the
    # /memories/ routing, and the prompt. Spy on the `create_deep_agent` that
    # `build_agent` invokes, exactly as
    # `test_open_agent_passes_a_backend_instance_not_a_factory` does.
    with _capture_calls(agent_module, "create_deep_agent") as calls:
        agent_module.build_agent()  # served-style: no persistence passed
    captured = calls[0]

    # Persistence is the server's job; neither may be passed.
    assert captured.get("checkpointer") is None, "must not pass a checkpointer"
    assert captured.get("store") is None, "must not pass a store"
    # The gate, the /memories/ routing, and the prompt are the SAME objects the CLI
    # uses (identity checks — a copy would defeat the point).
    assert captured.get("interrupt_on") is GATED_TOOLS
    assert captured.get("system_prompt") is SYSTEM_PROMPT
    assert isinstance(captured.get("backend"), CompositeBackend)
    # ...and so are the tools and subagents — the fields that actually diverge if a
    # second assembly is ever introduced, which the old served-graph test never
    # checked. Assert they are present, not silently empty.
    assert captured.get("tools"), "served agent lost its web-search tool"
    assert captured.get("subagents"), "served agent lost its researcher subagent"


def test_served_graph_delegates_to_the_shared_builder() -> None:
    # The anti-drift guard, made structural: `graph.py` must ROUTE THROUGH
    # `agent.build_agent` — the single source of truth — not re-inline
    # `create_deep_agent`, so tools/subagents/prompt/gate can never differ from the
    # CLI. Assert the delegation, and that the served call passes no persistence.
    #
    # `graph` imported inside the test for the same reason as
    # `test_served_graph_assembles_offline` — isolate import-time construction.
    from deep_research import graph as graph_module

    with _capture_calls(graph_module, "build_agent") as calls:
        graph_module.build_graph()

    assert len(calls) == 1, "build_graph must call the shared builder exactly once"
    assert calls[0].get("checkpointer") is None
    assert calls[0].get("store") is None
