"""Platform entry point: the same deep research agent, wired for `langgraph dev`.

`cli.py` (via `agent.open_agent()`) runs the agent *in-process* and owns its
persistence — a local `SqliteSaver` checkpointer and `SqliteStore`. The LangGraph
API server (`langgraph dev`, `langgraph up`, LangGraph Platform, and the
`deep-agents-ui` web app) is the mirror-image topology: the *server* owns
persistence and injects a checkpointer and store at runtime.

This module is thin on purpose. It delegates to `agent.build_agent()` — the single
source of truth for the agent's wiring — passing no checkpointer/store, so the
served agent and the CLI agent cannot drift (add a tool or subagent in `agent.py`
and both entry points gain it). `build_agent` documents why omitting persistence is
required, and why it keeps `interrupt_on` legal. The `/memories/` route keeps
working because deepagents' `StoreBackend` resolves the store from the runtime via
`get_store()`, so it transparently reads the *server's* store rather than the CLI's
`SqliteStore` — same route, same `MEMORY_NAMESPACE`, different concrete backend.
(Consequence: notes the CLI wrote to `.deep_research/memories.sqlite` are not
visible to a served instance; it is a different physical store under the same key.)

`langgraph.json` points at the module-level `graph` below — a COMPILED graph, built
ONCE at import. It is deliberately *not* pointed at the `build_graph` factory:
`langgraph_api` re-invokes a graph factory on every run (`invoke_factory`, measured
in `langgraph_api/graph.py`), which would reconstruct the model, tools, and subagent
per request; a compiled graph is built once and reused. `build_graph()` is still
factored out so tests can assemble the served graph and inspect its wiring — and it
is imported inside those tests, not at their module top, so this import-time
construction cannot turn a build break into a whole-file collection error.
"""

from __future__ import annotations

from typing import Any

from .agent import build_agent
from .config import missing_keys


def build_graph() -> Any:
    """Assemble the served research agent, delegating to the shared `build_agent`.

    Passes no checkpointer/store: the LangGraph API server injects both at runtime.
    Fails loudly if credentials are missing, so a misconfigured server reports a
    clear error at graph load. The CLI runs `missing_keys()` at startup, but nothing
    guarded the served path — without this the agent would load "healthy" and only
    fail with an opaque 401 on the first user request.
    """
    if missing := missing_keys():
        raise RuntimeError(
            "Cannot serve the research agent — missing required environment "
            f"variable(s): {', '.join(missing)}. "
            "Copy .env.example to .env and fill them in."
        )
    return build_agent()


# The compiled `CompiledStateGraph` `langgraph.json` loads (`:graph`), built once at
# import. `build_model()` reads keys from the environment, which `langgraph.json`'s
# `env: ./.env` populates before this module is imported.
graph = build_graph()
