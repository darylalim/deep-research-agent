"""Research tools.

Currently a single tool: Tavily web search. Add more tools here (e.g. a URL
fetcher, an arXiv search, a calculator) and pass them to the orchestrator or a
subagent in `agent.py` / `subagents.py`.
"""

from __future__ import annotations

from langchain_tavily import TavilySearch


def build_web_search(max_results: int = 5) -> TavilySearch:
    """Build the Tavily web-search tool.

    Reads `TAVILY_API_KEY` from the environment at call time. Construction does
    not hit the network, so it is safe to build eagerly.

    **Leave `search_depth`, `time_range`, `topic` and `include_domains` unset here.**
    They are deliberately absent, not forgotten. Each is in the tool's *args schema*,
    so the model chooses one per call — and `TavilySearch._run` resolves them as
    `self.X if self.X else X`, meaning a value set HERE silently overrides whatever the
    model asked for, on every search, forever. Setting `time_range="week"` to "get fresh
    results" would also stop the agent from ever researching anything older than a week.
    The prompts (`agent.py`, `subagents.py`) teach the agent to set them per query
    instead; that only works while these stay `None`.

    `max_results` is different — it is instantiation-only (`_run` rejects it as an
    invocation kwarg), so it belongs here.
    """
    return TavilySearch(max_results=max_results)
