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
    """
    return TavilySearch(max_results=max_results)
