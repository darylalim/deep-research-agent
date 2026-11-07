"""Subagent definitions.

The orchestrator delegates focused sub-questions to these via the built-in
`task` tool. Subagents run with an *isolated* context — they don't see the main
conversation — which keeps the orchestrator's own context lean. They are also
stateless: each `task` call is one-shot, so the instructions must be complete.
"""

from __future__ import annotations

from deepagents import SubAgent

from .config import RESEARCHER_EFFORT, RESEARCHER_MODEL, build_model
from .tools import build_web_search

RESEARCHER_PROMPT = """You are a focused web researcher. You are given exactly \
ONE specific sub-question to investigate.

- Use `tavily_search` to gather evidence. Search more than once when the first
  results are thin, stale, or conflicting — refine the query each time.
- Shape the search; do not just retype the question. `tavily_search` takes more
  than a query:
    - `include_domains` — go straight to the primary source when you know it (the
      vendor's own docs or pricing page, the project's own repo). A primary source
      is both more current and more citable than an SEO aggregator summarizing it.
    - `time_range` (or `start_date`) — for anything version-, price-, limit- or
      release-sensitive, which is most of what you will be asked. Without it you are
      as likely to cite a two-year-old blog post as this month's changelog.
    - `topic="news"` for events; `"finance"` for markets.
    - `search_depth="advanced"` — only once the basic results come back thin. It
      costs twice as much, so escalate on evidence, not by default.
- Return a concise synthesis (a few tight paragraphs or bullet points), not a
  raw dump of search results.
- Attribute every substantive claim to a source by including the URL inline.
- Surface disagreements and gaps in the evidence instead of papering over them.
- You have your own isolated context and cannot see the wider conversation, so
  answer only the sub-question you were given — fully — in a single reply."""


def build_research_subagent() -> SubAgent:
    """A specialized web-research subagent.

    Give it one independent sub-question per `task` call; fan several out in the
    same turn to research breadth in parallel while keeping the orchestrator's
    context focused on synthesis.

    **It carries its own `model`**, built from `RESEARCHER_MODEL` / `RESEARCHER_EFFORT`,
    which default to the orchestrator's — so by default this is the configuration it
    used to inherit. Read off deepagents 0.7 (`graph.py`), not assumed: a spec's `model`
    and its `interrupt_on` are resolved independently, so giving the researcher a model
    does not cost it the inherited `GATED_TOOLS`, and prompt caching is appended to every
    declarative subagent unconditionally. **Never add an `interrupt_on` key here**, not
    even `{}`: a present key REPLACES the inherited gate, and a researcher's `write_file`
    would then go through unreviewed. `test_agent_wiring.py` pins both.
    """
    return {
        "name": "researcher",
        "description": (
            "Investigates ONE focused sub-question via web search and returns a "
            "concise, source-cited summary. Delegate independent sub-questions "
            "to it (in parallel when possible) to keep the main context clean."
        ),
        "system_prompt": RESEARCHER_PROMPT,
        "tools": [build_web_search()],
        "model": build_model(model=RESEARCHER_MODEL, effort=RESEARCHER_EFFORT),
    }
