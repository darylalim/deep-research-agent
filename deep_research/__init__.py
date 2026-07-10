"""Deep Research Agent.

A persistent, subagent-orchestrating research agent built on Deep Agents
(LangChain 1.0 + LangGraph). It plans with a todo list, delegates focused
web-search sub-questions to a `researcher` subagent, keeps durable findings in
cross-session memory, and gates file writes behind human approval.

Entry point: `python -m deep_research`  (see cli.main).
"""

__version__ = "0.1.0"
