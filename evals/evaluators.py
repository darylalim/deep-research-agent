"""Evaluators: one metric each, as LangSmith requires.

Split deliberately into two kinds:

- **Code evaluators** grade the *trajectory* — the workflow `SYSTEM_PROMPT`
  promises (plan, check memory, delegate, search, persist). These are objective,
  free, and they are the only tests this repo has ever had of the prompt's
  contract. They earned their keep immediately: `write_todos` was never being
  called at all.
- **LLM judges** grade the *prose* — citation discipline and responsiveness,
  which no regex can settle.

**Grade the orchestrator's contract against `orchestrator_trajectory`, never
`trajectory`.** Those `SYSTEM_PROMPT` steps are addressed to the orchestrator, and
deepagents hands every subagent its own `write_todos`/`ls`/`write_file` — so a flat
trajectory lets a researcher's own bookkeeping score as if the orchestrator had done
it. Only `searched_the_web` deliberately counts the whole tree.

Every evaluator takes `(run, example)` and returns a single
`{"score": ..., "comment": ...}`. Returning several metrics from one function is
an error in LangSmith, so each check gets its own function.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, TypedDict

from langchain_anthropic import ChatAnthropic

# Actions that constitute "starting the research", i.e. the point by which the
# agent was supposed to have planned and consulted memory.
RESEARCH_ACTIONS = ("task", "tavily_search")

URL = re.compile(r"https?://[^\s)\]>,]+")


# The judge is Haiku, not the agent's own Opus 4.8, for two independent reasons:
# grading is a cheap, high-volume classification task that does not need Opus, and
# Opus 4.8 rejects `temperature` with a 400 — a judge wants `temperature=0`, so the
# app's own `build_model()` is the wrong constructor to reuse here.
# The `ty: ignore` is the same false positive `config.py::build_model` carries: ty
# builds the signature from the Pydantic aliases and does not model `populate_by_name`.
JUDGE = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0, max_tokens=1024)  # ty: ignore[unknown-argument, missing-argument]


def _outputs(record: Any) -> dict[str, Any]:
    """Read `.outputs` off a run/example.

    Local `evaluate()` hands these in as objects; an evaluator uploaded to
    LangSmith receives plain dicts. Support both — the same function has to work
    in either place.
    """
    if hasattr(record, "outputs"):
        return record.outputs or {}
    if isinstance(record, dict):
        return record.get("outputs") or {}
    return {}


def _inputs(record: Any) -> dict[str, Any]:
    if hasattr(record, "inputs"):
        return record.inputs or {}
    if isinstance(record, dict):
        return record.get("inputs") or {}
    return {}


def _first_research_index(trajectory: list[str]) -> int:
    """Where the agent stopped preparing and started researching."""
    for index, tool in enumerate(trajectory):
        if tool in RESEARCH_ACTIONS:
            return index
    return len(trajectory)


# --- Trajectory (code) evaluators ------------------------------------------


def plans_with_todos(run: Any, example: Any) -> dict[str, Any]:
    """SYSTEM_PROMPT step 1: call `write_todos` before starting the work.

    Exempt for a single quick lookup — the prompt says so, so the bar comes from
    the example (`expects_plan`) rather than being applied blindly. An agent that
    opens a todo list to answer "what version is Python" is over-planning, and
    scoring that as a pass would teach us the wrong thing.
    """
    trajectory = _outputs(run).get("orchestrator_trajectory", [])
    if not _outputs(example).get("expects_plan", True):
        return {"score": 1, "comment": "single lookup — no plan required"}

    start = _first_research_index(trajectory)
    planned = "write_todos" in trajectory[:start]
    return {
        "score": int(planned),
        "comment": (
            "planned before researching"
            if planned
            else f"no write_todos before the first research action; trajectory={trajectory}"
        ),
    }


def checks_memory_first(run: Any, example: Any) -> dict[str, Any]:
    """SYSTEM_PROMPT step 2: look in `/memories/` before researching from scratch."""
    trajectory = _outputs(run).get("orchestrator_trajectory", [])
    start = _first_research_index(trajectory)
    looked = any(tool in ("ls", "read_file") for tool in trajectory[:start])
    return {
        "score": int(looked),
        "comment": (
            "consulted memory first"
            if looked
            else f"started researching without reading /memories/; trajectory={trajectory}"
        ),
    }


def delegates_breadth(run: Any, example: Any) -> dict[str, Any]:
    """SYSTEM_PROMPT step 3: fan independent sub-questions out to `researcher`.

    The bar comes from the example (`min_delegations`), because a single quick
    lookup is *supposed* to skip delegation — the prompt says so explicitly.
    """
    trajectory = _outputs(run).get("orchestrator_trajectory", [])
    required = _outputs(example).get("min_delegations", 1)
    delegated = trajectory.count("task")
    return {
        "score": int(delegated >= required),
        "comment": f"{delegated} `task` dispatch(es), expected >= {required}",
    }


def searched_the_web(run: Any, example: Any) -> dict[str, Any]:
    """Did it actually research, or answer from the model's own memory?

    Counts searches anywhere in the tree, including inside subagents — where, in
    practice, all of them happen.
    """
    outputs = _outputs(run)
    searches = outputs.get("trajectory", []).count("tavily_search")
    in_subagents = outputs.get("subagent_tools", []).count("tavily_search")
    return {
        "score": int(searches > 0),
        "comment": f"{searches} web search(es), {in_subagents} of them inside a subagent",
    }


def persists_findings(run: Any, example: Any) -> dict[str, Any]:
    """SYSTEM_PROMPT step 5: write durable findings under `/memories/`.

    Per-example, like the other two workflow bars. The prompt says to persist
    *durable* findings and explicitly *not* to save ephemeral ones — so demanding a
    memory write for "what version is Python" (true until the next point release)
    would be scoring the agent down for obeying its instructions.

    Checked against the paths the agent asked to write, not the returned state:
    `/memories/` is routed to the Store, so an approved memory write leaves
    `state["files"]` empty — measured.
    """
    if not _outputs(example).get("expects_persist", True):
        return {"score": 1, "comment": "ephemeral finding — persisting not required"}

    writes = _outputs(run).get("proposed_writes", [])
    memories = [path for path in writes if path.startswith("/memories/")]
    return {
        "score": int(bool(memories)),
        "comment": f"wrote {memories}"
        if memories
        else f"no /memories/ write; wrote {writes or 'nothing'}",
    }


def response_cites_sources(run: Any, example: Any) -> dict[str, Any]:
    """Does the answer the *user actually sees* contain source URLs?

    `response` is rendered by `cli.render_turn`, so this grades the exact text the
    REPL prints — not the agent's internal reasoning, and not what its subagents
    wrote down. A report saved to a file with the URLs in it does not count: the
    user reading the terminal never opens that file.
    """
    urls = URL.findall(_outputs(run).get("response", ""))
    return {
        "score": int(bool(urls)),
        "comment": (
            f"{len(urls)} source URL(s) in what the user is shown"
            if urls
            else "the user is shown no source URLs"
        ),
    }


# --- LLM judges -------------------------------------------------------------


class _CitationGrade(TypedDict):
    reasoning: Annotated[str, ..., "One or two sentences justifying the verdict."]
    substantive_claims: Annotated[
        int,
        ...,
        "How many substantive factual claims the answer makes in total (cited or not).",
    ]
    uncited_claims: Annotated[
        list[str],
        ...,
        "Those claims, if any, with no source attributed to them.",
    ]


def _citation_score(substantive_claims: int, uncited: int) -> float:
    """The share of substantive claims that carry a source.

    A *proportion*, deliberately, where this used to be an all-or-nothing bool — and
    the difference is not a matter of taste. "Every claim is cited" is a conjunction
    over every claim in the report: at 95% per-claim compliance, a 30-claim report
    passes 0.95**30 ≈ 21% of the time, and at 90% it passes 4%. So the boolean was
    destined to read 0 on any answer long enough to be worth writing — it scored a
    report missing one citation exactly the same as one citing nothing at all, which
    left it with no gradient and no way to show that a fix had helped. Measured: it
    returned 0 on 4 of 5 sweep examples while `response_cites_sources` passed 5 of 5.

    Vacuously perfect when there is nothing to attribute; an answer that asserts
    nothing is `answers_the_question`'s problem, not this one's.
    """
    if substantive_claims <= 0:
        return 1.0
    return max(0.0, (substantive_claims - uncited) / substantive_claims)


class _AnswerGrade(TypedDict):
    reasoning: Annotated[str, ..., "One or two sentences justifying the verdict."]
    answers: Annotated[
        bool,
        ...,
        "True if the response directly and completely answers the question asked.",
    ]


def claims_are_cited(run: Any, example: Any) -> dict[str, Any]:
    """Judge: what share of the answer's substantive claims carry a source?

    Claim-level, where `response_cites_sources` is only URL-presence: an answer can
    carry one link and still assert six unsourced facts around it. Scored as a
    proportion — see `_citation_score` for why that is load-bearing rather than
    cosmetic.
    """
    prose = _outputs(run).get("response", "")
    if not prose.strip():
        return {"score": 0.0, "comment": "the agent produced no prose"}

    judge = JUDGE.with_structured_output(
        _CitationGrade, method="json_schema", strict=True
    )
    grade: Any = judge.invoke(
        [
            {
                "role": "user",
                "content": (
                    "Grade how well a research agent attributed its claims to sources.\n\n"
                    "A SUBSTANTIVE CLAIM is a specific, checkable assertion — a number, "
                    "date, version, limit, price, benchmark, or capability. Framing, "
                    "hedges, opinions, and summary sentences are not claims and need no "
                    "source.\n\n"
                    "A claim COUNTS AS CITED if a source is identifiable for it: a link, a "
                    "bare domain, or a named publication, either alongside the claim or on "
                    "the section/table/row it belongs to. Judge attribution, NOT "
                    "formatting — `docs.tavily.com/pricing` is a citation even though it is "
                    "not a clickable link, and a table whose row cites its source covers "
                    "the figures in that row.\n\n"
                    "Count every substantive claim, then list only those with no "
                    "identifiable source.\n\n"
                    f"QUESTION:\n{_inputs(example).get('question', '')}\n\n"
                    f"ANSWER:\n{prose}"
                ),
            }
        ]
    )
    total = grade.get("substantive_claims") or 0
    uncited = grade.get("uncited_claims") or []
    score = _citation_score(total, len(uncited))
    cited = total - len(uncited)
    return {
        "score": score,
        "comment": f"{cited}/{total} claims cited ({score:.0%}). "
        + (f"Uncited: {uncited[:8]}" if uncited else "all attributed."),
    }


def answers_the_question(run: Any, example: Any) -> dict[str, Any]:
    """Judge: does the user-visible response actually answer what was asked?"""
    response = _outputs(run).get("response", "")
    if not response.strip():
        return {"score": 0, "comment": "empty response"}

    judge = JUDGE.with_structured_output(
        _AnswerGrade, method="json_schema", strict=True
    )
    grade: Any = judge.invoke(
        [
            {
                "role": "user",
                "content": (
                    "Does this response directly and completely answer the question? A "
                    "response that defers to a file, or summarizes without answering, does "
                    "not.\n\n"
                    f"QUESTION:\n{_inputs(example).get('question', '')}\n\n"
                    f"RESPONSE:\n{response}"
                ),
            }
        ]
    )
    return {
        "score": int(bool(grade.get("answers"))),
        "comment": grade.get("reasoning", ""),
    }


CODE_EVALUATORS = [
    plans_with_todos,
    checks_memory_first,
    delegates_breadth,
    searched_the_web,
    persists_findings,
    response_cites_sources,
]

JUDGE_EVALUATORS = [claims_are_cited, answers_the_question]

ALL_EVALUATORS = [*CODE_EVALUATORS, *JUDGE_EVALUATORS]
