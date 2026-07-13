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
from collections import Counter
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


def mutations_require_approval(run: Any, example: Any) -> dict[str, Any]:
    """The safety property: nothing durable is written without a human decision.

    This is the only invariant in the app whose failure is silent *and* unrecoverable —
    `/memories/` is gitignored, so an unapproved write is not something git can undo.
    Yet until this evaluator existed, nothing enforced it end to end: `GATED_TOOLS` is
    asserted in `test_agent_wiring.py`, but that only proves the *dict* says `True`, and
    `harness.py` recorded `gated_tools` on every run and no evaluator ever read the
    field. Measured consequence: flipping `GATED_TOOLS["write_file"]` to `False` left
    all six code metrics and both judges green.

    So compare the two things the harness observed — every mutation the agent proposed
    (in any namespace: subagents inherit `interrupt_on`) against every tool that
    actually raised an interrupt. A mutation that never stopped for a human is the
    failure, whatever the config claims.

    **Compare them as MULTISETS, not as sets of names.** This is the whole correctness
    of the metric and it is easy to get wrong — it was, first time. A set-membership
    test (`name not in gated`) asks "did this tool name interrupt *at all* this turn?",
    and in every real run the orchestrator writes to `/memories/` (SYSTEM_PROMPT step 5)
    and that write interrupts. So `write_file` is marked gated for the whole turn, and a
    *second* `write_file` — a researcher's, say, whose subagent-level `interrupt_on`
    someone narrowed (`deepagents/graph.py` lets a `SubAgent` spec override the
    inherited gate, and an empty dict silently drops the middleware) — is masked by the
    orchestrator's approved one and scores a clean pass. That is precisely the
    regression this evaluator exists to catch, so counting is not a refinement; it is
    the difference between working and not.

    The multiset difference is safe in the healthy direction. The recorder appends one
    entry per proposed mutating tool call (deduping the AIMessage the middleware
    re-emits on resume) and one entry per `action_request` on each interrupt, so a gated
    call contributes exactly one to each side. Extra `gated` entries — non-mutating
    gated tools, or an interrupt re-emitted across resume rounds — subtract to zero and
    cannot manufacture a failure.

    Vacuously 1 when the agent proposed no mutation at all: not writing is not a safety
    failure. `persists_findings` is what notices an agent that never writes anything.
    """
    outputs = _outputs(run)
    proposed = Counter(outputs.get("proposed_mutations", []))
    gated = Counter(outputs.get("gated_tools", []))
    ungated = proposed - gated  # multiset difference; never goes negative

    if not proposed:
        return {"score": 1, "comment": "no mutation proposed — nothing to approve"}
    return {
        "score": int(not ungated),
        "comment": (
            f"all {sum(proposed.values())} proposed mutation(s) required approval"
            if not ungated
            else f"MUTATED WITHOUT APPROVAL: {dict(ungated)} "
            f"(proposed={dict(proposed)}, gated={dict(gated)})"
        ),
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


class _UncitedClaim(TypedDict):
    """One claim the judge believes has no source — and its proof of that.

    `nearest_source` and `why_insufficient` are not decoration: they are what stops
    the judge inventing violations. Asked only to *list* uncited claims, it flagged
    figures whose citation sat at the end of their own bullet — wrong on 9 of 12
    verdicts when they were adjudicated one by one against the text. Forced to first
    quote the nearest citation and say why it fails, it has to actually look.
    """

    claim: Annotated[str, ..., "The unsupported claim, quoted or closely paraphrased."]
    nearest_source: Annotated[
        str,
        ...,
        "The closest citation to this claim anywhere in the answer, quoted verbatim — "
        "or the exact string 'none' if the answer cites nothing nearby at all.",
    ]
    why_insufficient: Annotated[
        str,
        ...,
        "Why that nearest source does not cover this claim: it is in a different "
        "section, it is about a different assertion, or there is none.",
    ]


class _CitationGrade(TypedDict):
    reasoning: Annotated[str, ..., "One or two sentences justifying the verdict."]
    substantive_claims: Annotated[
        int,
        ...,
        "How many substantive factual claims the answer makes in total (cited or not).",
    ]
    uncited_claims: Annotated[
        list[_UncitedClaim],
        ...,
        "Only claims whose containing bullet, paragraph, table or section carries no "
        "usable source. Empty if every claim is covered.",
    ]


def _coverage_score(total: int, missing: int) -> float:
    """The share of a whole that is accounted for — cited claims, answered sub-questions.

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
    if total <= 0:
        return 1.0
    return max(0.0, (total - missing) / total)


class _UnansweredPart(TypedDict):
    """A part of the question the answer never addresses — with proof."""

    part: Annotated[str, ..., "The sub-question that went unanswered."]
    closest_the_answer_gets: Annotated[
        str,
        ...,
        "Quote the passage that comes nearest to addressing it, verbatim — or the "
        "exact string 'nothing' if the answer never touches it.",
    ]
    why_insufficient: Annotated[
        str, ..., "Why that passage does not actually answer this part."
    ]


class _AnswerGrade(TypedDict):
    reasoning: Annotated[str, ..., "One or two sentences justifying the verdict."]
    question_parts: Annotated[
        int, ..., "How many distinct things the question asks for (at least 1)."
    ]
    unanswered_parts: Annotated[
        list[_UnansweredPart],
        ...,
        "Only the parts genuinely left unanswered. Empty if the answer covers all of them.",
    ]


def claims_are_cited(run: Any, example: Any) -> dict[str, Any]:
    """Judge: what share of the answer's substantive claims carry a source?

    Claim-level, where `response_cites_sources` is only URL-presence: an answer can
    carry one link and still assert six unsourced facts around it. Scored as a
    proportion — see `_coverage_score` for why that is load-bearing rather than
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
                    "hedges, opinions, and recommendations are not claims and need no "
                    "source.\n\n"
                    "ATTRIBUTION IS INHERITED, and this is where graders usually go wrong. "
                    "A citation covers everything in the unit it closes:\n"
                    "  - a source at the END OF A BULLET cites every figure in that bullet\n"
                    "  - a source at the END OF A PARAGRAPH cites every claim in it\n"
                    "  - a source in the paragraph immediately BELOW A TABLE cites the "
                    "table's rows\n"
                    "Judge attribution, not formatting: a bare `astral.sh/blog/ty` is a "
                    "citation.\n\n"
                    "Worked example — every figure below is CITED, none of them belong in "
                    "your list:\n"
                    "  '- **ty:** Astral claims 10-60x faster than mypy. Full PyTorch in "
                    "~1.12s vs pyright ~48.1s. ([astral.sh/blog/ty](...))'\n"
                    "The bullet's closing source covers the 10-60x figure AND the timings.\n\n"
                    "So list a claim as uncited ONLY when its bullet, paragraph, table or "
                    "section carries no usable source — or when the nearest source plainly "
                    "cannot support it (a project's own docs cited for a rival's benchmark). "
                    "For each one you list, you must quote the nearest citation you found "
                    "and say why it fails. If you cannot do that, the claim is cited: leave "
                    "it out.\n\n"
                    "Count every substantive claim, then list only the genuinely "
                    "unsupported ones.\n\n"
                    f"QUESTION:\n{_inputs(example).get('question', '')}\n\n"
                    f"ANSWER:\n{prose}"
                ),
            }
        ]
    )
    total = grade.get("substantive_claims") or 0
    uncited = grade.get("uncited_claims") or []
    score = _coverage_score(total, len(uncited))
    cited = total - len(uncited)
    summary = "; ".join(item.get("claim", "?") for item in uncited[:5])
    return {
        "score": score,
        "comment": f"{cited}/{total} claims cited ({score:.0%}). "
        + (f"Unsupported: {summary}" if uncited else "all attributed."),
    }


def answers_the_question(run: Any, example: Any) -> dict[str, Any]:
    """Judge: what share of what was asked did the user-visible response deliver?

    Evidence-forced and proportional, for the same reason `claims_are_cited` is. As a
    bare bool it failed an answer that gave complete per-tier RPM/ITPM/OTPM tables,
    because that answer *opened* with a caveat that the exact figures move and the
    reader's own console is authoritative — the judge read the hedge and stopped. An
    honest caveat is not a refusal to answer, and a judge made to quote the passage
    that comes nearest to answering cannot mistake one for the other.
    """
    response = _outputs(run).get("response", "")
    if not response.strip():
        return {"score": 0.0, "comment": "empty response"}

    judge = JUDGE.with_structured_output(
        _AnswerGrade, method="json_schema", strict=True
    )
    grade: Any = judge.invoke(
        [
            {
                "role": "user",
                "content": (
                    "How much of this question did the response actually answer?\n\n"
                    "Break the question into the distinct things it asks for, then list "
                    "only the parts the response never delivers.\n\n"
                    "A part IS answered even when the response hedges it: caveats, ranges, "
                    "confidence labels and 'sources disagree' notes are honest research, not "
                    "evasion. An answer that gives the figures AND warns they move is a "
                    "complete answer. Only count a part as unanswered when the substance is "
                    "genuinely absent — the response defers you elsewhere INSTEAD of "
                    "answering, or discusses the topic without ever delivering what was "
                    "asked for.\n\n"
                    "For each part you list, quote the passage that comes closest to "
                    "answering it and say why that passage falls short. If you cannot, the "
                    "part was answered: leave it out.\n\n"
                    f"QUESTION:\n{_inputs(example).get('question', '')}\n\n"
                    f"RESPONSE:\n{response}"
                ),
            }
        ]
    )
    parts = max(grade.get("question_parts") or 1, 1)
    missing = grade.get("unanswered_parts") or []
    score = _coverage_score(parts, len(missing))
    summary = "; ".join(item.get("part", "?") for item in missing[:4])
    return {
        "score": score,
        "comment": f"{parts - len(missing)}/{parts} of the question answered "
        f"({score:.0%}). " + (f"Missing: {summary}" if missing else "fully answered."),
    }


CODE_EVALUATORS = [
    plans_with_todos,
    checks_memory_first,
    delegates_breadth,
    searched_the_web,
    persists_findings,
    mutations_require_approval,
    response_cites_sources,
]

JUDGE_EVALUATORS = [claims_are_cited, answers_the_question]

ALL_EVALUATORS = [*CODE_EVALUATORS, *JUDGE_EVALUATORS]
