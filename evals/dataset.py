"""The evaluation dataset.

**Deliberately reference-free.** Each example carries the question and a
*structural* expectation (`min_delegations`) — not a gold answer. Writing gold
answers by hand would mean inventing facts about live services, and grading a
research agent against my own unverified claims is worse than not grading it.
The judges are therefore reference-free too: they grade citation discipline and
responsiveness, which are properties of the answer, not of a key.

The right way to add references later is the one the `langsmith-dataset` skill
describes: run the agent, read the trace, curate the answers you have actually
verified, and upload those as `outputs`. Traces first, gold second.

The four structural columns — `min_delegations`, `max_delegations`, `expects_plan`,
`expects_persist` — are assertions about *judgment*, not just tool use, and they exist because
`SYSTEM_PROMPT` grants exemptions rather than issuing blanket rules: delegate
breadth **but** handle a single quick lookup yourself; plan first **unless** one
search settles it; persist durable findings **but not** ephemeral ones. Applying
any of those bars unconditionally scores the agent down for following its own
instructions — which is why the control example sets all three to their low bar.

Two properties of those columns govern every judgment call below, and both are
asymmetric. `min_delegations` is a FLOOR, so setting it low costs signal while setting
it high scores a *correct* agent down — genuine uncertainty about a bar therefore
resolves downward. The one deliberate exception is the packaging example's bar of 3,
which is an assertion rather than an estimate and carries its own fallback in place;
nothing else here sits above what its trajectory plainly requires. And
`expects_plan=False` / `expects_persist=False` make their evaluators
return 1 unconditionally: a False bar cannot be wrong, only uninformative. Vacuous is
safe; that is the whole reason the direct-path examples exempt planning rather than
asserting it.

One tension a reader will otherwise read as an oversight: example 4 (Anthropic's rate
limits) is also one vendor's limits table on one page and sets `min_delegations` to 2,
while the SQLite and OpenTelemetry examples below assume that same shape does *not* fan
out. Both are anchored on a single named document deliberately, to make the direct path
the natural one. Their columns cannot be wrong — they are unfalsifiable — so if
`orchestrator_trajectory` shows a `task` on either, the WORDING needs another pass, not
the bars.

Count the vacuity in BOTH directions, because it is easy to tally only the boolean
columns and conclude the delegation bar still has teeth everywhere: `min_delegations=0`
makes `delegates_breadth` return 1 unconditionally too, on exactly the same examples.
So after the direct-path additions `plans_with_todos` AND `delegates_breadth` are each
vacuous on the same 3 of 9, where each was 1 of 5. `plans_with_todos` is the one that
costs something — it carries the longest measurement history of any metric here, so
future movement in it is harder to read. Accepted rather than papered over with a bar
SYSTEM_PROMPT explicitly exempts.

`max_delegations` is what stops that vacuity being total, and it is the reason the
column exists. A floor alone cannot see over-orchestration — at `min_delegations=0`,
0, 1 and 5 dispatches all score 1 — so the control's comment could call fanning out
"over-orchestrating" while nothing checked it, and a direct-path example that fanned
out silently became a *delegated*-path example, grading the path that already measures
well while the gap it was added for stayed open, at a clean 1. The three
`min_delegations=0` examples therefore carry ceilings; nothing else does, because a
ceiling nobody has an argument for is a bar waiting to fail a correct agent. Absent
means NO ceiling, never zero — the mirror of the unknown-means-fail rule that governs
`GATED_TOOLS`, and inverted on purpose: a gate must refuse what it does not recognise,
an eval bar must not assert what nobody claimed. `delegates_breadth`'s docstring
carries the full argument, and two tests hold the ends — one that a missing ceiling
asserts nothing, one that the direct-path examples nonetheless have one.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

from langsmith import Client

DATASET_NAME = "deep-research-agent: research workflow"

DATASET_DESCRIPTION = (
    "Research questions for the deep research agent. Outputs hold structural "
    "expectations (min_delegations), not gold answers — see evals/dataset.py."
)

EXAMPLES: list[dict[str, Any]] = [
    {
        "inputs": {
            "question": (
                "Compare the free tiers of Tavily and LangSmith: what are the "
                "monthly usage limits of each?"
            )
        },
        # Two independent lookups — the textbook case for fanning out.
        "outputs": {
            "min_delegations": 2,
            "expects_plan": True,
            "expects_persist": True,
        },
    },
    {
        "inputs": {
            "question": (
                "How do LangGraph's SqliteSaver and PostgresSaver checkpointers "
                "differ in durability, concurrency support, and setup cost?"
            )
        },
        "outputs": {
            "min_delegations": 2,
            "expects_plan": True,
            "expects_persist": True,
        },
    },
    {
        "inputs": {
            "question": (
                "Compare mypy, pyright, and ty as Python type checkers: which are "
                "actively maintained, and how do they differ on speed and coverage?"
            )
        },
        "outputs": {
            "min_delegations": 2,
            "expects_plan": True,
            "expects_persist": True,
        },
    },
    {
        "inputs": {
            "question": (
                "What are Anthropic's published rate limits for the Claude API, and "
                "how do they differ between usage tiers?"
            )
        },
        "outputs": {
            "min_delegations": 2,
            "expects_plan": True,
            "expects_persist": True,
        },
    },
    {
        # GAP B: the highest bar here, and the reason it exists. BEFORE this example
        # every non-control question asked for exactly two delegations, so an agent
        # hard-capped at two dispatches scored a clean 5 of 5 and `delegates_breadth`
        # was never exercised higher. It is also the deliberate exception to the
        # resolve-downward rule in the module docstring — an assertion, not an
        # estimate, which is why the fallback below is written down rather than left
        # to be rediscovered from a failed sweep.
        # Three legs, three primary sources (docs.pypi.org, PEP 740, pip's docs), three
        # vocabularies. This is the sharpest column assertion here and its failure mode
        # is known in advance: a PEP 740 attestation is signed with the SAME OIDC
        # identity Trusted Publishing establishes, so legs 1 and 2 are causally coupled
        # and an agent may brief them together. They survive as separate dispatches on
        # independent *briefability*, not on unrelated sources — do not restate the
        # stronger claim. If `delegates_breadth` fails on a trace showing that bundling,
        # it is this example's defect: drop the floor to 2, leave SYSTEM_PROMPT alone.
        "inputs": {
            "question": (
                "In the Python packaging supply chain: how does PyPI's Trusted "
                "Publishing authenticate a CI workflow without a long-lived token, "
                "what does a PEP 740 attestation assert about a published distribution "
                "file, and what does pip's hash-checking mode verify at install time — "
                "and what does it demand of a requirements file?"
            )
        },
        "outputs": {
            "min_delegations": 3,
            "expects_plan": True,
            "expects_persist": True,
        },
    },
    {
        # GAP D: the hedging judge, which has never been tested on a hedge.
        # `answers_the_question` was rebuilt because it read an honest caveat as a
        # refusal to answer — and nothing in the dataset was shaped to produce one. Here
        # The DESIGN ASSUMPTION — labelled as one, because the docstring above forbids
        # inventing facts about live services and this is exactly that kind of claim:
        # providers do not commit to bit-exact reproducibility, and the public accounts
        # of the residual variation do not agree, so the honest answer is a qualified no
        # and the hedge is the substance rather than a wrapper around it. If the first
        # sweep instead produces a flat sourced summary, the assumption was wrong and
        # GAP D is still open. Nothing detects that: the score is 1 either way, so read
        # the prose, not the metric.
        #
        # The bar is 1, not 2, and that is a deliberate downgrade from what the
        # candidate was proposed at. The two legs are asymmetric — a vendor-docs survey
        # and a systems question — and SYSTEM_PROMPT step 3 tells the orchestrator to
        # reach for `include_domains` against vendor docs itself, so one `task` plus
        # direct searches is *prompt-compliant* and would have failed at 2. This example
        # exists for the judge, not for `delegates_breadth`; GAP B is carried above.
        "inputs": {
            "question": (
                "Can a commercial LLM API be made to return identical outputs for "
                "identical requests under greedy decoding (temperature 0) — what do the "
                "major providers actually guarantee, and what causes the variation that "
                "remains?"
            )
        },
        "outputs": {
            "min_delegations": 1,
            "expects_plan": True,
            "expects_persist": True,
        },
    },
    {
        # GAP A: claim density on the direct-search path. That path is this agent's
        # documented blind spot: memory checks and citation coverage both collapsed
        # there until SYSTEM_PROMPT steps 2 and 4 named the path explicitly. CLAUDE.md,
        # *What the evals found*, carries the measured before and after — deliberately
        # not restated here, because a second copy of a number is how the last one
        # drifted (`evals/__main__.py` states the same rule). What matters for this
        # example is the shape of what that fix left ungraded: before it, the only
        # direct-path example was the control, whose whole answer is two facts, so
        # `claims_are_cited` has a granularity of 0.5 there and a regression would
        # barely move the number. This carries a dozen attributable figures off one
        # decade-stable page.
        #
        # `expects_persist=True` puts a live `persists_findings` bar on the direct path
        # for the first time — this example and the OpenTelemetry one below, jointly;
        # neither is first on its own. It may well fail, because step 5 has never been
        # named for that path the way steps 2 and 4 now are. That is a finding about the
        # prompt, not a mis-set column: do not flip it, and do not act on a single run
        # (CLAUDE.md's write_todos study is the cautionary case — two samples of one
        # question disagreed, and only the pooled figure meant anything).
        "inputs": {
            "question": (
                "What hard upper limits does SQLite publish in its own "
                "implementation-limits documentation — on database size, columns per "
                "table, SQL statement length, host parameters, and attached databases — "
                "and which of those are compile-time defaults that can be raised rather "
                "than fixed ceilings?"
            )
        },
        "outputs": {
            "min_delegations": 0,
            # One `task` is defensible context hygiene — a researcher absorbing one
            # document read. Two is not, on a question one page answers: it is the
            # fan-out that converts this into a delegated-path example and quietly
            # reopens the gap it was added to close.
            "max_delegations": 1,
            "expects_plan": False,
            "expects_persist": True,
        },
    },
    {
        # The second direct-path example, and the second one is the point: at n=1 a
        # direct-path failure cannot be told from noise, and this repo has already very
        # nearly redesigned around a five-run result that evaporated at ten (CLAUDE.md,
        # *A tempting hypothesis that the data killed*). It takes `checks_memory_first`
        # to three observations on the direct path, and — together with the SQLite
        # example above, not ahead of it — is what puts a proposed mutation on that path
        # at all, which is the precondition for `mutations_require_approval` being
        # anything but vacuous there.
        #
        # What neither GAP A example buys, so nobody claims it later: attribution
        # DISCRIMINATION. The judge treats attribution as inherited, and both answers
        # sit on one source family, so a single trailing spec URL can score ~100%
        # whether the agent was disciplined or lazy. They raise claim COUNT, which is
        # what the control's two facts could not supply. (Name the control, never its
        # index: this commit inserted four examples above it and every positional
        # reference in the file went stale at once.)
        "inputs": {
            "question": (
                "In OpenTelemetry tracing, what is a span, what fields does the "
                "OpenTelemetry specification define on one, and what sizes or permitted "
                "values does it specify for the trace ID, span ID, span kind, and span "
                "status?"
            )
        },
        "outputs": {
            "min_delegations": 0,
            "max_delegations": 1,  # same argument as the SQLite example above
            "expects_plan": False,
            "expects_persist": True,
        },
    },
    {
        # The control. The prompt explicitly permits a direct `tavily_search` for a
        # single quick lookup, so delegating here is not required — an agent that
        # spins up a subagent for this is over-orchestrating, and one that answers
        # with no search at all still fails `searched_the_web`.
        "inputs": {
            "question": "What is the latest stable release of Python, and when was it released?"
        },
        "outputs": {
            "min_delegations": 0,
            # Zero, and this is the one example where the repo had already committed to
            # the judgement in prose: the comment above has always called delegating
            # here over-orchestration. It described nothing checkable until the ceiling
            # existed.
            "max_delegations": 0,
            "expects_plan": False,
            "expects_persist": False,
        },
    },
]


class DatasetReader(Protocol):
    """The slice of `langsmith.Client` that reading the dataset actually uses.

    A Protocol rather than `Client` so the comparison in `drift()` can be tested
    offline against a fake. The two alternatives were both worse: widening the
    parameter to `Any` gives up checking at the real call sites in order to
    accommodate the tests, and a `# ty: ignore` would be the only one in first-party
    code — see CLAUDE.md on why that count is deliberately zero and what it costs to
    reintroduce one.
    """

    def has_dataset(self, *, dataset_name: str) -> bool: ...

    def read_dataset(self, *, dataset_name: str) -> Any: ...

    def list_examples(self, *, dataset_id: Any) -> Iterable[Any]: ...


def _remote(client: DatasetReader, dataset_id: Any) -> dict[str | None, Any]:
    """Map question text -> the uploaded example.

    Keyed on the question because that is the only identity an example has on both
    sides: LangSmith assigns the ids, so a local entry has none to match on.

    `None` is an admitted key, not an oversight: an uploaded example with no `question`
    input is possible (nothing stops one being added through the LangSmith UI) and it
    is itself drift. Narrowing the annotation by filtering it out here would delete it
    from `drift()`'s orphan list — hiding the one example a reader of this file cannot
    see at all. It surfaces as an orphan keyed `None`, which is exactly what it is.
    """
    return {
        (example.inputs or {}).get("question"): example
        for example in client.list_examples(dataset_id=dataset_id)
    }


def drift(client: DatasetReader | None = None) -> list[str]:
    """Every way the uploaded dataset differs from `EXAMPLES`, as plain sentences.

    `--run` without `--limit` hands `evaluate()` the dataset NAME, so the sweep grades
    whatever LangSmith holds — never what you just edited. That is silent, and it costs
    a sweep to learn. Nothing else notices: the evaluators read the *example's* outputs,
    so a missing column falls back to its default and grades against the wrong bar
    without erroring, which is the same hazard `sync()` exists to close from the writing
    side.

    **Compare the outputs, never the count.** Measured, in this repo, one commit apart:
    adding `max_delegations` changed three examples and left the count at 9. A
    count check passes there while all three direct-path ceilings silently do not
    exist remotely — `delegates_breadth` reads a missing key as "no ceiling" — so the
    sweep scores exactly as it did before the column was added.

    Deliberately a WIDER comparison than the one `sync()` makes, and the difference is
    orphans. `sync()` adds and updates but never deletes, so a remote example that is
    not in `EXAMPLES` is invisible to it, by design. To a *run* that example is not
    invisible at all: `evaluate()` grades it, and its scores land in an experiment
    aggregate alongside examples you can read in this file. So it is reported here, and
    `--upload` is not its remedy.
    """
    client = client or Client()
    if not client.has_dataset(dataset_name=DATASET_NAME):
        return [f"the dataset {DATASET_NAME!r} does not exist yet — run `--upload`"]

    existing = _remote(client, client.read_dataset(dataset_name=DATASET_NAME).id)
    local = {e["inputs"]["question"]: e["outputs"] for e in EXAMPLES}

    differences = [
        f"not uploaded: {question!r}" for question in local if question not in existing
    ]
    differences += [
        f"outputs differ for {question!r}: "
        f"uploaded {existing[question].outputs or {}}, local {outputs}"
        for question, outputs in local.items()
        if question in existing and (existing[question].outputs or {}) != outputs
    ]
    differences += [
        f"uploaded but absent from EXAMPLES, and `--upload` will NOT remove it — "
        f"delete it in LangSmith if the sweep should not grade it: {question!r}"
        for question in existing
        if question not in local
    ]
    return differences


def sync(client: Client | None = None) -> str:
    """Create the dataset if absent, add new examples, and reconcile changed ones.

    Idempotent and keyed on the question text. Updating — not just inserting — is the
    part that matters: adding a column here (`expects_plan` was added after the first
    upload) leaves every already-uploaded example without it, and an evaluator reading
    a missing key falls back to its default and silently grades against the wrong bar.
    """
    client = client or Client()

    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME, description=DATASET_DESCRIPTION
        )

    existing = _remote(client, dataset.id)

    fresh = [e for e in EXAMPLES if e["inputs"]["question"] not in existing]
    stale = [
        (existing[e["inputs"]["question"]].id, e["outputs"])
        for e in EXAMPLES
        if e["inputs"]["question"] in existing
        and (existing[e["inputs"]["question"]].outputs or {}) != e["outputs"]
    ]

    if fresh:
        client.create_examples(
            inputs=[e["inputs"] for e in fresh],
            outputs=[e["outputs"] for e in fresh],
            dataset_id=dataset.id,
        )
    for example_id, outputs in stale:
        client.update_example(example_id=example_id, outputs=outputs)

    print(
        f"dataset '{DATASET_NAME}': {len(existing)} present, "
        f"{len(fresh)} added, {len(stale)} updated."
    )
    return DATASET_NAME
