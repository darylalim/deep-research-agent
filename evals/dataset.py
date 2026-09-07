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

The three structural columns — `min_delegations`, `expects_plan`, `expects_persist`
— are assertions about *judgment*, not just tool use, and they exist because
`SYSTEM_PROMPT` grants exemptions rather than issuing blanket rules: delegate
breadth **but** handle a single quick lookup yourself; plan first **unless** one
search settles it; persist durable findings **but not** ephemeral ones. Applying
any of those bars unconditionally scores the agent down for following its own
instructions — which is why the control example sets all three to their low bar.

Two properties of those columns govern every judgment call below, and both are
asymmetric. `min_delegations` is a FLOOR, so setting it low costs signal while setting
it high scores a *correct* agent down — every borderline bar here therefore resolves
downward. And `expects_plan=False` / `expects_persist=False` make their evaluators
return 1 unconditionally: a False bar cannot be wrong, only uninformative. Vacuous is
safe; that is the whole reason the direct-path examples exempt planning rather than
asserting it.

One tension a reader will otherwise read as an oversight: example 4 (Anthropic's rate
limits) is also one vendor's limits table on one page and sets `min_delegations` to 2,
while the SQLite and OpenTelemetry examples below assume that same shape does *not* fan
out. Both are anchored on a single named document deliberately, to make the direct path
the natural one. Their columns cannot be wrong — they are unfalsifiable — so if
`orchestrator_trajectory` shows a `task` on either, the WORDING needs another pass, not
the bars. The cost of the exemptions is real and accepted: `plans_with_todos` is now
vacuous on 3 of 9 examples, and it is the metric carrying 15 runs of history behind its
~80% figure, so future movement in it is harder to read than it was at 1 of 5.
"""

from __future__ import annotations

from typing import Any

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
        # GAP B: the only bar above 2 in the dataset. Every other non-control example
        # asks for exactly two delegations, so an agent hard-capped at two dispatches
        # scored a clean 5 of 5 and `delegates_breadth` was never exercised higher.
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
        # the honest answer IS a qualified no: no provider commits to bit-exact
        # reproducibility, and the public explanations of the residual variation
        # disagree. The hedge is the substance, not a wrapper around it.
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
        # documented blind spot — measured 0 of 5 lookups checked `/memories/` first,
        # and the one answer the orchestrator researched itself scored 25% citation
        # coverage against 83-100% on delegated ones. Naming the direct path in
        # SYSTEM_PROMPT steps 2 and 4 fixed it (77% -> 98% mean). But the control below
        # is the only other `min_delegations=0` example and its whole answer is two
        # facts, so `claims_are_cited` has a granularity of 0.5 there and a regression
        # of that fix would barely move the number. This carries a dozen attributable
        # figures off one decade-stable page.
        #
        # `expects_persist=True` is the first live `persists_findings` bar on the direct
        # path, and it may well fail: step 5 has never been named for that path the way
        # steps 2 and 4 now are. A failure is a finding about the prompt, not a mis-set
        # column — do not flip it, and do not act on one run (write_todos took 15).
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
            "expects_plan": False,
            "expects_persist": True,
        },
    },
    {
        # The second direct-path example, and the second one is the point: at n=1 a
        # direct-path failure cannot be told from noise, and this repo has already
        # redesigned around n=5 noise once (the plan/persist anti-correlation that
        # evaporated at n=10). It also gives `checks_memory_first` a third observation
        # on the path measured at 0 of 5, and gives `mutations_require_approval` — which
        # is vacuous unless a mutation is proposed — its first teeth there.
        #
        # What neither GAP A example buys, so nobody claims it later: attribution
        # DISCRIMINATION. The judge treats attribution as inherited, and both answers
        # sit on one source family, so a single trailing spec URL can score ~100%
        # whether the agent was disciplined or lazy. They raise claim COUNT, which is
        # what example 5's two facts could not supply.
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
            "expects_plan": False,
            "expects_persist": False,
        },
    },
]


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

    existing = {
        (example.inputs or {}).get("question"): example
        for example in client.list_examples(dataset_id=dataset.id)
    }

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
