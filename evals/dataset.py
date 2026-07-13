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
