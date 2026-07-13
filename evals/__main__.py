"""Entry point:  uv run python -m evals [--upload] [--run]

    --upload    create/sync the dataset in LangSmith (idempotent)
    --run       run the agent over the dataset and score it
    --code-only skip the LLM judges (free, and enough for a trajectory regression)

The env var juggling below is load-bearing. `deep_research.config` resolves
`STATE_DIR` into a module constant at *import* time, so the throwaway state dir
has to be in the environment before anything imports it — which is why
`evals.harness` is imported inside `main()` and not at the top of this file. The
same reason `tests/conftest.py` sets its state dir as top-level code.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from typing import Any

from dotenv import load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(prog="evals", description=__doc__)
    parser.add_argument("--upload", action="store_true", help="create/sync the dataset")
    parser.add_argument("--run", action="store_true", help="evaluate the agent")
    parser.add_argument(
        "--code-only", action="store_true", help="skip the LLM judges (free)"
    )
    parser.add_argument(
        "--prefix", default="workflow", help="experiment name prefix in LangSmith"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="only evaluate the first N examples (a full sweep is ~100k tokens each)",
    )
    args = parser.parse_args()
    if not (args.upload or args.run):
        parser.error("nothing to do — pass --upload and/or --run")

    load_dotenv()
    for key in ("ANTHROPIC_API_KEY", "TAVILY_API_KEY", "LANGSMITH_API_KEY"):
        if not os.environ.get(key):
            raise SystemExit(f"{key} is not set — evals need real credentials.")

    # OVERRIDE, never `setdefault`. `DEEP_RESEARCH_STATE_DIR` is a documented way to
    # relocate the agent's *real* state, and `load_dotenv()` above has just loaded the
    # user's `.env` into the environment. A `setdefault` would therefore no-op and hand
    # the eval — which drops the checkpointer and store between examples — whatever
    # directory the user keeps their durable memories in. (A *blank* `DEEP_RESEARCH_
    # STATE_DIR=` line is the same bug wearing a hat: `Path("").resolve()` is the repo
    # root.) Evals own their state dir, full stop; there is no reason to let anything
    # else name it. Must also happen before the imports below, since
    # `deep_research.config` freezes the path at import time.
    os.environ["DEEP_RESEARCH_STATE_DIR"] = tempfile.mkdtemp(
        prefix="deep_research_evals_"
    )

    from itertools import islice

    from langsmith import Client, evaluate

    from . import dataset
    from .evaluators import ALL_EVALUATORS, CODE_EVALUATORS
    from .harness import research

    if args.upload:
        dataset.sync()
    if not args.run:
        return

    evaluators = CODE_EVALUATORS if args.code_only else ALL_EVALUATORS
    # `evaluate` takes a dataset name or an iterable of examples; the latter is how
    # a smoke run stays cheap.
    data: Any = dataset.DATASET_NAME
    if args.limit:
        client = Client()
        data = list(
            islice(client.list_examples(dataset_name=dataset.DATASET_NAME), args.limit)
        )

    count = len(data) if isinstance(data, list) else len(dataset.EXAMPLES)
    print(
        f"running {count} example(s) with {len(evaluators)} evaluator(s); "
        f"state dir: {os.environ['DEEP_RESEARCH_STATE_DIR']}"
    )

    results = evaluate(
        research,
        data=data,
        evaluators=evaluators,
        experiment_prefix=args.prefix,
        # SERIAL, NOT A PERFORMANCE CHOICE. Every example wipes and recreates the
        # one state dir whose path `deep_research.config` froze at import, so two
        # examples in flight would delete each other's checkpoint database
        # mid-run. Raising this needs a process per example, not a thread.
        max_concurrency=1,
    )
    print(results)


if __name__ == "__main__":
    main()
