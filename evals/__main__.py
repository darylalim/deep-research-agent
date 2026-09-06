"""Entry point:  uv run python -m evals [--upload] [--run]

    --upload    create/sync the dataset in LangSmith (idempotent)
    --run       run the agent over the dataset and score it
    --code-only skip the LLM judges. NOT free — it saves ~$0.01 of judging and
                still runs the agent over every example, which is the whole bill.
                See CLAUDE.md, *What a sweep costs*, for the measured figures;
                they are deliberately not restated here, because five copies of
                a number is how the last one drifted.

`argparse` gets an explicit one-line `description=` rather than this docstring.
The default formatter reflows whatever it is handed into a single paragraph, so
passing `__doc__` printed the note below to end users as if it addressed them,
ran the cost warning straight into it, and left "below" pointing at nothing.

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
    parser = argparse.ArgumentParser(
        prog="evals",
        description="Run the agent against the LangSmith eval dataset and score it.",
    )
    parser.add_argument("--upload", action="store_true", help="create/sync the dataset")
    parser.add_argument("--run", action="store_true", help="evaluate the agent")
    parser.add_argument(
        "--code-only",
        action="store_true",
        help="skip the LLM judges (saves ~$0.01; the agent still runs on every example)",
    )
    parser.add_argument(
        "--prefix", default="workflow", help="experiment name prefix in LangSmith"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="only evaluate the first N examples (47k-1M tokens each, measured)",
    )
    args = parser.parse_args()
    if not (args.upload or args.run):
        parser.error("nothing to do — pass --upload and/or --run")
    # `--limit 0` used to be the most expensive way to ask for nothing: the guard
    # below was `if args.limit:`, so a falsy zero fell through to the *whole*
    # dataset. Measured, by doing it: one example got through before the kill, at
    # 232,865 tokens / $0.51. A negative limit reached `islice` and died there with
    # a raw ValueError. Both are argument errors; say so before spending anything.
    if args.limit is not None and args.limit < 1:
        parser.error(f"--limit must be at least 1 (got {args.limit})")

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
    # a smoke run stays *smaller*. Not cheap — one example measured 47k-1M tokens.
    data: Any = dataset.DATASET_NAME
    if args.limit is not None:
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
