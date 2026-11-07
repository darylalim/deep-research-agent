"""`python -m evals --regrade <experiment>`: grade an experiment's stored answers again.

This compares judges without re-running the agent, which is the expensive part of a
sweep. Each stored answer goes through the real evaluators several times, because
`temperature` is gone and one grade no longer shows how much a score moves.

**Nothing is written back to LangSmith.** `evaluate_existing` would add feedback to the
experiment and mix new scores with old ones, so the client here is only ever read, and
`ExperimentReader` holds just the two calls that needs. Results go to a local file.

**The stored scores are saved before the first judge call**, and the file is rewritten
after every grade, so a crash partway through keeps what has been paid for.

Stored scores are only as good as the instrument that made them. Experiments from before
2026-09-07 were graded by a judge capped at `max_tokens=1024`, whose truncated grades
were repaired and scored (see `CLAUDE.md`), so read their comments before trusting them.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol

from . import evaluators as graders


class Evaluator(Protocol):
    """An evaluator function. Its name is the feedback key its scores are stored under."""

    __name__: str

    def __call__(self, run: Any, example: Any, /) -> dict[str, Any]: ...


class ExperimentReader(Protocol):
    """The two read calls a regrade needs, so the tests can supply their own.

    `list_runs` is deprecated in favour of `client.runs.query`, with removal due after
    2027-01-31. That replacement is async and returns a different shape, which is not
    worth taking on for a tool this size until the old call is actually going away.
    """

    def list_runs(self, *, project_name: str, is_root: bool) -> Iterable[Any]: ...

    def list_feedback(
        self, *, run_ids: Sequence[Any], feedback_key: Sequence[str]
    ) -> Iterable[Any]: ...


def stored_answers(
    client: ExperimentReader, experiment: str, metrics: Sequence[str]
) -> list[tuple[Any, dict[str, Any]]]:
    """Each root run of `experiment`, paired with a report row holding its old scores.

    Two rows of feedback for one metric are refused rather than resolved. That means
    something already graded this experiment a second time, and picking one would
    report a comparison against a judge nobody can name.
    """
    runs = sorted(
        client.list_runs(project_name=experiment, is_root=True),
        key=lambda run: run.start_time,
    )
    if not runs:
        raise SystemExit(f"experiment {experiment!r} holds no runs to regrade")

    old: dict[tuple[Any, str], dict[str, Any]] = {}
    for row in client.list_feedback(
        run_ids=[run.id for run in runs], feedback_key=list(metrics)
    ):
        key = (row.run_id, row.key)
        if key in old:
            raise SystemExit(
                f"run {row.run_id} has more than one {row.key} score, so the old "
                "judge's grade cannot be told apart from a later one"
            )
        old[key] = {"score": row.score, "comment": row.comment}

    return [
        (
            run,
            {
                "run_id": str(run.id),
                "question": (run.inputs or {}).get("question", ""),
                "old": {metric: old.get((run.id, metric)) for metric in metrics},
                "new": {metric: [] for metric in metrics},
            },
        )
        for run in runs
    ]


def _save(report: dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


def regrade(
    client: ExperimentReader,
    experiment: str,
    *,
    out: Path,
    repeats: int,
    evaluators: Sequence[Evaluator] = graders.JUDGE_EVALUATORS,
) -> dict[str, Any]:
    """Grade every stored answer `repeats` times per metric, saving as it goes."""
    metrics = [evaluator.__name__ for evaluator in evaluators]
    answers = stored_answers(client, experiment, metrics)
    report: dict[str, Any] = {
        "experiment": experiment,
        "judge": str(getattr(graders.JUDGE, "model", "unknown")),
        "repeats": repeats,
        "answers": [row for _, row in answers],
    }
    _save(report, out)
    print(
        f"{len(answers)} answer(s) x {len(metrics)} metric(s) x {repeats} repeat(s) = "
        f"at most {len(answers) * len(metrics) * repeats} judge call(s), since a metric "
        f"with nothing to read stops before calling; saving to {out}"
    )

    for number, (run, row) in enumerate(answers, start=1):
        for evaluator in evaluators:
            for _ in range(repeats):
                # The run stands in for the example too: the evaluators read only
                # `question` from it, and a run stores the same inputs. The answers are
                # graded as they were asked, even if the dataset has changed since.
                grade = evaluator(run, run)
                row["new"][evaluator.__name__].append(grade)
                _save(report, out)
                print(f"  answer {number}: {evaluator.__name__} = {grade['score']}")
    return report


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _shown(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def summary(report: dict[str, Any]) -> list[str]:
    """One line per answer and metric, then a line per metric across all answers.

    A non-verdict (`score=None`) is counted and never averaged in. The new mean is a
    mean of per-answer means, so a repeat that went unmeasured does not reweight its
    answer against the others.
    """
    lines: list[str] = []
    metrics = list(report["answers"][0]["new"]) if report["answers"] else []
    for metric in metrics:
        old_scores: list[float] = []
        new_means: list[float] = []
        spreads: list[float] = []
        unmeasured = 0
        lines.append(metric)
        for row in report["answers"]:
            old = (row["old"].get(metric) or {}).get("score")
            grades = [grade["score"] for grade in row["new"][metric]]
            scores = [score for score in grades if score is not None]
            unmeasured += len(grades) - len(scores)
            if old is not None:
                old_scores.append(old)
            if scores:
                new_means.append(sum(scores) / len(scores))
                spreads.append(max(scores) - min(scores))
            shown = ", ".join(_shown(score) for score in grades)
            lines.append(f"  old {_shown(old)} | new {shown} | {row['question'][:60]}")
        lines.append(
            f"  mean: old {_shown(_mean(old_scores))} ({len(old_scores)} scored), "
            f"new {_shown(_mean(new_means))} ({len(new_means)} scored, "
            f"{unmeasured} grade(s) not measured); spread between repeats: mean "
            f"{_shown(_mean(spreads))}, max {_shown(max(spreads, default=None))}"
        )
    return lines
