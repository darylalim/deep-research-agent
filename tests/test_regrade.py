"""Offline tests for `python -m evals --regrade`.

The tool spends money and its output decides which judge the evals use, so these pin
what would otherwise fail quietly: old scores lost to a crash, a non-verdict averaged in
as a number, and two judges' grades mixed in one comparison.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from evals import __main__ as entry
from evals import evaluators
from evals.evaluators import JUDGE_EVALUATORS, claims_are_cited
from evals.regrade import regrade, stored_answers, summary

START = datetime(2026, 9, 6, tzinfo=UTC)


def _run(number: int, *, outputs: Any = None, error: str | None = None) -> Any:
    return SimpleNamespace(
        id=f"run{number}",
        start_time=START + timedelta(minutes=number),
        inputs={"question": f"question {number}"},
        outputs={"response": f"answer {number}"} if outputs is None else outputs,
        error=error,
    )


def _feedback(run: str, key: str, score: float) -> Any:
    return SimpleNamespace(run_id=run, key=key, score=score, comment=f"old {score}")


class _Reader:
    """Read calls only: a write to LangSmith would be an `AttributeError` here."""

    def __init__(self, runs: list[Any], feedback: list[Any]) -> None:
        self.runs = runs
        self.feedback = feedback

    def list_runs(self, *, project_name: str, is_root: bool) -> list[Any]:
        assert is_root is True, "child runs carry no answer to grade"
        return self.runs

    def list_feedback(self, *, run_ids: Any, feedback_key: Any) -> list[Any]:
        return [f for f in self.feedback if f.key in feedback_key]


def scoring(*scores: float | None) -> Any:
    """An evaluator that returns `scores` in turn, named like a real metric."""
    remaining = iter(scores)

    def claims_are_cited(run: Any, example: Any) -> dict[str, Any]:
        return {"score": next(remaining), "comment": "new"}

    return claims_are_cited


def test_old_scores_are_on_disk_before_the_first_judge_call(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "report.json"
    reader = _Reader([_run(1)], [_feedback("run1", "claims_are_cited", 0.9)])

    def raises(run: Any, example: Any) -> dict[str, Any]:
        raise RuntimeError("the API went away")

    raises.__name__ = "claims_are_cited"
    with pytest.raises(RuntimeError):
        regrade(reader, "exp", out=out, repeats=1, evaluators=[raises])

    saved = json.loads(out.read_text())
    assert saved["answers"][0]["old"]["claims_are_cited"]["score"] == 0.9


def test_a_crash_keeps_the_grades_already_paid_for(tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    reader = _Reader([_run(1)], [])
    grades = iter([{"score": 0.5, "comment": "new"}])

    def fails_second(run: Any, example: Any) -> dict[str, Any]:
        return next(grades)

    fails_second.__name__ = "claims_are_cited"
    with pytest.raises(StopIteration):
        regrade(reader, "exp", out=out, repeats=2, evaluators=[fails_second])

    saved = json.loads(out.read_text())
    assert [g["score"] for g in saved["answers"][0]["new"]["claims_are_cited"]] == [0.5]


def test_each_answer_is_graded_once_per_repeat_per_metric(tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []

    def metric(name: str) -> Any:
        def grade(run: Any, example: Any) -> dict[str, Any]:
            calls.append((run.id, name))
            return {"score": 1.0, "comment": ""}

        grade.__name__ = name
        return grade

    regrade(
        _Reader([_run(1), _run(2)], []),
        "exp",
        out=tmp_path / "r.json",
        repeats=3,
        evaluators=[metric("a"), metric("b")],
    )
    assert len(calls) == 12
    assert calls.count(("run2", "b")) == 3


def test_the_real_evaluators_read_the_stored_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A stored run stands in for both arguments, so the question has to come through
    # it. Without this, a regrade could grade every answer against an empty question.
    prompts: list[str] = []

    class Judge:
        def with_structured_output(self, schema: Any, **kwargs: Any) -> Judge:
            return self

        def invoke(self, messages: Any) -> dict[str, Any]:
            prompts.append(messages[0]["content"])
            grade = {"substantive_claims": 4, "uncited_claims": [{"claim": "x"}]}
            return {
                "raw": AIMessage("", response_metadata={"stop_reason": "end_turn"}),
                "parsed": grade,
                "parsing_error": None,
            }

    monkeypatch.setattr(evaluators, "JUDGE", Judge())
    report = regrade(
        _Reader([_run(1)], []),
        "exp",
        out=tmp_path / "r.json",
        repeats=1,
        evaluators=[claims_are_cited],
    )
    assert "question 1" in prompts[0]
    assert "answer 1" in prompts[0]
    assert report["answers"][0]["new"]["claims_are_cited"][0]["score"] == 0.75


def test_a_crashed_stored_run_is_a_non_verdict_not_a_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(evaluators, "JUDGE", None)  # must not be reached
    crashed = _run(1, outputs={}, error="boom")
    report = regrade(
        _Reader([crashed], []),
        "exp",
        out=tmp_path / "r.json",
        repeats=1,
        evaluators=[claims_are_cited],
    )
    assert report["answers"][0]["new"]["claims_are_cited"][0]["score"] is None


def test_a_non_verdict_is_counted_and_never_averaged_in(tmp_path: Path) -> None:
    reader = _Reader([_run(1), _run(2)], [])
    report = regrade(
        reader,
        "exp",
        out=tmp_path / "r.json",
        repeats=2,
        evaluators=[scoring(1.0, None, 0.5, 0.25)],
    )
    last = summary(report)[-1]
    # Per-answer means are 1.0 and 0.375. Counted as a zero, the first would be 0.5.
    assert "new 0.69 (2 scored, 1 grade(s) not measured)" in last
    assert "mean 0.12, max 0.25" in last


def test_an_answer_with_no_old_score_stays_out_of_the_old_mean(tmp_path: Path) -> None:
    reader = _Reader([_run(1), _run(2)], [_feedback("run1", "claims_are_cited", 0.8)])
    report = regrade(
        reader,
        "exp",
        out=tmp_path / "r.json",
        repeats=1,
        evaluators=[scoring(1.0, 1.0)],
    )
    assert "old 0.80 (1 scored)" in summary(report)[-1]


def test_a_metric_graded_twice_is_refused_before_spending(tmp_path: Path) -> None:
    reader = _Reader(
        [_run(1)],
        [
            _feedback("run1", "claims_are_cited", 0.9),
            _feedback("run1", "claims_are_cited", 0.4),
        ],
    )
    with pytest.raises(SystemExit, match="more than one claims_are_cited"):
        stored_answers(reader, "exp", ["claims_are_cited"])


def test_an_empty_experiment_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="holds no runs"):
        regrade(_Reader([], []), "exp", out=tmp_path / "r.json", repeats=1)


def test_the_default_metrics_are_the_judges(tmp_path: Path) -> None:
    rows = stored_answers(
        _Reader([_run(1)], []), "exp", [e.__name__ for e in JUDGE_EVALUATORS]
    )
    assert set(rows[0][1]["new"]) == {
        "claims_are_cited",
        "answers_the_question",
        "claims_are_supported",
    }


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--regrade", "exp", "--run"], "runs alone"),
        (["--regrade", "exp", "--upload"], "runs alone"),
        (["--regrade", "exp", "--repeats", "0"], "at least 1"),
        ([], "nothing to do"),
    ],
)
def test_bad_arguments_stop_before_any_key_or_call_is_needed(
    argv: list[str],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.argv", ["evals", *argv])
    with pytest.raises(SystemExit) as stopped:
        entry.main()
    assert stopped.value.code == 2
    assert message in capsys.readouterr().err


def test_main_regrades_with_tracing_off_and_saves_outside_the_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # `setenv` first, so that the values `main` writes are undone after the test.
    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
        monkeypatch.setenv(name, "true")
    monkeypatch.setenv("DEEP_RESEARCH_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LANGSMITH_API_KEY", "x")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["evals", "--regrade", "exp", "--repeats", "1"])
    tracing_seen: list[str] = []

    def client() -> _Reader:
        tracing_seen.extend(
            os.environ[name] for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2")
        )
        # A crashed run needs no judge, so this reaches the summary with no network.
        return _Reader([_run(1, outputs={}, error="boom")], [])

    monkeypatch.setattr("langsmith.Client", client)
    entry.main()

    assert tracing_seen == ["false", "false"]
    saved = list((tmp_path / "deep-research-regrades").glob("exp-*.json"))
    assert len(saved) == 1
    assert "1 grade(s) not measured" in capsys.readouterr().out
    assert json.loads(saved[0].read_text())["repeats"] == 1
