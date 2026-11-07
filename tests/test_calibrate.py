"""Offline tests for `python -m evals --calibrate`.

Its verdict decides whether the judge may grade Phase 3's sweeps, so these pin the
ways it could pass a judge it should fail: a flag matched to the wrong claim, an
unmeasured round counted as agreement, the labelled file drifting from its reports,
and a calibration graded by a different prompt from the one a sweep uses.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from evals import calibrate as cal
from evals import evaluators


def _report(
    claims: list[tuple[str, str, str]], parts: Sequence[tuple[str, str, str]] = ()
) -> dict:
    return {
        "question": "q?",
        "response": " ".join(text for _, text, _ in claims) or "answer",
        "claims": [{"id": i, "claim": t, "label": lab} for i, t, lab in claims],
        "parts": [{"id": i, "part": t, "label": lab} for i, t, lab in parts],
    }


CLAIMS = [
    ("c1", "Tavily's free plan gives you 1,000 API credits a month.", "cited"),
    ("c2", "Many third-party articles still say 400 days.", "uncited"),
    ("c3", "Python 3.13.14 was released the same day, June 10, 2026.", "cited"),
]


def flagging(*flags_per_call: list[str] | None) -> Any:
    """A grader that lists the given uncited claims, call by call; `None` = no grade."""
    calls = iter(flags_per_call)

    def grade(question: str, prose: str) -> tuple[dict | None, str]:
        flags = next(calls)
        if flags is None:
            return None, "the judge's own answer hit max_tokens and was cut off"
        grade = {
            "substantive_claims": 3,
            "uncited_claims": [{"claim": f} for f in flags],
        }
        return grade, ""

    return grade


def _run(
    reports: list[dict], tmp_path: Path, *flags: list[str] | None, repeats: int = 1
):
    result = cal.calibrate(
        reports,
        out=tmp_path / "out.json",
        repeats=repeats,
        graders_by_metric={"claims_are_cited": flagging(*flags)},
    )
    return cal.tally(result, reports, "claims_are_cited")


# --- the labelled file itself ---------------------------------------------------


def test_every_labelled_claim_occurs_exactly_once_in_its_own_report():
    for report in cal.load():
        for item in report["claims"]:
            assert report["response"].count(item["claim"]) == 1, item["id"]


def test_every_label_is_one_the_tally_understands():
    seen: set[str] = set()
    for report in cal.load():
        for kind, spec in cal.METRICS.items():
            for item in report[spec["labelled"]]:
                assert item["id"] not in seen, f"duplicate id {item['id']}"
                seen.add(item["id"])
                allowed = {
                    spec["flag_means"],
                    {"claims_are_cited": "cited"}.get(kind, "answered"),
                }
                assert item["label"] in allowed, item
                assert item.get("adjudicated", item["label"]) in allowed, item


def test_an_adjudicated_label_is_scored_only_when_asked_for(tmp_path):
    """The blind score must stay the default: it is the one nobody could steer."""
    report = _report(CLAIMS)
    report["claims"][0]["adjudicated"] = "uncited"  # c1: blind "cited"
    result = cal.calibrate(
        [report],
        out=tmp_path / "out.json",
        repeats=1,
        graders_by_metric={"claims_are_cited": flagging([CLAIMS[0][1], CLAIMS[1][1]])},
    )
    blind = cal.tally(result, [report], "claims_are_cited")
    adjudicated = cal.tally(result, [report], "claims_are_cited", "adjudicated")
    assert blind["false_accusations"] == ["c1"]
    assert adjudicated["false_accusations"] == []
    assert (blind["agree"], adjudicated["agree"]) == (2, 3)
    text = "\n".join(cal.summary(result, [report]))
    assert "(blind labels)" in text and "(adjudicated labels)" in text


def test_the_blind_label_is_never_overwritten_in_the_file():
    for report in cal.load():
        for item in report["claims"]:
            if "adjudicated" in item:
                assert item["adjudicated"] != item["label"], item["id"]
                assert item.get("adjudication_note"), item["id"]


# --- matching ------------------------------------------------------------------


def test_every_real_claim_matches_itself_and_no_neighbour():
    """A flag quoting one claim verbatim must not be credited to the claim beside it."""
    for report in cal.load():
        for spec in cal.METRICS.values():
            labelled = report[spec["labelled"]]
            for item in labelled:
                found, _ = cal.match(item[spec["text"]], labelled, spec["text"])
                assert found == item["id"], item["id"]
                for other in labelled:
                    if other is not item:
                        assert cal.overlap(item[spec["text"]], other[spec["text"]]) < (
                            cal.MIN_OVERLAP
                        ), (item["id"], other["id"])


def test_a_paraphrase_with_markdown_still_matches_its_claim():
    labelled = [{"id": c, "claim": t} for c, t, _ in CLAIMS]
    found, _ = cal.match(
        "**Many** third-party articles say `400 days`", labelled, "claim"
    )
    assert found == "c2"


def test_a_different_version_number_does_not_match():
    """3.13.14 and 3.14.6 must stay distinct tokens, or a flag about one counts for both."""
    labelled = [{"id": "c3", "claim": CLAIMS[2][1]}]
    found, _ = cal.match("Python 3.14.6 released June 2026", labelled, "claim")
    assert found is None


def test_a_flag_too_short_to_identify_a_claim_matches_nothing():
    labelled = [{"id": c, "claim": t} for c, t, _ in CLAIMS]
    assert cal.match("1,000 credits", labelled, "claim")[0] is None


def test_an_unrelated_flag_matches_nothing():
    labelled = [{"id": c, "claim": t} for c, t, _ in CLAIMS]
    assert cal.match("Pyright is maintained by Microsoft", labelled, "claim")[0] is None


# --- counting ------------------------------------------------------------------


def test_agreement_false_accusations_and_misses_are_counted_apart(tmp_path):
    # Flags c1 (labelled cited: a false accusation) and misses c2 (labelled uncited).
    counts = _run([_report(CLAIMS)], tmp_path, [CLAIMS[0][1]])
    assert counts["agree"] == 1  # c3, correctly left alone
    assert counts["false_accusations"] == ["c1"]
    assert counts["misses"] == ["c2"]
    assert counts["compared"] == 3


def test_a_perfect_judge_agrees_on_every_comparison(tmp_path):
    counts = _run(
        [_report(CLAIMS)], tmp_path, [CLAIMS[1][1]], [CLAIMS[1][1]], repeats=2
    )
    assert (counts["agree"], counts["compared"]) == (6, 6)
    assert cal.verdict(counts) == "PASS"


def test_an_unmatched_flag_is_reported_and_not_scored(tmp_path):
    counts = _run(
        [_report(CLAIMS)],
        tmp_path,
        [CLAIMS[1][1], "Pyright is maintained by Microsoft"],
    )
    assert counts["unmatched"] == ["Pyright is maintained by Microsoft"]
    assert counts["agree"] == 3


def test_an_unmeasured_round_is_never_counted_as_agreement(tmp_path):
    """No grade means no flags, which would otherwise read as 'every claim cited'."""
    counts = _run([_report(CLAIMS)], tmp_path, None, [CLAIMS[1][1]], repeats=2)
    assert counts["not_measured"] == 1
    assert counts["compared"] == 3
    assert cal.verdict(counts) == "INCOMPLETE"


def test_flags_match_only_claims_in_the_report_they_were_graded_on(tmp_path):
    other = _report([("c9", "Many third-party articles still say 400 days.", "cited")])
    first = _report(CLAIMS)
    # The second report's judge flags text identical to a claim in the FIRST report.
    counts = _run([first, other], tmp_path, [CLAIMS[1][1]], [])
    assert counts["false_accusations"] == []
    assert counts["agree"] == 4


def test_the_bar_sits_at_85_percent():
    def at(agree: int, compared: int) -> str:
        return cal.verdict(
            {
                "agree": agree,
                "compared": compared,
                "not_measured": 0,
                "agreement": agree / compared,
            }
        )

    assert at(22, 25) == "PASS"  # 88%
    assert at(21, 25) == "FAIL"  # 84%
    assert cal.PASS_AGREEMENT == 0.85


def test_results_are_on_disk_after_every_grade(tmp_path):
    out = tmp_path / "out.json"

    calls: list[str] = []

    def crashes_second(question: str, prose: str):
        if calls:
            raise RuntimeError("the API went away")
        calls.append(question)
        return {"substantive_claims": 1, "uncited_claims": []}, ""

    with pytest.raises(RuntimeError):
        cal.calibrate(
            [_report(CLAIMS)],
            out=out,
            repeats=2,
            graders_by_metric={"claims_are_cited": crashes_second},
        )
    assert len(json.loads(out.read_text())["rounds"]) == 1


# --- the real grading path -----------------------------------------------------


class _Judge:
    """Records the prompt and returns a fixed grade, as `JUDGE` would."""

    def __init__(self, grade: dict) -> None:
        self.grade = grade
        self.prompts: list[str] = []

    def with_structured_output(self, schema: Any, **kwargs: Any) -> _Judge:
        return self

    def invoke(self, messages: Any) -> dict:
        self.prompts.append(messages[0]["content"])
        raw = AIMessage(content="", response_metadata={"stop_reason": "end_turn"})
        return {"raw": raw, "parsed": self.grade, "parsing_error": None}


def test_the_default_grader_sends_the_full_report_through_the_sweeps_prompt(
    monkeypatch, tmp_path
):
    """Calibrating a copy of the prompt would pass a judge the sweep never uses."""
    report = cal.load()[0]
    judge = _Judge({"substantive_claims": 5, "uncited_claims": []})
    monkeypatch.setattr(evaluators, "JUDGE", judge)

    cal.calibrate([report], out=tmp_path / "out.json", repeats=1)

    citation_prompt = next(p for p in judge.prompts if "attributed its claims" in p)
    assert report["response"] in citation_prompt
    run = type(
        "Run", (), {"outputs": {"response": report["response"]}, "error": None}
    )()
    example = type("Ex", (), {"inputs": {"question": report["question"]}})()
    evaluators.claims_are_cited(run, example)
    assert judge.prompts[-1] == citation_prompt
