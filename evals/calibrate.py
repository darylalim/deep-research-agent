"""`python -m evals --calibrate`: measure the judges against human labels.

`--regrade` compares one judge with another, so it shows whether they agree, not
whether either is right. This compares the judge with `evals/calibration.json`, which
the project owner labelled without seeing any judge verdict.

**It grades through the same functions a sweep does.** `grade_citations` and
`grade_answer` in `evaluators.py` hold the prompts and schemas `claims_are_cited` and
`answers_the_question` score, so a pass here is a pass for the metric Phase 3 reads,
not for a copy of it. Each call gets the full report, because whether a claim is
cited depends on the bullet, paragraph or table around it.

**Flags are matched to labels by word overlap, not exact text.** The judge's `claim`
field is "quoted or closely paraphrased", so exact matching would turn a paraphrase
into a miss and an unmatched flag at once. Every match is saved with its overlap, and
every flag that matched nothing is printed, so a wrong match can be seen and checked.

**Nothing is written to LangSmith.** The caller turns tracing off, and this module
never builds a client. Results go to a local file, rewritten after every grade so a
crash keeps what has been paid for.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from . import evaluators as graders

CALIBRATION = Path(__file__).with_name("calibration.json")

# Decided before the first calibration run, so the result cannot move the bar.
PASS_AGREEMENT = 0.85
# Share of the shorter text's words the other must contain to count as the same claim.
MIN_OVERLAP = 0.6
# Below this many words, overlap is too easy to reach by chance ("1,000 credits").
MIN_WORDS = 3

Grader = Callable[[str, str], tuple[dict[str, Any] | None, str]]

# Which judge, which list in its grade, which text field in an entry of that list,
# which labels in calibration.json it is matched against, and which label means
# "the judge should flag this".
METRICS: dict[str, dict[str, str]] = {
    "claims_are_cited": {
        "listed": "uncited_claims",
        "field": "claim",
        "labelled": "claims",
        "text": "claim",
        "flag_means": "uncited",
    },
    "answers_the_question": {
        "listed": "unanswered_parts",
        "field": "part",
        "labelled": "parts",
        "text": "part",
        "flag_means": "unanswered",
    },
}


def load(path: Path = CALIBRATION) -> list[dict[str, Any]]:
    return json.loads(path.read_text())["reports"]


def _words(text: str) -> set[str]:
    """Lowercased words and numbers, with Markdown and punctuation stripped.

    Numbers keep their inner separators, so `2,000,000` and `3.14.6` stay one token
    each and a flag about 3.13 cannot match a claim about 3.14.
    """
    return set(re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*", text.lower()))


def _numbers(words: set[str]) -> set[str]:
    return {word for word in words if word[0].isdigit()}


def overlap(a: str, b: str) -> float:
    """Share of the shorter text's words found in the other; 0.0 below `MIN_WORDS`.

    **The numbers must agree first.** In these reports a claim is usually told apart
    from its neighbours by one figure: "Python 3.14.6 released June 2026" shares four
    of five words with "Python 3.13.14 was released the same day, June 10, 2026", and
    plain overlap matched them. So one side's numbers must all appear in the other,
    either way round, because the judge may quote part of a claim or a wider passage.
    """
    wa, wb = _words(a), _words(b)
    if min(len(wa), len(wb)) < MIN_WORDS:
        return 0.0
    na, nb = _numbers(wa), _numbers(wb)
    if not (na <= nb or nb <= na):
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def match(
    flag: str, labelled: Sequence[dict[str, Any]], text: str
) -> tuple[str | None, float]:
    """The id of the labelled item `flag` refers to, and the overlap that decided it.

    The best match wins, and only at `MIN_OVERLAP` or above. A flag that matches
    nothing returns `None`: the judge listed something nobody labelled, which is
    printed for checking, not counted as right or wrong.
    """
    best_id, best = None, 0.0
    for item in labelled:
        score = overlap(flag, item[text])
        if score > best:
            best_id, best = item["id"], score
    return (best_id, best) if best >= MIN_OVERLAP else (None, best)


def _save(result: dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


def calibrate(
    reports: Sequence[dict[str, Any]],
    *,
    out: Path,
    repeats: int,
    graders_by_metric: dict[str, Grader] | None = None,
) -> dict[str, Any]:
    """Grade every labelled report `repeats` times with each judge, saving as it goes.

    A report with nothing labelled for a metric is skipped for that metric, which
    saves a call that could not change any count.
    """
    graders_by_metric = graders_by_metric or {
        "claims_are_cited": graders.grade_citations,
        "answers_the_question": graders.grade_answer,
    }
    result: dict[str, Any] = {
        "judge": str(getattr(graders.JUDGE, "model", "unknown")),
        "repeats": repeats,
        "pass_agreement": PASS_AGREEMENT,
        "rounds": [],
    }
    _save(result, out)
    calls = sum(
        1
        for report in reports
        for metric, spec in METRICS.items()
        if metric in graders_by_metric and report[spec["labelled"]]
    )
    print(
        f"{calls} report-metric pair(s) x {repeats} repeat(s) = {calls * repeats} judge call(s)"
    )
    print(f"saving to {out}")

    for number, report in enumerate(reports):
        for metric, grade in graders_by_metric.items():
            spec = METRICS[metric]
            labelled = report[spec["labelled"]]
            if not labelled:
                continue
            for repeat in range(repeats):
                raw, refusal = grade(report["question"], report["response"])
                listed = (raw or {}).get(spec["listed"]) or []
                flags = []
                for entry in listed:
                    flag = str(entry.get(spec["field"], ""))
                    found, score = match(flag, labelled, spec["text"])
                    flags.append(
                        {"flag": flag, "matched": found, "overlap": round(score, 2)}
                    )
                result["rounds"].append(
                    {
                        "report": number,
                        "metric": metric,
                        "repeat": repeat,
                        "graded": raw is not None,
                        "refusal": refusal,
                        "flags": flags,
                        "grade": raw,
                    }
                )
                _save(result, out)
                state = (
                    f"{len(flags)} flag(s)"
                    if raw is not None
                    else f"not measured: {refusal}"
                )
                print(f"  report {number + 1} {metric} repeat {repeat + 1}: {state}")
    return result


def tally(
    result: dict[str, Any], reports: Sequence[dict[str, Any]], metric: str
) -> dict[str, Any]:
    """Counts for one metric, pooled over every graded round.

    Each labelled item is compared once per graded round, so 25 claims over 3 repeats
    is 75 comparisons. A round with no grade is counted as not measured and left out of
    the comparisons, never scored as agreement.
    """
    spec = METRICS[metric]
    counts = {
        "agree": 0,
        "false_accusations": [],
        "misses": [],
        "unmatched": [],
        "not_measured": 0,
    }
    per_repeat: dict[int, list[int]] = {}
    for rnd in result["rounds"]:
        if rnd["metric"] != metric:
            continue
        if not rnd["graded"]:
            counts["not_measured"] += 1
            continue
        flagged = {f["matched"] for f in rnd["flags"] if f["matched"]}
        counts["unmatched"].extend(f["flag"] for f in rnd["flags"] if not f["matched"])
        tallies = per_repeat.setdefault(rnd["repeat"], [0, 0])
        for item in reports[rnd["report"]][spec["labelled"]]:
            should_flag = item["label"] == spec["flag_means"]
            tallies[1] += 1
            if (item["id"] in flagged) == should_flag:
                counts["agree"] += 1
                tallies[0] += 1
            elif should_flag:
                counts["misses"].append(item["id"])
            else:
                counts["false_accusations"].append(item["id"])
    compared = sum(total for _, total in per_repeat.values())
    counts["compared"] = compared
    counts["agreement"] = counts["agree"] / compared if compared else None
    counts["per_repeat"] = {r: f"{a}/{t}" for r, (a, t) in sorted(per_repeat.items())}
    return counts


def verdict(counts: dict[str, Any]) -> str:
    """PASS, FAIL, or INCOMPLETE when any round went unmeasured.

    An unmeasured round is not a pass: dropping it would let the judge clear the bar
    on the reports it happened to grade.
    """
    if counts["not_measured"] or counts["agreement"] is None:
        return "INCOMPLETE"
    return "PASS" if counts["agreement"] >= PASS_AGREEMENT else "FAIL"


def summary(result: dict[str, Any], reports: Sequence[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for metric, spec in METRICS.items():
        if not any(rnd["metric"] == metric for rnd in result["rounds"]):
            continue
        counts = tally(result, reports, metric)
        shown = "n/a" if counts["agreement"] is None else f"{counts['agreement']:.0%}"
        lines.append(
            f"{metric}: agreement {counts['agree']}/{counts['compared']} ({shown}); "
            f"per repeat {counts['per_repeat']}; {counts['not_measured']} round(s) not measured"
        )
        lines.append(
            f"  false accusations (flagged, labelled not {spec['flag_means']}): "
            f"{len(counts['false_accusations'])} {sorted(counts['false_accusations'])}"
        )
        lines.append(
            f"  misses (labelled {spec['flag_means']}, not flagged): "
            f"{len(counts['misses'])} {sorted(counts['misses'])}"
        )
        lines.append(f"  flags matching no labelled item: {len(counts['unmatched'])}")
        lines.extend(f"    - {flag[:110]}" for flag in counts["unmatched"])
        if metric == "claims_are_cited":
            lines.append(f"  verdict at {PASS_AGREEMENT:.0%}: {verdict(counts)}")
    return lines
