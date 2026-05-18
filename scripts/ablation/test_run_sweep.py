"""Tests for the ablation orchestrator's first-5 fail-fast kill heuristic.

Background — the bug:
    The old kill condition fired on 5/5 score=0 regardless of answer content.
    GPT-5-mini on GAIA scored 0/5 with REAL wrong answers ("Herbert Simon" vs
    ref "Claude Shannon"), got killed by the orchestrator, but the cell would
    have ended at acc=0.18 over 100 tasks. The fix tightens the 5-zero kill
    to require empty answers (= wiring broken), not merely wrong ones.

Run:
    .venv/bin/python -m pytest scripts/ablation/test_run_sweep.py -v
"""
from __future__ import annotations

from scripts.ablation.run_sweep import _check_first5_kill


def _errored_row(msg: str = "boom") -> dict:
    return {"task_id": "t", "error": msg, "score": None, "answer": ""}


def _scored_row(score: float, answer: str = "") -> dict:
    return {
        "task_id": "t",
        "error": None,
        "score": {"score": score},
        "answer": answer,
    }


def test_kill_5_errors() -> None:
    """5 errored rows -> killed-5error."""
    rows = [_errored_row(f"err {i}") for i in range(5)]
    assert _check_first5_kill(rows) == "killed-5error"


def test_kill_5_empty_answers() -> None:
    """5 rows with score=0.0 AND empty answer -> killed-5zero (wiring broken)."""
    rows = [_scored_row(0.0, "") for _ in range(5)]
    assert _check_first5_kill(rows) == "killed-5zero"


def test_no_kill_5_wrong_answers() -> None:
    """5 rows with score=0.0 AND non-empty wrong answer -> NO kill.

    REGRESSION TEST for the original bug: GPT-5-mini answering "Herbert Simon"
    where the reference was "Claude Shannon" is legit poor performance, not
    broken wiring. The orchestrator must let it ride.
    """
    rows = [
        _scored_row(0.0, "Herbert Simon"),
        _scored_row(0.0, "Alan Turing"),
        _scored_row(0.0, "John von Neumann"),
        _scored_row(0.0, "Marvin Minsky"),
        _scored_row(0.0, "Donald Knuth"),
    ]
    assert _check_first5_kill(rows) is None


def test_no_kill_mixed() -> None:
    """4 errors + 1 success -> no kill (not all errored, not all scored-zero)."""
    rows = [_errored_row() for _ in range(4)] + [_scored_row(1.0, "right answer")]
    assert _check_first5_kill(rows) is None


def test_no_kill_under_5_rows() -> None:
    """Fewer than 5 rows -> None (not yet evaluable)."""
    rows = [_errored_row() for _ in range(4)]
    assert _check_first5_kill(rows) is None


def test_no_kill_4_empty_plus_1_nonempty() -> None:
    """4 empty + 1 non-empty wrong answer -> no kill (the non-empty saves it)."""
    rows = [_scored_row(0.0, "") for _ in range(4)] + [_scored_row(0.0, "wrong")]
    assert _check_first5_kill(rows) is None


def test_no_kill_whitespace_only_answer_treated_as_empty_but_with_score() -> None:
    """Whitespace-only answer is treated as empty -> killed-5zero fires."""
    rows = [_scored_row(0.0, "   \n\t") for _ in range(5)]
    assert _check_first5_kill(rows) == "killed-5zero"


def test_only_first_5_inspected() -> None:
    """If the first 5 trigger a kill, later rows don't rescue it."""
    rows = [_errored_row() for _ in range(5)] + [_scored_row(1.0, "ok")]
    assert _check_first5_kill(rows) == "killed-5error"
