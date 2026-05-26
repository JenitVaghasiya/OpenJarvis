"""Regression test for the SWE-bench variant bug.

`SWEBenchDataset()` defaulted to the 50-task `verified_mini` variant, but
hybrid n=100 subsets reference task_ids from the full 500-task
`princeton-nlp/SWE-bench_Verified`. The runner now passes
``variant="verified"`` explicitly; this test pins that behavior so the
bug can't silently regress.

Requires HF_TOKEN and a populated HF cache (or network). Cap each test
at 60s; HF download can be slow on first run but the cache should be
warm on the mkt cluster.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from openjarvis.agents.hybrid.runner import DEFAULT_SUBSETS_DIR, _load_swebench_tasks


SUBSET_PATH = Path(
    os.environ.get(
        "HYBRID_SWEBENCH_N100_SUBSET",
        DEFAULT_SUBSETS_DIR / "swebench_verified_n100_seed42.json",
    )
)
KNOWN_FULL_TASK_ID = "pytest-dev__pytest-10081"


def test_loads_full_verified_not_mini() -> None:
    """No max_samples → at least 500 records (full Verified set, not 50)."""
    tasks = _load_swebench_tasks(n=None)
    assert len(tasks) >= 500, (
        f"expected >=500 from full SWE-bench_Verified, got {len(tasks)}; "
        "runner may have regressed to the 50-task verified_mini variant"
    )


def test_n100_returns_100_records() -> None:
    """n=100 against the full variant → 100 records (not capped at 50)."""
    tasks = _load_swebench_tasks(n=100)
    assert len(tasks) == 100, (
        f"expected 100 records, got {len(tasks)}; if 50 the runner is "
        "loading verified_mini again"
    )


def test_known_full_task_id_present() -> None:
    """`pytest-dev__pytest-10081` (from subset n=100 seed=42) must load.

    This is the exact bug we hit: 94/100 subset ids were missing because
    the dataset only had 50 mini-variant rows.
    """
    if SUBSET_PATH.exists():
        data = json.loads(SUBSET_PATH.read_text())
        ids = data["task_ids"] if isinstance(data, dict) else list(data)
        assert KNOWN_FULL_TASK_ID in ids, (
            f"sanity: {KNOWN_FULL_TASK_ID} should be in {SUBSET_PATH.name}"
        )

    tasks = _load_swebench_tasks(n=None)
    loaded_ids = {t["task_id"] for t in tasks}
    assert KNOWN_FULL_TASK_ID in loaded_ids, (
        f"{KNOWN_FULL_TASK_ID} missing from loaded SWE-bench records; "
        "runner is probably back on verified_mini"
    )
