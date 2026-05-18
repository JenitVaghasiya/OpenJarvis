"""Regression tests for the swebench Modal cgroup-v2 patch.

See ``src/openjarvis/evals/scorers/swebench_harness.py`` for context. The
upstream swebench ``ModalSandboxRuntime.__init__`` writes to a cgroup-v1
path (``/sys/fs/cgroup/cpu/cpu.shares``) that doesn't exist in Modal v1.4+
sandboxes (cgroup-v2). Without the patch every grade silently scores 0.

Cheap tests (idempotency, garbage-patch handling) run by default. The
real Modal grade is marked ``@pytest.mark.modal`` and skipped unless
``SWEBENCH_RUN_MODAL_TESTS=1`` is set and a Modal token is configured.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from openjarvis.evals.core.types import EvalRecord
from openjarvis.evals.scorers.swebench_harness import (
    _CGROUP_SOURCE_SENTINEL,
    _apply_patches_once,
    _run_harness,
    SWEBenchHarnessScorer,
)


# Known-good patch from cloud-only-opus47-swe-n100/results.jsonl row 0
# (astropy__astropy-14539, scored 1.0 with the harness).
ASTROPY_14539_PATCH = (
    'diff --git a/astropy/io/fits/diff.py b/astropy/io/fits/diff.py\n'
    '--- a/astropy/io/fits/diff.py\n'
    '+++ b/astropy/io/fits/diff.py\n'
    '@@ -1449,7 +1449,7 @@ class TableDataDiff(_BaseDiff):\n'
    '                 arrb.dtype, np.floating\n'
    '             ):\n'
    '                 diffs = where_not_allclose(arra, arrb, rtol=self.rtol, atol=self.atol)\n'
    '-            elif "P" in col.format:\n'
    '+            elif "P" in col.format or "Q" in col.format:\n'
    '                 diffs = (\n'
    '                     [\n'
    '                         idx\n'
)


def _swebench_modal_src() -> Path:
    from swebench.harness.modal_eval import run_evaluation_modal as _m

    return Path(_m.__file__)


def test_patches_idempotent():
    """Applying patches twice must leave exactly one sentinel in the source."""
    _apply_patches_once()
    _apply_patches_once()  # second call is a no-op when sentinel is present
    src = _swebench_modal_src().read_text()
    count = src.count(_CGROUP_SOURCE_SENTINEL)
    assert count == 1, f"Expected sentinel exactly once, got {count}"
    # Confirm the broken bare write is gone and the try/except is in place.
    assert "try:" in src and "except FileNotFoundError:" in src


def _modal_creds_present() -> bool:
    if os.environ.get("SWEBENCH_RUN_MODAL_TESTS", "0") != "1":
        return False
    if os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"):
        return True
    # Modal also stores creds in ~/.modal.toml.
    return Path("~/.modal.toml").expanduser().exists()


@pytest.mark.modal
@pytest.mark.slow
@pytest.mark.skipif(
    not _modal_creds_present(),
    reason="set SWEBENCH_RUN_MODAL_TESTS=1 and configure Modal token to run",
)
def test_grade_known_good_patch():
    """End-to-end: known-good astropy fix must grade as resolved on Modal."""
    result = _run_harness("astropy__astropy-14539", ASTROPY_14539_PATCH, 1800)
    assert result["success"] is True, result.get("details", {})
    assert result["score"] == 1.0


@pytest.mark.modal
@pytest.mark.slow
@pytest.mark.skipif(
    not _modal_creds_present(),
    reason="set SWEBENCH_RUN_MODAL_TESTS=1 and configure Modal token to run",
)
def test_grade_bad_patch():
    """Garbage patch must score 0.0 cleanly — no harness crash, no exception."""
    scorer = SWEBenchHarnessScorer(timeout_s=600)
    record = EvalRecord(
        record_id="astropy__astropy-14539",
        problem="",
        reference="",
        category="swebench",
        metadata={"instance_id": "astropy__astropy-14539"},
    )
    # Wrap in a code fence so extract_patch returns something, exercising
    # the harness end-to-end rather than the early "no_patch_extracted" path.
    answer = "```diff\nnot a real diff\n```"
    is_correct, details = scorer.score(record, answer)
    assert is_correct is False
    # Either the harness rejected the patch (report present, instance not in
    # resolved_ids) or it bailed early with reason="no_report". Both are fine
    # — what we're checking is that we didn't raise.
    assert "patch" in details
