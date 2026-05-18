"""Regression tests for hybrid registry validation (Bug 5, 2026-05-15).

Skillorchestra's SWE codepath gates on `method_cfg.swe_use_agent_loop`.
Without that flag the cell silently falls back to a one-shot cloud call
on SWE-bench tasks, which is almost never the intent. The registry
loader (`load_registry`) now validates every skillorchestra SWE cell has
the flag set and raises ValueError at load time if any are missing.

Tests:
1. Real registries on disk pass validation (no missing flags today).
2. A synthetic registry with the flag missing fails with a clear error
   naming the offending cell.
3. The same synthetic cell with the flag set loads cleanly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openjarvis.agents.hybrid.runner import (
    DEFAULT_REGISTRY_DIR,
    _SWE_BENCHES,
    load_registry,
)


# ---------------------------------------------------------------------------
# 1. Real on-disk registries: every skillorchestra SWE cell must already
#    carry `swe_use_agent_loop = true`. This guards against future edits
#    that drop the flag (the exact failure mode of Bug 5).
# ---------------------------------------------------------------------------

def test_skillorchestra_swe_has_loop_flag() -> None:
    cells = load_registry(DEFAULT_REGISTRY_DIR)
    swe_cells = {
        name: cell
        for name, cell in cells.items()
        if cell.get("method") == "skillorchestra"
        and cell.get("bench") in _SWE_BENCHES
    }
    assert swe_cells, (
        "expected at least one skillorchestra SWE cell in the bundled "
        "registry; if you deleted them all, drop this test too."
    )
    missing = [
        name
        for name, cell in swe_cells.items()
        if not bool((cell.get("method_cfg") or {}).get("swe_use_agent_loop"))
    ]
    assert not missing, (
        f"skillorchestra SWE cells missing swe_use_agent_loop: {missing}"
    )


# ---------------------------------------------------------------------------
# 2/3. Synthetic registry: round-trip a single SWE cell with and without
#      the flag to confirm the validator fires (and only fires) at the
#      right time.
# ---------------------------------------------------------------------------

_BAD_TOML = """\
[cells.skillorchestra-fake-swe-3]
method = "skillorchestra"
bench  = "swebench-verified"
n      = 3
local  = { model = "Qwen/Qwen3.5-27B-FP8", endpoint = "http://localhost:8001/v1" }
cloud  = { model = "claude-opus-4-7",      endpoint = "anthropic" }
method_cfg = {}
"""

_GOOD_TOML = """\
[cells.skillorchestra-fake-swe-3]
method = "skillorchestra"
bench  = "swebench-verified"
n      = 3
local  = { model = "Qwen/Qwen3.5-27B-FP8", endpoint = "http://localhost:8001/v1" }
cloud  = { model = "claude-opus-4-7",      endpoint = "anthropic" }
method_cfg = { swe_use_agent_loop = true, swe_max_turns = 10 }
"""


def test_registry_loader_rejects_missing_flag(tmp_path: Path) -> None:
    (tmp_path / "fake.toml").write_text(_BAD_TOML)
    with pytest.raises(ValueError) as ei:
        load_registry(tmp_path)
    msg = str(ei.value)
    assert "skillorchestra-fake-swe-3" in msg, msg
    assert "swe_use_agent_loop" in msg, msg


def test_registry_loader_accepts_with_flag(tmp_path: Path) -> None:
    (tmp_path / "fake.toml").write_text(_GOOD_TOML)
    cells = load_registry(tmp_path)
    assert "skillorchestra-fake-swe-3" in cells
    mcfg = cells["skillorchestra-fake-swe-3"]["method_cfg"]
    assert mcfg["swe_use_agent_loop"] is True


# ---------------------------------------------------------------------------
# Sanity: non-SWE benches and non-skillorchestra methods are unaffected.
# ---------------------------------------------------------------------------

_GAIA_NO_FLAG = """\
[cells.skillorchestra-fake-gaia-3]
method = "skillorchestra"
bench  = "gaia"
n      = 3
local  = { model = "Qwen/Qwen3.5-27B-FP8", endpoint = "http://localhost:8001/v1" }
cloud  = { model = "claude-opus-4-7",      endpoint = "anthropic" }
method_cfg = {}
"""


def test_validator_ignores_non_swe_skillorchestra(tmp_path: Path) -> None:
    (tmp_path / "fake.toml").write_text(_GAIA_NO_FLAG)
    cells = load_registry(tmp_path)
    assert "skillorchestra-fake-gaia-3" in cells
