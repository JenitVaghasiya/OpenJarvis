"""Backfill `tool_calls_total` into existing hybrid n=100 summaries.

The `tool_calls_total` field was added to the canonical row-write path in
commit `7f432453` (2026-05-18). Earlier n=100 cells don't have it in
either `results.jsonl` rows or `summary.json`. Rerunning is too expensive,
but per-task log files at
``experiments/hybrid/runs/<cell>/logs/<task_id>.json`` preserve enough
state to derive the count.

Per-paradigm derivation (matches the in-process counter installed by
commit `7f432453`):

* ``baseline_cloud`` (cells named ``cloud-only-*``):
  - SWE: ``metadata.turns`` (== bash invocations + final patch turn).
  - GAIA one-shot: 0 (no tools).
  - GAIA opus47 with web_search: ``traces.n_web_searches`` if present,
    else 0 (older cells pre-date the wiring).
* ``advisors``:
  - SWE: ``metadata.turns - 1`` (subtract the advisor critique pass).
  - GAIA: ``metadata.turns - 2`` is NOT meaningful here — old cells did
    no web_search and the executor passes were one-shot, so 0.
* ``minions``:
  - GAIA: ``traces.prefetch.n_searches`` (only the prefetch hits tools;
    supervisor↔worker protocol is text-only).
  - SWE: ``metadata.turns - 1`` (subtract supervisor turn; worker runs
    the bash agent loop).
* ``skillorchestra`` (old single-file version):
  - GAIA: 0 (one-shot routed call).
  - SWE: count ``skillorch_(local|cloud)_turn`` events in the log
    (router's ``anthropic`` event is the routing call, not a tool turn).

The script writes the derived total into ``summary.json`` under the same
``tool_calls_total`` key and preserves all other keys. Rows that were
already populated get re-confirmed (idempotent — no overwrite unless
``--force``). Cells whose logs are missing (no `logs/` dir) get skipped
with a warning.

Usage::

    .venv/bin/python scripts/ablation/backfill_tool_calls.py
    .venv/bin/python scripts/ablation/backfill_tool_calls.py --dry-run
    .venv/bin/python scripts/ablation/backfill_tool_calls.py --force
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HYBRID_RUNS = Path(
    os.environ.get(
        "HYBRID_RUNS_DIR",
        Path(
            os.environ.get(
                "OPENJARVIS_HYBRID_EXPERIMENTS_DIR",
                Path.home() / ".openjarvis" / "experiments" / "hybrid",
            )
        )
        / "runs",
    )
)


def _load_log(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def _count_event_kind(log: Dict[str, Any], kind_suffix: str) -> int:
    """Count events whose ``kind`` ends with ``kind_suffix``."""
    n = 0
    for ev in log.get("events", []) or []:
        if isinstance(ev, dict) and str(ev.get("kind", "")).endswith(kind_suffix):
            n += 1
    return n


def _derive_one(cell_name: str, log: Dict[str, Any]) -> Optional[int]:
    """Derive tool_calls for a single task log. ``None`` if undetermined."""
    md = log.get("metadata", {}) or {}
    tr = (md.get("traces") or {}) if isinstance(md, dict) else {}
    turns = int(md.get("turns") or 0)

    if cell_name.startswith("cloud-only-"):
        # baseline_cloud
        if tr.get("mode") in ("anthropic_agent_loop",):
            return int(tr.get("n_web_searches") or 0)
        if tr.get("mode") == "one_shot":
            return 0
        if tr.get("backbone") == "cloud":
            # SWE branch: turns == LLM turns; matches the post-7f432453
            # convention (`tool_calls = int(out["turns"])`).
            return turns
        # GAIA one-shot fall-through
        if tr.get("is_swe") is False:
            return 0
        return turns if turns > 0 else 0

    if cell_name.startswith("advisors-"):
        if tr.get("swe_mode"):
            # initial_out["turns"] + 1 (advisor) + final_out["turns"] = turns
            # post-7f432453 stores tool_calls = initial+final = turns - 1.
            return max(0, turns - 1)
        # GAIA: newer cells (post commit 24da67d2) ran web_search and
        # logged ``n_web_searches`` in traces. Older cells did neither.
        if tr.get("web_search_enabled"):
            return int(tr.get("n_web_searches") or 0)
        return 0

    if cell_name.startswith("minions-"):
        if tr.get("swe_mode"):
            # 1 (supervisor) + worker bash-agent turns = turns.
            return max(0, turns - 1)
        # GAIA: prefetch n_searches is the only tool surface.
        prefetch = tr.get("prefetch") or {}
        return int(prefetch.get("n_searches") or 0)

    if cell_name.startswith("skillorchestra-"):
        # Old single-file version routes via `chosen_agent`; SWE goes
        # through run_swe_agent_loop with trace_prefix=skillorch_(local|cloud).
        if "chosen_agent" in tr:
            # Count `*_turn` events specific to the SWE loop. The
            # ``skillorchestra_route`` and bare ``anthropic`` events are
            # the routing call, NOT a tool turn.
            n_local = _count_event_kind(log, "skillorch_local_turn")
            n_cloud = _count_event_kind(log, "skillorch_cloud_turn")
            n = n_local + n_cloud
            if n > 0:
                return n
            # GAIA path: one-shot routed call, no tools.
            return 0

    return None


def _derive_cell(cell_dir: Path) -> Tuple[int, int, List[str], str]:
    """Sum tool_calls across all per-task evidence in ``cell_dir/``.

    Prefers the canonical ``results.jsonl`` ``tool_calls`` field
    (post-7f432453 wiring writes it directly). Falls back to per-task
    log derivation for pre-wiring cells.

    Returns ``(total, n_tasks_counted, warnings, source)`` where ``source``
    is ``"results.jsonl"``, ``"logs"``, or ``"none"``.
    """
    cell = cell_dir.name
    warnings: List[str] = []

    # Source 1: results.jsonl with tool_calls field (canonical).
    res_p = cell_dir / "results.jsonl"
    rows: List[Dict[str, Any]] = []
    if res_p.exists():
        try:
            rows = [json.loads(line) for line in res_p.read_text().splitlines() if line.strip()]
        except Exception as e:  # noqa: BLE001
            warnings.append(f"{cell}: failed to parse results.jsonl ({e})")
            rows = []
        if rows and all("tool_calls" in r for r in rows):
            return sum(int(r.get("tool_calls") or 0) for r in rows), len(rows), warnings, "results.jsonl"

    # Source 2: per-task log files (pre-wiring fallback).
    logs_dir = cell_dir / "logs"
    have_logs = logs_dir.is_dir() and any(logs_dir.glob("*.json"))
    if have_logs:
        total = 0
        n = 0
        for log_path in sorted(logs_dir.glob("*.json")):
            log = _load_log(log_path)
            if log is None:
                warnings.append(f"{cell}: failed to parse {log_path.name}")
                continue
            v = _derive_one(cell, log)
            if v is None:
                warnings.append(
                    f"{cell}: undetermined for {log_path.stem} "
                    f"(traces.keys={sorted((log.get('metadata') or {}).get('traces', {}).keys())})"
                )
                continue
            total += v
            n += 1
        if n > 0:
            return total, n, warnings, "logs"

    # Source 3: legacy results.jsonl `traces.n_searches` (pre-mini-SWE-agent
    # cells stored the GAIA web_search count under this key, before the
    # tool_calls schema landed). Only safe for cells that actually used
    # web_search (advisors/cloud GAIA cells); for SWE the field is
    # spurious. Heuristic: name contains "gaia" => use it.
    if rows and "gaia" in cell.lower() and any(
        "n_searches" in (r.get("traces") or {}) for r in rows
    ):
        total = sum(int((r.get("traces") or {}).get("n_searches") or 0) for r in rows)
        return total, len(rows), warnings, "results.jsonl(legacy n_searches)"

    return 0, 0, warnings + [f"{cell}: no derivable evidence"], "none"


def _load_summary(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def _save_summary(p: Path, d: Dict[str, Any]) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(d, indent=2))
    tmp.replace(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Don't write summaries.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing tool_calls_total too.")
    ap.add_argument("--glob", default="*-n100",
                    help="Cell directory glob (default: *-n100).")
    ap.add_argument("--runs-dir", default=str(HYBRID_RUNS),
                    help="hybrid runs root dir.")
    args = ap.parse_args()

    root = Path(args.runs_dir)
    rows: List[Tuple[str, Any, Any, str, int]] = []
    all_warnings: List[str] = []

    for cell_dir in sorted(root.glob(args.glob)):
        if not cell_dir.is_dir():
            continue
        summ_p = cell_dir / "summary.json"
        if not summ_p.exists():
            continue
        summ = _load_summary(summ_p)
        if summ is None:
            all_warnings.append(f"{cell_dir.name}: failed to parse summary.json")
            continue
        before = summ.get("tool_calls_total", None)
        derived, n_tasks, warnings, source = _derive_cell(cell_dir)
        all_warnings.extend(warnings)

        action = ""
        if before is None:
            if n_tasks == 0:
                action = "skip (no derivable evidence)"
            else:
                action = f"backfill -> {derived} (from {n_tasks} {source})"
                if not args.dry_run:
                    summ["tool_calls_total"] = int(derived)
                    _save_summary(summ_p, summ)
        elif args.force and isinstance(before, int) and before != derived and n_tasks > 0:
            action = f"force-update {before} -> {derived} (from {source})"
            if not args.dry_run:
                summ["tool_calls_total"] = int(derived)
                _save_summary(summ_p, summ)
        elif n_tasks > 0 and isinstance(before, int) and before != derived:
            action = f"mismatch (have {before}, {source} sum {derived}) — keep, use --force to overwrite"
        else:
            action = f"already has {before}"

        rows.append((cell_dir.name, before, derived, action, n_tasks))

    # Print table
    col1 = max(len(r[0]) for r in rows) if rows else 10
    print(f"{'cell'.ljust(col1)}  {'before':>8}  {'logs_n':>6}  {'derived':>8}  action")
    print("-" * (col1 + 40))
    for name, before, derived, action, n_tasks in rows:
        b = "—" if before is None else str(before)
        print(f"{name.ljust(col1)}  {b:>8}  {n_tasks:>6}  {derived:>8}  {action}")

    if all_warnings:
        print(f"\n{len(all_warnings)} warnings:")
        for w in all_warnings[:50]:
            print(f"  ! {w}")
        if len(all_warnings) > 50:
            print(f"  ... and {len(all_warnings) - 50} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
