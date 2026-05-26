#!/usr/bin/env python3
"""Orchestrator for the n=100 hybrid-paradigm ablation sweep.

Drives a batch of ``openjarvis.agents.hybrid.runner`` invocations with:

  * per-tier concurrency caps (anthropic-opus / anthropic-haiku / openai / gemini)
  * 30s monitor thread per cell, 120/300s heartbeat in main
  * first-5 fail-fast (5/5 errored OR 5/5 scored-zero → kill the cell)
  * exit-code + summary.json driven retry loop (resume mode, runner-handled)
  * incremental atomic updates of ``results-table.md`` (auto-section only)
  * Ctrl-C → SIGTERM the whole process group of each child cleanly
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_PY = Path(os.environ.get("OPENJARVIS_VENV_PY", REPO_ROOT / ".venv" / "bin" / "python"))
EXPERIMENTS_DIR = Path(
    os.environ.get(
        "OPENJARVIS_HYBRID_EXPERIMENTS_DIR",
        Path.home() / ".openjarvis" / "experiments" / "hybrid",
    )
)
RUNS_DIR = EXPERIMENTS_DIR / "runs"
RESULTS_TABLE = EXPERIMENTS_DIR / "docs" / "results-table.md"
LOG_DIR = Path(os.environ.get("HYBRID_SWEEP_LOG_DIR", "/tmp/hybrid-sweep-logs"))
AUTO_SECTION_HEADER = "## n=100 sweep — auto-generated"

# ---------------------------------------------------------------------------
# Batch definitions (cell names exactly as in the registries)
# ---------------------------------------------------------------------------

WAVE_A = [
    "skillorchestra-qwen-haiku45-gaia-n100",
    "skillorchestra-qwen-haiku45-swe-n100",
    "cloud-only-haiku45-gaia-n100",
    "cloud-only-haiku45-swe-n100",
    "skillorchestra-qwen-gpt5mini-gaia-n100",
    "skillorchestra-qwen-gpt5mini-swe-n100",
    "cloud-only-gpt5mini-gaia-n100",
    "cloud-only-gpt5mini-swe-n100",
    "skillorchestra-qwen-gemini25flash-gaia-n100",
    "skillorchestra-qwen-gemini25flash-swe-n100",
    "cloud-only-gemini25flash-gaia-n100",
    "cloud-only-gemini25flash-swe-n100",
]

WAVE_B = [
    "skillorchestra-qwen-gpt5-gaia-n100",
    "skillorchestra-qwen-gpt5-swe-n100",
    "cloud-only-gpt5-gaia-n100",
    "cloud-only-gpt5-swe-n100",
    "skillorchestra-qwen-gemini25pro-gaia-n100",
    "skillorchestra-qwen-gemini25pro-swe-n100",
    "cloud-only-gemini25pro-gaia-n100",
    "cloud-only-gemini25pro-swe-n100",
]

WAVE_C = [
    "skillorchestra-qwen-opus47-gaia-n100",
    "skillorchestra-qwen-opus47-swe-n100",
    "cloud-only-opus47-gaia-n100",
    "cloud-only-opus47-swe-n100",
]

SMOKE = [
    "mini-swe-agent-swebenchverified-opus-3",
    "minions-swe-agent-swebenchverified-qwen27b-opus-3",
    "conductor-swe-agent-swebenchverified-opus-3",
    "advisors-swe-agent-swebenchverified-qwen9b-opus-3",
    "skillorchestra-swe-agent-swebenchverified-qwen27b-opus-3",
    "toolorchestra-swe-agent-swebenchverified-qwen27b-opus-3",
    "archon-swe-agent-swebenchverified-qwen27b-opus-3",
]

ALL = WAVE_A + WAVE_B + WAVE_C

BATCHES: Dict[str, List[str]] = {
    "smoke": SMOKE,
    "wave-a": WAVE_A,
    "wave-b": WAVE_B,
    "wave-c": WAVE_C,
    "all": ALL,
}

# ---------------------------------------------------------------------------
# Tiers / concurrency
# ---------------------------------------------------------------------------

TIER_CAPS: Dict[str, int] = {
    "anthropic-opus": 2,
    "anthropic-haiku": 4,
    "openai": 4,
    "gemini": 4,
}


def tier_of(cell: str) -> str:
    """Map a cell name to a tier. Order matters: gpt5mini before gpt5."""
    c = cell.lower()
    if "opus47" in c or "-opus-" in c:
        return "anthropic-opus"
    if "haiku45" in c:
        return "anthropic-haiku"
    if "gpt5mini" in c:
        return "openai"
    if "gpt5" in c:
        return "openai"
    if "gemini25pro" in c or "gemini25flash" in c:
        return "gemini"
    return "anthropic-opus"  # safest cap for unknowns


# ---------------------------------------------------------------------------
# Cell-name parsing (for the table)
# ---------------------------------------------------------------------------

CLOUD_LABELS = {
    "opus47": "claude-opus-4-7",
    "haiku45": "claude-haiku-4-5",
    "gpt5mini": "gpt-5-mini",
    "gpt5": "gpt-5",
    "gemini25pro": "gemini-2.5-pro",
    "gemini25flash": "gemini-2.5-flash",
}

BENCH_LABELS = {"gaia": "GAIA", "swe": "SWE-bench"}


def parse_cell(cell: str) -> Dict[str, str]:
    """Pull (paradigm, local, cloud, bench) out of an ablation cell name.

    Two naming conventions are supported:
      * ``cloud-only-<cloud>-<bench>-n100``                (no local)
      * ``<paradigm>-<local>-<cloud>-<bench>-n100``        (everything else,
        e.g. ``skillorchestra-qwen-...``, ``advisors-qwen27b-...``)

    We anchor on the ``-n100`` suffix and walk left to recover ``bench`` and
    ``cloud`` so paradigm/local prefixes can vary without breaking parsing.
    """
    c = cell
    if c.startswith("cloud-only-"):
        paradigm = "cloud-only"
        local = ""
        rest = c[len("cloud-only-"):]
        rparts = rest.split("-")
        # rest is "<cloud>-<bench>-n100" (or "<cloud>-<bench>" historically)
        cloud_key = rparts[0] if rparts else ""
        bench_key = rparts[1] if len(rparts) > 1 else ""
    else:
        parts = c.split("-")
        # Strip a trailing "n100"-style scale token so it can't be mistaken
        # for the bench.
        if parts and parts[-1].startswith("n") and parts[-1][1:].isdigit():
            parts = parts[:-1]
        # Expect at least <paradigm>-<local>-<cloud>-<bench>.
        paradigm = parts[0] if parts else ""
        local = parts[1] if len(parts) > 1 else ""
        cloud_key = parts[2] if len(parts) > 2 else ""
        bench_key = parts[3] if len(parts) > 3 else ""
    return {
        "paradigm": paradigm,
        "local": local,
        "cloud_key": cloud_key,
        "cloud": CLOUD_LABELS.get(cloud_key, cloud_key),
        "bench_key": bench_key,
        "bench": BENCH_LABELS.get(bench_key, bench_key),
    }


# ---------------------------------------------------------------------------
# results.jsonl scanning
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    try:
        text = path.read_text()
    except Exception:
        return []
    out: List[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            # Partial / corrupt write — skip
            continue
    return out


def _row_is_errored(row: dict) -> bool:
    """Truth: runner writes top-level ``error`` (str or None) + ``score`` (dict or None).

    A row is errored if ``error`` is non-null OR ``score`` is missing/null.
    """
    if row.get("error"):
        return True
    score = row.get("score")
    if score is None:
        return True
    return False


def _row_score(row: dict) -> Optional[float]:
    score = row.get("score")
    if not isinstance(score, dict):
        return None
    v = score.get("score")
    if not isinstance(v, (int, float)):
        return None
    return float(v)


def _row_last_error(row: dict) -> str:
    err = row.get("error")
    if isinstance(err, str) and err:
        # First line only — runner tucks traceback in there.
        return err.splitlines()[0][:160]
    return ""


def _check_first5_kill(rows: List[dict]) -> Optional[str]:
    """Pure kill-decision for the first-5 fail-fast heuristic.

    Returns:
        - ``"killed-5error"`` if all of the first 5 rows are errored.
        - ``"killed-5zero"`` if all of the first 5 rows scored 0.0 AND
          have empty answers (= wiring broken, not legit poor performance).
          A non-empty wrong answer is legit poor performance — e.g. GAIA's
          first 5 may genuinely stump a weak model — so we do NOT kill.
        - ``None`` if fewer than 5 rows or neither condition holds.

    Threshold rationale: at n=100 on hard benchmarks (e.g. GAIA),
    5 consecutive wrong-but-nonempty answers is plausible — the cell
    can still end at acc≈0.18. Errors and empty-answers, by contrast,
    indicate broken wiring (auth, parser, tool config) that won't recover.
    """
    if len(rows) < 5:
        return None
    first5 = rows[:5]
    if all(_row_is_errored(r) for r in first5):
        return "killed-5error"
    all_zero_empty = all(
        (not _row_is_errored(r))
        and (_row_score(r) == 0.0)
        and (not (r.get("answer") or "").strip())
        for r in first5
    )
    if all_zero_empty:
        return "killed-5zero"
    return None


# ---------------------------------------------------------------------------
# Cell state
# ---------------------------------------------------------------------------


@dataclass
class CellState:
    name: str
    tier: str
    expected_n: int = 100
    round: int = 0
    status: str = "queued"  # queued | running | done | killed-5error | killed-5zero | blocked-bug | error
    proc: Optional[subprocess.Popen] = None
    log_path: Optional[Path] = None
    monitor_thread: Optional[threading.Thread] = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    first5_checked: bool = False
    kill_reason: str = ""
    last_error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    last_done: int = 0
    last_acc: float = 0.0
    last_err_count: int = 0
    stall_ticks: int = 0  # consecutive 30s ticks where done didn't increase
    final_summary: Optional[dict] = None
    next_hb_at: float = 0.0  # earliest time we should emit a heartbeat line for this cell

    @property
    def out_dir(self) -> Path:
        return RUNS_DIR / self.name

    @property
    def results_path(self) -> Path:
        return self.out_dir / "results.jsonl"

    @property
    def summary_path(self) -> Path:
        return self.out_dir / "summary.json"

    @property
    def lock_path(self) -> Path:
        return self.out_dir / ".lock"


# ---------------------------------------------------------------------------
# Lock probe (read-only — never grabs)
# ---------------------------------------------------------------------------


def lock_is_held(lock_path: Path) -> Tuple[bool, str]:
    """Return ``(held, holder_pid_str)``. Non-destructive: tries an LOCK_EX|NB,
    immediately releases. If the lock file doesn't exist, treat as free."""
    if not lock_path.exists():
        return False, ""
    try:
        f = lock_path.open("a+")
    except Exception:
        return False, ""
    try:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return False, ""
        except BlockingIOError:
            f.seek(0)
            pid = (f.read() or "?").strip() or "?"
            return True, pid
    finally:
        try:
            f.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Table updater (atomic, idempotent, hands-off other sections)
# ---------------------------------------------------------------------------

_TABLE_LOCK = threading.Lock()


def _format_table_row(cell_name: str, summary: dict) -> str:
    """One pipe-table row for the auto-generated section."""
    acc = float(summary.get("accuracy", 0.0) or 0.0)
    cost = float(summary.get("cost_usd_total", 0.0) or 0.0)
    wall = float(summary.get("wall_time_s", 0.0) or 0.0)
    n_done = int(summary.get("n_done", 0) or 0)
    n_target = int(summary.get("n_target", 0) or 0)
    tokens_local = int(summary.get("tokens_local_total", 0) or 0)
    tokens_cloud = int(summary.get("tokens_cloud_total", 0) or 0)
    bench_key = parse_cell(cell_name).get("bench_key", "")
    bench = BENCH_LABELS.get(bench_key, bench_key or "?")
    # tools_per_task: not in summary today — derive from results.jsonl if we have time.
    # For now leave it as "—" so the column stays.
    tools_str = "—"
    return (
        f"| `{cell_name}` | {bench} | {acc:.3f} · ${cost:.2f} | "
        f"{wall:.0f}s | tools={tools_str} | tokens_local={tokens_local} | "
        f"tokens_cloud={tokens_cloud} | {n_done}/{n_target} |"
    )


_AUTO_HEADER_LINE = AUTO_SECTION_HEADER  # exact match key
_TABLE_HEADER = (
    "| cell | bench | acc · $cost | wall | tools/task | tokens_local | "
    "tokens_cloud | done/target |\n"
    "|---|---|---|---|---|---|---|---|"
)


def update_table(cell_name: str, summary: dict) -> None:
    """Splice (or insert) the row for ``cell_name`` under the auto section.

    Atomic: write to ``.tmp``, then ``os.replace``. Idempotent: same cell
    overwrites its own row in place.
    """
    with _TABLE_LOCK:
        RESULTS_TABLE.parent.mkdir(parents=True, exist_ok=True)
        if RESULTS_TABLE.exists():
            text = RESULTS_TABLE.read_text()
        else:
            text = "# Results table\n"

        new_row = _format_table_row(cell_name, summary)
        cell_key = f"| `{cell_name}` |"

        lines = text.splitlines()

        # Find the auto-section header.
        try:
            hdr_idx = next(i for i, ln in enumerate(lines) if ln.strip() == _AUTO_HEADER_LINE)
        except StopIteration:
            # Append a fresh section.
            if lines and lines[-1].strip():
                lines.append("")
            lines.append(_AUTO_HEADER_LINE)
            lines.append("")
            lines.append(_TABLE_HEADER)
            lines.append(new_row)
            _atomic_write_lines(RESULTS_TABLE, lines)
            return

        # Find the end of the auto-section: next H2 / H1 OR EOF.
        end_idx = len(lines)
        for i in range(hdr_idx + 1, len(lines)):
            ln = lines[i]
            if ln.startswith("## ") and ln.strip() != _AUTO_HEADER_LINE:
                end_idx = i
                break
            if ln.startswith("# "):
                end_idx = i
                break

        section = lines[hdr_idx:end_idx]

        # Make sure section has the header line(s).
        has_header = any("| cell | bench |" in ln for ln in section)
        if not has_header:
            # Re-seed: keep title, drop everything else inside section.
            section = [_AUTO_HEADER_LINE, "", _TABLE_HEADER]

        # Drop any prior row for this cell.
        section = [ln for ln in section if not ln.startswith(cell_key)]

        # Append new row.
        section.append(new_row)

        new_lines = lines[:hdr_idx] + section + lines[end_idx:]
        _atomic_write_lines(RESULTS_TABLE, new_lines)


def _atomic_write_lines(path: Path, lines: List[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines).rstrip() + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Monitor thread
# ---------------------------------------------------------------------------


def monitor_cell(state: CellState) -> None:
    """30s polling on results.jsonl. First-5 fail-fast. Sets ``last_*`` on state."""
    expected = state.expected_n
    while not state.stop_event.is_set():
        rows = _read_jsonl(state.results_path)
        done = len(rows)
        errs = sum(1 for r in rows if _row_is_errored(r))
        scores = [_row_score(r) for r in rows if not _row_is_errored(r)]
        scores = [s for s in scores if s is not None]
        acc = (sum(scores) / len(scores)) if scores else 0.0

        # stall detector (informational; we don't auto-kill on this)
        if done == state.last_done:
            state.stall_ticks += 1
        else:
            state.stall_ticks = 0

        state.last_done = done
        state.last_acc = acc
        state.last_err_count = errs
        # last_error: take from the latest errored row, if any.
        for r in reversed(rows):
            if _row_is_errored(r):
                err = _row_last_error(r)
                if err:
                    state.last_error = err
                break

        # First-5 fail-fast — runs at most once per cell, on the same 5 rows.
        # See _check_first5_kill for the kill-decision rules and rationale.
        if (not state.first5_checked) and done >= 5:
            state.first5_checked = True
            first5 = rows[:5]
            kill = _check_first5_kill(first5)
            if kill == "killed-5error":
                state.kill_reason = kill
                _kill_proc(state)
                _log_kill(state, first5, "all 5 errored")
                state.stop_event.set()
                return
            if kill == "killed-5zero":
                state.kill_reason = kill
                _kill_proc(state)
                _log_kill(state, first5, "all 5 scored 0.0 with empty answers")
                state.stop_event.set()
                return

        # Subprocess died?
        if state.proc is not None and state.proc.poll() is not None:
            return

        # Sleep 30s, but wake on stop.
        state.stop_event.wait(30.0)


def _kill_proc(state: CellState) -> None:
    p = state.proc
    if p is None:
        return
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            p.terminate()
        except Exception:
            pass
    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass


def _log_kill(state: CellState, rows: List[dict], reason: str) -> None:
    print(
        f"[KILL] {state.name}: {state.kill_reason} — {reason}",
        flush=True,
    )
    for i, r in enumerate(rows):
        tid = r.get("task_id", "?")
        err = _row_last_error(r) or "(no error string)"
        sc = _row_score(r)
        print(f"  row[{i}] task={tid} score={sc} error={err}", flush=True)


# ---------------------------------------------------------------------------
# Launch / retry loop
# ---------------------------------------------------------------------------


def launch_cell(state: CellState) -> None:
    """Spawn the runner subprocess for this cell (one round).

    Sets ``proc``, ``log_path``, ``monitor_thread``, ``started_at``.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    state.log_path = LOG_DIR / f"{state.name}.r{state.round}.{ts}.log"
    state.stop_event = threading.Event()
    state.first5_checked = False
    state.kill_reason = ""
    state.last_done = 0
    state.stall_ticks = 0
    state.started_at = time.time()

    log_file = state.log_path.open("a")
    cmd = [
        str(VENV_PY),
        "-m",
        "openjarvis.agents.hybrid.runner",
        "--cell",
        state.name,
    ]
    log_file.write(f"# $ {' '.join(cmd)}\n# cwd={REPO_ROOT}\n# round={state.round}\n")
    log_file.flush()
    state.proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # so we can SIGTERM the whole pgrp
    )
    state.status = "running"

    state.monitor_thread = threading.Thread(
        target=monitor_cell, args=(state,), daemon=True, name=f"mon-{state.name}"
    )
    state.monitor_thread.start()


def collect_cell(state: CellState) -> str:
    """Wait for ``state.proc`` to exit, then classify outcome.

    Returns one of:
      ``done``           — clean run, full summary, no errored tasks
      ``retry``          — non-fatal: some errored rows / non-zero exit / partial
      ``killed-5error``  — fail-fast triggered (5/5 errored on first batch)
      ``killed-5zero``   — fail-fast triggered (5/5 scored 0)
    """
    assert state.proc is not None
    rc = state.proc.wait()
    state.stop_event.set()
    if state.monitor_thread is not None:
        state.monitor_thread.join(timeout=5)

    if state.kill_reason in ("killed-5error", "killed-5zero"):
        return state.kill_reason

    # Inspect summary + jsonl
    summary = None
    if state.summary_path.exists():
        try:
            summary = json.loads(state.summary_path.read_text())
        except Exception:
            summary = None

    rows = _read_jsonl(state.results_path)
    err_rows = sum(1 for r in rows if _row_is_errored(r))

    if (
        rc == 0
        and summary is not None
        and int(summary.get("task_count", 0)) == state.expected_n
        and int(summary.get("n_done", 0)) == state.expected_n
        and err_rows == 0
    ):
        state.final_summary = summary
        return "done"

    return "retry"


# ---------------------------------------------------------------------------
# Heartbeat printer (main thread)
# ---------------------------------------------------------------------------


def _fmt_eta(eta_s: float) -> str:
    if eta_s <= 0 or eta_s != eta_s:  # NaN guard
        return "--:--"
    m, s = divmod(int(eta_s), 60)
    return f"{m:02d}:{s:02d}"


def heartbeat_line(state: CellState) -> str:
    elapsed = time.time() - state.started_at if state.started_at else 0.0
    done = state.last_done
    expected = state.expected_n
    if done > 0:
        eta_s = max(0.0, (expected - done) * (elapsed / done))
    else:
        eta_s = 0.0
    status_tag = "OK"
    tail = "OK"
    # Detect issues
    if state.kill_reason:
        status_tag = "ALERT"
        tail = state.kill_reason
    elif state.last_err_count > 0:
        tail = f"err_rows={state.last_err_count} last={state.last_error[:80]}"
    elif state.stall_ticks >= 3:
        status_tag = "ALERT"
        tail = f"STALL ({state.stall_ticks} ticks no progress)"

    return (
        f"[{state.name}] round={state.round} done={done}/{expected} "
        f"acc={state.last_acc:.3f} err={state.last_err_count} "
        f"elapsed={_fmt_eta(elapsed)} eta={_fmt_eta(eta_s)} | "
        f"{status_tag} | {tail}"
    )


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


_SHUTDOWN = threading.Event()


def _install_sigint(states: List[CellState]) -> None:
    def _handler(signum, frame):
        if _SHUTDOWN.is_set():
            return
        _SHUTDOWN.set()
        print(
            "\n[main] caught SIGINT — sending SIGTERM to all running cells.",
            flush=True,
        )
        for s in states:
            if s.proc is not None and s.proc.poll() is None:
                _kill_proc(s)
                s.stop_event.set()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def run_sweep(cells: List[str], max_rounds: int, smoke_n: bool = False) -> int:
    """Drive the sweep. Returns process exit code (0 ok, 1 on any kill/blocked)."""
    # Build initial states
    states: Dict[str, CellState] = {}
    for c in cells:
        expected = 3 if smoke_n else 100
        states[c] = CellState(name=c, tier=tier_of(c), expected_n=expected)

    _install_sigint(list(states.values()))

    # Build per-tier semaphores. We don't use threading.BoundedSemaphore because
    # we drive launches from the main thread one-by-one; we just count active
    # cells per tier and only launch when capacity allows.
    active_per_tier: Dict[str, int] = {t: 0 for t in TIER_CAPS}

    # Skip cells that already have a held lock (refuse to double-launch).
    runnable: List[CellState] = []
    for s in states.values():
        held, pid = lock_is_held(s.lock_path)
        if held:
            print(
                f"[skip] {s.name}: lock held by pid {pid}; refusing to launch a second instance.",
                flush=True,
            )
            s.status = "skipped-locked"
        else:
            # If a complete summary already exists with full coverage, skip.
            if s.summary_path.exists():
                try:
                    smry = json.loads(s.summary_path.read_text())
                    if (
                        int(smry.get("n_done", 0)) == s.expected_n
                        and int(smry.get("task_count", 0)) == s.expected_n
                        and int(smry.get("n_err", 0)) == 0
                    ):
                        s.status = "done"
                        s.final_summary = smry
                        update_table(s.name, smry)
                        print(f"[skip] {s.name}: already complete (summary.json full).", flush=True)
                        continue
                except Exception:
                    pass
            runnable.append(s)

    # Queue: list of CellState we still need to schedule (round 0 launches).
    pending: List[CellState] = list(runnable)
    running: List[CellState] = []

    last_hb = 0.0

    while pending or running:
        if _SHUTDOWN.is_set():
            break

        # Launch as many as tier caps allow.
        i = 0
        while i < len(pending):
            s = pending[i]
            if active_per_tier[s.tier] < TIER_CAPS[s.tier]:
                s.round += 1
                if s.round > max_rounds:
                    s.status = "blocked-bug"
                    print(f"[blocked] {s.name}: exceeded max_rounds={max_rounds}", flush=True)
                    pending.pop(i)
                    continue
                launch_cell(s)
                active_per_tier[s.tier] += 1
                running.append(s)
                pending.pop(i)
                print(
                    f"[launch] {s.name} (round={s.round}, tier={s.tier}, "
                    f"slot={active_per_tier[s.tier]}/{TIER_CAPS[s.tier]}, "
                    f"log={s.log_path})",
                    flush=True,
                )
            else:
                i += 1

        # Reap any that are done.
        still_running: List[CellState] = []
        for s in running:
            if s.proc is None or s.proc.poll() is None:
                # Also reap if monitor killed it
                if s.kill_reason:
                    pass  # will collect below
                else:
                    still_running.append(s)
                    continue
            outcome = collect_cell(s)
            active_per_tier[s.tier] = max(0, active_per_tier[s.tier] - 1)
            s.finished_at = time.time()

            if outcome == "done":
                s.status = "done"
                assert s.final_summary is not None
                try:
                    update_table(s.name, s.final_summary)
                except Exception as e:
                    print(f"[table-error] {s.name}: {e}", flush=True)
                elapsed = s.finished_at - s.started_at
                print(
                    f"[DONE] {s.name} acc={s.final_summary.get('accuracy', 0.0):.3f} "
                    f"cost=${s.final_summary.get('cost_usd_total', 0.0):.2f} "
                    f"done={s.final_summary.get('n_done', 0)}/{s.expected_n} "
                    f"elapsed={_fmt_eta(elapsed)}",
                    flush=True,
                )
            elif outcome in ("killed-5error", "killed-5zero"):
                s.status = outcome
                print(
                    f"[KILLED] {s.name} reason={outcome} last_err={s.last_error[:120]}",
                    flush=True,
                )
            else:  # retry
                if s.round >= max_rounds:
                    s.status = "blocked-bug"
                    print(
                        f"[BLOCKED] {s.name}: hit max_rounds={max_rounds} without clean exit. "
                        f"last_err={s.last_error[:120]}",
                        flush=True,
                    )
                else:
                    print(
                        f"[retry] {s.name} round {s.round} → re-queue (resume). "
                        f"err_rows={s.last_err_count} last={s.last_error[:120]}",
                        flush=True,
                    )
                    pending.append(s)
        running = still_running

        # Heartbeat every 120s (downshift to 300s past first-5 check).
        now = time.time()
        if now - last_hb >= 120.0 and running:
            for s in running:
                # Per-cell heartbeat cadence: 120s before first-5, 300s after.
                cadence = 300.0 if s.first5_checked else 120.0
                if now >= s.next_hb_at:
                    print(heartbeat_line(s), flush=True)
                    s.next_hb_at = now + cadence
            last_hb = now

        if running:
            time.sleep(5.0)

    # Final summary
    print("\n" + "=" * 72, flush=True)
    print("Final sweep summary:", flush=True)
    print("=" * 72, flush=True)
    any_bad = False
    for s in states.values():
        if s.status == "done":
            smry = s.final_summary or {}
            print(
                f"DONE   {s.name}  acc={smry.get('accuracy', 0.0):.3f}  "
                f"${smry.get('cost_usd_total', 0.0):.2f}  "
                f"{smry.get('n_done', 0)}/{s.expected_n}",
                flush=True,
            )
        elif s.status.startswith("killed-"):
            any_bad = True
            print(
                f"KILL   {s.name:<60s} {s.status}  (last err: {s.last_error[:100]})",
                flush=True,
            )
        elif s.status == "blocked-bug":
            any_bad = True
            print(
                f"BLOCK  {s.name:<60s} blocked-bug  (last err: {s.last_error[:100]})",
                flush=True,
            )
        elif s.status == "skipped-locked":
            print(f"SKIP   {s.name:<60s} skipped-locked", flush=True)
        else:
            any_bad = True
            print(f"?      {s.name:<60s} status={s.status}", flush=True)

    return 1 if any_bad else 0


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def dry_run(cells: List[str]) -> int:
    print(f"Plan: {len(cells)} cells")
    by_tier: Dict[str, List[str]] = {}
    for c in cells:
        by_tier.setdefault(tier_of(c), []).append(c)
    print()
    print("Cells (in order):")
    for c in cells:
        t = tier_of(c)
        print(f"  {c:<60s} tier={t}")
    print()
    print("Concurrency plan (per-tier semaphores):")
    for t, names in sorted(by_tier.items()):
        cap = TIER_CAPS.get(t, "?")
        print(f"  tier={t:<18s} cap={cap}  cells={len(names)}")
    print()
    print(f"Logs:    {LOG_DIR}/<cell>.r<round>.<ts>.log")
    print(f"Outputs: {RUNS_DIR}/<cell>/{{results.jsonl,summary.json}}")
    print(f"Table:   {RESULTS_TABLE}  (auto section: '{AUTO_SECTION_HEADER}')")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="run_sweep.py",
        description="Hybrid-paradigm ablation sweep orchestrator (n=100).",
    )
    p.add_argument(
        "--batch",
        choices=sorted(BATCHES.keys()),
        default="wave-a",
        help="Which preset batch to run (default: wave-a).",
    )
    p.add_argument(
        "--cells",
        default=None,
        help="Comma-separated explicit cell list (overrides --batch).",
    )
    p.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Max retry rounds per cell (default: 3).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan and exit without launching anything.",
    )
    args = p.parse_args(argv)

    if args.cells:
        cells = [c.strip() for c in args.cells.split(",") if c.strip()]
        smoke = False
    else:
        cells = list(BATCHES[args.batch])
        smoke = args.batch == "smoke"

    if not cells:
        print("[error] no cells to run.", file=sys.stderr)
        return 2

    if args.dry_run:
        return dry_run(cells)

    if not VENV_PY.exists():
        print(f"[error] python not found: {VENV_PY}", file=sys.stderr)
        return 2

    return run_sweep(cells, max_rounds=args.max_rounds, smoke_n=smoke)


if __name__ == "__main__":
    sys.exit(main())
