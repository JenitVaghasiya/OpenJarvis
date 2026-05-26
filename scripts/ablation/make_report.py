#!/usr/bin/env python3
"""Build ``$OPENJARVIS_HYBRID_EXPERIMENTS_DIR/docs/index.html`` plus a set
of PNG pareto plots in the same dir. The HTML embeds the PNGs via
``<img src="...">`` so the browser never has to render anything itself —
just shows static images.

Re-running is idempotent: regenerates index.html + all PNGs from whatever
cells have summary.json at run-time.
"""
from __future__ import annotations

import html as html_lib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Callable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HYBRID_ROOT = Path(
    os.environ.get(
        "OPENJARVIS_HYBRID_EXPERIMENTS_DIR",
        Path.home() / ".openjarvis" / "experiments" / "hybrid",
    )
)
RUNS_DIR = HYBRID_ROOT / "runs"
DOCS_DIR = HYBRID_ROOT / "docs"
OUT_HTML = DOCS_DIR / "index.html"
RAW_MD = DOCS_DIR / "results-table.md"
PLOTS_DIR = DOCS_DIR / "plots-n100"
PLOTS_DIR.mkdir(exist_ok=True)

CLOUD_TOKENS = {
    "opus47": "Opus 4.7",
    "haiku45": "Haiku 4.5",
    "gpt5mini": "GPT-5 mini",
    "gpt5": "GPT-5.5",
    "gemini25pro": "Gemini 3.1 Pro",
    "gemini25flash": "Gemini 3.1 Flash",
}
CLOUD_TOKEN_ORDER = [
    "gpt5mini", "gemini25flash", "gemini25pro", "gpt5", "opus47", "haiku45",
]
BENCH_LABELS = {"gaia": "GAIA", "swe": "SWE-bench"}

PARADIGM_COLOR = {
    "skillorchestra": "#3b82f6",
    "cloud-only":     "#9ca3af",
    "minions":        "#10b981",
    "advisors":       "#f59e0b",
}
PARADIGM_ORDER = ["cloud-only", "skillorchestra", "minions", "advisors"]


def parse_cell_name(name: str) -> Optional[dict]:
    if not name.endswith("-n100"):
        return None
    parts = name[:-len("-n100")].split("-")
    if not parts:
        return None
    paradigm = parts[0]
    rest = parts[1:]
    if paradigm == "cloud":
        if rest and rest[0] == "only":
            rest = rest[1:]
        paradigm = "cloud-only"
        local = None
    elif paradigm in ("skillorchestra", "minions", "advisors"):
        if not rest:
            return None
        local = rest[0]
        rest = rest[1:]
    else:
        return None
    if len(rest) < 2:
        return None
    cloud_token = None
    for tok in CLOUD_TOKEN_ORDER:
        if tok in rest:
            cloud_token = tok
            break
    if cloud_token is None:
        return None
    bench = rest[-1]
    if bench not in BENCH_LABELS:
        return None
    return {"paradigm": paradigm, "local": local, "cloud": cloud_token, "bench": bench}


@dataclass
class Cell:
    name: str
    paradigm: str
    local: Optional[str]
    cloud: str
    bench: str
    accuracy: float
    cost_usd: float
    tokens_local_total: int
    tokens_cloud_total: int
    n_done: int
    latency_med_s: Optional[float]
    tool_calls_mean: Optional[float]
    web_searches_mean: Optional[float]


def load_cells() -> list[Cell]:
    out = []
    for d in sorted(RUNS_DIR.iterdir()):
        if not (d.is_dir() and d.name.endswith("-n100")):
            continue
        sj = d / "summary.json"
        if not sj.exists():
            continue
        parsed = parse_cell_name(d.name)
        if parsed is None:
            continue
        try:
            s = json.loads(sj.read_text())
        except Exception:
            continue
        # latency median
        rj = d / "results.jsonl"
        lat_med = None
        if rj.exists():
            lats = []
            for line in rj.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                v = row.get("latency_s")
                if isinstance(v, (int, float)):
                    lats.append(float(v))
            if lats:
                lat_med = median(lats)
        # tool calls mean — count any flavor of tool use per task:
        #   * SWE / mini-swe-agent fires events with kind="<something>_bash" (one per bash exec)
        #   * GAIA / agentic flows record tool calls inline on anthropic/openai events
        #     as `tool_calls: [...]` (anthropic server_tool_use, openai function calls)
        #     and Anthropic-side web search uses `n_web_searches`
        #   * minions does a hidden pre-fetch step recorded under metadata.traces.prefetch.n_searches
        # Sum all of them so the metric isn't silently zero on GAIA.
        #
        # `web_searches_mean` is the same idea but ONLY counts Anthropic
        # server-side web_search invocations — useful to see how often
        # GAIA cells actually leveraged the new opt-in web_search tool.
        tc_mean = None
        ws_mean = None
        logs = d / "logs"
        if logs.is_dir():
            counts = []
            ws_counts = []
            for log_p in logs.iterdir():
                if not log_p.name.endswith(".json"):
                    continue
                try:
                    lg = json.loads(log_p.read_text())
                except Exception:
                    continue
                ev = lg.get("events") or []
                bash_c = sum(1 for e in ev if isinstance(e, dict)
                             and isinstance(e.get("kind"), str)
                             and "_bash" in e["kind"])
                tc_c = sum(len(e.get("tool_calls") or [])
                           for e in ev if isinstance(e, dict))
                ws_c = sum(int(e.get("n_web_searches") or 0)
                           for e in ev if isinstance(e, dict))
                # n_web_searches is Anthropic's server-side web_search count; it's
                # already represented as one tool_call per search above, so don't
                # double-count — only fall back to it if tool_calls list is empty.
                if tc_c == 0:
                    tc_c = ws_c
                # prefetch (minions) — recorded out-of-band in metadata.traces.prefetch
                meta = lg.get("metadata") or {}
                pf = ((meta.get("traces") or {}).get("prefetch") or {}) if isinstance(meta, dict) else {}
                pf_c = int(pf.get("n_searches") or 0) if isinstance(pf, dict) else 0
                counts.append(bash_c + tc_c + pf_c)
                ws_counts.append(ws_c + pf_c)
            if counts:
                tc_mean = sum(counts) / len(counts)
            if ws_counts:
                ws_mean = sum(ws_counts) / len(ws_counts)
        out.append(Cell(
            name=d.name,
            paradigm=parsed["paradigm"],
            local=parsed["local"],
            cloud=parsed["cloud"],
            bench=parsed["bench"],
            accuracy=float(s.get("accuracy") or 0.0),
            cost_usd=float(s.get("cost_usd_total") or 0.0),
            tokens_local_total=int(s.get("tokens_local_total") or 0),
            tokens_cloud_total=int(s.get("tokens_cloud_total") or 0),
            n_done=int(s.get("n_done") or 0),
            latency_med_s=lat_med,
            tool_calls_mean=tc_mean,
            web_searches_mean=ws_mean,
        ))
    return out


@dataclass
class Axis:
    key: str
    title: str
    description: str
    filter_fn: Callable[[Cell], bool]


def axis_definitions() -> list[Axis]:
    return [
        Axis("all-cells", "0. All cells overview",
             "Every cell on one plot. Color = paradigm. Pareto frontier at a glance.",
             lambda c: True),
        Axis("cloud-anthropic", "1. Cloud-size within Anthropic",
             "Local = Qwen-27B; vary Anthropic cloud (Opus 4.7 vs Haiku 4.5). All paradigms overlaid.",
             lambda c: c.cloud in ("opus47", "haiku45")),
        Axis("cloud-openai", "2. Cloud-size within OpenAI",
             "Local = Qwen-27B; vary OpenAI cloud (GPT-5.5 vs GPT-5 mini). All paradigms overlaid.",
             lambda c: c.cloud in ("gpt5", "gpt5mini")),
        Axis("cloud-google", "3. Cloud-size within Google",
             "Local = Qwen-27B; vary Google cloud (Gemini 3.1 Pro vs Flash). All paradigms overlaid.",
             lambda c: c.cloud in ("gemini25pro", "gemini25flash")),
        Axis("cloud-family-frontier", "4. Cloud-family — frontier tier",
             "Compare frontier clouds across vendors (Opus 4.7, GPT-5.5, Gemini 3.1 Pro).",
             lambda c: c.cloud in ("opus47", "gpt5", "gemini25pro")),
        Axis("cloud-family-mini", "5. Cloud-family — mini/flash tier",
             "Compare cost-floor clouds across vendors (Haiku 4.5, GPT-5 mini, Gemini 3.1 Flash).",
             lambda c: c.cloud in ("haiku45", "gpt5mini", "gemini25flash")),
        Axis("paradigm-skillorch", "6. Skillorchestra only",
             "Just the skillorchestra cells. See how its router behaves across cloud choices.",
             lambda c: c.paradigm == "skillorchestra"),
        Axis("paradigm-cloud-only", "7. Cloud-only baseline",
             "Baseline cloud-only runs across all 6 clouds — no local model in the loop.",
             lambda c: c.paradigm == "cloud-only"),
    ]


METRIC_SPECS = [
    ("cost_usd",           "Cost (USD)",            True),
    ("latency_med_s",      "Latency median (s)",    True),
    ("tokens_cloud_total", "Tokens cloud (total)",  True),
]


def render_axis_png(axis: Axis, cells: list[Cell]) -> Optional[Path]:
    pts = [c for c in cells if axis.filter_fn(c)]
    if not pts:
        return None
    n_metrics = len(METRIC_SPECS)
    fig, axes = plt.subplots(
        n_metrics, 2,
        figsize=(15, 4.2 * n_metrics),
        squeeze=False,
    )
    fig.suptitle(axis.title, fontsize=16, fontweight="bold", y=0.998)
    used_paradigms = sorted({c.paradigm for c in pts}, key=lambda p: PARADIGM_ORDER.index(p) if p in PARADIGM_ORDER else 99)

    for row_idx, (mkey, mlabel, logx) in enumerate(METRIC_SPECS):
        for col_idx, bench in enumerate(("gaia", "swe")):
            ax = axes[row_idx][col_idx]
            bench_pts = [c for c in pts if c.bench == bench]
            for p in used_paradigms:
                p_pts = [c for c in bench_pts if c.paradigm == p]
                xs, ys, labels = [], [], []
                for c in p_pts:
                    v = getattr(c, mkey)
                    if v is None:
                        continue
                    if logx and v <= 0:
                        continue
                    xs.append(v); ys.append(c.accuracy); labels.append(CLOUD_TOKENS.get(c.cloud, c.cloud))
                if xs:
                    ax.scatter(xs, ys, c=PARADIGM_COLOR.get(p, "#000"),
                               s=120, alpha=0.85, edgecolors="white", linewidths=1.5,
                               label=p if (row_idx == 0 and col_idx == 0) else None)
                    for x, y, lbl in zip(xs, ys, labels):
                        ax.annotate(lbl, (x, y), xytext=(6, 6), textcoords="offset points",
                                    fontsize=9, color="#1f2937", alpha=0.9)
            if logx:
                ax.set_xscale("log")
            # Generous y-axis: 0 → max(observed)+0.15, floored at 0.7 so small numbers don't look cramped
            ax.set_ylim(0, max(0.75, max((c.accuracy for c in bench_pts), default=0.5) + 0.15))
            ax.set_xlabel(mlabel, fontsize=10)
            ax.set_ylabel("accuracy" if col_idx == 0 else "")
            ax.set_title(f"{mlabel} — {BENCH_LABELS[bench]}", fontsize=11, fontweight="bold")
            ax.grid(True, color="#e5e7eb", linewidth=0.5)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_color("#d1d5db")

    # legend on top
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.99),
                   ncol=len(used_paradigms), frameon=True, facecolor="white",
                   edgecolor="#d1d5db", fontsize=10)
    plt.subplots_adjust(left=0.07, right=0.97, top=0.96, bottom=0.04,
                        hspace=0.55, wspace=0.18)
    out = PLOTS_DIR / f"{axis.key}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------- Markdown → HTML (small) ----------

def md_to_html(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    in_table = False
    table_rows: list[list[str]] = []

    def flush_table():
        nonlocal table_rows
        if not table_rows:
            return
        head = table_rows[0]
        body = table_rows[2:] if len(table_rows) > 2 else []
        out.append("<table class='md'><thead><tr>" +
                   "".join(f"<th>{html_lib.escape(h.strip())}</th>" for h in head) +
                   "</tr></thead><tbody>")
        for row in body:
            out.append("<tr>" + "".join(
                f"<td>{html_lib.escape(cell.strip())}</td>" for cell in row) + "</tr>")
        out.append("</tbody></table>")
        table_rows = []

    def inline(s: str) -> str:
        s = html_lib.escape(s)
        # bold
        import re as _re
        s = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        s = _re.sub(r"`(.+?)`", r"<code>\1</code>", s)
        return s

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("|") and "|" in line[1:]:
            if not in_table:
                in_table = True
                table_rows = []
            cells = [c for c in line.strip().strip("|").split("|")]
            table_rows.append(cells)
            continue
        if in_table:
            flush_table()
            in_table = False
        if line.startswith("### "):
            out.append(f"<h3>{inline(line[4:])}</h3>")
        elif line.startswith("## "):
            out.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.startswith("# "):
            out.append(f"<h1>{inline(line[2:])}</h1>")
        elif line.startswith("> "):
            out.append(f"<blockquote>{inline(line[2:])}</blockquote>")
        elif line.startswith("- "):
            out.append(f"<li>{inline(line[2:])}</li>")
        elif line.strip() == "---":
            out.append("<hr>")
        elif line.strip() == "":
            out.append("")
        else:
            out.append(f"<p>{inline(line)}</p>")
    if in_table:
        flush_table()
    return "\n".join(out)


# ---------- HTML output ----------

CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #fafbfc; color: #1a1a1a; max-width: 1280px; margin: 0 auto;
       padding: 24px 16px 60px; line-height: 1.55; }
h1 { border-bottom: 2px solid #1a1a1a; padding-bottom: 8px; }
h2 { margin-top: 40px; border-bottom: 1px solid #d1d5db; padding-bottom: 6px; }
.card { background: white; border: 1px solid #e5e7eb; border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05); padding: 16px 20px; margin: 18px 0; }
.hero { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 12px; margin: 16px 0 24px; }
.hero .item { background: white; border: 1px solid #e5e7eb; border-left: 3px solid #3b82f6;
              border-radius: 6px; padding: 12px 14px; }
.hero .item.cost { border-left-color: #10b981; }
.hero .label { font-size: 0.78em; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; }
.hero .value { font-size: 1.15em; font-weight: 600; margin-top: 4px; }
.hero .sub { font-size: 0.85em; color: #4b5563; margin-top: 2px; }
table.summary { width: 100%; border-collapse: collapse; font-size: 0.85em; margin-top: 12px; }
table.summary th, table.summary td { border: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; }
table.summary th { background: #f3f4f6; cursor: pointer; user-select: none; }
table.summary tr:nth-child(even) { background: #fafafa; }
table.summary .best { background: #dcfce7 !important; }
section { margin-top: 28px; }
section img { max-width: 100%; height: auto; display: block; margin: 8px 0;
              border: 1px solid #e5e7eb; border-radius: 6px; background: white; }
.legend { font-size: 0.85em; color: #6b7280; }
.legend .sw { display: inline-block; width: 11px; height: 11px; border-radius: 50%;
              vertical-align: middle; margin-right: 4px; }
.desc { color: #4b5563; }
table.md { width: 100%; border-collapse: collapse; font-size: 0.85em; margin: 12px 0; }
table.md th, table.md td { border: 1px solid #e5e7eb; padding: 4px 8px; text-align: left; }
table.md th { background: #f3f4f6; }
table.md tr:nth-child(even) { background: #fafafa; }
code { background: #f3f4f6; padding: 1px 4px; border-radius: 3px; font-size: 0.92em; }
hr { border: none; border-top: 1px solid #d1d5db; margin: 20px 0; }
blockquote { border-left: 3px solid #d1d5db; padding-left: 12px; color: #4b5563; margin: 8px 0; }
"""

SORT_JS = """
document.querySelectorAll('table.summary th').forEach((th, idx) => {
  th.addEventListener('click', () => {
    const tbl = th.closest('table');
    const tbody = tbl.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const asc = th.dataset.dir !== 'asc';
    rows.sort((a, b) => {
      const av = a.cells[idx].dataset.sort ?? a.cells[idx].innerText;
      const bv = b.cells[idx].dataset.sort ?? b.cells[idx].innerText;
      const af = parseFloat(av), bf = parseFloat(bv);
      const an = isNaN(af) ? av : af, bn = isNaN(bf) ? bv : bf;
      return (an < bn ? -1 : an > bn ? 1 : 0) * (asc ? 1 : -1);
    });
    rows.forEach(r => tbody.appendChild(r));
    tbl.querySelectorAll('th').forEach(t => delete t.dataset.dir);
    th.dataset.dir = asc ? 'asc' : 'desc';
  });
});
"""


def build_html(cells: list[Cell], plots: list[tuple[Axis, Path]]) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    # headline cards
    by_bench = {"gaia": [c for c in cells if c.bench == "gaia"],
                "swe":  [c for c in cells if c.bench == "swe"]}
    def best_acc(lst):
        return max(lst, key=lambda c: c.accuracy) if lst else None
    def best_acc_per_dollar(lst):
        cand = [c for c in lst if c.cost_usd > 0]
        return max(cand, key=lambda c: c.accuracy / c.cost_usd) if cand else None

    hero = ['<div class="hero">']
    for bench_key, label in (("gaia", "GAIA"), ("swe", "SWE-bench")):
        b = best_acc(by_bench[bench_key])
        if b:
            hero.append(f"""<div class="item"><div class="label">Best acc — {label}</div>
                <div class="value">{b.accuracy:.3f} · {b.paradigm} × {CLOUD_TOKENS.get(b.cloud, b.cloud)}</div>
                <div class="sub">cost: ${b.cost_usd:.2f}</div></div>""")
    for bench_key, label in (("gaia", "GAIA"), ("swe", "SWE-bench")):
        b = best_acc_per_dollar(by_bench[bench_key])
        if b:
            ratio = b.accuracy / b.cost_usd
            hero.append(f"""<div class="item cost"><div class="label">Best acc / $ — {label}</div>
                <div class="value">{b.accuracy:.3f} for ${b.cost_usd:.2f}</div>
                <div class="sub">{b.paradigm} × {CLOUD_TOKENS.get(b.cloud, b.cloud)} ({ratio:.3f} acc/$)</div></div>""")
    hero.append("</div>")

    # summary table
    legend_html = '<div class="legend">'
    for p in PARADIGM_ORDER:
        col = PARADIGM_COLOR.get(p, "#000")
        legend_html += f'<span class="sw" style="background:{col}"></span>{p}&nbsp;&nbsp;'
    legend_html += '</div>'

    rows_html = []
    sorted_cells = sorted(cells, key=lambda c: (c.bench, -c.accuracy))
    # mark best per bench
    best_per_bench = {b: best_acc(by_bench[b]) for b in by_bench}
    for c in sorted_cells:
        is_best = best_per_bench.get(c.bench) is c
        cls = "best" if is_best else ""
        rows_html.append(f"""<tr class="{cls}">
            <td>{c.paradigm}</td>
            <td>{c.local or '—'}</td>
            <td>{CLOUD_TOKENS.get(c.cloud, c.cloud)}</td>
            <td>{BENCH_LABELS[c.bench]}</td>
            <td data-sort="{c.accuracy}">{c.accuracy:.3f}</td>
            <td data-sort="{c.cost_usd}">${c.cost_usd:.2f}</td>
            <td data-sort="{c.latency_med_s or 0}">{('%.1f' % c.latency_med_s + 's') if c.latency_med_s else '—'}</td>
            <td data-sort="{c.tool_calls_mean if c.tool_calls_mean is not None else 0}">{('%.1f' % c.tool_calls_mean) if c.tool_calls_mean is not None else '—'}</td>
            <td data-sort="{c.web_searches_mean if c.web_searches_mean is not None else 0}">{('%.2f' % c.web_searches_mean) if c.web_searches_mean is not None else '—'}</td>
            <td data-sort="{c.tokens_local_total}">{c.tokens_local_total:,}</td>
            <td data-sort="{c.tokens_cloud_total}">{c.tokens_cloud_total:,}</td>
        </tr>""")
    summary_table = f"""<table class="summary"><thead><tr>
      <th>paradigm</th><th>local</th><th>cloud</th><th>bench</th>
      <th>accuracy ↕</th><th>cost ↕</th><th>latency_med ↕</th><th>tool_calls ↕</th>
      <th>web_searches ↕</th>
      <th>tokens_local ↕</th><th>tokens_cloud ↕</th>
    </tr></thead><tbody>{''.join(rows_html)}</tbody></table>"""

    # axis sections (embed PNGs)
    axis_sections = []
    for axis, png_path in plots:
        rel = f"plots-n100/{png_path.name}"
        axis_sections.append(f"""<section>
            <h2>{html_lib.escape(axis.title)}</h2>
            <p class="desc">{html_lib.escape(axis.description)}</p>
            <img src="{rel}" alt="{axis.title}">
        </section>""")

    md_html = ""
    if RAW_MD.exists():
        md_html = md_to_html(RAW_MD.read_text())

    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>OpenJarvis Hybrid n=100 Ablation</title>
<style>{CSS}</style>
</head><body>
<h1>OpenJarvis Hybrid n=100 Ablation</h1>
<p class="desc">Comparing local-cloud paradigms across 6 cloud models on GAIA + SWE-bench-Verified · {len(cells)} cells · generated {ts}</p>

<div class="card">
<h2 style="border:0; margin-top:0">Headline findings</h2>
{''.join(hero)}
</div>

<div class="card">
<h2 style="border:0; margin-top:0">All cells (sortable)</h2>
{legend_html}
{summary_table}
</div>

{''.join(axis_sections)}

<div class="card">
<h2 style="border:0; margin-top:0">results-table.md (full)</h2>
{md_html}
</div>

<script>{SORT_JS}</script>
</body></html>
"""


def main():
    cells = load_cells()
    print(f"loaded {len(cells)} cells")
    plots = []
    for axis in axis_definitions():
        png = render_axis_png(axis, cells)
        if png is not None:
            plots.append((axis, png))
            print(f"  rendered {png.name}")
    html = build_html(cells, plots)
    OUT_HTML.write_text(html)
    print(f"wrote {OUT_HTML} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
