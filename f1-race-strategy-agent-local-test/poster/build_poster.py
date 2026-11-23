"""Build the 24 x 36 in PITWALL poster from the evaluation's committed outputs (tag eval-v3).

    python poster/build_poster.py --export
    python poster/build_poster.py --dev reports/eval/rerun/v3/dev20 --test reports/eval/rerun/v3/test9 \
        --fidelity reports/calibration/fidelity --export

`--dev` / `--test`: the `--out-dir` of `python -m src.eval.rerun` for the development set (dev20) and the test set
(test9), after its `tables` stage. `--fidelity`: the fidelity scorecard's outputs (the scorecard directory is
`--scorecard`, default poster_stats.SCORECARD). Every number comes from `poster_stats.compute()`; `numbers.md`
lists each one with its file and computation, and the build stops if a cross-check against the pipeline's own
tables fails. Verdict words come from the pre-registered (race, driver) interval, never from text.

Writes poster/PITWALL_poster_24x36in.html and poster/numbers.md; with --export also the PDF (Chromium), the
300 ppi PNG and the 1200 px preview (poppler). Reads files only: nothing from `src` is imported.
"""
from __future__ import annotations

import argparse
import html as _html
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import poster_stats as ps  # noqa: E402

OUT_HTML = HERE / "PITWALL_poster_24x36in.html"
OUT_PDF = HERE / "PITWALL_poster_24x36in.pdf"
OUT_PNG = HERE / "PITWALL_poster_24x36in_300ppi.png"
OUT_PREVIEW = HERE / "PITWALL_poster_24x36in_preview.png"
MAP = "2023_F1_CourseLayout_Spain.svg"

PANEL, INK, MUTED, GRID = "#1a1930", "#f4f2f8", "#c9c6d8", "#4a4866"
PEND_INK = "#b9b6cc"
# one colour (Okabe-Ito, colour-blind safe) and one marker shape per policy; test set filled, development hollow
POL = {"agent": ("Agent", "#56B4E9", "circle"),
       "belief_optimizer": ("Look-ahead", "#E69F00", "square"),
       "run_to_flag": ("Run to the flag", "#F0E442", "down"),
       "one_stop_total": ("One stop in total", "#D55E00", "up"),
       "actual_strategy": ("Teams’ actual", "#2BB88A", "diamond"),
       "fixed_one_stop": ("Fixed one-stop", "#CC79A7", "square"),
       "fixed_two_stop": ("Fixed two-stop", "#BDBDBD", "diamond")}
SETS = (("h", "hall", "Test set"), ("d", "d", "Development set"))
BOLD = ' font-weight="700"'
USED: list[str] = []
N: dict = {}


# ---------------------------------------------------------------- numbers as printed
def used(key):
    if key not in USED:
        USED.append(key)
    return N[key]


def V(key):
    n = used(key)
    if n.pending:
        return f'<span class="pend" title="{n.tag}">pending</span>'
    return _html.escape(n.text)


def T(key):
    n = used(key)
    return "pending" if n.pending else _html.escape(n.text)


def ready(*keys):
    return all(not used(k).pending for k in keys)


def val(key):
    return used(key).value


def versus(g, x, name):
    """'beat X by 2.3 s' / 'was level with X (+0.1 s)' / 'trailed X by 1.0 s', from the computed verdict."""
    if not ready(f"{g}_{x}_verdict"):
        return f'<span class="pend">pending</span> {name}'
    v, d = val(f"{g}_{x}_verdict"), val(f"{g}_{x}_diff")
    used(f"{g}_{x}_diff")
    mag = _html.escape(ps.s(abs(d), 1))
    if v == "ahead":
        return f"beat {name} by {mag} s"
    if v == "behind":
        return f"trailed {name} by {mag} s"
    return f"matched {name}"


def rules_phrase():
    """Both simple rules at once, from the two computed verdicts."""
    v = [val(f"hall_{x}_verdict") for x in ("run_to_flag", "one_stop_total")]
    d = [V(f"hall_{x}_diff") for x in ("run_to_flag", "one_stop_total")]
    mags = [_html.escape(ps.s(abs(val(f"hall_{x}_diff")), 1)) for x in ("run_to_flag", "one_stop_total")]
    if v == ["ahead", "ahead"]:
        return f"beat both simple rules by {mags[0]} s and {mags[1]} s"
    if v == ["level", "level"]:
        return f"matched both simple rules ({d[0]} and {d[1]} s)"
    if v == ["behind", "behind"]:
        return f"trailed both simple rules by {mags[0]} s and {mags[1]} s"
    return (f"{versus('hall', 'run_to_flag', 'run to the flag')} and "
            f"{versus('hall', 'one_stop_total', 'one stop in total')}")


# ---------------------------------------------------------------- svg pieces (1 user unit = 1 pt)
def marker(shape, x, y, color, r=8.0, hollow=False):
    fill = PANEL if hollow else color
    sw = 3 if hollow else 1.5
    stroke = color if hollow else PANEL
    st = f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"'
    if shape == "circle":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" {st}/>'
    if shape == "square":
        a = r * 1.7
        return f'<rect x="{x - a / 2:.1f}" y="{y - a / 2:.1f}" width="{a:.1f}" height="{a:.1f}" {st}/>'
    if shape == "diamond":
        d = r * 1.3
        return f'<path d="M{x:.1f},{y - d:.1f}L{x + d:.1f},{y:.1f}L{x:.1f},{y + d:.1f}L{x - d:.1f},{y:.1f}Z" {st}/>'
    d, k = r * 1.25, (1 if shape == "up" else -1)
    return (f'<path d="M{x:.1f},{y - k * d:.1f}L{x + d:.1f},{y + k * d * 0.8:.1f}L{x - d:.1f},{y + k * d * 0.8:.1f}Z" '
            f'{st}/>')


def nice_ticks(lo, hi, n=5):
    span = max(hi - lo, 1e-9)
    step = min((st for st in (1, 2, 2.5, 5, 10, 20, 25, 50) if span / st <= n), default=50)
    t0 = np.floor(lo / step) * step
    return [round(t0 + i * step, 6) for i in range(int(np.ceil((hi - t0) / step - 1e-9)) + 1)]


def fmt_tick(t):
    return (f"{t:.0f}" if float(t).is_integer() else f"{t:g}").replace("-", "−")


def dot_chart(rows, key_mean, key_ci, key_txt, zero_line, width=760, aria="", sub=32, gap=16, inline=False):
    """One block per row: a test-set line (filled marker) and a development line (hollow), each with its 95%
    interval and its value printed at the right."""
    vals = []
    for r in rows:
        for t, g, _ in SETS:
            if ready(key_ci(t, g, r)):
                vals += [v for v in (*val(key_ci(t, g, r)), val(key_mean(t, g, r))) if np.isfinite(v)]
    ticks = nice_ticks(min(vals + [0.0]), max(vals + [0.0]), n=4)
    LO, HI = ticks[0], ticks[-1]
    L, R, VX = 225, 525, width
    sx = lambda v: L + (R - L) * (v - LO) / (HI - LO)  # noqa: E731
    top = 34 if zero_line else (26 if inline else 8)
    blk = 2 * sub + gap
    yb = top + blk * len(rows) - gap / 2
    H = yb + 32
    s = [f'<svg class="chart" viewBox="0 0 {width} {H:.0f}" role="img" aria-label="{aria}">']
    for tk in ticks:
        x = sx(tk)
        s.append(f'<line x1="{x:.1f}" y1="{top - 4}" x2="{x:.1f}" y2="{yb:.1f}" stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{x:.1f}" y="{yb + 26:.1f}" text-anchor="middle" font-size="20" fill="{MUTED}">'
                 f'{fmt_tick(tk)}{" s" if tk == ticks[-1] else ""}</text>')
    if inline:
        s.append(f'<text x="{VX - 100}" y="{top + 2}" text-anchor="end" font-size="19" fill="{MUTED}">test</text>')
        s.append(f'<text x="{VX}" y="{top + 2}" text-anchor="end" font-size="19" fill="{MUTED}">dev</text>')
    if zero_line and LO < 0 < HI:
        x0 = sx(0)
        s.append(f'<line x1="{x0:.1f}" y1="{top - 10}" x2="{x0:.1f}" y2="{yb:.1f}" stroke="{INK}" stroke-width="2"/>')
        s.append(f'<text x="{x0 - 8:.1f}" y="{top - 14}" text-anchor="end" font-size="20" fill="{MUTED}">'
                 f'← agent better</text>')
        s.append(f'<text x="{x0 + 8:.1f}" y="{top - 14}" font-size="20" fill="{MUTED}">worse →</text>')
    for i, r in enumerate(rows):
        name, color, shape = POL[r]
        y0 = top + blk * i
        s.append(f'<text x="0" y="{y0 + sub + 8:.1f}" font-size="23" font-weight="700" fill="{INK}">{name}</text>')
        for j, (t, g, _) in enumerate(SETS):
            y = y0 + sub * j + sub / 2
            lo, hi = val(key_ci(t, g, r))
            m = val(key_mean(t, g, r))
            s.append(f'<line x1="{sx(lo):.1f}" y1="{y:.1f}" x2="{sx(hi):.1f}" y2="{y:.1f}" stroke="{color}" '
                     f'stroke-width="5"/>')
            s.append(marker(shape, sx(m), y, color, r=10, hollow=(t == "d")))
            if inline:   # test and development values side by side, on the block's middle line
                s.append(f'<text x="{VX - 100 if t == "h" else VX}" y="{y0 + sub + 8:.1f}" text-anchor="end" '
                         f'font-size="22"{BOLD if t == "h" else ""} fill="{INK if t == "h" else MUTED}">'
                         f'{key_txt(t, g, r)}</text>')
                continue
            s.append(f'<text x="{VX}" y="{y + 8:.1f}" text-anchor="end" font-size="22"'
                     f"{BOLD if t == 'h' else ''} fill=\"{INK if t == 'h' else MUTED}\">"
                     f'{key_txt(t, g, r)}</text>')
    s.append("</svg>")
    return "".join(s)


def forest():
    return dot_chart(("run_to_flag", "one_stop_total", "actual_strategy", "belief_optimizer"),
                     lambda t, g, x: f"{g}_{x}_diff", lambda t, g, x: f"{g}_{x}_ci",
                     lambda t, g, x: f'{T(f"{g}_{x}_diff")} {T(f"{g}_{x}_ci")}', True,
                     aria="Agent minus each alternative, mean time lost, 95% intervals, test and development",
                     sub=26, gap=10)


def levels():
    return dot_chart(("agent", "belief_optimizer", "run_to_flag", "one_stop_total", "actual_strategy"),
                     lambda t, g, p: f"{t}_mean_{p}", lambda t, g, p: f"{t}_mci_{p}",
                     lambda t, g, p: f'{T(f"{t}_mean_{p}")}', False,
                     aria="Mean time lost of each strategy, 95% intervals, test and development", sub=17, gap=5, inline=True)


def per_race(width=760):
    """Agent minus run_to_flag per test race, 95% (race, driver) intervals."""
    races = list(val("hall_names"))
    lo_hi = [v for r in races for v in val(f"pr_{r}_agent_ci")]
    ticks = nice_ticks(min(lo_hi + [0]), max(lo_hi + [0]), n=5)
    LO, HI = ticks[0], ticks[-1]
    L, R, VX = 190, 545, width
    sx = lambda v: L + (R - L) * (v - LO) / (HI - LO)  # noqa: E731
    top, rowh = 32, 25
    yb = top + rowh * len(races)
    H = yb + 32
    s = [f'<svg class="chart" viewBox="0 0 {width} {H}" role="img" aria-label="Agent minus run to the flag per test '
         f'race, with 95% intervals">']
    for tk in ticks:
        s.append(f'<line x1="{sx(tk):.1f}" y1="{top - 6}" x2="{sx(tk):.1f}" y2="{yb}" stroke="{GRID}"/>')
        s.append(f'<text x="{sx(tk):.1f}" y="{yb + 26}" text-anchor="middle" font-size="20" fill="{MUTED}">'
                 f'{fmt_tick(tk)}{" s" if tk == ticks[-1] else ""}</text>')
    s.append(f'<line x1="{sx(0):.1f}" y1="{top - 10}" x2="{sx(0):.1f}" y2="{yb}" stroke="{INK}" stroke-width="2"/>')
    s.append(f'<text x="{sx(0) - 8:.1f}" y="{top - 14}" text-anchor="end" font-size="20" fill="{MUTED}">'
             f'← agent better</text>')
    s.append(f'<text x="{sx(0) + 8:.1f}" y="{top - 14}" font-size="20" fill="{MUTED}">worse →</text>')
    for i, r in enumerate(races):
        y = top + rowh * i + rowh / 2
        m, (lo, hi) = val(f"pr_{r}_agent"), val(f"pr_{r}_agent_ci")
        color = "#D55E00" if m > 0 else "#56B4E9"
        s.append(f'<text x="0" y="{y + 8:.1f}" font-size="23" fill="{INK}">{ps.race_name(r)}</text>')
        s.append(f'<rect x="{min(sx(0), sx(m)):.1f}" y="{y - 11:.1f}" width="{abs(sx(m) - sx(0)):.1f}" height="22" '
                 f'fill="{color}"/>')
        s.append(f'<line x1="{sx(lo):.1f}" y1="{y:.1f}" x2="{sx(hi):.1f}" y2="{y:.1f}" stroke="{INK}" stroke-width="2.5"/>')
        s.append(f'<text x="{VX}" y="{y + 8:.1f}" text-anchor="end" font-size="22" font-weight="700" fill="{INK}">'
                 f'{T(f"pr_{r}_agent")} {T(f"pr_{r}_agent_ci")}</text>')
    s.append("</svg>")
    return "".join(s)


def problem_diagram(width=470):
    """Illustration (no data): lap time rises as tyres wear; a pit stop costs time and resets the tyres."""
    s = [f'<svg class="chart" viewBox="0 0 {width} 215" role="img" aria-label="Illustration: lap time rises as tyres '
         f'wear; a pit stop costs time once and gives fresh, faster tyres">']
    X0, X1, Y0, Y1 = 40, width - 10, 40, 178
    s.append(f'<line x1="{X0}" y1="{Y1}" x2="{X1}" y2="{Y1}" stroke="{MUTED}" stroke-width="2"/>')
    s.append(f'<line x1="{X0}" y1="{Y0}" x2="{X0}" y2="{Y1}" stroke="{MUTED}" stroke-width="2"/>')
    s.append(f'<text x="{X1}" y="{Y1 + 28}" text-anchor="end" font-size="23" fill="{MUTED}">laps →</text>')
    s.append(f'<text x="{X0 - 10}" y="{Y0 - 14}" font-size="23" fill="{MUTED}">lap time ↑</text>')
    xp = X0 + (X1 - X0) * 0.52
    s.append(f'<path d="M{X0 + 8},{Y1 - 60} Q{(X0 + xp) / 2},{Y1 - 78} {xp - 8},{Y1 - 128}" fill="none" '
             f'stroke="#F0E442" stroke-width="5"/>')
    s.append(f'<path d="M{xp + 26},{Y1 - 52} Q{(xp + X1) / 2},{Y1 - 68} {X1 - 8},{Y1 - 112}" fill="none" '
             f'stroke="#F0E442" stroke-width="5"/>')
    s.append(f'<rect x="{xp - 4}" y="{Y0 - 4}" width="26" height="{Y1 - Y0 + 4}" fill="#ff4d3d" opacity="0.85"/>')
    s.append(f'<text x="{xp + 9}" y="{Y0 - 14}" text-anchor="middle" font-size="23" font-weight="800" fill="#ff8a7d">'
             f'pit stop</text>')
    s.append(f'<text x="{X0 + 14}" y="{Y1 - 22}" font-size="23" fill="{INK}">tyres wear: slower</text>')
    s.append(f'<text x="{xp + 34}" y="{Y1 - 22}" font-size="23" fill="{INK}">fresh: faster</text>')
    s.append("</svg>")
    return "".join(s)


def sim_chart(width=640):
    """The belief's predicted lap time per lap for the example's two top plans (poster/sim_curve.csv)."""
    c = val("sim_curve")
    plans = (("plan1_s", T("ex_p1_words"), "#56B4E9", ""), ("plan2_s", T("ex_p2_words"), "#E69F00", ' stroke-dasharray="14 7"'))
    laps = c["lap"].to_numpy()
    allv = np.concatenate([c[k].to_numpy() for k, *_ in plans])
    yt = list(range(int(np.floor(allv.min())), int(np.ceil(allv.max())) + 1))
    yt = [y for y in yt if y % 2 == 0] if len(yt) > 5 else yt
    LO, HI = min(yt[0], np.floor(allv.min())), max(yt[-1], np.ceil(allv.max()))
    L, R, T0, B = 52, width - 8, 104, 272
    sx = lambda v: L + (R - L) * (v - (laps[0] - 0.5)) / (laps[-1] - laps[0] + 1)  # noqa: E731
    sy = lambda v: B - (B - T0) * (v - LO) / (HI - LO)  # noqa: E731
    H = B + 40
    s = [f'<svg class="chart" viewBox="0 0 {width} {H}" role="img" aria-label="Predicted lap time by lap for two '
         f'plans at the example decision">']
    for i, (_, words, color, dash) in enumerate(plans):
        y = 22 + 38 * i
        s.append(f'<line x1="0" y1="{y - 8}" x2="44" y2="{y - 8}" stroke="{color}" stroke-width="6"{dash}/>')
        s.append(f'<text x="56" y="{y}" font-size="24" fill="{INK}"><tspan font-weight="800">Plan {i + 1}:</tspan> '
                 f'{words}</text>')
    for y in yt:
        s.append(f'<line x1="{L}" y1="{sy(y):.1f}" x2="{R}" y2="{sy(y):.1f}" stroke="{GRID}"/>')
        s.append(f'<text x="{L - 10}" y="{sy(y) + 8:.1f}" text-anchor="end" font-size="24" fill="{MUTED}">{y}</text>')
    for x in [v for v in range(int(laps[0]), int(laps[-1]) + 1) if v % 10 == 0]:
        s.append(f'<text x="{sx(x):.1f}" y="{B + 30}" text-anchor="middle" font-size="24" fill="{MUTED}">{x}</text>')
    for k, _, color, dash in plans:
        pts = []
        for lap, v in zip(laps, c[k]):
            pts += [f"{sx(lap - 0.5):.1f},{sy(v):.1f}", f"{sx(lap + 0.5):.1f},{sy(v):.1f}"]
        s.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="5"{dash}/>')
    s.append(f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" stroke="{MUTED}" stroke-width="2"/>')
    s.append("</svg>")
    return "".join(s)


def agent_loop(model, width=900):
    """The agent's loop: the race so far in, tool calls and results for up to the turn limit, a decision out, the
    harness's legality check."""
    H = 512
    s = [f'<svg class="chart" viewBox="0 0 {width} {H}" role="img" aria-label="How the agent works: the language model '
         f'reads the race so far, calls tools and reads their results, then submits a decision that the harness checks">',
         f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" '
         f'orient="auto"><path d="M0,0L10,5L0,10Z" fill="{INK}"/></marker></defs>']

    def box(x, y, w, h, color, fill, titles, lines, tsize=30, lsize=26, title_y=42, lh=32):
        s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{color}" stroke-width="2.5"/>')
        for k, tt in enumerate(titles):
            s.append(f'<text x="{x + w / 2}" y="{y + title_y + (tsize + 4) * k}" text-anchor="middle" font-size="{tsize}" '
                     f'font-weight="800" fill="{INK}">{tt}</text>')
        y0 = y + title_y + (tsize + 4) * (len(titles) - 1) + 8
        for k, line in enumerate(lines):
            s.append(f'<text x="{x + w / 2}" y="{y0 + lh * (k + 1)}" text-anchor="middle" '
                     f'font-size="{lsize}" fill="{MUTED}">{line}</text>')

    def arrow(x1, y1, x2, y2):
        s.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{INK}" stroke-width="3.5" '
                 f'marker-end="url(#ah)"/>')

    box(0, 70, 170, 190, "#9cc8ff", "#16263a", ["Race", "so far"], ["laps, tyres,", "gaps, flags"])
    arrow(172, 165, 212, 165)
    box(218, 70, 222, 190, "#56B4E9", "#16263a", ["LLM agent"], [model, "reads, asks,", "decides"])
    tools = (("Ranked plans", ["the look-ahead’s best", "plan per strategy"]),
             ("Race picture", ["position, rivals’ tyres,", "observed wear, sets left"]),
             ("Evaluate plan", ["times any plan", "the agent proposes"]))
    for i, (title, lines) in enumerate(tools):
        box(530, 112 * i, 370, 108, "#E69F00", "#2c2614", [title], lines, tsize=28, lsize=25, title_y=30, lh=26)
    arrow(442, 122, 522, 122)
    s.append(f'<text x="484" y="108" text-anchor="middle" font-size="25" fill="{MUTED}">calls</text>')
    arrow(526, 220, 446, 220)
    s.append(f'<text x="486" y="253" text-anchor="middle" font-size="25" fill="{MUTED}">results</text>')
    s.append(f'<text x="329" y="300" text-anchor="middle" font-size="27" font-weight="800" fill="{INK}">'
             f'up to {T("agent_max_turns")} turns</text>')
    arrow(329, 310, 329, 352)
    box(60, 358, 500, 152, "#ffcf4a", "#2c2614", ["Decision"],
        ["PIT + tyre, or", "STAY OUT + planned stop", "any legal plan"], title_y=38, lh=31)
    arrow(562, 434, 612, 434)
    box(618, 358, 282, 152, "#2BB88A", "#12291f", ["Harness"], ["checks the", "plan is legal"], title_y=36, lh=30)
    s.append("</svg>")
    return "".join(s)


def less_lost():
    """'averaged 2.2 s less lost time than simple rules' (the smaller of the two paired means)."""
    v = [val(f"hall_{x}_verdict") for x in ("run_to_flag", "one_stop_total")]
    if v != ["ahead", "ahead"]:
        return rules_phrase()
    m = min(abs(val(f"hall_{x}_diff")) for x in ("run_to_flag", "one_stop_total"))
    for x in ("run_to_flag", "one_stop_total"):
        used(f"hall_{x}_diff")
    return f"averaged {_html.escape(ps.s(m, 1))} s less lost time than simple rules"


def html(args_line):
    D, H = N["_sets"].value
    model = T("h_model") if ready("h_model") else T("d_model")
    legal_all = val("h_ag_ok") == val("h_ag_scored") and val("d_ag_ok") == val("d_ag_scored")
    used("h_ag_ok"), used("d_ag_ok")
    lv = [val(f"{g}_belief_optimizer_verdict") for g in ("hall", "d")]
    edge = ("; its edge comes from its simulator, not the language model" if lv == ["level", "level"] else "")
    answer = (f"it makes a legal, explained call every time and {less_lost()} on {V('races_total')} test races{edge}."
              if legal_all else f"it {less_lost()} on {V('races_total')} test races{edge}.")
    match = ("The agent matches its look-ahead on both sets: the simulator and search do the work."
             if lv == ["level", "level"] else "The agent differs from its look-ahead.")
    ribbon = ""
    if H.fixture:
        ribbon = '<div class="ribbon fix">LAYOUT FIXTURE: development rows in the test column. NOT A RESULT.</div>'
    body = f"""
<div class="poster">
{ribbon}
<header class="top">
  <div class="title">
    <div class="kicker">CSE 598 Agentic AI &middot; Arizona State University</div>
    <h1>PITWALL</h1>
    <div class="acro">Predictive Intelligence for Tactical When-to-pit Agentic Lap-wise Logic</div>
    <div class="authors">Jason Lunder &middot; Rishabh Baral &middot; Sarah Maisha &middot; Quan Nguyen &middot; Manya Jain<br>
      <span>Stakeholder: Manav Ananthakumar &middot; Professor: Hua Wei</span></div>
  </div>
  <figure class="map">
    <img src="{MAP}" alt="Circuit de Barcelona-Catalunya, 2023 layout">
  </figure>
</header>

<div class="answer">
  <div><b>Question:</b> can a language-model agent decide when a Formula 1 car should pit?</div>
  <div><b>Answer:</b> {answer}</div>
</div>

<main class="main">
  <div class="row r1">
  <article class="panel prob-panel">
    <h2><b>1</b>The problem</h2>
    <p>Tyres get slower every lap. A pit stop costs about {V("pit_loss_median")} but gives fresh tyres. When to stop,
    and onto which tyre, decides races. The call is made mid-race, without knowing what comes next.</p>
    <div class="prob">{problem_diagram()}</div>
  </article>
  <article class="panel sim">
    <h2><b>2</b>How the simulator works</h2>
    <div class="formula"><span class="lhs">Lap time</span><span class="op">=</span><span class="term t1">tyre&rsquo;s base pace</span><span class="op">+</span><span class="term t2">wear &times; tyre age</span><span class="op">+</span><span class="term t3">fuel load</span><span class="op">+</span><span class="term t4">driver&rsquo;s pace</span></div>
    <div class="simgrid">
      <div><div class="caption">Predicted lap time (s), example decision</div>{sim_chart()}
        <p class="note">Each pit stop adds {V("sim_stop_s")}.</p></div>
      <ul class="facts">
        <li><b>Learns from</b> this race&rsquo;s laps so far, the weekend&rsquo;s practice and earlier races.</li>
        <li><b>Also knows</b> pit-stop cost, tyre life, safety cars and tyre sets left.</li>
        <li><b>Uncertainty:</b> it plays out the rest of the race {V("sim_n_samples")} times per plan.</li>
        <li><b>Look-ahead search</b> tries every sensible plan of stop laps and tyres ({V("sim_plans")} here) and
        ranks them by expected race time.</li>
      </ul>
    </div>
  </article>
  </div>

  <div class="row r2">
  <article class="panel">
    <h2><b>3</b>How the agent works</h2>
    {agent_loop(model)}
  </article>
  <article class="panel ex">
    <h2><b>4</b>An example decision</h2>
    <ol class="trace">
      <li><b>Situation.</b> {V("ex_race")}, after lap {V("ex_lap")}, {V("ex_left")} laps left; {V("ex_set")} tyres,
      {V("ex_age")} laps old; {V("ex_status")} flag.</li>
      <li><b>Tools.</b> Best plan: {V("ex_p1_words")}.<br>Next best: {V("ex_p2_words")}: <span class="nw">{V("ex_p2_slower")}.</span><br>
      Tyre wear {V("ex_wear_obs")} s/lap observed, {V("ex_wear_model")} predicted.</li>
      <li><b>Decision.</b> {V("ex_dec")}, confidence {V("ex_conf")}.</li>
      <li><b>Reason.</b> <q>{V("ex_reason")}</q></li>
    </ol>
    <p class="note">Time lost against the judge&rsquo;s best plan: {V("ex_regret")} s.</p>
  </article>
  </div>

  <article class="panel score">
    <h2><b>5</b>How we score it</h2>
    <div class="scoregrid">
      <p>A second simulator, the <b>judge</b>, uses the whole race to time every plan. <b>Time lost</b> = a plan&rsquo;s
      race time minus the judge&rsquo;s best plan. The judge orders close finishers correctly {V("fid_close_pct")} of
      the time.</p>
      <p><b>Compared against:</b> two simple rules (run to the flag; one stop in total), the teams&rsquo; actual
      strategies and the look-ahead alone.<br>
      <b>Test set:</b> {V("h_races")} races, {V("h_ag_scored")} decisions.<br><b>Development set:</b> {V("d_races")}
      races, {V("d_ag_scored")} decisions.</p>
    </div>
  </article>

  <div class="row r3">
  <article class="panel res">
    <h2><b>6</b>Results <span class="key">filled = test set, hollow = development</span></h2>
    <div class="caption">Agent minus each alternative: time lost (s) [95% interval]</div>{forest()}
    <div class="caption">Mean time lost (s)</div>{levels()}
  </article>
  <article class="panel">
    <h2><b>7</b>Analysis</h2>
    <div class="caption">By test race: agent minus run to the flag (s)</div>{per_race()}
    <ul>
      <li>{match}</li>
      <li>The language model adds a legal, explained call every time, in about {V("h_lat")}.</li>
      <li>It wins where tyre behaviour is well known and loses where it is not: Australia 2022,
      {V("pr_2022_03_australia_agent")} s.</li>
      <li>Not modelled: track position and pace management. Teams also choose pace, fuel mix and setup.</li>
      <li>Reasoned departures from the top plan: {V("h_sdep_helped")} helped, {V("h_sdep_hurt")} hurt.</li>
    </ul>
  </article>
  </div>

  <article class="panel take">
    <h2><b>8</b>Takeaways and next</h2>
    <div class="takegrid">
      <p>The agent matched or beat simple rules on {V("races_matched")} of {V("races_total")} test races.</p>
      <p><b>Next:</b> tyre knowledge from telemetry; more of the team&rsquo;s decisions; a stronger model.</p>
    </div>
  </article>
</main>

<footer>
  <div class="foot">Model {model} &middot; tag <code>{V("h_tag")}</code> ({V("h_commit")}) &middot; one agent pass &middot; github.com/f1-t/agentic-ai &middot; data: FastF1 &middot; map:
    <span lang="ja">ごひょううべこ</span>, CC BY-SA 4.0</div>
</footer>
</div>"""
    return ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<!-- built by poster/build_poster.py {_html.escape(args_line)} -->"
            "<title>PITWALL poster</title><style>" + CSS + "</style></head><body>" + body + "</body></html>\n")


CSS = """
@page{size:24in 36in;margin:0}
:root{--bg:#0f0e1c;--panel:#1a1930;--line:#3d3b58;--ink:#f4f2f8;--muted:#c9c6d8;--red:#ff4d3d;
      --amber:#ffcf4a;--dev:#9cc8ff;--pend:#b9b6cc}
*{box-sizing:border-box}
html,body{margin:0;padding:0;width:24in;height:36in;background:var(--bg);
  -webkit-print-color-adjust:exact;print-color-adjust:exact}
body{font-family:"Noto Sans","Liberation Sans",Arial,sans-serif;color:var(--ink);font-size:27pt;line-height:1.2}
code{font-family:"DejaVu Sans Mono","Liberation Mono",monospace;font-size:.9em}
.poster{position:relative;width:24in;height:36in;padding:.6in .7in .6in;display:flex;flex-direction:column;
  gap:.2in;overflow:hidden;border-top:.12in solid var(--red)}
.ribbon{flex:none;border:3px dashed #ff4d3d;color:#ff8a7d;font-size:20pt;font-weight:800;padding:.04in .2in;
  text-align:center}
.top{display:flex;justify-content:space-between;align-items:center;gap:.4in;flex:none}
.kicker{font-size:21pt;letter-spacing:.12em;text-transform:uppercase;color:var(--red);font-weight:700}
h1{font-size:96pt;line-height:1;margin:0;letter-spacing:.02em;font-weight:900}
.acro{font-size:23pt;color:var(--amber);font-weight:700;margin-bottom:.06in}
.authors{font-size:24pt;line-height:1.26}
.authors span{font-size:21pt;color:var(--muted)}
.map{margin:0;width:5.4in;flex:none}
.map img{width:5.4in;height:auto;display:block}
.answer{flex:none;background:#2a1418;border-left:.14in solid var(--red);padding:.12in .3in .14in;font-size:31pt;
  line-height:1.24;display:flex;flex-direction:column;gap:.06in}
.answer b{color:var(--amber)}
.pend{display:inline-block;font-size:.88em;font-style:italic;font-weight:700;color:var(--pend);
  border:2px dashed var(--pend);padding:0 .07in}
.main{flex:1;display:flex;flex-direction:column;gap:.18in;min-height:0}
.row{display:grid;gap:.3in;flex:none}
.r1{grid-template-columns:1fr 2.2fr}
.r2{grid-template-columns:.96fr 1.04fr}
.formula{display:flex;align-items:center;gap:.12in;flex-wrap:nowrap;font-size:27pt;margin:.04in 0 .14in}
.formula .lhs{font-weight:800}
.formula .op{font-weight:800;color:var(--muted)}
.formula .term{padding:.06in .14in;border-radius:.08in;border:2px solid;white-space:nowrap;font-weight:700}
.t1{border-color:#56B4E9 !important;background:#16263a}
.t2{border-color:#E69F00 !important;background:#2c2614}
.t3{border-color:#CC79A7 !important;background:#2a1a28}
.t4{border-color:#2BB88A !important;background:#12291f}
.simgrid{display:grid;grid-template-columns:1.08fr 1fr;gap:.35in;align-items:start}
.facts{font-size:26pt;padding-left:.36in}
.facts li{margin-bottom:.1in}
.score{display:grid;grid-template-columns:auto 1fr;column-gap:.45in;align-items:start}
.score h2{margin:0}
.score .scoregrid{display:grid;grid-template-columns:1fr 1fr;gap:.4in;font-size:26pt}
.score .scoregrid p{margin:0}
.r3{grid-template-columns:1fr 1fr;flex:1;min-height:0}
.r3 ul{font-size:26pt;margin-top:.12in}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:.1in;padding:.16in .28in .2in;min-height:0;
  overflow:hidden}
h2{font-size:34pt;line-height:1.1;margin:0 0 .12in;display:flex;align-items:center;gap:.14in;white-space:nowrap}
h2>b{display:inline-flex;align-items:center;justify-content:center;width:.56in;height:.56in;border-radius:50%;
  background:#c42f26;font-size:24pt;flex:none}
h2 .key{margin-left:auto;font-size:22pt;font-weight:600;color:var(--muted)}
p{margin:0 0 .1in}
p:last-child{margin-bottom:0}
.note{font-size:24pt;line-height:1.2;color:var(--muted)}
.note b{color:var(--ink)}
ul,ol{margin:0;padding-left:.42in}
li{margin-bottom:.08in}
.trace li{margin-bottom:.1in}
.trace li::marker{color:var(--amber);font-weight:800}
.plan{font-family:"DejaVu Sans Mono","Liberation Mono",monospace;font-size:.82em;color:var(--amber);white-space:nowrap}
q{font-style:italic;color:var(--muted)}
.chart{display:block;width:100%;height:auto;margin:.04in 0 .1in;font-family:"Noto Sans","Liberation Sans",Arial,sans-serif}
.caption{font-size:25pt;font-weight:800;margin:.08in 0 .04in}
.take{background:#16263a;border-color:var(--dev)}
.take p{font-size:30pt;line-height:1.24;margin:0}
.takegrid{display:grid;grid-template-columns:1fr 1fr;gap:.45in}
.prob{margin:.15in auto 0;max-width:5.6in}
.ex{font-size:27.5pt}
.nw{white-space:nowrap}
.ex .note{font-size:26pt}
footer{flex:none;border-top:2px solid var(--line);padding-top:.08in}
.foot{font-size:15pt;color:var(--muted);white-space:nowrap}
.foot code{font-size:1em}
.foot [lang=ja]{font-family:"Noto Sans CJK JP","Noto Sans CJK TC",sans-serif}
"""


def export():
    chrome = os.environ.get("CHROMIUM") or shutil.which("chromium") or shutil.which("google-chrome") or shutil.which("chrome")
    if not chrome:
        raise SystemExit("no Chromium found; set $CHROMIUM")
    subprocess.run([chrome, "--headless=new", "--no-pdf-header-footer", "--virtual-time-budget=5000",
                    f"--print-to-pdf={OUT_PDF}", OUT_HTML.as_uri()], check=True, capture_output=True)
    info = subprocess.run(["pdfinfo", str(OUT_PDF)], check=True, capture_output=True, text=True).stdout
    pages = re.search(r"Pages:\s+(\d+)", info).group(1)
    size = re.search(r"Page size:\s+(.*)", info).group(1)
    if pages != "1" or not size.startswith("1728 x 2592"):
        raise SystemExit(f"PDF is {pages} page(s) of {size}, expected 1 page of 1728 x 2592 pts")
    subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", str(OUT_PDF), str(OUT_PNG.with_suffix(""))], check=True)
    subprocess.run(["pdftoppm", "-scale-to-x", "1200", "-scale-to-y", "-1", "-png", "-singlefile", str(OUT_PDF),
                    str(OUT_PREVIEW.with_suffix(""))], check=True)
    print("wrote", OUT_PDF.name, f"({pages} page, {size}),", OUT_PNG.name, OUT_PREVIEW.name)


def main():
    global N
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dev", default=ps.DEV_DEFAULT, help=f"development set out-dir (default {ps.DEV_DEFAULT})")
    ap.add_argument("--test", default=ps.TEST_DEFAULT, help=f"test set out-dir (default {ps.TEST_DEFAULT})")
    ap.add_argument("--fidelity", default=str(ps.PROJECT / "reports" / "calibration" / "fidelity"),
                    help="the fidelity scorecard's outputs (default reports/calibration/fidelity)")
    ap.add_argument("--scorecard", default=ps.SCORECARD, help=f"scorecard under --fidelity (default {ps.SCORECARD})")
    ap.add_argument("--ablation", default=ps.ABLATION, help=argparse.SUPPRESS)
    ap.add_argument("--noise-floor", help="a ceiling_pooled.csv (default: the scorecard's own)")
    ap.add_argument("--fixture", action="store_true",
                    help="layout test only: accept a development set as --test and mark the poster NOT A RESULT")
    ap.add_argument("--export", action="store_true", help="also write the PDF, the 300 ppi PNG and the preview")
    ap.add_argument("--group", help="with --export: also copy the PDF to poster/GroupNN_Poster.pdf (NN = this number)")
    a = ap.parse_args()
    dev = a.dev if Path(a.dev).is_absolute() else str(ps.PROJECT / a.dev)
    test = a.test if Path(a.test).is_absolute() else str(ps.PROJECT / a.test)
    N = ps.compute(dev, test, a.fidelity, a.noise_floor, allow_dev_as_test=a.fixture, scorecard=a.scorecard,
                   ablation=a.ablation)
    fid_txt = "reports/calibration/fidelity" if Path(a.fidelity).resolve() == (ps.PROJECT / "reports" / "calibration" / "fidelity").resolve() else str(a.fidelity)
    args_line = " ".join(["--dev", str(a.dev), "--test", str(a.test), "--fidelity", fid_txt,
                          "--scorecard", a.scorecard])
    OUT_HTML.write_text(html(args_line), encoding="utf-8")
    (HERE / "numbers.md").write_text(ps.provenance_md(N, USED, args_line), encoding="utf-8")
    pend = [k for k in USED if N[k].pending]
    print(f"wrote {OUT_HTML.name} and numbers.md: {len(USED)} numbers, {len(pend)} pending")
    if a.export:
        export()
        if a.group:
            dst = HERE / f"Group{int(a.group):02d}_Poster.pdf"
            shutil.copyfile(OUT_PDF, dst)
            print("wrote", dst.name)


if __name__ == "__main__":
    main()
