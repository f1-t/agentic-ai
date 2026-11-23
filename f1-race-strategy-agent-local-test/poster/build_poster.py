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
    return f"matched {name} ({V(f'{g}_{x}_diff')} s)"


def rules_phrase():
    """Both simple rules at once, from the two computed verdicts."""
    v = [val(f"hall_{x}_verdict") for x in ("run_to_flag", "one_stop_total")]
    d = [V(f"hall_{x}_diff") for x in ("run_to_flag", "one_stop_total")]
    mags = [_html.escape(ps.s(abs(val(f"hall_{x}_diff")), 1)) for x in ("run_to_flag", "one_stop_total")]
    if v == ["ahead", "ahead"]:
        return f"beat both simple rules (by {mags[0]} and {mags[1]} s)"
    if v == ["level", "level"]:
        return f"matched both simple rules ({d[0]} and {d[1]} s)"
    if v == ["behind", "behind"]:
        return f"trailed both simple rules (by {mags[0]} and {mags[1]} s)"
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


def dot_chart(rows, key_mean, key_ci, key_txt, zero_line, width=540, label=None, aria=""):
    """One block per row: a test-set line (filled marker) and a development line (hollow), each with its 95%
    interval and its value printed at the right."""
    vals = []
    for r in rows:
        for t, g, _ in SETS:
            if ready(key_ci(t, g, r)):
                vals += [v for v in (*val(key_ci(t, g, r)), val(key_mean(t, g, r))) if np.isfinite(v)]
    ticks = nice_ticks(min(vals + [0.0]), max(vals + [0.0]), n=3)
    LO, HI = ticks[0], ticks[-1]
    L, R, VX = 178, 318, width
    sx = lambda v: L + (R - L) * (v - LO) / (HI - LO)  # noqa: E731
    top, sub, gap = (32 if zero_line else 8), 29, 13
    blk = 2 * sub + gap
    yb = top + blk * len(rows) - gap / 2
    H = yb + 30
    s = [f'<svg class="chart" viewBox="0 0 {width} {H:.0f}" role="img" aria-label="{aria}">']
    for tk in ticks:
        x = sx(tk)
        s.append(f'<line x1="{x:.1f}" y1="{top - 4}" x2="{x:.1f}" y2="{yb:.1f}" stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{x:.1f}" y="{yb + 24:.1f}" text-anchor="middle" font-size="19" fill="{MUTED}">'
                 f'{fmt_tick(tk)}{" s" if tk == ticks[-1] else ""}</text>')
    if zero_line and LO < 0 < HI:
        x0 = sx(0)
        s.append(f'<line x1="{x0:.1f}" y1="{top - 10}" x2="{x0:.1f}" y2="{yb:.1f}" stroke="{INK}" stroke-width="2"/>')
        s.append(f'<text x="{x0 - 6:.1f}" y="{top - 14}" text-anchor="end" font-size="18.5" fill="{MUTED}">'
                 f'← agent better</text>')
        s.append(f'<text x="{x0 + 6:.1f}" y="{top - 14}" font-size="18.5" fill="{MUTED}">worse →</text>')
    for i, r in enumerate(rows):
        name, color, shape = POL[r]
        y0 = top + blk * i
        s.append(f'<text x="0" y="{y0 + sub + 7:.1f}" font-size="21" font-weight="700" fill="{INK}">'
                 f'{label(name) if label else name}</text>')
        for j, (t, g, _) in enumerate(SETS):
            y = y0 + sub * j + sub / 2
            lo, hi = val(key_ci(t, g, r))
            m = val(key_mean(t, g, r))
            s.append(f'<line x1="{sx(lo):.1f}" y1="{y:.1f}" x2="{sx(hi):.1f}" y2="{y:.1f}" stroke="{color}" '
                     f'stroke-width="4"/>')
            s.append(marker(shape, sx(m), y, color, r=9, hollow=(t == "d")))
            s.append(f'<text x="{VX}" y="{y + 7:.1f}" text-anchor="end" font-size="19.5"'
                     f'{" font-weight=\"700\"" if t == "h" else ""} fill="{INK if t == "h" else MUTED}">'
                     f'{key_txt(t, g, r)}</text>')
    s.append("</svg>")
    return "".join(s)


def forest():
    return dot_chart(("run_to_flag", "one_stop_total", "actual_strategy", "belief_optimizer"),
                     lambda t, g, x: f"{g}_{x}_diff", lambda t, g, x: f"{g}_{x}_ci",
                     lambda t, g, x: f'{T(f"{g}_{x}_diff")} {T(f"{g}_{x}_ci")} {T(f"{g}_{x}_verdict")}', True,
                     label=lambda n: "− " + (n[0].lower() + n[1:] if not n.startswith("Teams") else "teams’ actual"),
                     aria="Agent minus each comparator, mean regret difference, 95% intervals, test and development")


def levels():
    return dot_chart(("agent", "belief_optimizer", "run_to_flag", "one_stop_total", "actual_strategy"),
                     lambda t, g, p: f"{t}_mean_{p}", lambda t, g, p: f"{t}_mci_{p}",
                     lambda t, g, p: f'{T(f"{t}_mean_{p}")} {T(f"{t}_mci_{p}")}', False,
                     aria="Mean regret per policy, 95% intervals, test and development")


def per_race(width=1000):
    """Agent minus run_to_flag per test race, 95% (race, driver) intervals."""
    races = list(val("hall_names"))
    lo_hi = [v for r in races for v in val(f"pr_{r}_agent_ci")]
    ticks = nice_ticks(min(lo_hi + [0]), max(lo_hi + [0]), n=5)
    LO, HI = ticks[0], ticks[-1]
    L, R, VX = 220, 880, width
    sx = lambda v: L + (R - L) * (v - LO) / (HI - LO)  # noqa: E731
    top, rowh = 30, 30
    yb = top + rowh * len(races)
    H = yb + 30
    s = [f'<svg class="chart" viewBox="0 0 {width} {H}" role="img" aria-label="Agent minus run to the flag per test '
         f'race, with 95% intervals">']
    for tk in ticks:
        s.append(f'<line x1="{sx(tk):.1f}" y1="{top - 6}" x2="{sx(tk):.1f}" y2="{yb}" stroke="{GRID}"/>')
        s.append(f'<text x="{sx(tk):.1f}" y="{yb + 24}" text-anchor="middle" font-size="19" fill="{MUTED}">'
                 f'{fmt_tick(tk)}{" s" if tk == ticks[-1] else ""}</text>')
    s.append(f'<line x1="{sx(0):.1f}" y1="{top - 10}" x2="{sx(0):.1f}" y2="{yb}" stroke="{INK}" stroke-width="2"/>')
    s.append(f'<text x="{sx(0) - 6:.1f}" y="{top - 12}" text-anchor="end" font-size="19" fill="{MUTED}">'
             f'← agent better</text>')
    s.append(f'<text x="{sx(0) + 6:.1f}" y="{top - 12}" font-size="19" fill="{MUTED}">worse →</text>')
    for i, r in enumerate(races):
        y = top + rowh * i + rowh / 2
        m, (lo, hi) = val(f"pr_{r}_agent"), val(f"pr_{r}_agent_ci")
        color = "#D55E00" if m > 0 else "#56B4E9"
        s.append(f'<text x="0" y="{y + 7:.1f}" font-size="21" fill="{INK}">{ps.race_name(r)}</text>')
        s.append(f'<rect x="{min(sx(0), sx(m)):.1f}" y="{y - 10:.1f}" width="{abs(sx(m) - sx(0)):.1f}" height="20" '
                 f'fill="{color}"/>')
        s.append(f'<line x1="{sx(lo):.1f}" y1="{y:.1f}" x2="{sx(hi):.1f}" y2="{y:.1f}" stroke="{INK}" stroke-width="2"/>')
        s.append(f'<text x="{VX}" y="{y + 7:.1f}" text-anchor="end" font-size="21" font-weight="700" fill="{INK}">'
                 f'{T(f"pr_{r}_agent")} {T(f"pr_{r}_agent_ci")}</text>')
        used(f"pr_{r}_agent_ci")
    s.append("</svg>")
    return "".join(s)


def judge_chart(width=470):
    """Close-pair order and relative gap error: the judge against the measured ceiling (noise floor)."""
    s = [f'<svg class="chart" viewBox="0 0 {width} 226" role="img" aria-label="The judge against real races: '
         f'close-pair order and relative gap error, the judge and the measured ceiling">']
    L, R = 92, width - 14

    def row(y, label, key, ckey, x_of, color, shape, above):
        lo, hi = val(ckey)
        s.append(f'<text x="0" y="{y + 6}" font-size="18.5" font-weight="700" fill="{INK}">{label}</text>')
        s.append(f'<line x1="{x_of(lo):.1f}" y1="{y}" x2="{x_of(hi):.1f}" y2="{y}" stroke="{color}" stroke-width="4"/>')
        s.append(marker(shape, x_of(val(key)), y, color))
        txt = f'{T(key)}{" s" if "gap" in ckey else ""}'
        if above:
            s.append(f'<text x="{x_of(val(key)):.1f}" y="{y - 13}" text-anchor="middle" font-size="18" font-weight="700" '
                     f'fill="{INK}">{txt}</text>')
        else:
            s.append(f'<text x="{x_of(hi) + 8:.1f}" y="{y + 6}" font-size="18" font-weight="700" fill="{INK}">{txt}</text>')

    xo = lambda v: L + (R - L) * (v - 0.5) / 0.5  # noqa: E731
    s.append(f'<text x="0" y="18" font-size="18.5" font-weight="800" fill="{INK}">Close finishers in the right order</text>')
    for tk in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        s.append(f'<line x1="{xo(tk):.1f}" y1="38" x2="{xo(tk):.1f}" y2="98" stroke="{GRID}"/>')
        s.append(f'<text x="{xo(tk):.1f}" y="114" text-anchor="middle" font-size="16.5" fill="{MUTED}">{tk:.1f}</text>')
    row(52, "Judge", "fid_close", "fid_close_ci", xo, "#56B4E9", "circle", True)
    row(76, "Ceiling", "ceil_close", "ceil_close_ci", xo, "#F0E442", "diamond", False)
    xg = lambda v: L + (R - L) * v / 15  # noqa: E731
    s.append(f'<text x="0" y="142" font-size="18.5" font-weight="800" fill="{INK}">Gap between cars: error</text>')
    for tk in (0, 3, 6, 9, 12, 15):
        s.append(f'<line x1="{xg(tk):.1f}" y1="150" x2="{xg(tk):.1f}" y2="206" stroke="{GRID}"/>')
        s.append(f'<text x="{xg(tk):.1f}" y="223" text-anchor="middle" font-size="16.5" fill="{MUTED}">'
                 f'{tk}{" s" if tk == 15 else ""}</text>')
    row(168, "Judge", "fid_relgap", "fid_relgap_ci", xg, "#56B4E9", "circle", True)
    row(190, "Floor", "ceil_relgap", "ceil_relgap_ci", xg, "#F0E442", "diamond", False)
    s.append("</svg>")
    return "".join(s)


def blind_diagram(width=380):
    """The future-blindness test: cut the race at the decision moment, alter the future, compare."""
    s = [f'<svg class="chart" viewBox="0 0 {width} 330" role="img" aria-label="Future-blindness test: the race is cut '
         f'at the decision moment, everything after it is altered, and everything a policy sees must stay identical">',
         '<defs><pattern id="scr" width="12" height="12" patternUnits="userSpaceOnUse" patternTransform="rotate(30)">'
         '<rect width="12" height="12" fill="#3a2430"/><line x1="0" y1="0" x2="0" y2="12" stroke="#ff4d3d" '
         'stroke-width="3"/></pattern></defs>']
    X0, XC, X1 = 0, 168, width
    for i, lab in enumerate(("as raced", "cut off", "scrambled")):
        y = 34 + 54 * i
        s.append(f'<rect x="{X0}" y="{y}" width="{XC - X0}" height="42" fill="#2c3d55" stroke="{MUTED}" stroke-width="1"/>')
        s.append(f'<text x="{(X0 + XC) / 2:.0f}" y="{y + 28}" text-anchor="middle" font-size="20" fill="{INK}">'
                 f'laps so far</text>')
        if lab == "cut off":
            s.append(f'<rect x="{XC}" y="{y}" width="{X1 - XC}" height="42" fill="none" stroke="{GRID}" '
                     f'stroke-width="1.5" stroke-dasharray="5 4"/>')
        else:
            s.append(f'<rect x="{XC}" y="{y}" width="{X1 - XC}" height="42" fill="{"#2c3d55" if i == 0 else "url(#scr)"}" '
                     f'stroke="{MUTED}" stroke-width="1"/>')
        s.append(f'<text x="{XC + 12}" y="{y + 28}" font-size="21" font-weight="800" fill="{INK}">{lab}</text>')
    s.append(f'<line x1="{XC}" y1="26" x2="{XC}" y2="204" stroke="#ff4d3d" stroke-width="4"/>')
    s.append(f'<text x="{XC}" y="20" text-anchor="middle" font-size="19" font-weight="800" fill="#ff8a7d">'
             f'decision moment</text>')
    s.append(f'<text x="0" y="242" font-size="19" fill="{MUTED}">scrambled: lap times, tyres, stops,</text>')
    s.append(f'<text x="0" y="266" font-size="19" fill="{MUTED}">safety cars, race length</text>')
    s.append(f'<text x="0" y="312" font-size="22" font-weight="800" fill="{INK}">All a policy sees: equal? '
             f'<tspan fill="#2BB88A">✓ yes</tspan></text>')
    s.append("</svg>")
    return "".join(s)


def system_diagram(model, width=470):
    """Predictive simulator -> look-ahead search -> LLM agent -> decision, scored by the judge."""
    def box(y, h, title, lines, color, fill):
        t = (f'<rect x="0" y="{y}" width="{width}" height="{h}" rx="9" fill="{fill}" stroke="{color}" stroke-width="2"/>'
             f'<text x="16" y="{y + 30}" font-size="22" font-weight="800" fill="{INK}">{title}</text>')
        for i, line in enumerate(lines):
            t += f'<text x="16" y="{y + 58 + 24 * i}" font-size="20" fill="{MUTED}">{line}</text>'
        return t

    def arrow(y):
        return (f'<line x1="{width / 2}" y1="{y}" x2="{width / 2}" y2="{y + 20}" stroke="{INK}" stroke-width="3" '
                f'marker-end="url(#ah)"/>')
    s = [f'<svg class="chart" viewBox="0 0 {width} 450" role="img" aria-label="The system">',
         f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
         f'orient="auto"><path d="M0,0L10,5L0,10Z" fill="{INK}"/></marker></defs>']
    s.append(box(0, 78, "Predictive simulator", ["only what is known at the decision moment"], "#9cc8ff", "#16263a"))
    s.append(arrow(80))
    s.append(box(104, 78, "Look-ahead search", ["ranks full race plans on that simulator"], "#E69F00", "#2c2614"))
    s.append(arrow(184))
    s.append(box(208, 104, f"LLM agent ({model})", ["reads the ranked plans and the race picture,",
                                                     "then makes the call: pit + tyre, or stay out"], "#56B4E9", "#16263a"))
    s.append(arrow(314))
    s.append(box(338, 78, "Judge", ["scores the plan against the whole race"], "#ffcf4a", "#2c2614"))
    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------- tables
def tail_table():
    pols = ("agent", "belief_optimizer", "run_to_flag", "one_stop_total", "actual_strategy")
    head = ('<tr><th></th><th>&gt;20 s off, test</th><th>&gt;20 s off, development</th><th>median, test</th>'
            '<th>worst, test</th></tr>')
    body = []
    for pol in pols:
        name, color, _ = POL[pol]
        body.append(f'<tr><th class="rl"><i class="sw" style="background:{color}"></i>{name}</th>'
                    f'<td><b>{V(f"hall_t_over20_{pol}")}</b></td><td class="dvc">{V(f"d_t_over20_{pol}")}</td>'
                    f'<td>{V(f"hall_t_median_{pol}")} s</td><td>{V(f"hall_t_worst_{pol}")} s</td></tr>')
    return f'<table class="tbl tail"><thead>{head}</thead><tbody>{"".join(body)}</tbody></table>'


def adopt_table():
    a, n = val("adopted"), val("not_adopted")
    rows = "".join(f"<tr><td>{x}</td><td>{y}</td></tr>" for x, y in
                   zip(a + [""] * (len(n) - len(a)), n + [""] * (len(a) - len(n))))
    return ('<table class="tbl adopt"><thead><tr><th>Adopted</th><th>Tested, not adopted</th></tr></thead>'
            f"<tbody>{rows}</tbody></table>")


# ---------------------------------------------------------------- page
def html(args_line):
    D, H = N["_sets"].value
    model = T("h_model") if ready("h_model") else T("d_model")
    ribbon = ""
    if H.fixture:
        ribbon = '<div class="ribbon fix">LAYOUT FIXTURE: development rows in the test column. NOT A RESULT.</div>'
    elif not H.present or not H.has_agent:
        ribbon = '<div class="ribbon">DRAFT: hatched boxes are filled by the test-set run.</div>'
    body = f"""
<div class="poster">
{ribbon}
<header class="top">
  <div class="title">
    <div class="kicker">CSE 598 Agentic AI &middot; Arizona State University</div>
    <h1>PITWALL</h1>
    <div class="acro">Predictive Intelligence for Tactical When-to-pit Agentic Lap-wise Logic</div>
    <div class="q">Can a language model call the pit stop?</div>
    <div class="authors">Jason Lunder &middot; Rishabh Baral &middot; Sarah Maisha &middot; Quan Nguyen &middot; Manya Jain<br>
      <span>Stakeholder: Manav Ananthakumar &middot; TA: Xiaoou Liu &middot; Professor: Hua Wei</span></div>
  </div>
  <figure class="map">
    <img src="{MAP}" alt="Circuit de Barcelona-Catalunya, 2023 layout">
    <figcaption>Barcelona-Catalunya [4]</figcaption>
  </figure>
</header>

<div class="answer">
  <div class="pts">
    <div><b>Future-blind.</b> Scrambling a race after the decision moment changes nothing a policy sees.</div>
    <div><b>A measured judge.</b> It orders close finishers correctly {V("fid_close")} of the time (ceiling
      {V("ceil_close")}); gap error {V("fid_relgap")} s (floor {V("ceil_relgap")} s).</div>
    <div><b>Test set, {V("h_ag_scored")} decisions in {V("h_races")} races.</b> The agent
      {versus("hall", "run_to_flag", "run to the flag")}, {versus("hall", "one_stop_total", "one stop in total")}
      and {versus("hall", "actual_strategy", "the teams’ actual strategy")}, and
      {versus("hall", "belief_optimizer", "its own look-ahead")}.</div>
  </div>
</div>

<main class="grid">
  <div class="row1">
  <article class="panel p1">
    <h2><b>1</b>The system</h2>
    <p>Each lap a strategist decides: pit now or stay out.</p>
    {system_diagram(model)}
    <p class="note"><b>Simple rules:</b> run to the flag (stop only if tyre life forces it); one stop in total.</p>
  </article>

  <article class="panel p2">
    <h2><b>2</b>Evaluation integrity</h2>
    <p>A policy reads only the race up to the decision moment, this weekend&rsquo;s practice, the rules and tables
    built from earlier races. A test checks it:</p>
    {blind_diagram()}
  </article>

  <article class="panel p3">
    <h2><b>3</b>How good is the judge?</h2>
    <p>The judge replays every car&rsquo;s real strategy in {V("fid_races")} races. The ceiling is the judge given
    every stop, safety-car lap and start as recorded.</p>
    {judge_chart()}
    <p>The look-ahead&rsquo;s favourite plan is within {V("fid_b2regret")} s of the judge&rsquo;s best.</p>
    {adopt_table()}
  </article>
  </div>

  <article class="panel p4">
    <h2><b>4</b>Results</h2>
    <p class="def"><b>Regret</b>: a plan&rsquo;s extra race time over the judge&rsquo;s best plan. Lower is better.
    Paired on the same decisions; 95% intervals resample drivers within races. <b>Filled</b>: test set
    ({V("h_ag_scored")} decisions, {V("h_races")} races). <b>Hollow</b>: development set ({V("d_ag_scored")} decisions,
    {V("d_races")} races).</p>
    <div class="figs">
      <div><div class="caption">Agent minus each comparator (s)</div>{forest()}</div>
      <div><div class="caption">Mean regret (s)</div>{levels()}</div>
    </div>
    <div class="caption">Per test race: agent minus run to the flag (s), 95% intervals</div>
    {per_race()}
    <div class="caption">Decisions more than 20 s off, median and worst regret</div>
    {tail_table()}
  </article>

  <div class="stack">
  <article class="panel p5">
    <h2><b>5</b>What the LLM adds</h2>
    <ul>
      <li>It matches its look-ahead: {V("hall_belief_optimizer_diff")} s on the test set,
      {V("d_belief_optimizer_diff")} s on development.</li>
      <li>A legal, explained call every time ({V("h_ag_ok")}; {V("d_ag_ok")}), in about {V("h_lat")}, {V("h_calls")}
      tool calls and {V("h_tok")} input tokens.</li>
      <li>Its departures from the top plan helped {V("h_dep_helped")} and hurt {V("h_dep_hurt")} times on the test set,
      {V("d_dep_helped")} and {V("d_dep_hurt")} on development.</li>
    </ul>
    <p>The gains come from the simulator and the search.</p>
  </article>

  <article class="panel p6">
    <h2><b>6</b>Limitations and next</h2>
    <ul>
      <li>The look-ahead misjudges hard-tyre pace early in races at circuits without history: Australia and Miami
      2022.</li>
      <li>One modest model, {model}; one pass.</li>
      <li>The judge scores race time, not position.</li>
    </ul>
    <p><b>Next:</b> tyre knowledge from telemetry; a stronger model.</p>
  </article>

  <article class="panel p7">
    <h2><b>7</b>For the stakeholder</h2>
    <p>On nine test races the agent {rules_phrase()} and
    {versus("hall", "actual_strategy", "the teams’ actual strategies")}, with a legal call every time. Its edge comes from the predictive simulator and the look-ahead; the
    language model matches the search. The next gains are tyre knowledge at new circuits and a stronger model.</p>
  </article>
  </div>
</main>

<footer>
  <div class="refs">
    [1] Code and protocol (<code>docs/eval.md</code>): github.com/f1-t/agentic-ai. [2] Results:
    <code>reports/eval/rerun/v3/{{dev20,test9}}</code>; scorecard <code>reports/calibration/fidelity</code>; every number
    and its computation: <code>poster/numbers.md</code>. [3] Timing data: FastF1. [4] Map: &ldquo;2023 F1 CourseLayout
    Spain.svg&rdquo; by <span lang="ja">ごひょううべこ</span>, Wikimedia Commons, CC BY-SA 4.0. Independent student
    project, not affiliated with Formula 1, the FIA or Pirelli.
  </div>
  <div class="foot">Model {model} &middot; tag <code>{V("h_tag")}</code> ({V("h_commit")}) &middot; test set
    {V("h_races")} races, n = {V("h_ag_scored")} &middot; development set {V("d_races")} races, n = {V("d_ag_scored")}
    &middot; one agent pass</div>
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
body{font-family:"Noto Sans","Liberation Sans",Arial,sans-serif;color:var(--ink);font-size:26pt;line-height:1.22}
code{font-family:"DejaVu Sans Mono","Liberation Mono",monospace;font-size:.86em}
.poster{position:relative;width:24in;height:36in;padding:.6in .7in .6in;display:flex;flex-direction:column;
  gap:.2in;overflow:hidden;border-top:.12in solid var(--red)}
.ribbon{flex:none;border:3px dashed var(--pend);color:var(--pend);font-size:19pt;font-weight:800;
  padding:.04in .2in;border-radius:.08in;text-align:center}
.ribbon.fix{border-color:#ff4d3d;color:#ff8a7d}
.top{display:flex;justify-content:space-between;align-items:center;gap:.4in;flex:none}
.kicker{font-size:20pt;letter-spacing:.12em;text-transform:uppercase;color:var(--red);font-weight:700}
h1{font-size:96pt;line-height:1;margin:0;letter-spacing:.02em;font-weight:900}
.acro{font-size:22pt;color:var(--amber);font-weight:700}
.q{font-size:40pt;font-weight:700;margin:.02in 0 .05in;line-height:1.12}
.authors{font-size:23pt;line-height:1.26}
.authors span{font-size:20pt;color:var(--muted)}
.map{margin:0;width:5.6in;flex:none;text-align:center}
.map img{width:5.6in;height:auto;display:block}
.map figcaption{font-size:16pt;color:var(--muted);margin-top:.02in}
.answer{flex:none;background:#2a1418;border-left:.14in solid var(--red);padding:.12in .26in .14in;font-size:24pt;
  line-height:1.22}
.pts{display:grid;grid-template-columns:.9fr 1fr 1.6fr;gap:.32in}
.pts b{font-size:25pt;color:var(--amber)}
.pend{display:inline-block;font-size:.88em;font-style:italic;font-weight:700;color:var(--pend);
  border:2px dashed var(--pend);border-radius:.06in;padding:0 .07in}
.grid{flex:1;display:grid;grid-template-columns:1fr 1fr 1fr;grid-template-rows:auto 1fr;gap:.24in .32in;min-height:0}
.row1{grid-column:1 / span 3;display:grid;grid-template-columns:1fr .86fr 1.14fr;gap:.32in}
.p4{grid-column:1 / span 2}
.stack{display:flex;flex-direction:column;gap:.24in;min-height:0}
.stack .panel{flex:none}
.stack .p7{flex:1}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:.1in;padding:.16in .26in .18in;min-height:0;
  overflow:hidden}
h2{font-size:33pt;line-height:1.1;margin:0 0 .1in;display:flex;align-items:center;gap:.14in;white-space:nowrap}
h2>b{display:inline-flex;align-items:center;justify-content:center;width:.54in;height:.54in;border-radius:50%;
  background:#c42f26;font-size:23pt;flex:none}
p{margin:0 0 .1in}
p:last-child{margin-bottom:0}
.note{font-size:23pt;line-height:1.2;color:var(--muted)}
.note b{color:var(--ink)}
.def{border-left:.08in solid var(--amber);padding-left:.14in;font-size:23pt;line-height:1.22}
ul{margin:0 0 .1in;padding-left:.36in}
li{margin-bottom:.06in}
.chart{display:block;width:100%;height:auto;margin:.04in 0 .1in;font-family:"Noto Sans","Liberation Sans",Arial,sans-serif}
.caption{font-size:23pt;font-weight:800;margin:.1in 0 .02in}
.figs{display:grid;grid-template-columns:1fr 1fr;gap:.36in;margin:.06in 0}
.figs.b{grid-template-columns:1.15fr 1fr}
.tbl{width:100%;border-collapse:collapse;font-size:22pt;line-height:1.18;margin:.06in 0 .06in}
.tbl th,.tbl td{padding:.04in .06in;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
.tbl thead th{font-size:20pt;color:var(--muted);font-weight:700;vertical-align:bottom}
.tbl th.rl,.tbl thead th:first-child{text-align:left;font-weight:700}
.tbl td.dvc{color:var(--muted)}
.tbl.adopt td,.tbl.adopt th{text-align:left;white-space:normal;width:50%}
.tbl.adopt thead th{color:var(--amber);font-size:20pt}
.sw{display:inline-block;width:.18in;height:.18in;border-radius:.03in;margin-right:.08in;vertical-align:.02em}
.p7{background:#16263a;border-color:var(--dev)}
footer{flex:none;border-top:2px solid var(--line);padding-top:.08in;display:flex;flex-direction:column;gap:.04in}
.refs{font-size:13pt;line-height:1.3;color:var(--muted)}
.refs code{font-size:1em}
.refs [lang=ja]{font-family:"Noto Sans CJK JP","Noto Sans CJK TC",sans-serif}
.foot{font-size:16pt;font-weight:700}
.foot code{font-size:1em}
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


if __name__ == "__main__":
    main()
