"""Build the 24 x 36 in PITWALL poster from the corrected evaluation pipeline's committed outputs.

    python poster/build_poster.py --dev reports/eval/rerun/dev20 --test reports/eval/rerun/test9 \
        --fidelity reports/calibration/fidelity --export

`--dev` / `--test`: the `--out-dir` of `python -m src.eval.rerun` for the development set (dev20) and the report
set (test9), after its `tables` stage. `--fidelity`: the fidelity scorecard's outputs. Without `--test`, or
without agent rows, every number that needs them prints as a visibly marked placeholder and the build still
succeeds. Every number comes from `poster_stats.compute()`; `numbers.md` lists each one with its file and
computation, and the build stops if a cross-check against the pipeline's own tables fails. Verdict words come
from the intervals (reporting.yaml's rule), never from text.

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
# one colour (Okabe-Ito, colour-blind safe) and one marker shape per comparator; values always printed beside
POL = {"agent": ("Agent", "#56B4E9", "circle"),
       "belief_optimizer": ("Optimizer, no LLM", "#E69F00", "square"),
       "actual_strategy": ("Teams’ actual", "#2BB88A", "diamond"),
       "one_stop_total": ("One stop in total", "#D55E00", "up"),
       "run_to_flag": ("Run to the flag", "#F0E442", "down")}
GROUP_ROWS = (("hfresh", "fresh 5 (primary)"), ("hearlier", "earlier 4"), ("hall", "all 9"), ("d", "development"))
USED: list[str] = []
N: dict = {}


# ---------------------------------------------------------------- numbers as printed
def used(key):
    if key not in USED:
        USED.append(key)
    return N[key]


def V(key, chip="pending"):
    n = used(key)
    if n.pending:
        return f'<span class="pend" title="{n.tag}">{chip}</span>'
    return _html.escape(n.text)


def T(key):
    """Plain text of a number for SVG (no markup); marks it used."""
    n = used(key)
    return "pending" if n.pending else _html.escape(n.text)


def ready(*keys):
    return all(not used(k).pending for k in keys)


def val(key):
    return used(key).value


VERB = {"ahead": "ahead of", "behind": "behind", "level": "level with"}


def vword(key):
    """'ahead of' / 'level with' / 'behind' from a computed verdict."""
    if not ready(key):
        return '<span class="pend">pending</span>'
    return VERB[val(key)]


# ---------------------------------------------------------------- svg pieces (1 user unit = 1 pt)
def marker(shape, x, y, color, r=8.0, hollow=False, faint=False):
    fill = PANEL if hollow else color
    sw = 3 if hollow else 1.5
    stroke = color if hollow else PANEL
    op = ' fill-opacity="0.55"' if faint else ""
    st = f'fill="{fill}"{op} stroke="{stroke}" stroke-width="{sw}"'
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


def hatch_defs(pid):
    return (f'<defs><pattern id="{pid}" width="10" height="10" patternUnits="userSpaceOnUse" '
            f'patternTransform="rotate(45)"><rect width="10" height="10" fill="#22213a"/>'
            f'<line x1="0" y1="0" x2="0" y2="10" stroke="#4d4b69" stroke-width="4"/></pattern></defs>')


def pend_band(pid, x0, x1, y, h, text, size=16.5):
    return (f'<rect x="{x0:.1f}" y="{y - h / 2:.1f}" width="{x1 - x0:.1f}" height="{h:.1f}" rx="5" fill="url(#{pid})" '
            f'stroke="{PEND_INK}" stroke-width="1.5" stroke-dasharray="6 4"/>'
            f'<text x="{(x0 + x1) / 2:.1f}" y="{y + size * 0.36:.1f}" text-anchor="middle" font-size="{size}" '
            f'font-style="italic" font-weight="700" fill="{PEND_INK}">{text}</text>')


def nice_ticks(lo, hi, n=6):
    span = max(hi - lo, 1e-9)
    step = min((st for st in (1, 2, 2.5, 5, 10, 20, 25, 50) if span / st <= n), default=50)
    t0 = np.floor(lo / step) * step
    return [round(t0 + i * step, 6) for i in range(int(np.ceil((hi - t0) / step - 1e-9)) + 1)]


def fmt_tick(t):
    return (f"{t:.0f}" if float(t).is_integer() else f"{t:g}").replace("-", "−")


def forest(width=1000):
    """Agent minus each comparator: fresh five (the pre-registered primary test), earlier four, all nine and
    development, each with its 95% (race, driver) interval, value and computed verdict."""
    comps = ("run_to_flag", "one_stop_total", "actual_strategy", "belief_optimizer")
    vals = []
    for x in comps:
        for g, _ in GROUP_ROWS:
            if ready(f"{g}_{x}_ci"):
                vals += [v for v in (*val(f"{g}_{x}_ci"), val(f"{g}_{x}_diff")) if np.isfinite(v)]
    ticks = nice_ticks(min(vals + [-2.0]), max(vals + [2.0]))
    LO, HI = ticks[0], ticks[-1]
    L, R, VX = 380, 735, width
    sx = lambda v: L + (R - L) * (v - LO) / (HI - LO)  # noqa: E731
    top, sub, gap = 34, 21, 9
    blk = sub * len(GROUP_ROWS) + gap
    yb = top + blk * len(comps) - gap / 2
    H = yb + 28
    pid = "hfo"
    s = [f'<svg class="chart" viewBox="0 0 {width} {H:.0f}" role="img" aria-label="Agent minus each comparator, '
         f'mean regret difference in seconds with 95% intervals, by group">', hatch_defs(pid)]
    for i, x in enumerate(comps):   # the primary rows, shaded
        y0 = top + blk * i
        s.append(f'<rect x="{L - 172}" y="{y0 - 1}" width="{width - L + 172}" height="{sub}" rx="4" fill="#2d2a4a"/>')
    for tk in ticks:
        xx = sx(tk)
        s.append(f'<line x1="{xx:.1f}" y1="{top - 4}" x2="{xx:.1f}" y2="{yb:.1f}" stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{xx:.1f}" y="{yb + 22:.1f}" text-anchor="middle" font-size="17" fill="{MUTED}">'
                 f'{fmt_tick(tk)}{" s" if tk == ticks[-1] else ""}</text>')
    if LO < 0 < HI:
        x0 = sx(0)
        s.append(f'<line x1="{x0:.1f}" y1="{top - 10}" x2="{x0:.1f}" y2="{yb:.1f}" stroke="{INK}" stroke-width="2"/>')
        s.append(f'<text x="{x0 - 6:.1f}" y="{top - 14}" text-anchor="end" font-size="17" fill="{MUTED}">'
                 f'← agent better</text>')
        s.append(f'<text x="{x0 + 6:.1f}" y="{top - 14}" font-size="17" fill="{MUTED}">agent worse →</text>')
    s.append(f'<text x="{VX}" y="{top - 14}" text-anchor="end" font-size="17" fill="{MUTED}">agent − X [95%] verdict</text>')
    for i, x in enumerate(comps):
        name, color, shape = POL[x]
        y0 = top + blk * i
        s.append(f'<text x="0" y="{y0 + sub * 2 + 2:.1f}" font-size="19.5" font-weight="800" fill="{INK}">'
                 f'minus {name[0].lower() + name[1:] if x != "actual_strategy" else "teams’ actual"}</text>')
        for j, (g, glab) in enumerate(GROUP_ROWS):
            y = y0 + sub * j + sub / 2 - 1
            bold = ' font-weight="800"' if j == 0 else ""
            s.append(f'<text x="{L - 12}" y="{y + 6:.1f}" text-anchor="end" font-size="17.5"{bold} '
                     f'fill="{INK if j == 0 else MUTED}">{glab}</text>')
            if ready(f"{g}_{x}_ci"):
                lo, hi = val(f"{g}_{x}_ci")
                m = val(f"{g}_{x}_diff")
                if np.isfinite(lo):
                    s.append(f'<line x1="{sx(lo):.1f}" y1="{y:.1f}" x2="{sx(hi):.1f}" y2="{y:.1f}" stroke="{color}" '
                             f'stroke-width="{4 if j == 0 else 3}"/>')
                s.append(marker(shape, sx(m), y, color, r=8 if j == 0 else 7, hollow=(g == "d"), faint=(j in (1, 2))))
                txt = f'{T(f"{g}_{x}_diff")} {T(f"{g}_{x}_ci")} {T(f"{g}_{x}_verdict")}'
                s.append(f'<text x="{VX}" y="{y + 6:.1f}" text-anchor="end" font-size="17.5"{bold} '
                         f'fill="{INK if j == 0 else MUTED}">{txt}</text>')
            else:
                s.append(pend_band(pid, L, VX, y, sub - 5, f"{used(f'{g}_{x}_ci').tag} pending"))
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
    xg = lambda v: L + (R - L) * v / 25  # noqa: E731
    s.append(f'<text x="0" y="142" font-size="18.5" font-weight="800" fill="{INK}">Gap between cars: error</text>')
    for tk in (0, 5, 10, 15, 20, 25):
        s.append(f'<line x1="{xg(tk):.1f}" y1="150" x2="{xg(tk):.1f}" y2="206" stroke="{GRID}"/>')
        s.append(f'<text x="{xg(tk):.1f}" y="223" text-anchor="middle" font-size="16.5" fill="{MUTED}">'
                 f'{tk}{" s" if tk == 25 else ""}</text>')
    row(168, "Judge", "fid_relgap", "fid_relgap_ci", xg, "#56B4E9", "circle", True)
    row(190, "Floor", "ceil_relgap", "ceil_relgap_ci", xg, "#F0E442", "diamond", False)
    s.append("</svg>")
    return "".join(s)


def blind_diagram(width=470):
    """The future-blindness test: cut the race at the decision moment, alter the future, compare."""
    s = [f'<svg class="chart" viewBox="0 0 {width} 178" role="img" aria-label="Future-blindness test: the race is cut '
         f'at the decision moment, everything after it is altered, and everything a policy sees must stay identical">',
         '<defs><pattern id="scr" width="12" height="12" patternUnits="userSpaceOnUse" patternTransform="rotate(30)">'
         '<rect width="12" height="12" fill="#3a2430"/><line x1="0" y1="0" x2="0" y2="12" stroke="#ff4d3d" '
         'stroke-width="3"/></pattern></defs>']
    X0, XC, X1 = 0, 150, 300
    for i, lab in enumerate(("as raced", "cut off", "scrambled")):
        y = 28 + 38 * i
        s.append(f'<rect x="{X0}" y="{y}" width="{XC - X0}" height="28" fill="#2c3d55" stroke="{MUTED}" stroke-width="1"/>')
        s.append(f'<text x="{(X0 + XC) / 2:.0f}" y="{y + 20}" text-anchor="middle" font-size="17" fill="{INK}">'
                 f'laps so far</text>')
        if lab == "cut off":
            s.append(f'<rect x="{XC}" y="{y}" width="{X1 - XC}" height="28" fill="none" stroke="{GRID}" '
                     f'stroke-width="1.5" stroke-dasharray="5 4"/>')
        else:
            s.append(f'<rect x="{XC}" y="{y}" width="{X1 - XC}" height="28" fill="{"#2c3d55" if i == 0 else "url(#scr)"}" '
                     f'stroke="{MUTED}" stroke-width="1"/>')
        s.append(f'<text x="{XC + 10}" y="{y + 20}" font-size="17.5" font-weight="800" fill="{INK}">{lab}</text>')
    s.append(f'<line x1="{XC}" y1="24" x2="{XC}" y2="146" stroke="#ff4d3d" stroke-width="4"/>')
    s.append(f'<text x="{XC}" y="18" text-anchor="middle" font-size="16.5" font-weight="800" fill="#ff8a7d">'
             f'decision moment</text>')
    s.append(f'<path d="M{X1 + 8},30 q10,0 10,12 v32 q0,10 10,10 q-10,0 -10,10 v32 q0,12 -10,12" fill="none" '
             f'stroke="{MUTED}" stroke-width="2"/>')
    s.append(f'<text x="{X1 + 34}" y="66" font-size="17.5" font-weight="700" fill="{INK}">all a policy</text>')
    s.append(f'<text x="{X1 + 34}" y="88" font-size="17.5" font-weight="700" fill="{INK}">sees: equal?</text>')
    s.append(f'<text x="{X1 + 34}" y="116" font-size="19" font-weight="800" fill="#2BB88A">✓ yes</text>')
    s.append(f'<text x="0" y="172" font-size="16.5" fill="{MUTED}">scrambled: lap times, tyres, stops, safety cars, '
             f'race length</text>')
    s.append("</svg>")
    return "".join(s)


def loop_diagram(model, width=470):
    """The agent's loop: race view -> LLM <-> tools -> decision -> optimizer completes the plan."""
    def box(x, y, w, h, title, sub, color, fill="#24233d"):
        t = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{color}" stroke-width="2"/>'
             f'<text x="{x + w / 2}" y="{y + 24}" text-anchor="middle" font-size="18.5" font-weight="800" '
             f'fill="{INK}">{title}</text>')
        for i, line in enumerate(sub):
            t += (f'<text x="{x + w / 2}" y="{y + 46 + 21 * i}" text-anchor="middle" font-size="16.5" '
                  f'fill="{MUTED}">{line}</text>')
        return t

    def arrow(x1, y1, x2, y2, both=False):
        return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{INK}" stroke-width="2.5" marker-end="url(#ah)"'
                + (' marker-start="url(#ah)"' if both else "") + "/>")
    s = [f'<svg class="chart" viewBox="0 0 {width} 236" role="img" aria-label="The agent loop">',
         f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         f'orient="auto-start-reverse"><path d="M0,0L10,5L0,10Z" fill="{INK}"/></marker></defs>']
    s.append(box(0, 0, 122, 92, "Race view", ["anonymised,", "laps so far"], "#9cc8ff", "#16263a"))
    s.append(box(150, 0, 140, 92, "LLM", [model, "plans, asks"], "#56B4E9"))
    s.append(box(340, 0, 130, 92, "Tools", ["tyre wear,", "pit loss, plans"], "#9cc8ff", "#16263a"))
    s.append(arrow(124, 46, 147, 46))
    s.append(arrow(296, 46, 334, 46, both=True))
    s.append(f'<text x="315" y="112" text-anchor="middle" font-size="16.5" fill="{MUTED}">up to 10 turns</text>')
    s.append(arrow(220, 95, 220, 136))
    s.append(box(120, 140, 160, 92, "Decision", ["PIT + tyre", "or STAY OUT"], "#F0E442"))
    s.append(box(322, 140, 148, 92, "Optimizer", ["completes", "the plan"], "#E69F00"))
    s.append(arrow(282, 186, 319, 186))
    for i, line in enumerate(("2 submit-only", "turns force", "a decision")):
        s.append(f'<text x="0" y="{170 + 21 * i}" font-size="16.5" fill="{MUTED}">{line}</text>')
    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------- tables
def tail_table():
    pols = ("agent", "belief_optimizer", "run_to_flag", "one_stop_total", "actual_strategy")
    head = ('<tr><th>share of decisions<br>more than x s off</th><th>&gt;20 s</th><th>&gt;10 s</th><th>&gt;30 s</th>'
            '<th>median</th><th>worst</th><th class="sep">&gt;20 s,<br>all 9</th><th>&gt;20 s,<br>dev</th></tr>')
    body = []
    for pol in pols:
        name, color, _ = POL[pol]
        cells = (f'<td><b>{V(f"hfresh_t_over20_{pol}")}</b></td><td>{V(f"hfresh_t_over10_{pol}")}</td>'
                 f'<td>{V(f"hfresh_t_over30_{pol}")}</td><td>{V(f"hfresh_t_median_{pol}")} s</td>'
                 f'<td>{V(f"hfresh_t_worst_{pol}")} s</td><td class="sep dvc">{V(f"hall_t_over20_{pol}")}</td>'
                 f'<td class="dvc">{V(f"d_t_over20_{pol}")}</td>')
        body.append(f'<tr><th class="rl"><i class="sw" style="background:{color}"></i>{name}</th>{cells}</tr>')
    return f'<table class="tbl tail"><thead>{head}</thead><tbody>{"".join(body)}</tbody></table>'


def era_table():
    rows = []
    for e in val("h_eras"):
        cells = []
        for x in ("run_to_flag", "actual_strategy"):
            cells.append(f'<td><b>{V(f"era_{e}_{x}_diff")}</b> <span class="cil">{V(f"era_{e}_{x}_ci")}</span><br>'
                         f'<span class="vd">{V(f"era_{e}_{x}_verdict")}</span></td>')
        rows.append(f'<tr><th class="rl">{e}<br><span class="races">{V(f"era_{e}_names")}</span></th>{"".join(cells)}</tr>')
    return ('<table class="tbl era"><thead><tr><th>era (held-out)</th><th>agent − run to flag</th>'
            '<th>agent − teams</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table>")


def factor_table():
    def r(label, *items):
        out = []
        for k, unit, what in items:
            d, lo, hi = val(k)
            nd = 3 if unit == "" else 1
            out.append(f'<b>{_html.escape(ps.s(d, nd, sign=True))}{unit}</b> {what} '
                       f'<span class="cil">{_html.escape(ps.ci(lo, hi, nd))}</span>')
        return f'<tr><th class="rl">{label}</th><td>{"<br>".join(out)}</td></tr>'
    return ('<table class="tbl fac"><tbody>'
            '<tr class="grp"><th colspan="2">Adopted: switching each off costs</th></tr>'
            + r("Safety cars, judge", ("abl_world_events_off_A_close_pair_acc", "", "order"),
                ("abl_world_events_off_A_rel_gap_mae_s", " s", "gaps"))
            + r("Tyre life from history only", ("abl_cliff_edge_history_A_close_pair_acc", "", "order"))
            + r("Driver effects, predictive", ("abl_belief_no_driver_effects_B2_regret_s", " s", "plan loss"))
            + '<tr class="grp"><th colspan="2">Rejected: switching each on costs</th></tr>'
            + r("Traffic", ("abl_traffic_both_A_close_pair_acc", "", "order"))
            + r("Wear hinge, judge", ("abl_world_hinge_A_close_pair_acc", "", "order"))
            + r("Learned calibrator*", ("abl_learned_trained_B2_regret_s", " s", "plan loss"))
            + "</tbody></table>")


def config_line(t):
    on = [lab for k, lab in (("sc", "safety cars"), ("edge", "history-only tyre life"), ("bde", "predictive driver effects"))
          if ready(f"{t}_{k}") and val(f"{t}_{k}")]
    off = [lab for k, lab in (("traffic", "traffic"), ("hinge", "hinge"), ("learned", "learned calibrator"))
           if ready(f"{t}_{k}") and not val(f"{t}_{k}")]
    for k in ("sc", "edge", "bde", "traffic", "hinge", "learned"):
        used(f"{t}_{k}")
    return ", ".join(on) or "none of the adopted factors"


def leak_list():
    return "; ".join(ps.LEAK_SHORT.get(k, k.replace("_", " ")) for k in val("h_fb_fixed"))


# ---------------------------------------------------------------- the sentences that depend on the result
def lead_lost():
    """'did not generalise' when development is ahead and the fresh five are not."""
    if not ready("d_run_to_flag_verdict", "hfresh_run_to_flag_verdict"):
        return '<span class="pend">pending</span>'
    d, f = val("d_run_to_flag_verdict"), val("hfresh_run_to_flag_verdict")
    if d == "ahead" and f != "ahead":
        return "did not generalise"
    return "held" if d == f else "changed"


def fresh_hist_clause():
    """None of the fresh five had an earlier race at its circuit; the earlier four are not a clean comparison."""
    if not ready("fresh_hist_n", "earlier_hist_n"):
        return '<span class="pend">pending</span>'
    none_ = val("fresh_hist_n") == 0
    return (f"{'None' if none_ else V('fresh_hist_n')} of the {V('hfresh_word')} fresh races had an earlier race at "
            f"{'its' if none_ else 'their'} circuit, so tyre life came from a generic prior; the earlier four are no "
            f"clean comparison ({V('earlier_hist_n')} had circuit history, all were examined before), so the fresh "
            f"five is the test that counts.")


def per_race_chart(width=1000):
    """Agent minus run_to_flag per held-out race, fresh five (left) and earlier four (right), 95% intervals;
    a dagger marks a race whose tools had a same-circuit tyre-life prior."""
    halves = (("hfresh", "fresh five"), ("hearlier", "earlier four"))
    races = {g: list(val(f"{g}_names")) for g, _ in halves}
    lo_hi = [v for g in races for r in races[g] for v in val(f"pr_{r}_agent_ci")]
    ticks = nice_ticks(min(lo_hi + [0]), max(lo_hi + [0]), n=4)
    LO, HI = ticks[0], ticks[-1]
    top, rowh = 52, 21
    n = max(len(v) for v in races.values())
    yb = top + rowh * n
    H = yb + 26
    s = [f'<svg class="chart" viewBox="0 0 {width} {H}" role="img" aria-label="Agent minus run to the flag per '
         f'held-out race, with 95% intervals">',
         f'<text x="0" y="17" font-size="19" font-weight="800" fill="{INK}">Per race: agent minus run to the flag (s), '
         f'95% intervals</text>']
    for h, (g, glab) in enumerate(halves):
        x0 = h * width / 2
        L, R, VX = x0 + 150, x0 + 400, x0 + 482
        sx = lambda v, L=L, R=R: L + (R - L) * (v - LO) / (HI - LO)  # noqa: E731
        s.append(f'<text x="{x0}" y="42" font-size="17.5" font-weight="800" fill="{MUTED}">{glab}</text>')
        for tk in ticks:
            s.append(f'<line x1="{sx(tk):.1f}" y1="{top - 6}" x2="{sx(tk):.1f}" y2="{yb}" stroke="{GRID}"/>')
            s.append(f'<text x="{sx(tk):.1f}" y="{yb + 20}" text-anchor="middle" font-size="16.5" fill="{MUTED}">'
                     f'{fmt_tick(tk)}{" s" if tk == ticks[-1] else ""}</text>')
        s.append(f'<line x1="{sx(0):.1f}" y1="{top - 8}" x2="{sx(0):.1f}" y2="{yb}" stroke="{INK}" stroke-width="2"/>')
        color = "#56B4E9" if g == "hfresh" else "#9e9bb5"
        for i, r in enumerate(races[g]):
            y = top + rowh * i + rowh / 2
            m, (lo, hi) = val(f"pr_{r}_agent"), val(f"pr_{r}_agent_ci")
            dag = "†" if "1" in val(f"pr_{r}_tier") else ""
            s.append(f'<text x="{x0}" y="{y + 6:.1f}" font-size="17.5" fill="{INK}">{ps.race_name(r)}{dag}</text>')
            s.append(f'<rect x="{min(sx(0), sx(m)):.1f}" y="{y - 7:.1f}" width="{abs(sx(m) - sx(0)):.1f}" height="14" '
                     f'fill="{color}"/>')
            s.append(f'<line x1="{sx(lo):.1f}" y1="{y:.1f}" x2="{sx(hi):.1f}" y2="{y:.1f}" stroke="{INK}" '
                     f'stroke-width="2"/>')
            s.append(f'<text x="{VX}" y="{y + 6:.1f}" text-anchor="end" font-size="17.5" font-weight="700" '
                     f'fill="{INK}">{T(f"pr_{r}_agent")}</text>')
            used(f"pr_{r}_agent_ci")
            used(f"pr_{r}_tier")
    s.append(f'<text x="{width}" y="17" text-anchor="end" font-size="16.5" fill="{MUTED}">† circuit history in the '
             f'tyre-life prior</text>')
    s.append("</svg>")
    return "".join(s)


def banner_agent():
    allok = ready("h_ag_ok", "h_ag_scored") and val("h_ag_ok") == val("h_ag_scored")
    n = ("all " + V("h_ag_scored")) if allok else V("h_ag_ok")
    return (f"PITWALL made a legal, explained call on {n} held-out decisions. On the {V('hfresh_word')} races it had "
            f"never seen it was {vword('hfresh_belief_optimizer_verdict')} the optimizer in its tools "
            f"({V('hfresh_belief_optimizer_diff')} s), but {vword('hfresh_run_to_flag_verdict')} two fixed rules "
            f"({V('hfresh_run_to_flag_diff')} s {V('hfresh_run_to_flag_ci')}) and "
            f"{vword('hfresh_actual_strategy_verdict')} the teams ({V('hfresh_actual_strategy_diff')} s). Its "
            f"development lead over the rules ({V('d_run_to_flag_diff')} s) {lead_lost()}.")


def stakeholder():
    beat = ready("hfresh_run_to_flag_verdict") and val("hfresh_run_to_flag_verdict") == "ahead"
    return (f"<p>The test bench holds: future-blind by test, its judge at its measured ceiling for order. On "
            f"{V('hfresh_word')} unseen races the agent was {vword('hfresh_run_to_flag_verdict')} two fixed rules and "
            f"{vword('hfresh_actual_strategy_verdict')} the teams: with this system and this model it "
            f"{'beats' if beat else 'does not yet beat'} simple rules. The language model was "
            f"{vword('hfresh_belief_optimizer_verdict')} its own optimizer; the likeliest next gain is "
            f"tyre-life knowledge at new circuits, then a stronger model.</p>")


def html(args_line):
    D, H = N["_sets"].value
    ribbon = ""
    if H.fixture:
        ribbon = '<div class="ribbon fix">LAYOUT FIXTURE: development rows in the held-out column. NOT A RESULT.</div>'
    elif not H.present or not H.has_agent:
        ribbon = ('<div class="ribbon">DRAFT: every hatched <span class="pend">pending</span> box is filled by the '
                  'held-out run; nothing in one is a result.</div>')
    t = "h" if ready("h_sc") else "d"
    sc_on = val(f"{t}_sc")
    model = T("h_model") if ready("h_model") else T("d_model")
    sc_judge = ("with safety cars: the one in progress as recorded, later ones drawn" if sc_on else "without safety cars")
    sc_limit = ("Safety cars after the decision moment are drawn, not recorded." if sc_on
                else "The judge draws no safety cars: a stop under one is priced as green.")
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
    <figcaption>Barcelona-Catalunya, a development race [4]</figcaption>
  </figure>
</header>

<div class="answer">
  <div class="pts">
    <div><b>1 Integrity, proven by test.</b> Scrambling a race after the decision moment changes nothing a policy
      sees; {V("h_fb_fixed")} leaks removed; reporting pre-registered.</div>
    <div><b>2 The judge, measured.</b> Replaying {V("fid_races")} real races, it orders close finishers correctly
      {V("fid_close")} of the time, at its measured ceiling ({V("ceil_close")}); gap error {V("fid_relgap")} s
      (floor {V("ceil_relgap")} s).</div>
    <div><b>3 The agent, held-out, in our simulator.</b> {banner_agent()}</div>
  </div>
</div>

<main class="grid">
  <article class="panel p1">
    <h2><b>1</b>Question and system</h2>
    <p>Each lap a strategist decides: pit now or stay out. Can a language-model agent make that call from the race so
    far, better than simple rules?</p>
    <div class="diagram">
      <div class="box belief"><div class="bh">Predictive simulator <span>every policy plans on it</span></div>
        <ul><li>race so far, practice, earlier races; nothing later</li></ul></div>
      <div class="wall"><span class="arrow">&darr; each policy&rsquo;s plan</span><span class="hidden">the future stays hidden</span></div>
      <div class="box world"><div class="bh">Judge <span>scores only</span></div>
        <ul><li>whole race, {sc_judge}; times every plan</li></ul></div>
    </div>
    <div class="caption">The agent&rsquo;s loop</div>
    {loop_diagram(model)}
    <p class="note"><b>Fixed rules</b>: run to the flag (stop only if tyre life forces it); one stop in total.</p>
  </article>

  <article class="panel p2">
    <h2><b>2</b>Evaluation integrity</h2>
    <p class="def">A policy reads only the race up to the decision moment, this weekend&rsquo;s practice, the rules,
    and tables built from earlier races.</p>
    {blind_diagram()}
    <p class="small"><b>{V("h_fb_fixed")} leaks removed</b>, each now tested ({V("h_fb_known")} open): {leak_list()}.</p>
    <p class="small"><b>Test set:</b> {V("h_races_reg")} held-out races: {V("hfresh_word")} never examined (the
    pre-registered primary test), {V("hearlier_word")} scored once by an earlier evaluation that leaked race-wide tyre
    life and was discarded. Nothing changed once the held-out set existed.</p>
    <p class="note">Harness at <code>{V("h_tag")}</code>: {V("h_harness")}. Anonymised race view; cross-race tables
    as of the race.</p>
  </article>

  <article class="panel p3">
    <h2><b>3</b>How good is the judge?</h2>
    <p class="small">Every car&rsquo;s real strategy replayed on {V("fid_races")} calibration races. The ceiling: the
    judge handed every stop, safety-car lap and start as recorded.</p>
    {judge_chart()}
    <p class="note">Order is at the ceiling; gaps are not. Bars: 95% over races. Plan loss: judge time lost trusting the
    predictive plan. *Earlier base.</p>
    {factor_table()}
    <p class="note">Used in the held-out run: {config_line(t)}.</p>
  </article>

  <article class="panel p4">
    <h2><b>4</b>Results: agent minus each comparator</h2>
    <p class="def"><b>Regret</b>: a plan&rsquo;s extra race time, timed by the judge, over its best plan in hindsight.
    Paired on the same decisions; 95% intervals resample drivers within races; verdict from the interval.</p>
    {forest()}
    <div class="figs">
      <div><div class="caption">The tail, fresh five (pre-registered: &gt;20 s)</div>{tail_table()}
      <p class="note">On the fresh five the agent is more than 20 s off on {V("hfresh_t_over20_agent")} of decisions,
      the two rules on {V("hfresh_t_over20_run_to_flag")}, the teams on {V("hfresh_t_over20_actual_strategy")}
      (development: agent {V("d_t_over20_agent")}, teams {V("d_t_over20_actual_strategy")}).</p></div>
      <div><div class="caption">By era</div>{era_table()}</div>
    </div>
    {per_race_chart()}
    <p class="caveat2"><b>One race drives &ldquo;ahead of the teams&rdquo;:</b> at Monaco 2019 the judge scores the
    teams {V("race_2019_06_monaco_actual_strategy_mean")} s off (worst {V("race_2019_06_monaco_actual_strategy_max")}
    s); Monaco strategy is about track position, which the judge does not model. Without Monaco, all nine: agent − teams {V("hall_nomon_actual_strategy_diff")}
    {V("hall_nomon_actual_strategy_ci")}, − run to the flag {V("hall_nomon_run_to_flag_diff")}
    {V("hall_nomon_run_to_flag_ci")}; earlier four: − teams {V("hearlier_nomon_actual_strategy_diff")}, − run to the
    flag {V("hearlier_nomon_run_to_flag_diff")}.</p>
  </article>

  <div class="stack">
  <article class="panel p5">
    <h2><b>5</b>Why the lead did not hold</h2>
    <p class="small"><b>Measured:</b> Australia and Miami 2022 carry the gap (agent regret
    {V("race_2022_03_australia_agent_mean")} and {V("race_2022_05_miami_agent_mean")} s, run to the flag
    {V("race_2022_03_australia_run_to_flag_mean")} and {V("race_2022_05_miami_run_to_flag_mean")}); the optimizer
    alone has the same gap.</p>
    <p class="small"><b>Leading explanation, not established:</b> there the predictive simulator plans two stops where
    one long hard-tyre stint was best. {fresh_hist_clause()} On development the lead is {V("dev_hist_with_diff")} s
    at {V("dev_hist_with_n")} races with circuit history, {V("dev_hist_without_diff")} s at {V("dev_hist_without_n")}
    without.</p>
    <p class="note">History alone is unlikely to explain all of it: the configuration was chosen on development races.</p>
  </article>

  <article class="panel p6">
    <h2><b>6</b>Limitations, next</h2>
    <ul>
      <li>One pass of one modest model, {model}; {V("hearlier_word")} held-out races were scored once before.</li>
      <li>No tyre-life history at new circuits; added history is 2021-22 only.</li>
      <li>Soft&ndash;medium gaps not fixed; no safety-car queue. {sc_limit}</li>
      <li>The model misreads tool output, invents arguments.</li>
    </ul>
    <p class="small"><b>Next:</b> circuit tyre-life history, compound gaps, a stronger model, repeat passes.</p>
  </article>

  <article class="panel p7">
    <h2><b>7</b>For the stakeholder</h2>
    {stakeholder()}
  </article>
  </div>
</main>

<footer>
  <div class="refs">
    [1] Code and protocol (<code>docs/eval.md</code>): github.com/f1-t/agentic-ai. [2] Results:
    <code>reports/eval/rerun/{{dev20,test9}}</code>; scorecard <code>reports/calibration/fidelity</code>; pre-registration
    <code>configs/eval/reporting.yaml</code>; every number and its computation: <code>poster/numbers.md</code>.
    [3] Timing data: FastF1. [4] Map: &ldquo;2023 F1 CourseLayout Spain.svg&rdquo; by <span lang="ja">ごひょううべこ</span>,
    Wikimedia Commons, CC BY-SA 4.0. Independent student project, not affiliated with Formula 1, the FIA or Pirelli.
  </div>
  <div class="foot">Model {model} &middot; held-out <code>{V("h_tag")}</code> ({V("h_commit")}), {V("h_races")} races,
    n = {V("h_n")} ({V("h_ag_scored")} scored) &middot; development <code>{V("d_tag")}</code> ({V("d_commit")}),
    {V("d_races")} races, n = {V("d_n")} &middot; one agent pass &middot; reporting.yaml {V("h_rep")}</div>
</footer>
</div>"""
    return ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<!-- built by poster/build_poster.py {_html.escape(args_line)} -->"
            "<title>PITWALL poster</title><style>" + CSS + "</style></head><body>" + body + "</body></html>\n")


CSS = """
@page{size:24in 36in;margin:0}
:root{--bg:#0f0e1c;--panel:#1a1930;--line:#3d3b58;--ink:#f4f2f8;--muted:#c9c6d8;--red:#ff4d3d;
      --amber:#ffcf4a;--held:#ffcf4a;--dev:#9cc8ff;--pend:#b9b6cc}
*{box-sizing:border-box}
html,body{margin:0;padding:0;width:24in;height:36in;background:var(--bg);
  -webkit-print-color-adjust:exact;print-color-adjust:exact}
body{font-family:"Noto Sans","Liberation Sans",Arial,sans-serif;color:var(--ink);font-size:22pt;line-height:1.22}
code{font-family:"DejaVu Sans Mono","Liberation Mono",monospace;font-size:.86em}
.poster{position:relative;width:24in;height:36in;padding:.6in .7in .6in;display:flex;flex-direction:column;
  gap:.18in;overflow:hidden;border-top:.12in solid var(--red)}
.ribbon{flex:none;border:3px dashed var(--pend);color:var(--pend);font-size:19pt;font-weight:800;
  padding:.04in .2in;border-radius:.08in;text-align:center;
  background:repeating-linear-gradient(135deg,#1c1b31 0 .12in,#26253f .12in .24in)}
.ribbon .pend{font-size:.9em}
.ribbon.fix{border-color:#ff4d3d;color:#ff8a7d}
.top{display:flex;justify-content:space-between;align-items:center;gap:.4in;flex:none}
.kicker{font-size:19pt;letter-spacing:.12em;text-transform:uppercase;color:var(--red);font-weight:700}
h1{font-size:88pt;line-height:1;margin:0;letter-spacing:.02em;font-weight:900}
.acro{font-size:21pt;color:var(--amber);font-weight:700}
.q{font-size:36pt;font-weight:700;margin:.02in 0 .05in;line-height:1.12}
.authors{font-size:22pt;line-height:1.26}
.authors span{font-size:19pt;color:var(--muted)}
.map{margin:0;width:5.2in;flex:none;text-align:center}
.map img{width:5.2in;height:auto;display:block}
.map figcaption{font-size:16pt;color:var(--muted);margin-top:.02in}
.answer{flex:none;background:#2a1418;border-left:.14in solid var(--red);padding:.1in .24in .12in;font-size:22pt;
  line-height:1.2}
.pts{display:grid;grid-template-columns:.95fr 1fr 1.7fr;gap:.3in}
.pts b{font-size:23pt;color:var(--amber)}
.pend{display:inline-block;font-size:.88em;line-height:1.15;font-style:italic;font-weight:700;color:var(--pend);
  border:2px dashed var(--pend);border-radius:.06in;padding:0 .07in;vertical-align:.04em;white-space:nowrap;
  background:repeating-linear-gradient(135deg,#22213a 0 .07in,#2f2e4a .07in .14in)}
.grid{flex:1;display:grid;grid-template-columns:1fr 1fr 1fr;grid-template-rows:auto 1fr;gap:.22in .32in;min-height:0}
.p4{grid-column:1 / span 2}
.stack{display:flex;flex-direction:column;gap:.22in;min-height:0}
.stack .panel{flex:none}
.stack .p7{flex:1}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:.1in;padding:.14in .24in .16in;min-height:0;
  overflow:hidden}
h2{font-size:29pt;line-height:1.1;margin:0 0 .08in;display:flex;align-items:center;gap:.13in;white-space:nowrap}
h2>b{display:inline-flex;align-items:center;justify-content:center;width:.5in;height:.5in;border-radius:50%;
  background:#c42f26;font-size:21pt;flex:none}
p{margin:0 0 .08in}
p:last-child{margin-bottom:0}
.small{font-size:22pt;line-height:1.2}
.note{font-size:19pt;line-height:1.2;color:var(--muted)}
.note b{color:var(--ink)}
.def{border-left:.08in solid var(--amber);padding-left:.13in;font-size:22pt;line-height:1.2}
ul{margin:0;padding-left:.34in}
li{margin-bottom:.03in}
.diagram{display:flex;flex-direction:column;margin:.02in 0 .08in}
.box{border:2px solid;border-radius:.08in;padding:.05in .13in}
.box.belief{border-color:var(--dev);background:#16263a}
.box.world{border-color:var(--held);background:#2c2614}
.bh{font-size:21pt;font-weight:800;line-height:1.2}
.bh span{font-size:17pt;font-weight:600;color:var(--muted);margin-left:.05in}
.box ul{font-size:20pt;line-height:1.18;padding-left:.26in;margin-top:.02in}
.box li{margin-bottom:.02in}
.wall{display:flex;justify-content:space-between;align-items:center;padding:.02in .13in;font-size:18pt}
.wall .arrow{font-weight:700}
.wall .hidden{color:var(--muted);font-style:italic}
.pols{font-size:20pt;line-height:1.18;margin:.04in 0 0}
.pols li{margin-bottom:.03in}
.leaks{font-size:20pt;line-height:1.16;margin:.01in 0 .08in;columns:2;column-gap:.3in}
.leaks li{margin-bottom:.03in;break-inside:avoid}
.chart{display:block;width:100%;height:auto;margin:.03in 0 .06in;font-family:"Noto Sans","Liberation Sans",Arial,sans-serif}
.caption{font-size:20pt;font-weight:800;margin:.06in 0 .02in}
.figs{display:grid;grid-template-columns:1.25fr 1fr;gap:.3in;margin:.04in 0 .06in}
.tbl{width:100%;border-collapse:collapse;font-size:18pt;line-height:1.14;margin:.03in 0 .05in}
.tbl th,.tbl td{padding:.025in .05in;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
.tbl thead th{font-size:16.5pt;color:var(--muted);font-weight:700;vertical-align:bottom}
.tbl th.rl,.tbl thead th:first-child{text-align:left;font-weight:700}
.tbl td{font-variant-numeric:tabular-nums}
.tbl td.dvc{color:var(--muted)}
.tbl .sep{border-left:2px solid var(--line)}
.tbl .cil{font-size:16.5pt;color:var(--muted)}
.tbl .vd{font-size:16.5pt;font-weight:700}
.tbl .races{font-size:16.5pt;font-weight:400;color:var(--muted);white-space:normal}
.tbl.era th.rl{white-space:normal}
.tbl.fac td{font-size:18pt;text-align:left;white-space:nowrap}
.tbl.fac th.rl{font-weight:600;white-space:normal}
.tbl tr.grp th{text-align:left;font-size:18pt;font-weight:800;color:var(--amber);padding-top:.05in}
.sw{display:inline-block;width:.16in;height:.16in;border-radius:.03in;margin-right:.08in;vertical-align:.02em}
.caveat2{font-size:20pt;line-height:1.2;background:#33213a;border-left:.08in solid #CC79A7;padding:.06in .14in;margin:.04in 0 0}
.p6 ul{font-size:22pt;line-height:1.18}
.p7{background:#16263a;border-color:var(--dev)}
.p7 p{font-size:22pt;line-height:1.22}
footer{flex:none;border-top:2px solid var(--line);padding-top:.07in;display:flex;flex-direction:column;gap:.04in}
.refs{font-size:13pt;line-height:1.3;color:var(--muted)}
.refs code{font-size:1em}
.refs [lang=ja]{font-family:"Noto Sans CJK JP","Noto Sans CJK TC",sans-serif}
.foot{font-size:15pt;font-weight:700}
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
    ap.add_argument("--dev", required=True, help="development set out-dir (reports/eval/rerun/dev20)")
    ap.add_argument("--test", help="held-out set out-dir (reports/eval/rerun/test9); absent: placeholders")
    ap.add_argument("--fidelity", default=str(ps.PROJECT / "reports" / "calibration" / "fidelity"),
                    help="the fidelity scorecard's outputs (default reports/calibration/fidelity)")
    ap.add_argument("--scorecard", default=ps.SCORECARD, help=f"scorecard under --fidelity (default {ps.SCORECARD})")
    ap.add_argument("--ablation", default=ps.ABLATION, help=f"ablation under --fidelity (default {ps.ABLATION})")
    ap.add_argument("--noise-floor", help="a ceiling_pooled.csv (default: the scorecard's own)")
    ap.add_argument("--fixture", action="store_true",
                    help="layout test only: accept a development set as --test and mark the poster NOT A RESULT")
    ap.add_argument("--export", action="store_true", help="also write the PDF, the 300 ppi PNG and the preview")
    a = ap.parse_args()
    N = ps.compute(a.dev, a.test, a.fidelity, a.noise_floor, allow_dev_as_test=a.fixture, scorecard=a.scorecard,
                   ablation=a.ablation)
    args_line = " ".join(["--dev", str(a.dev)] + (["--test", str(a.test)] if a.test else [])
                         + ["--fidelity", str(a.fidelity), "--scorecard", a.scorecard, "--ablation", a.ablation])
    OUT_HTML.write_text(html(args_line), encoding="utf-8")
    (HERE / "numbers.md").write_text(ps.provenance_md(N, USED, args_line), encoding="utf-8")
    pend = [k for k in USED if N[k].pending]
    print(f"wrote {OUT_HTML.name} and numbers.md: {len(USED)} numbers, {len(pend)} pending")
    if a.export:
        export()


if __name__ == "__main__":
    main()
