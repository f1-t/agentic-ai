"""Build the 24 x 36 in PITWALL poster from the committed result files.

    python poster/build_poster.py            # writes poster/PITWALL_poster_24x36in.html and poster/numbers.md
    python poster/build_poster.py --export   # also the PDF, the 300 ppi PNG and the 1200 px preview

Every number on the poster comes from `poster_stats.compute()` (held-out `reports/eval/d4_*`,
development `reports/eval/dev6_run2_*`); `numbers.md` lists each one with its file and computation.
The export uses system Chromium (`chromium` on PATH, or $CHROMIUM) and poppler (`pdftoppm`).
Fonts: Noto Sans (and Noto Sans CJK for the map author's name); install them for an identical layout.
"""
from __future__ import annotations

import argparse
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

# one colour and one marker shape per comparator (Okabe-Ito, colour-blind safe); held-out = filled,
# development = hollow. Values are always printed next to the marks.
COMP = {"fixed_one_stop": ("Fixed one-stop rule", "#CC79A7", "square"),
        "actual_strategy": ("Teams’ actual strategy", "#2BB88A", "diamond"),
        "belief_optimizer": ("Belief optimizer (no LLM)", "#E69F00", "circle")}
PANEL, INK, MUTED, GRID = "#1a1930", "#f4f2f8", "#c9c6d8", "#4a4866"
USED: list[str] = []


def v(N, key):
    """The printed text of a number, recorded for the provenance table."""
    if key not in USED:
        USED.append(key)
    return N[key].text


# ---------------------------------------------------------------- svg pieces (1 user unit = 1 pt)
CHART_W = 472


def marker(shape, x, y, color, r=9.0, hollow=False):
    fill = PANEL if hollow else color
    sw = 3 if hollow else 1.5
    stroke = color if hollow else PANEL
    st = f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"'
    if shape == "circle":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" {st}/>'
    if shape == "square":
        a = r * 1.7
        return f'<rect x="{x - a / 2:.1f}" y="{y - a / 2:.1f}" width="{a:.1f}" height="{a:.1f}" {st}/>'
    d = r * 1.3
    return f'<path d="M{x:.1f},{y - d:.1f}L{x + d:.1f},{y:.1f}L{x:.1f},{y + d:.1f}L{x - d:.1f},{y:.1f}Z" {st}/>'


LO, HI, X0, X1 = -8.0, 3.0, 14.0, 458.0


def sx(val):
    return X0 + (X1 - X0) * (val - LO) / (HI - LO)


def axis(y_top, y_bot):
    s = []
    for t in range(-8, 3, 2):
        x = sx(t)
        s.append(f'<line x1="{x:.1f}" y1="{y_top}" x2="{x:.1f}" y2="{y_bot}" stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{x:.1f}" y="{y_bot + 24}" text-anchor="middle" font-size="18" fill="{MUTED}">'
                 f'{str(t).replace("-", chr(0x2212))}{" s" if t == 2 else ""}</text>')
    x = sx(0)
    s.append(f'<line x1="{x:.1f}" y1="{y_top - 6}" x2="{x:.1f}" y2="{y_bot}" stroke="{INK}" stroke-width="2"/>')
    s.append(f'<text x="{sx(-0.25):.1f}" y="{y_top - 12}" text-anchor="end" font-size="18" fill="{MUTED}">← agent better</text>')
    s.append(f'<text x="{sx(0.25):.1f}" y="{y_top - 12}" font-size="18" fill="{MUTED}">agent worse →</text>')
    return "".join(s)


def interval(lo, hi, mean, y, shape, color, hollow=False):
    return (f'<line x1="{sx(lo):.1f}" y1="{y:.1f}" x2="{sx(hi):.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="4"/>'
            f'<line x1="{sx(lo):.1f}" y1="{y - 8:.1f}" x2="{sx(lo):.1f}" y2="{y + 8:.1f}" stroke="{color}" stroke-width="3"/>'
            f'<line x1="{sx(hi):.1f}" y1="{y - 8:.1f}" x2="{sx(hi):.1f}" y2="{y + 8:.1f}" stroke="{color}" stroke-width="3"/>'
            + marker(shape, sx(mean), y, color, hollow=hollow))

SYSTEM_C, TEAMS_C = "#56B4E9", "#2BB88A"


def survival(N, tag="h"):
    """Share of the common scenarios whose regret is above x, PITWALL against the teams' actual strategy."""
    ra, rt = N[f"{tag}_tail_curve"].value
    USED.append(f"{tag}_tail_curve")
    W, H, L, R, T, B, XM = CHART_W, 330, 62, 444, 20, 276, 50.0
    px = lambda x: L + (R - L) * x / XM
    py = lambda f: B - (B - T) * f
    s = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="Share of decisions more than x seconds '
         f'from the hindsight reference, PITWALL and the teams’ actual strategy">']
    for f in (0, .25, .5, .75, 1):
        s.append(f'<line x1="{L}" y1="{py(f):.1f}" x2="{R}" y2="{py(f):.1f}" stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{L - 8}" y="{py(f) + 6:.1f}" text-anchor="end" font-size="18" fill="{MUTED}">{int(f * 100)}%</text>')
    for x in range(0, 51, 10):
        s.append(f'<line x1="{px(x):.1f}" y1="{T}" x2="{px(x):.1f}" y2="{B}" stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{px(x):.1f}" y="{B + 24}" text-anchor="middle" font-size="18" fill="{MUTED}">{x}{" s" if x == 50 else ""}</text>')
    s.append(f'<text x="{(L + R) / 2:.1f}" y="{H - 4}" text-anchor="middle" font-size="18" fill="{MUTED}">'
             f'x: seconds from the hindsight reference</text>')
    for vals, color, dash, label, ly in ((rt, TEAMS_C, "10 6", "Teams’ actual", 0), (ra, SYSTEM_C, "", "PITWALL", 1)):
        v_ = np.sort(np.asarray(vals))
        n = len(v_)
        d = [f"M{px(0):.1f},{py((v_ > 0).mean()):.1f}"]
        for u in np.unique(v_[(v_ > 0) & (v_ < XM)]):
            d.append(f"H{px(u):.1f}V{py((v_ > u).sum() / n):.1f}")
        d.append(f"H{px(XM):.1f}")
        s.append(f'<path d="{"".join(d)}" fill="none" stroke="{color}" stroke-width="4" stroke-dasharray="{dash}"/>')
    # direct labels, and the two values at 20 s
    s.append(f'<text x="{px(21):.1f}" y="{py(.78):.1f}" font-size="20" font-weight="700" fill="{SYSTEM_C}">PITWALL (solid)</text>')
    s.append(f'<text x="{px(21):.1f}" y="{py(.78) + 26:.1f}" font-size="20" font-weight="700" fill="{TEAMS_C}">Teams’ actual (dashed)</text>')
    s.append(f'<line x1="{px(20):.1f}" y1="{T}" x2="{px(20):.1f}" y2="{B}" stroke="{INK}" stroke-width="1.5" stroke-dasharray="3 4"/>')
    s.append("</svg>")
    return "".join(s)


SECONDARY = [("belief_optimizer",) + COMP["belief_optimizer"], ("fixed_one_stop",) + COMP["fixed_one_stop"],
             (ps.FAIR_RULES[0][0], ps.FAIR_RULES[0][1], "#F0E442", "square"),
             (ps.FAIR_RULES[1][0], ps.FAIR_RULES[1][1], "#D55E00", "diamond")]


def forest_secondary(N):
    """Held-out: agent minus the other comparators; fairer rule baselines print 'pending' until scored."""
    top, row = 40, 68
    h = top + row * len(SECONDARY) + 34
    s = [f'<svg class="chart" viewBox="0 0 {CHART_W} {h}" role="img" '
         f'aria-label="Held-out: agent minus other comparators, with 95% intervals">']
    s.append(axis(top, top + row * len(SECONDARY)))
    for i, (pol, name, color, shape) in enumerate(SECONDARY):
        y = top + row * i
        s.append(f'<rect x="0" y="{y + 4}" width="{CHART_W}" height="32" fill="{PANEL}" opacity="0.9"/>')
        s.append(f'<text x="4" y="{y + 28}" font-size="21" font-weight="700" fill="{INK}">{name}</text>')
        if f"h_diff_{pol}" in N:
            lo, hi = N[f"h_ci_{pol}"].value
            s.append(f'<text x="{CHART_W - 2}" y="{y + 28}" text-anchor="end" font-size="21" fill="{INK}">'
                     f'<tspan font-weight="700">{v(N, f"h_diff_{pol}")}</tspan> {v(N, f"h_ci_{pol}")}</text>')
            s.append(interval(lo, hi, N[f"h_diff_{pol}"].value, y + 54, shape, color))
        else:
            s.append(f'<text x="{CHART_W - 2}" y="{y + 28}" text-anchor="end" font-size="20" font-style="italic" '
                     f'fill="{MUTED}">pending: being scored</text>')
    s.append("</svg>")
    return "".join(s)


# ---------------------------------------------------------------- html
SHORT = {"fixed_one_stop": "vs one-stop rule", "actual_strategy": "vs teams’ actual",
         "belief_optimizer": "vs belief optimizer"}
RACE_ROWS = ("actual_strategy", "belief_optimizer", "fixed_one_stop")


def per_race_table(N):
    races = ps.HELD_RACES
    head = "".join(f"<th>{ps.RACE_NAMES[r].replace(' ', '<br>')}</th>" for r in races)
    rows = ['<tr class="lvl"><th class="rl">Agent regret (s)</th>'
            + "".join(f"<td>{v(N, f'h_race_mean_agent_{r}')}</td>" for r in races) + "</tr>"]
    for pol in RACE_ROWS:
        color = COMP[pol][1]
        cells = []
        for r in races:
            key = f"h_race_diff_{pol}_{r}"
            N[key].text = ps.s(N[key].value, 2 if pol == "belief_optimizer" else 1, sign=True)
            cells.append(f"<td>{v(N, key)}</td>")
        rows.append(f'<tr><th class="rl"><i class="sw" style="background:{color}"></i>{SHORT[pol]}</th>{"".join(cells)}</tr>')
    return f'<table class="tbl race"><thead><tr><th>agent minus &hellip;</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def lead_table(N):
    def row(label, key, diff=None, unit=""):
        d = f'<td class="d"><b>{v(N, diff[0])}</b> {v(N, diff[1])}</td>' if diff else '<td class="d"></td>'
        return (f'<tr><th class="rl">{label}</th><td>{v(N, f"h_tail_{key}_agent")}{unit}</td>'
                f'<td>{v(N, f"h_tail_{key}_actual")}{unit}</td>{d}</tr>')
    return ('<table class="tbl lead"><thead><tr><th>held-out, n ' + v(N, "h_tail_n") + '</th>'
            f'<th><i class="sw" style="background:{SYSTEM_C}"></i>PITWALL</th><th><i class="sw" style="background:{TEAMS_C}"></i>Teams</th>'
            '<th>difference [95%]</th></tr></thead><tbody>'
            + row("Mean regret", "mean", ("h_diff_actual_strategy", "h_ci_actual_strategy"), " s")
            + row("Median", "median", None, " s")
            + row("Within 1 s", "1", ("h_tail_1_diff", "h_tail_1_ci"))
            + row("Over 20 s", "20", ("h_tail_20_diff", "h_tail_20_ci"))
            + row("Over 30 s", "30")
            + row("90th percentile", "p90", ("h_tail_p90_diff", "h_tail_p90_ci"), " s")
            + row("Worst", "max", None, " s")
            + "</tbody></table>")


def contrast_table(N):
    def r(label, dk, ck, hk, hck):
        return (f'<tr><th class="rl">{label}</th><td><b>{v(N, dk)}</b> {v(N, ck)}</td>'
                f'<td><b>{v(N, hk)}</b> {v(N, hck)}</td></tr>')
    return ('<table class="tbl contrast"><thead><tr><th>agent minus</th><th>development</th>'
            '<th>held-out</th></tr></thead><tbody>'
            + '<tr class="grp"><th colspan="3"><i class="sw" style="background:' + TEAMS_C + '"></i>teams&rsquo; actual strategy</th></tr>'
            + r("mean (s)", "d_diff_actual_strategy", "d_ci_actual_strategy", "h_diff_actual_strategy", "h_ci_actual_strategy")
            + r("share over 20 s", "d_tail_20_diff", "d_tail_20_ci", "h_tail_20_diff", "h_tail_20_ci")
            + r("90th pct (s)", "d_tail_p90_diff", "d_tail_p90_ci", "h_tail_p90_diff", "h_tail_p90_ci")
            + '<tr class="grp"><th colspan="3"><i class="sw" style="background:' + COMP["belief_optimizer"][1] + '"></i>belief optimizer</th></tr>'
            + r("mean (s)", "d_diff_belief_optimizer", "d_ci_belief_optimizer", "h_diff_belief_optimizer", "h_ci_belief_optimizer")
            + "</tbody></table>")


def html(N):
    body = f"""
<div class="poster">
<header class="top">
  <div class="title">
    <div class="kicker">CSE 598 Agentic AI &middot; Arizona State University</div>
    <h1>PITWALL</h1>
    <div class="acro">Predictive Intelligence for Tactical When-to-pit Agentic Lap-wise Logic</div>
    <div class="q">Can a language model call the pit stop?</div>
    <div class="authors">Jason Lunder &middot; Rishabh Baral &middot; Sara Maisha &middot; Quan Nguyen &middot; Manya Jain<br>
      <span>Stakeholder: Manav Ananthakumar &middot; TA: Xiaoou Liu &middot; Professor: Hua Wei</span></div>
  </div>
  <figure class="map">
    <img src="{MAP}" alt="Circuit de Barcelona-Catalunya, 2023 layout">
    <figcaption>Barcelona-Catalunya 2023, a development race [4]</figcaption>
  </figure>
</header>

<div class="answer"><span class="tag held">Held-out, in our simulator</span>On {v(N, "h_races")} races the agent had
  never been run on, PITWALL&rsquo;s decisions cost the same on average as the teams&rsquo; real strategies, with fewer
  large errors. The optimizer alone does as well: the LLM added nothing measurable.</div>

<main class="cols">
<section class="col">
  <article class="panel">
    <h2><b>1</b>The question</h2>
    <p>Each lap a strategist decides: pit now or stay out. The stakeholder&rsquo;s baseline agent planned with the
    simulator that graded it. We split the two: can a language-model (LLM) agent make good pit calls seeing only
    the race so far?</p>
  </article>

  <article class="panel">
    <h2><b>2</b>How it works</h2>
    <div class="diagram">
      <div class="box belief"><div class="bh">Belief simulator <span>what every policy sees</span></div>
        <ul><li>practice + race laps so far, with uncertainty</li>
        <li>feeds the agent&rsquo;s tools and the belief optimizer</li></ul></div>
      <div class="wall"><span class="arrow">&darr; each policy&rsquo;s plan</span><span class="hidden">later laps stay hidden</span></div>
      <div class="box world"><div class="bh">World simulator <span>scoring only</span></div>
        <ul><li>every lap of the race (hindsight); no safety cars or traffic</li>
        <li>finds the hindsight reference, times every plan</li></ul></div>
      <div class="caveat">Not fully blind: stint-length limits and pit loss are read from the whole race.</div>
    </div>
    <p>Per scenario (race, driver, lap) the LLM (qwen3-30b-a3b) calls tools for up to 10 turns, then submits PIT
    plus a tyre, or STAY OUT; two submit-only turns force a call. The belief optimizer turns the call into a plan.
    <b>PITWALL</b> = belief simulator + optimizer + LLM front end.</p>
    <p class="def"><b>Regret</b> is the extra remaining-race time (s) of a plan over the <b>hindsight
    reference</b>: the best plan with at most {v(N, "max_stops")} more stops, timed in a deterministic replay of the race
    fitted on all its laps, with no safety cars or traffic. Lower is better.</p>
  </article>

  <article class="panel grow">
    <h2><b>3</b>Test design</h2>
    <p><span class="tag held">Held-out</span>{v(N, "h_races")} races &times; {v(N, "h_per_race")} scenarios: Monaco 2019,
    Sakhir 2020, Austria 2024 and 2025. Registered in advance, run once at tag <code>d4-final-eval</code>.
    The agent had never been run on them.</p>
    <p class="note">Prior exposure: baselines were scored once on Monaco 2019 under an earlier simulator; Sakhir 2020
    and Austria 2025 were in an earlier model comparison; Austria 2024 had none.</p>
    <p><span class="tag dev">Development</span>{v(N, "d_races")} races the simulator was calibrated on,
    {v(N, "d_n")} scenarios.</p>
    <p>Comparators: the teams&rsquo; actual strategy; the belief optimizer (same belief, no LLM); rule baselines.
    Differences are agent minus comparator on the same scenarios. Brackets: 95% interval, resampling drivers
    within races.</p>
  </article>
</section>

<section class="col">
  <article class="panel">
    <h2><b>4</b>PITWALL vs the teams</h2>
    <div class="tiles">
      <div><b>{v(N, "h_scored")}</b><span>decisions scored</span></div>
      <div><b>{v(N, "h_voluntary_pct")}</b><span>before the turn limit</span></div>
      <div><b>{v(N, "h_latency_median")}</b><span>median per decision</span></div>
    </div>
    <p class="lead"><span class="tag held">Held-out</span>Same average cost; fewer large errors (suggestive here,
    clear on development races). The teams are more often near-perfect.</p>
    <div class="caption">Share of decisions more than x s from the hindsight reference</div>
    {survival(N)}
    {lead_table(N)}
    <p class="note"><b>Without Austria 2024</b> ({v(N, "aut24_share_agent_regret")} of the agent&rsquo;s regret): agent mean
    {v(N, "h_noaut_mean_agent")} s; minus teams {v(N, "h_noaut_diff_actual_strategy")} {v(N, "h_noaut_ci_actual_strategy")},
    minus belief optimizer {v(N, "h_noaut_diff_belief_optimizer")} {v(N, "h_noaut_ci_belief_optimizer")},
    minus one-stop {v(N, "h_noaut_diff_fixed_one_stop")} {v(N, "h_noaut_ci_fixed_one_stop")}. Conclusions unchanged.</p>
    <div class="caption">Per race ({v(N, "h_per_race")} scenarios each); four races are the real sample</div>
    {per_race_table(N)}
    <p class="note">Austria 2024: the hindsight reference uses Soft in {v(N, "aut24_ref_soft")}, priced from few race laps
    (the sampled drivers&rsquo; only Soft stint is a final {v(N, "aut24_soft_stint")} laps); the belief priced Soft from practice.
    Agent, belief optimizer and teams sit about {v(N, "aut24_about")} s from the reference; fixed one-stop
    {v(N, "h_race_mean_fixed_one_stop_2024_11_austria")} s, fixed two-stop {v(N, "aut24_mean_fixed_two_stop")} s.</p>
  </article>

  <article class="panel finding grow">
    <h2><b>5</b>What the LLM adds <span class="tag held">held-out</span></h2>
    <p>Nothing measurable over the belief optimizer: {v(N, "h_diff_belief_optimizer")} s {v(N, "h_ci_belief_optimizer")}.
    Equal on average, not identical: {v(N, "h_ties_belief_optimizer")} exact ties, same plan in {v(N, "h_same_plan")}
    of {v(N, "h_n")}. Where its immediate call differs ({v(N, "h_diff_call_n")}), the optimizer is better in
    {v(N, "h_diff_call_bo_better")}; over those {v(N, "h_diff_call_n")} the agent is {v(N, "h_diff_call_mean")} s worse on
    average. Pit-now calls agree with the reference {v(N, "h_pitnow_agree_agent")} of the time; &ldquo;always stay
    out&rdquo; agrees {v(N, "h_pitnow_agree_stay")} ({v(N, "h_pitnow_base")} of scenarios are pit-now).</p>
  </article>
</section>

<section class="col">
  <article class="panel">
    <h2><b>6</b>Other comparators <span class="tag held">held-out</span></h2>
    <div class="caption">Agent minus comparator, mean regret (s)</div>
    {forest_secondary(N)}
    <p class="caveat2"><b>One-stop caveat:</b> that rule always makes one more stop. Agent minus rule:
    {v(N, "h_f1s_before")} s {v(N, "h_f1s_before_ci")} before the driver&rsquo;s first stop (n {v(N, "h_f1s_before_n")}),
    {v(N, "h_f1s_after")} s {v(N, "h_f1s_after_ci")} after it (n {v(N, "h_f1s_after_n")}).</p>
  </article>

  <article class="panel">
    <h2><b>7</b>Development vs held-out</h2>
    {contrast_table(N)}
    <p class="note">Clear on development, suggestive on held-out. Mean regret roughly doubles for both policies
    that plan on the belief: agent {v(N, "d_mean_agent")} &rarr; {v(N, "h_mean_agent")} s, belief optimizer
    {v(N, "d_mean_belief_optimizer")} &rarr; {v(N, "h_mean_belief_optimizer")} s (teams {v(N, "d_mean_actual_strategy")}
    &rarr; {v(N, "h_mean_actual_strategy")} s).</p>
  </article>

  <article class="panel limits">
    <h2><b>8</b>Limitations</h2>
    <ul>
      <li>One agent pass, one model; {v(N, "h_races")} held-out races, {v(N, "h_circuits")} circuits.</li>
      <li>The scoring world is itself a fitted simulator.</li>
      <li>Stint limits and pit loss use hindsight; teams raced with traffic, rivals and live tyre data.</li>
      <li>{v(N, "h_grace")} of {v(N, "h_n")} held-out decisions came only in the forced-submission turns.</li>
    </ul>
  </article>

  <article class="panel stake">
    <h2><b>9</b>For the stakeholder</h2>
    <p>Without the oracle, PITWALL matches the teams on average in our simulator. Its quality comes from the
    belief simulator and optimizer, not the LLM, at a median {v(N, "h_latency_median")} and about
    {v(N, "h_tokens_mean")} input tokens per decision. The LLM&rsquo;s value is a legal, explained call; better tyre
    models are where gains are.</p>
  </article>

  <article class="panel grow">
    <h2><b>10</b>Next steps</h2>
    <ul>
      <li>Fairer rule baselines (slots in panel 6).</li>
      <li>Repeat passes; a second model.</li>
      <li>Safety cars and traffic in scoring; driver effects in the wear fit.</li>
    </ul>
  </article>
</section>
</main>

<footer>
  <div class="refs">
    [1] Code, evaluation protocol (<code>docs/eval.md</code>) and tag <code>d4-final-eval</code>: github.com/f1-t/agentic-ai.
    Project page: rishabhbaral-asu.github.io/PITWALL.<br>
    [2] Held-out: <code>reports/eval/d4_combined.csv</code>, <code>d4_compare_regret_s_paired.csv</code>,
    <code>d4_compare_per_race_regret_s_paired.csv</code>. Development: <code>dev6_run2_combined.csv</code>,
    <code>dev6_run2_compare_regret_s_paired.csv</code>. Every number and its computation: <code>poster/numbers.md</code>.<br>
    [3] Race timing data: FastF1 Python package, offline cache in the repository.<br>
    [4] Map: &ldquo;2023 F1 CourseLayout Spain.svg&rdquo; by <span lang="ja">ごひょううべこ</span>, Wikimedia Commons,
    CC BY-SA 4.0, unmodified. PITWALL is an independent student project, not affiliated with Formula 1, the FIA or Pirelli.
  </div>
  <div class="foot">Model qwen3-30b-a3b via ASU Research Computing &middot; tag <code>d4-final-eval</code> &middot;
    one agent pass &middot; n = {v(N, "h_n")} held-out / {v(N, "d_n")} development</div>
</footer>
</div>"""
    return ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            "<title>PITWALL poster</title><style>" + CSS + "</style></head><body>" + body + "</body></html>\n")


CSS = """
@page{size:24in 36in;margin:0}
:root{--bg:#0f0e1c;--panel:#1a1930;--line:#3d3b58;--ink:#f4f2f8;--muted:#c9c6d8;--red:#ff4d3d;
      --amber:#ffcf4a;--held:#ffcf4a;--dev:#9cc8ff}
*{box-sizing:border-box}
html,body{margin:0;padding:0;width:24in;height:36in;background:var(--bg);
  -webkit-print-color-adjust:exact;print-color-adjust:exact}
body{font-family:"Noto Sans","Liberation Sans",Arial,sans-serif;color:var(--ink);font-size:23pt;line-height:1.26}
code{font-family:"DejaVu Sans Mono","Liberation Mono",monospace;font-size:.9em}
.poster{width:24in;height:36in;padding:.65in .75in .7in;display:flex;flex-direction:column;gap:.26in;overflow:hidden;
  border-top:.12in solid var(--red)}
.top{display:flex;justify-content:space-between;align-items:center;gap:.5in;flex:none}
.kicker{font-size:20pt;letter-spacing:.12em;text-transform:uppercase;color:var(--red);font-weight:700}
h1{font-size:104pt;line-height:1;margin:.02in 0 0;letter-spacing:.02em;font-weight:900}
.acro{font-size:22pt;color:var(--amber);font-weight:700}
.q{font-size:42pt;font-weight:700;margin:.04in 0 .08in;line-height:1.15}
.authors{font-size:23pt;line-height:1.3}
.authors span{font-size:20pt;color:var(--muted)}
.map{margin:0;width:7in;flex:none;text-align:center}
.map img{width:7in;height:auto;display:block}
.map figcaption{font-size:17pt;color:var(--muted);margin-top:.02in}
.answer{flex:none;background:#2a1418;border-left:.14in solid var(--red);padding:.14in .3in;font-size:28pt;
  line-height:1.25;font-weight:600}
.tag{display:inline-block;font-size:17pt;line-height:1.3;font-weight:800;letter-spacing:.04em;text-transform:uppercase;
  padding:.01in .1in;border-radius:.06in;vertical-align:.18em;margin-right:.12in}
.tag.held{background:var(--held);color:#141225}
.tag.dev{background:var(--dev);color:#141225}
.answer .tag{font-size:18pt}
.cols{flex:1;display:grid;grid-template-columns:1fr 1fr 1fr;gap:.42in;min-height:0}
.col{display:flex;flex-direction:column;gap:.24in;min-height:0;overflow:hidden}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:.1in;padding:.22in .28in;flex:none}
.panel.grow{flex:1}
h2{font-size:32pt;line-height:1.1;margin:0 0 .12in;display:flex;align-items:center;gap:.14in;white-space:nowrap}
h2>b{display:inline-flex;align-items:center;justify-content:center;width:.56in;height:.56in;border-radius:50%;
  background:#c42f26;font-size:24pt;flex:none}
h2 .tag{margin:0 0 0 auto;vertical-align:0}
p{margin:0 0 .12in}
p:last-child{margin-bottom:0}
.note{font-size:20pt;line-height:1.25;color:var(--muted)}
.note b{color:var(--ink)}
.lead{font-size:24pt;font-weight:700;line-height:1.22}
.def{border-left:.08in solid var(--amber);padding-left:.16in}
ul,ol{margin:0;padding-left:.4in}
li{margin-bottom:.05in}
.diagram{display:flex;flex-direction:column;margin:.02in 0 .14in}
.box{border:2px solid;border-radius:.08in;padding:.08in .16in}
.box.belief{border-color:var(--dev);background:#16263a}
.box.world{border-color:var(--held);background:#2c2614}
.bh{font-size:23pt;font-weight:800;line-height:1.2}
.bh span{font-size:18pt;font-weight:600;color:var(--muted);margin-left:.08in}
.box ul{font-size:20pt;line-height:1.2;padding-left:.3in;margin-top:.02in}
.box li{margin-bottom:.02in}
.wall{display:flex;justify-content:space-between;align-items:center;padding:.03in .16in;font-size:19pt}
.wall .arrow{font-weight:700}
.wall .hidden{color:var(--muted);font-style:italic}
.diagram .caveat{margin-top:.08in;font-size:20pt;line-height:1.22;background:#3a1d22;border-radius:.06in;padding:.06in .16in}
.tiles{display:grid;grid-template-columns:1fr 1fr 1fr;gap:.12in;margin-bottom:.12in}
.tiles div{background:#24233d;border-radius:.08in;padding:.06in .12in}
.tiles b{display:block;font-size:34pt;line-height:1.05;white-space:nowrap}
.tiles span{display:block;font-size:18pt;line-height:1.15;color:var(--muted)}
.caption{font-size:20pt;font-weight:700;margin:.04in 0 .02in}
.chart{display:block;width:100%;height:auto;font-family:"Noto Sans","Liberation Sans",Arial,sans-serif}
.caveat2{font-size:20pt;line-height:1.25;background:#33213a;border-left:.08in solid #CC79A7;padding:.08in .16in;margin:.08in 0 0}
.tbl{width:100%;border-collapse:collapse;font-size:19pt;line-height:1.2;margin:.04in 0 .12in}
.tbl th,.tbl td{padding:.035in .05in;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
.tbl thead th{font-size:17pt;color:var(--muted);font-weight:700;vertical-align:bottom}
.tbl th.rl,.tbl thead th:first-child{text-align:left;font-weight:600}
.tbl td{font-variant-numeric:tabular-nums}
.tbl tr.lvl td{font-weight:700}
.tbl.lead td.d{font-size:18pt}
.tbl.contrast td{font-size:18pt}
.tbl tr.grp th{text-align:left;font-size:19pt;font-weight:800;padding-top:.08in}
.sw{display:inline-block;width:.16in;height:.16in;border-radius:.03in;margin-right:.08in;vertical-align:.02em}
.finding{border-color:var(--red)}
.limits ul{font-size:21pt;line-height:1.24}
.stake{background:#16263a;border-color:var(--dev)}
footer{flex:none;border-top:2px solid var(--line);padding-top:.1in;display:flex;flex-direction:column;gap:.05in}
.refs{font-size:13pt;line-height:1.35;color:var(--muted)}
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
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--export", action="store_true", help="also write the PDF, the 300 ppi PNG and the preview")
    args = ap.parse_args()
    N = ps.compute()
    OUT_HTML.write_text(html(N), encoding="utf-8")
    (HERE / "numbers.md").write_text(ps.provenance_md(N, USED), encoding="utf-8")
    print("wrote", OUT_HTML.name, "and numbers.md,", len(USED), "numbers")
    if args.export:
        export()


if __name__ == "__main__":
    main()
