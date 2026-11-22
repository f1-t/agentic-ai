# PITWALL poster: changes for the final 24 x 36 in build

Earlier versions: the Nov 18 draft in this folder (`PITWALL_research_poster.*`, Milestone 2 numbers) and a
Nov 21 24 x 36 in draft built outside the repo (`PITWALL_poster_24x36in.pdf` from the base file with
patched panels). This build replaces both. It is generated from scratch by `build_poster.py`; every number
comes from `reports/eval/d4_combined.csv` (held-out) or `reports/eval/dev6_run2_combined.csv`
(development), listed in `numbers.md`.

## Content

- **Lead result.** The comparison with the teams' actual strategy, held-out: same mean cost (agent minus
  teams -0.19 s [-1.88, 1.43]) and fewer large errors (over 20 s: 13% vs 19%, difference -6 points
  [-13, 0]; 90th percentile 22.6 vs 28.0 s, -5.5 [-10.4, 0.2]), shown beside the fact that the teams are
  more often near-perfect (median 2.6 vs 6.2 s; within 1 s 36% vs 30%). New figure: the share of decisions
  more than x seconds from the hindsight reference, both curves, with the value table. Worded "suggestive
  on held-out, clear on development" and "in our simulator"; credited to the whole system.
- **Held-out first, labelled everywhere.** No pooled numbers. The tiles (100/100 scored, 71% before the turn
  limit, 26 s median) are held-out figures. Development figures appear only in panel 7.
- **Intervals.** (race, driver) cluster bootstrap everywhere, as in `docs/eval.md`; stated once ("resampling
  drivers within races"). The four per-race differences are in a table.
- **Development vs held-out (panel 7).** The development lead over the teams (-3.25 [-5.50, -1.25]) against
  held-out (-0.19 [-1.88, 1.43]), plus the tail measures on both sets, and the regret roughly doubling for
  the agent (3.95 to 8.93 s) and the belief optimizer (4.11 to 8.95 s).
- **Negative results as findings (panel 5).** No measurable gain over the belief optimizer (-0.03
  [-0.29, 0.22]); equal on average but not identical (39 ties, same plan in 33 of 100); where the immediate
  call differs (11), the optimizer is better in 8; pit-now agreement 83% against 90% for "always stay out",
  with the 10% base rate.
- **Fixed one-stop rule** moved to a secondary panel with its caveat beside it (agent minus rule +7.44 s
  before the driver's first stop, -8.53 s after), and two marked "pending" slots for the fairer rule
  baselines.
- **Robustness line without Austria 2024** (45% of the agent's held-out regret).
- **Austria 2024 note** rewritten: which policies are about 16 s (agent, belief optimizer, teams) and which are
  not (fixed one-stop 23.0, fixed two-stop 35.9); the reference uses Soft in 23 of 25; the only Soft stint in
  the sampled drivers' real strategies is a final 7 laps; the belief priced Soft from practice.
- **Wording.** "Races the system had never been run on" became "the agent had never been run on them", with
  the exposure disclosure from `docs/eval.md`. One regret definition with "hindsight reference"; belief and
  world simulators named once, in a diagram. "No peeking" now carries its caveat (stint limits and pit loss
  read from the whole race).
- **Next steps** list only work not done yet (the optimizer comparison is done).
- **Footer**: model qwen3-30b-a3b via ASU Research Computing, tag `d4-final-eval`, one agent pass,
  n = 100 held-out / 150 development. References point at the d4 and dev6 run 2 files and `numbers.md`.
- **Cut**: the Pirelli tyre-facts panel (2023 Spain, now a calibration race, not checkable in the repo), the
  Albon example, the pace schematic, the schema table, the circuit hero band, and the pooled tiles.
- **Added**: Limitations and "For the stakeholder" boxes.

## Design

- About 1,140 words in total, of which about 150 are references and footer (draft: about 1,650). Body
  23 pt, nothing below 17 pt outside the references (13 pt) and footer (16 pt).
- Colour-blind-safe series (Okabe-Ito blue, bluish green, orange, reddish purple), each also told apart by
  dash or marker shape and by a value table; held-out vs development by fill and labels.
- The two low-contrast boxes the audit named (the scorer's "Full remaining race" box and the pace schematic)
  are gone with their panels; all text meets 4.5:1 or better on its background.
- Circuit map: the vector original from Wikimedia Commons replaces the 634 px raster (40 ppi at print
  size). The PDF has no raster images.
- 0.65-0.75 in from the trim to any text; checked by `check_layout.py`.

## Update after #85 (rule baselines) and #86 (development repeat pass)

- Author name corrected to "Sarah Maisha".
- Banner rewritten as three held-out points, "in our simulator": **Reliable** (a legal, explained call on
  all 100 decisions; 29 needed the forced-submission turns), **Parity** (same average cost as the teams,
  -0.19 s [-1.88, 1.43], fewer large errors, 13% vs 19% more than 20 s off; the same as the optimizer),
  **Limit** (two simple fixed rules do equally well; safety cars, traffic and rivals are not scored).
- Panel 6 "Rule baselines" fills the two slots from `d4_combined_rules.csv`: run to the flag -0.33 s
  [-2.14, 1.47], one stop in total +1.66 [-0.33, 3.65], with the disclosure that both were frozen on
  development races (5a2bc41) and scored once on held-out after its results were known. The fixed one-stop
  caveat now says its 4.3 s deficit is the forced extra stop (after the driver's first stop, agent minus one
  stop in total is -0.40 [-2.55, 1.81]). Pooled mean regret of every policy is listed for context.
- Per-race table (panel 4) gains the two rules: the agent loses to them at Monaco and Sakhir (+3.5 / +6.9,
  +4.3 / +5.5) and beats them at both Austrias (-2.8 / -1.2, -6.4 / -5.4).
- Development vs held-out table (panel 7) gains the two rules (development -0.31, -0.24).
- Limitations: the development repeat pass (mean regret 3.95 to 4.36 s, same pit/stay call in 145 of 150,
  agent minus optimizer -0.16 to +0.24) and its 4 stalled episodes (one illegal).
- "For the stakeholder" and Next steps rewritten for the clean-air reading and merged into one panel; the
  belief optimizer row left the rule chart (it is in panels 4, 7 and 8).
- Words: about 1,250 in total including about 110 in the references and footer.

## Still open

- Bleed: none; see README "Printing".
