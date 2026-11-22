# PITWALL poster

**The poster is `PITWALL_poster_24x36in.pdf`**: 24 x 36 in portrait, one page of 1728 x 2592 pt, fully vector,
with fonts embedded. `build_poster.py` writes it from the outputs of the corrected evaluation pipeline. The
build reads files only: it imports nothing from `src` and loads or simulates nothing.

`PITWALL_research_poster.{html,pdf}` and `PITWALL_preview.png` are the Nov 18 draft, kept as a record. Their
numbers are superseded.

## Rebuild (one command), then the layout check

Run from `f1-race-strategy-agent-local-test/` with the project's Python (pandas, numpy, pyyaml), Chromium and
poppler:

```bash
python poster/build_poster.py --dev reports/eval/rerun/dev20 --test reports/eval/rerun/test9 --fidelity reports/calibration/fidelity --export
python3 poster/check_layout.py
```

`--dev` and `--test` are the `--out-dir` of `python -m src.eval.rerun` (by default `reports/eval/rerun/dev20`
and `reports/eval/rerun/test9`), after its `tables` stage. For the held-out set, run that stage with
`--harness`, so that `provenance.json` records the future-blindness harness counts.

**Placeholders.** If you leave out `--test`, or a set has no agent rows, the build still succeeds:

- each number that needs those rows prints as a hatched "pending" box;
- each figure row prints as a hatched band;
- a draft ribbon heads the page.

**Outputs.**

- Without `--export`, the build writes the HTML and `numbers.md`.
- With `--export`, it also writes the PDF, `PITWALL_poster_24x36in_300ppi.png` and
  `PITWALL_poster_24x36in_preview.png` (1200 px wide).
- It stops if the PDF is not exactly one page of 1728 x 2592 pt.

**The layout check.** `check_layout.py` exits 1 on any of these:

- a box whose content is clipped;
- text within 0.5 in of the trim;
- text below 16 pt, or below 12 pt in the references and footer.

It also prints the word count and the words at each type size.

## What each input directory must contain

The build reads the committed files only. It never runs the `tables` stage or the lint on test9, because both
load held-out race data.

| input | files read |
|---|---|
| `--dev`, `--test` | `provenance.json`, `agent.csv`, `tables/combined.csv`, `tables/reporting_paired.csv`, `tables/reporting_tail.csv`, `tables/agent_latency.csv` |
| `--test` only | `tables/reporting_paired_by_era.csv`, `tables/summary.csv`, `tables/reporting_header.txt`, `trajectories/*.jsonl` (text, for the tyre-life prior tier) |
| `--fidelity` | `sc-align-997d69d/main/` (`run.json`, `scorecard.md`, `ledger.csv`, `ceiling_pooled.csv`), the adopted configuration's scorecard; `adopt-c67c9a7/ablation/` (each adopted choice switched back off); `ablation/` (f7be251, the learned calibrator only). `--scorecard`, `--ablation` and `--noise-floor` override them |
| always | `data/eval_set.json` (race ids only) |

Where the numbers come from:

- **Groups.** The held-out groups come from the reporting choices recorded verbatim in `provenance.json`
  (`configs/eval/reporting.yaml`; its sha256 is verified). They are the fresh five (the primary test), the
  earlier four and all nine.
- **Model id.** The model id comes from `agent_run.passes[].env`.
- **Judge description.** The judge's description, and which factors are on, come from the world and belief
  `choices` in `provenance.json`.

**Cross-checks.** `poster_stats.py` recomputes every paired difference, (race, driver) and race interval, verdict
and tail share from `combined.csv`, using the same cluster bootstrap as `src.eval.compare`. It stops on any
disagreement with:

- `reporting_paired.csv`, `reporting_paired_by_era.csv`, `reporting_tail.csv` and `summary.csv`;
- the scorecard and its noise floor;
- the pre-registered fidelity sentence;
- the ablation.

Some numbers are computed here with no pipeline table to check them against. `numbers.md` says so for each:

- the one-race (Monaco) sensitivity;
- the fresh races' per-race differences;
- the development split by circuit history.

**Refusals.** It refuses:

- a development set as `--test`;
- a `--test` that does not cover `data/eval_set.json`;
- an agent model other than `qwen3-30b-a3b`.

## Files

| file | what |
|---|---|
| `build_poster.py` | layout, charts and text; computed verdict words ("ahead of", "level with", "behind") |
| `poster_stats.py` | every number, its file and its computation; the cross-checks; placeholders |
| `numbers.md` | generated: every printed number (or pending placeholder) with its file and computation |
| `check_layout.py` | layout check in headless Chromium |
| `PITWALL_poster_24x36in.html` / `.pdf` | the poster |
| `PITWALL_poster_24x36in_300ppi.png` | 7200 x 10800 px, 300 ppi |
| `PITWALL_poster_24x36in_preview.png` | 1200 px wide preview |
| `2023_F1_CourseLayout_Spain.svg` | circuit map, unmodified from Wikimedia Commons ([file page](https://commons.wikimedia.org/wiki/File:2023_F1_CourseLayout_Spain.svg)), by ごひょううべこ, CC BY-SA 4.0 |
| `CHANGES.md` | what changed from the earlier build (PR #84) |

## Printing

The page is exactly 24 x 36 in with no bleed. All text sits at least 0.5 in inside the trim.

The background is dark to the edge. Ask the print shop whether it prints full-bleed dark backgrounds at this
size without a bleed margin. If it wants one, it will need a 0.125 in bleed, which means a 24.25 x 36.25 in
page.
