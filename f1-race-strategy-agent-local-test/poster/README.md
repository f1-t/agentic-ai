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
python poster/build_poster.py --dev <dev dir> --test <test dir> --fidelity reports/calibration/fidelity --export
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

| input | files read |
|---|---|
| `--dev`, `--test` | `provenance.json`, `scenarios.json`, `tables/combined.csv`, `tables/summary.csv`, `tables/chart_regret_s.csv`, `tables/tail.csv`, `tables/paired_agent_minus_x.csv` (with agent rows) or `tables/paired_belief_optimizer_minus_x.csv` (without) |
| `--test` only (a set with eras) | `tables/tail_by_era.csv`, `tables/paired_agent_minus_x_era_<era>.csv` |
| `--fidelity` | `main/run.json`, `main/scorecard.md`, `main/ledger.csv`, `ablation/headlines.csv`, `ablation/ablation.csv`; the noise floor from a `ceiling_pooled.csv` anywhere below it, else the table under "The noise floor" in `docs/eval.md` (`--noise-floor FILE` overrides) |
| always | `data/eval_set.json` (race ids only) |

**Cross-checks.** `poster_stats.py` recomputes every number from `combined.csv`, using the same cluster
bootstrap as `src.eval.compare`. It stops on any disagreement with the pipeline's own tables, the scorecard or
the ablation.

**Refusals.** It refuses:

- a development set passed as `--test` (`--fixture` allows one for layout tests only, and stamps the page NOT
  A RESULT);
- a `--test` whose races are not those of `data/eval_set.json`;
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
