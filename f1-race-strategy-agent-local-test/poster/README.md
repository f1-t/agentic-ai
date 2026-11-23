# PITWALL poster

**The poster is `PITWALL_poster_24x36in.pdf`**: 24 x 36 in portrait, one page of 1728 x 2592 pt, fully vector,
with fonts embedded. `build_poster.py` builds it from the committed evaluation outputs at tag `eval-v3`.

The build reads files only. It imports nothing from `src`, runs no pipeline stage or lint, and loads no race data.

## Rebuild (one command), then the layout check

From `f1-race-strategy-agent-local-test/`, with the project's Python (pandas, numpy, pyyaml), Chromium and poppler:

```bash
python poster/build_poster.py --export      # defaults: --dev reports/eval/rerun/v3/dev20 --test reports/eval/rerun/v3/test9
python3 poster/check_layout.py
```

`--group NN` (with `--export`) also writes the PDF as `poster/GroupNN_Poster.pdf`. Without the option, no copy is
written.

The example decision is the development episode `poster_stats.EXAMPLE` (`2022_06_spain:MAG:25`). It is read from
its trajectory, and the build checks it against the scored agent row.

Defaults:

- `--dev` and `--test` default to the committed v3 outputs.
- `--fidelity` defaults to `reports/calibration/fidelity`.
- `--scorecard` defaults to `belief-guard-243bf40/main`, the latest committed scorecard of the adopted configuration.

Outputs:

- Without `--export`, the build writes the HTML and `numbers.md`.
- With `--export`, it also writes the PDF, the 300 ppi PNG and the 1200 px preview.

`check_layout.py` exits 1 on any of these:

- a clipped box;
- text within 0.5 in of the trim;
- text below 16 pt, or below 12 pt in the references and footer.

## What each input must contain

| input | files read |
|---|---|
| `--dev`, `--test` | `provenance.json`, `agent.csv`, `run_support/*departures.csv`, `tables/combined.csv`, `tables/chart_regret_s.csv`, `tables/reporting_paired.csv`, `tables/reporting_tail.csv`, `tables/summary.csv`, `tables/agent_latency.csv` |
| `--test` only | `tables/reporting_paired_by_era.csv`, `trajectories/*.jsonl` (text, for the tyre-life prior tier) |
| `--fidelity` | `<scorecard>/run.json`, `scorecard.md`, `ledger.csv`, `ceiling_pooled.csv` (the noise floor) |
| always | `data/eval_set.json` (race ids only) |

**Sets.** The test set is the reporting group "all nine". The development set is dev20.

**Verdicts.** Every verdict word ("beat", "matched", "trailed") comes from the pre-registered 95% (race, driver)
interval.

**Cross-checks.** `poster_stats.py` recomputes every paired difference, interval, verdict, mean regret and tail share
from `combined.csv`, using the same cluster bootstrap as `src.eval.compare`. It stops on any disagreement with the
pipeline's own tables:

- `reporting_paired.csv`;
- `chart_regret_s.csv`;
- `reporting_tail.csv`;
- `summary.csv`.

Other checks:

- **Departures from the look-ahead.** These are checked against `combined.csv`.
- **The adopted / not adopted list.** This is checked against the world and belief choices recorded in
  `provenance.json`.
- **The scorecard.** Its headline is checked against its own noise floor.

**`numbers.md`.** It lists every printed number with its file and computation.

## Files

| file | what |
|---|---|
| `build_poster.py` | layout, charts and text |
| `poster_stats.py` | every number, its file and its computation; the cross-checks |
| `numbers.md` | generated: every printed number with file and computation |
| `check_layout.py` | layout check in headless Chromium |
| `PITWALL_poster_24x36in.html` / `.pdf` / `_300ppi.png` / `_preview.png` | the poster |
| `2023_F1_CourseLayout_Spain.svg` | circuit map, unmodified from Wikimedia Commons ([file page](https://commons.wikimedia.org/wiki/File:2023_F1_CourseLayout_Spain.svg)), by ごひょううべこ, CC BY-SA 4.0 |

## Printing

The page is exactly 24 x 36 in with no bleed. All text sits at least 0.5 in inside the trim. The background is dark
to the edge, so ask the print shop whether it prints full-bleed dark backgrounds at this size without a bleed margin.
