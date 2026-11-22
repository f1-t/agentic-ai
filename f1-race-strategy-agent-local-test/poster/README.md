# PITWALL poster

**Current poster: `PITWALL_poster_24x36in.pdf`** (24 x 36 in portrait, one page, 1728 x 2592 pt),
built from `PITWALL_poster_24x36in.html` by `build_poster.py`. `PITWALL_research_poster.{html,pdf}` and
`PITWALL_preview.png` are the Nov 18 draft, kept for reference; their numbers are superseded (Milestone 2
and pre-R8 files).

## Rebuild and export (one command)

From `f1-race-strategy-agent-local-test/`:

```bash
.venv/bin/python poster/build_poster.py --export   # HTML, numbers.md, PDF, 300 ppi PNG, 1200 px preview
python3 poster/check_layout.py                     # clipped boxes, 0.5 in safe area, sizes, word count
```

`build_poster.py` without `--export` writes only the HTML and `numbers.md`. The export needs Chromium
(`chromium` on PATH, or set `$CHROMIUM`) and poppler (`pdfinfo`, `pdftoppm`); it stops if the PDF is not one
page of 1728 x 2592 pt. Fonts: Noto Sans and Noto Sans CJK JP (for the map author's name in reference 4);
without them the browser substitutes fonts and the layout can shift, so rerun `check_layout.py` after a
build on another machine. `check_layout.py` exits 1 if anything is clipped, sits within 0.5 in of the trim,
or prints below 16 pt (12 pt in the references and footer).

## Files

| file | what |
|---|---|
| `build_poster.py` | layout, charts and text; every number comes from `poster_stats.py` |
| `poster_stats.py` | all numbers, from the committed result files in `reports/eval/` only (see "Data used"); stops on any mismatch with the committed paired tables |
| `numbers.md` | generated: every number on the poster, its file and its computation |
| `check_layout.py` | layout check in headless Chromium |
| `PITWALL_poster_24x36in.html` / `.pdf` | the poster (vector PDF, fonts embedded, no raster images) |
| `PITWALL_poster_24x36in_300ppi.png` | 7200 x 10800 px, 300 ppi at 24 x 36 in |
| `PITWALL_poster_24x36in_preview.png` | 1200 px wide preview |
| `2023_F1_CourseLayout_Spain.svg` | circuit map, unmodified from Wikimedia Commons ([file page](https://commons.wikimedia.org/wiki/File:2023_F1_CourseLayout_Spain.svg)), by ごひょううべこ, CC BY-SA 4.0, SHA-1 `7efd092b460d872da5a14db6c60c87c31e03c03c` (as listed on the file page) |
| `CHANGES.md` | what changed from the earlier drafts |

## Data used

`poster_stats.py` reads `reports/eval/d4_combined_rules.csv` and `dev6_run2_combined_rules.csv` (the
combined rows plus the rule baselines `run_to_flag` and `one_stop_total`, frozen at `5a2bc41` and scored
once on the held-out races after its results were known). It checks their other rows against
`d4_combined.csv` / `dev6_run2_combined.csv`, and their paired results against the committed
`*_rules_compare_*_paired.csv`. Run-to-run variation comes from `dev6_run3_combined.csv`, an identical
repeat pass on development.

## Printing

The page is exactly 24 x 36 in with no bleed; text keeps at least 0.5 in from the trim. The background is
dark to the edge: ask the print shop whether they print full-bleed dark backgrounds on this size without a
bleed margin, or whether they want a 0.125 in bleed (which would need a 24.25 x 36.25 in page) or accept a
thin white rim.
