# data/derived

Tables derived from public sources or from the committed race cache, read at run time by
`src/sim/tyre_history.py` (docs/eval.md, "Historical tyre-life prior").

## tyre_nominations.csv

One row per race weekend in the offline cache (column `cached_on`: on main from the start, or
added by `data-more-races`, #92): the Pirelli dry compound nominated for each role (`hard`, `medium`,
`soft`, as the C-number Pirelli published), the season's compound `scale` (compound_scale.csv),
the `scheduled_laps` of the race, the circuit key (`event_rates.CIRCUITS`), and two sources.

This is regulation information Pirelli publishes before each race weekend; it says nothing about
what happened in any race. Held-out races (data/eval_set.json) have a row, since the nomination
is public pre-race information, but their race data never feeds the history. The same holds for
the five 2021-22 draws, which are registered in data/eval_set.json now (#94 listed them in
`held_out_extra`): 2021_07_france, 2021_18_mexico, 2022_03_australia, 2022_05_miami,
2022_14_belgium.

Sources, fetched 2026-11-21 with a generic web client, no personal details:
- `source_url`: Pirelli's pre-race preview or compound announcement (press.pirelli.com,
  pirelli.com) wherever one could be fetched; otherwise formula1.com's "what tyres will the
  teams have" article (which reports Pirelli's choice) or a Pitpass repost of Pirelli's preview.
- `second_source_url`: the race's English Wikipedia article (or formula1.com, motorsport.com,
  RaceFans' copy of Pirelli's season table where Wikipedia gives no C-numbers).
- `scheduled_laps`: the race's Wikipedia infobox / race text; races run short (aborted starts)
  are noted, the scheduled distance is kept.

All 38 rows are confirmed by two fetched pages; none is blank. The 15 rows of #92's second
group were fetched on 2026-11-21 as well, each from Pirelli's race-week preview plus Wikipedia
or formula1.com (laps from Wikipedia); 2022 Australia's trio is non-consecutive (C2, C3, C5). Rows that would need a guess are
to be left blank with a note: `tyre_life_prior` then compares compounds by role.
Judgment calls: 2021 Austrian GP uses the race-week preview (C3/C4/C5), not Pirelli's February
pre-season table (C2/C3/C4), which was changed; 2021 Spanish GP has no Pirelli page (404), so
its sources are Wikipedia and RaceFans' copy of Pirelli's season table.

## compound_scale.csv

The C-numbers changed meaning between seasons, so every (scale, compound) maps to one
`common_index` in steps of the 2019-2022 scale (larger is softer):

| scale | seasons | mapping | source |
|---|---|---|---|
| C1-C5_2019 | 2019-2021 | Cn -> n | Pirelli, 18 Feb 2019 (press.pirelli.com/whats-new-with-pirellis-2019-formula-1-tyres/): five compounds numbered 1 to 5, hardest to softest; FIA 10 Dec 2019: 2019 tyres kept for 2020; Pirelli 2021 Bahrain preview: "the actual compounds are the same as last year", new construction |
| C1-C5_2022 | 2022 | Cn -> n | Pirelli, 15 Jul 2021 (press.pirelli.com/a-look-into-the-future-of-formula-1-with-new-18-inch-tyres-for-2022-cars/): 18-inch compounds "completely new", five grades |
| C0-C5_2023 | 2023 | C0 -> 1, C1 -> 1.5, Cn -> n | The Race, quoting Pirelli's Mario Isola (the-race.com/formula-1/pirellis-new-for-2023-sixth-f1-tyre-compound-explained/): "the old C1 is now the C0", "the new C1 is much closer to the C2" |
| C1-C5_2024 | 2024 | C1 -> 1.5, Cn -> n | RacingNews365, 21 Nov 2023 (racingnews365.com/pirelli-axe-f1-tyre-as-no-new-compounds-to-be-introduced-in-2024): C0 eliminated, "the 2024 compounds will be the same as this year" |
| C1-C6_2025 | 2025 | C1 -> 1.5, Cn -> n, C6 -> 6 | Pirelli, 12 May 2025 (press.pirelli.com/the-c6-to-make-its-debut-in-imola/): C6 the softest of the 2025 range; The Race (the-race.com/formula-1/bahrain-f1-2025-test-tyres/): "the hardest C1 remains similar to the previous version" |

The 2023 C1 is placed half a step above the old C1: Pirelli calls it "much closer to the C2",
which would argue for more than 0.5; 0.5 is the plain midpoint, stated rather than tuned. The
index is ordinal: a C3 of 2019 and a C3 of 2024 share an index but are not the same rubber
(the regulation-era weight in configs/tyre_history.yaml covers the 2022 change).

## tyre_history.csv, tyre_history_stints.csv, tyre_history_manifest.json

Built by `python -m src.sim.tyre_history build` from every cached race not held out
(data/eval_set.json plus `held_out_extra` in configs/tyre_history.yaml); the manifest lists
the races and their dates. Rerun it when races are added to the cache.

## circuit_ratings.csv, prerace_guidance.csv

Two tables of what the tyre supplier (Pirelli) publishes BEFORE each race, one row per weekend of
tyre_nominations.csv (all 38, the nine held-out weekends too: like the nomination, this is public
pre-race information). Nothing about any race's outcome is recorded, and no race data was read to
build them. A blank cell means the item was not published or could not be confirmed; readers then
fall back to what they did without the tables. Fetched 2026-11-22 with a generic web client, no
personal details.

**circuit_ratings.csv**: the "track characteristics" ratings of the supplier's race-week preview
(published Monday to Wednesday of race week), read off the preview's infographic: `traction`,
`braking`, `lateral`, `tyre_stress`, `asphalt_grip`, `asphalt_abrasion`, `track_evolution`,
`downforce`, each 1 (low) to 5 (high); `other_ratings` for a category outside these; the minimum
starting pressures for slicks (`min_pressure_front_psi`, `min_pressure_rear_psi`) and camber limits
(`camber_limit_front_deg`, `camber_limit_rear_deg`) where the preview gives them; `published_date` of
the preview; `source_url` (the preview page, or a repost of it) and `image_url` (the infographic read).
- 2019 previews carry five categories (tyre stress, lateral, asphalt grip, asphalt abrasion,
  downforce): traction, braking and track evolution are blank. 2021 previews have no track evolution.
- 2019-21 infographics print the number beside each bar; from 2022 they are needle gauges, read by
  needle position (near-edge reads are named in `notes`).
- 2020 Sakhir: the joint Bahrain/Sakhir preview's second infographic (outer circuit). 2021 Austrian:
  the joint Styrian/Austrian preview (same layout). 2022 Emilia Romagna and 2023 Spain: a Pitpass
  repost of the preview's infographic (Pirelli's own page not found); 2023 Spain's pressures agree
  with the FIA-published Pirelli preview of 2023-05-31.
- Coverage: 38/38 rows with ratings; the four life covariates (abrasion, tyre stress, lateral,
  traction) for 36 (not 2019).

**prerace_guidance.csv**: the supplier's pre-race strategy guidance, from its release after
qualifying (the "Saturday" / "qualifying" release; after the sprint on a sprint weekend,
`published_stage` "other pre-race"), always published before the race start (2022 Miami and Canada
are dated the race day in CEST but went out on Saturday evening local time: see `notes`).
- Up to three expected strategies in the order given (`s1` the one called fastest where the release
  ranks them, otherwise the order listed, said in `notes`): `sN_stops`, `sN_sequence` (role names
  SOFT/MEDIUM/HARD by the weekend's nomination, e.g. `MEDIUM-HARD`), `sN_windows` (the stated pit-stop
  laps, one `a-b` range per stop, `;`-separated; stated stint lengths converted to stop laps and said
  in `notes`). A sequence the release does not state in full is recorded as worded (e.g. `MEDIUM/HARD
  mix`, `?-?-SOFT`); readers use only fully stated sequences and still read the stop count.
- `pace_gap_soft_medium_s`, `pace_gap_medium_hard_s`: stated per-lap gaps (positive: the harder is
  slower); `degradation_note`: a short paraphrase of the stated wear expectation.
- `quote`: the sentence(s) the row was taken from (short; a few are close paraphrases, said in
  `notes`); `source_url`: the release.
- Coverage: an expected stop count for 38/38, a fully stated compound sequence for 32, pit-window
  laps for only 9 (2019 Monaco and UK, 2020 Sakhir, 2021 Bahrain, 2022 Singapore and Abu Dhabi, 2023
  Spain, 2024 Monaco, 2025 Qatar), stated pace gaps for 4.
- Only documents published before the race; no race report, result or post-race release was opened
  to fill a gap. 2024 Brazil's qualifying moved to race morning, so its row is the Saturday release
  written before qualifying. 2021 Emilia Romagna and Spain give a stop count only.

Read by `src/sim/prerace.py` (registered in `src/sim/provenance.py` as kind "pre_race").
