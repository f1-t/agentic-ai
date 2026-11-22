# PITWALL poster: version 2

Version 2 is built from the corrected evaluation pipeline. It replaces the 24 x 36 in build of PR #84. That
build's numbers came from an evaluation that leaked race-wide information (tyre stint limit, pit loss, race
length and others) into the policies. They are discarded, and none of them appears here.

## What changed

- **Inputs.** `build_poster.py` takes `--dev`, `--test` and `--fidelity`:
  - `--dev` and `--test` are the out-dirs of `python -m src.eval.rerun` (tables stage).
  - `--fidelity` is the fidelity scorecard's outputs.
  - The build no longer reads `reports/eval/d4_*` or `dev6_*`.
- **Placeholders.** A number that needs a run not done yet prints as a hatched, dashed "pending" box, and a
  figure row prints as a hatched band. While held-out agent rows are missing, a draft ribbon heads the page.
  `numbers.md` marks each pending number and names the run that fills it.
- **Cross-checks.** Each of these must equal the pipeline's own tables, or the build stops:
  - every mean, interval, tail share and paired difference (`chart_regret_s.csv`, `tail.csv`, `summary.csv`,
    `paired_*_minus_x*.csv`, `tail_by_era.csv`);
  - the scorecard headline (`run.json` against `scorecard.md`);
  - the ablation deltas (`ablation.csv` against `headlines.csv`);
  - the noise floor's level 0 against the headline.
- **Safety checks.**
  - The build refuses a development set given as `--test`. Only `--fixture` accepts one, and it stamps the
    page "NOT A RESULT".
  - It refuses a `--test` that does not cover `data/eval_set.json`.
  - It refuses an agent model other than qwen3-30b-a3b.
- **Content, in the owner's framing.**
  - The banner states three outcomes: integrity, the judge's measured accuracy, and the agent against simple
    rules and the teams.
  - Panels:
    1. the system, with the predictive simulator against the judge;
    2. evaluation integrity: the future-blindness test, the leaks removed and the test set;
    3. the judge's accuracy against its ceiling, and what each factor did;
    4. results: paired differences, levels, the tail, and by era, development beside held-out;
    5. what the LLM adds over the optimizer, and its reliability;
    6. limitations;
    7. for the stakeholder.
  - Verdict words ("ahead of", "level with", "behind") are computed from the intervals.
- **Layout.**
  - Body text is 22 pt. Nothing outside the references and footer is below 16 pt.
  - The page is fully vector, with fonts embedded, and `check_layout.py` passes.
