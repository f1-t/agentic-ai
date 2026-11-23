# PITWALL: an F1 race-strategy agent, evaluated without hindsight

CSE 598 (Agentic AI, ASU) capstone. It started from a stakeholder baseline: one question about
one race, answered by Claude with a deterministic simulator. That baseline is kept as the
"Legacy baseline" section below. The project is now an evaluation of a tool-using language-model
agent that makes one pit-stop decision at a time, from a snapshot of a real race.

- **The decision.** At a snapshot (race R, driver D, after lap L), the agent decides: pit now
  onto a compound, or stay out for k laps and reassess.
- **The loop.** It runs a hand-written plan → call tools → observe → refine loop (`src/agent.py`,
  at most 10 steps). The tools (`src/tools.py`) read the snapshot and give:
  - the race picture (`race_picture`);
  - a ranking of the best plan in each strategy family (`rank_plans`, the look-ahead);
  - tyre wear and pit loss, with ranges and probabilities.

  The agent may choose any plan (protocol `v2free`). It ends the episode by submitting a decision
  (`submit_recommendation`).

## How it is judged

There are two simulators over the same race data, from the committed FastF1 cache (`cache/`,
read offline):

- **The belief** (`configs/sim/belief.yaml`) is what every policy plans on, the agent's tools
  included. It is fitted only on what was known at the snapshot:
  - the race laps up to lap L;
  - this weekend's practice;
  - priors from earlier races only.

  Under SC or VSC, it draws the period from rates and lengths fitted on earlier races.
- **The world** (the judge) knows the whole race. It scores every policy's plan and finds the
  best plan from the same snapshot under the same random draws. The **regret** is the
  difference.

**The information contract** (docs/eval.md, "Information contract"): nothing a policy can read
depends on anything after the snapshot. Tests enforce it (`tests/test_future_blindness.py`,
`tests/test_eval_information_contract.py`), including that the model never sees the real race,
season or driver names.

**The comparators**, all scored the same way:
- simple rules: `run_to_flag` and `one_stop_total`;
- the teams' actual strategy;
- `belief_optimizer`, the optimizer the agent's tools run, deciding alone.

## Result

**Test set** (`reports/eval/rerun/v3/test9/`, tag `eval-v3`, one pass of `qwen3-30b-a3b`):

- Over all nine test-set races, the agent is ahead of both simple rules (-2.28 and -2.21 s mean
  regret; the race-cluster interval includes zero). It is also ahead of the teams' actual
  strategies (-5.03 s), and level with its look-ahead, the optimizer its tools run. Most of
  those leads come from 2019 Monaco. The field ran a managed pace there, and the simulators model
  neither managed pace nor track position ("Unmodelled team decisions" in docs/eval.md).
- On the five races examined for the first time (the pre-registered primary test), the agent is
  level with the rules (+1.67 s). The shortfall sits at 2022 Australia and 2022 Miami, where the
  look-ahead has the same regret.
- On the development races (`reports/eval/rerun/v3/dev20/`) it is ahead of the rules by 5.29 s.

This is the second evaluation. The first, at `eval-v2-test` (`reports/eval/rerun/test9/` and
`dev20/`), is kept as run.

The full tables, the method and the limitations are in docs/eval.md:
- "Results at eval-v3";
- "Second evaluation";
- "Held-out result (test9) at eval-v2-test";
- "Changes after the freeze";
- "Limitations".

Every other file in `reports/eval/` is a superseded record ("Result files").

## Setup (Python 3.12)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt        # runtime + pytest + pyyaml
pip install -r requirements-hf.txt         # only for a local Hugging Face model
cp .env.example .env                       # then set the backend variables below
```

Run everything from this directory (`f1-race-strategy-agent-local-test/`).

**Model backend.** `LLM_BACKEND` chooses it. Every value comes from `.env`, and no key is ever
committed or recorded.

| backend | `LLM_BACKEND` | variables |
| --- | --- | --- |
| ASU RC Voyager (the evaluated model) | `voyager` | `VOYAGER_API_KEY`, `VOYAGER_BASE_URL`, `VOYAGER_MODEL`, `VOYAGER_MAX_TOKENS`, `VOYAGER_TEMPERATURE` |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` |
| local Hugging Face (Transformers in-process) | `huggingface` | `HF_MODEL`, `HF_DEVICE`, `HF_MAX_NEW_TOKENS`, `HF_TOKEN` (gated models only) |

## Tests and the smoke run

| what | command | needs a model |
| --- | --- | --- |
| fast tier (about 2.5 min) | `python -m pytest -q` | no |
| slow tier (real races, end to end) | `python -m pytest -q -m "slow and not legacy"` | no |
| legacy tier (superseded switches still load) | `python -m pytest -q -m legacy` | no |
| smoke: 8 development scenarios through every stage, then the trajectory lint | `python -m src.eval.rerun smoke --n 8 --jobs 4` | yes |
| one agent episode on 2021 Spain, HAM at lap 0, graded | `python tests/test_case.py` | yes |
| one decision from a `RaceSnapshot` JSON (`src/snapshot.py`; a built-in 2021 Spain demo without one) | `python -m src.run [snapshot.json]` | yes |

CI (`.github/workflows/tests.yml`) runs the fast tier and the slow tier split into four jobs.
`tests/test_case.py` is a script, not collected by pytest.

**Rerunning the evaluation** takes one command per stage: `python -m src.eval.rerun
{scenarios,references,baselines,agent,tables} --set dev20 ...`. The details are in docs/eval.md,
"Rerunning the evaluation". The test set `test9` was built once and run twice (`eval-v2-test`,
`eval-v3`), and is not rebuilt.

## Where things are

| path | what |
| --- | --- |
| `src/agent.py`, `src/tools.py`, `src/llm/` | the agent loop, its tools, the model backends |
| `src/sim/` | the simulators (belief and world), fits, priors, the optimizer bridge |
| `src/eval/` | scenarios, references, policies, scoring, the rerun pipeline, the lint |
| `configs/sim/`, `configs/eval/` | simulator configs (current and legacy), sets and the pre-registered reporting choices |
| `reports/eval/rerun/v3/{dev20,test9}/` | the current development and test-set results (`eval-v3`) |
| `reports/eval/rerun/{dev20,test9}/` | the first evaluation (`eval-v2-*`), kept as run |
| `docs/eval.md`, `docs/rules.md`, `docs/baselines.md`, `docs/traffic.md` | the protocol, the race rules, the baselines, the traffic model |
| `poster/` | the research poster build |

## Legacy baseline (the original stakeholder project)

The original baseline asked Claude one question about the 2021 Spanish GP:

> Hamilton starts on SOFT; one stop or two?

Claude answered with four full-race tools and a deterministic linear simulator, which:
- had no safety cars, traffic or fuel effect;
- used a 22 s pit loss and a 10-lap minimum stint.

The answer was graded against that simulator's brute-force optimum, 5510.1 s
(`data/ground_truth.json`). That code is frozen in `src/legacy_tools.py` so the result stays
reproducible. It runs with no key and no network:

```bash
python -c "from src import legacy_tools; print(legacy_tools.optimal_strategy('SOFT'))"
# -> {'num_stops': 2, 'first_pit_lap': 28, ..., 'total_time_s': 5510.1}
```

`docs/sample_run.txt` is a captured run of that baseline. The current agent no longer uses these
tools. `tests/test_case.py` now runs the snapshot agent, and its legacy check lives in the test
suite. The proposal is `CSE598_Capstone_Proposal_Manav.docx`.

## Bird's-eye race demo

`python scripts/run_game_demo.py`, then open http://127.0.0.1:8000.

- The moving field is an illustrative JavaScript animation, not the project's simulator.
- The "Project Strategy Agent" panel runs the real `src.agent.run_agent` loop on a snapshot from
  `data/snapshots/`. It uses the configured backend and may spend provider quota.
- `python scripts/build_track_geometries.py` (network) adds real circuit outlines.
