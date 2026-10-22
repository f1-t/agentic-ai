# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

CSE 598 (Agentic AI, ASU) capstone. The repo root is a thin monorepo; all code lives in
`f1-race-strategy-agent-local-test/`, which is the stakeholder's baseline (Manav Ananthakumar,
upstream `DoomDust7/f1-race-strategy-agent`) imported wholesale in commit `65407b4`. Our team
extends it. The proposal that defines the project, success criteria, evaluation plan and
"next phase" roadmap is `f1-race-strategy-agent-local-test/CSE598_Capstone_Proposal_Manav.docx`
(read it with `unzip -p <file> word/document.xml | sed 's/<[^>]*>//g'`).

**Run every command below from `f1-race-strategy-agent-local-test/`.** `tests/test_case.py`
inserts its parent dir onto `sys.path` and `src/tools.py` resolves `cache/` relative to itself,
so other working directories break.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt           # anthropic, fastf1, pandas, numpy, dotenv
pip install -r requirements-hf.txt        # only for LLM_BACKEND=huggingface (torch, transformers)
cp .env.example .env                      # then edit; .env.example defaults to the huggingface backend

python tests/test_case.py                 # THE graded test: runs the agent, prints PASS/FAIL, exit 0/1
python -m src.run "Hamilton starts on SOFT. One-stop or two-stop, and when should he first pit?"

# Offline sanity check: no API key, no network, no model. Exercises data + simulator only.
python -c "from src import tools; print(tools.optimal_strategy('SOFT'))"
# -> {'num_stops': 2, 'first_pit_lap': 28, ..., 'total_time_s': 5510.1}  (must match data/ground_truth.json)

python scripts/build_cache.py             # regenerate cache/ (needs network); only when adding/refreshing race data
```

There is no pytest suite, linter, or formatter configured. `tests/test_case.py` is a plain
script, not a pytest module. The agent test needs a working model backend (an Anthropic key
with credit, or a local HF model download), so prefer the offline sanity check for anything
touching `tools.py`.

Backend is chosen by `LLM_BACKEND` (`anthropic`|`claude` or `huggingface`|`hf`|`local`).
Anthropic uses `ANTHROPIC_MODEL` (default `claude-opus-5`). HF uses `HF_MODEL`
(default `Qwen/Qwen3-4B-Instruct-2507`), `HF_DEVICE` (`auto` prefers MPS, then CUDA, then CPU),
`HF_MAX_NEW_TOKENS`, optional `HF_TOKEN`.

## Architecture

Four layers, deliberately decoupled so model providers can be swapped without touching F1 logic:

1. **`src/tools.py` — data, tools, simulator, and ground truth, all in one module.**
   - One FastF1 session is lazily loaded once (`_load_session`) in **offline mode** from the
     committed `cache/`. `cache/` is intentionally committed (see its `.gitignore` note).
   - The race is hardcoded by module constants `SEASON`, `GP_NAME`, `SESSION`, plus the
     physics knobs `PIT_LOSS_S`, `MIN_STINT_LAPS`, `OUTLIER_FACTOR`. Multi-race support means
     threading these through rather than editing constants.
   - `TOOL_SPECS` is the registry the agent consumes: each entry is `name`, `description`,
     `input_schema` (JSON schema sent to the model) and `fn`. **Adding a tool = adding an
     entry here**; nothing else needs to change for either backend.
   - `simulate_strategy` is the deterministic oracle: linear per-compound degradation fit
     (`estimate_tyre_degradation`, from green-flag non-pit laps under 107% of the compound's
     fastest), additive stint times, constant pit loss. It validates laps sum to race length,
     every stint >= `MIN_STINT_LAPS`, and >= 2 distinct compounds.
   - `optimal_strategy` brute-forces the simulator and is **not** exposed to the agent. It is
     the source of `data/ground_truth.json`.

   **Invariant:** the simulator is both the agent's tool and the grader's reference. Any change
   to `simulate_strategy`, `estimate_tyre_degradation`, or the physics constants silently
   invalidates `data/ground_truth.json`; regenerate it from `tools.optimal_strategy()` (and
   update `docs/sample_run.txt` if the expected numbers change). The proposal's stated next
   phase is to break this coupling (learned/uncertain degradation model, safety cars, traffic,
   a rival agent) so the agent can no longer "lean on the oracle".

2. **`src/agent.py` — the hand-written plan→act→observe→refine loop.** No agent framework, on
   purpose, so the cycle is visible. `MAX_STEPS = 10`. The loop ends when the model calls the
   synthetic `submit_recommendation` tool (`SUBMIT_TOOL`, defined here, not in `TOOL_SPECS`).
   `_dispatch` strips any argument not declared in the tool's schema before calling `fn`,
   because local models invent keys.
   - **The message history format branches on `backend.name == "anthropic"`**: Anthropic gets
     native content blocks / `tool_result` objects; every other backend gets a textual
     transcript where tool calls and results are JSON-dumped into plain user/assistant strings.
     A new backend must pick one of these two paths (today, anything not named `anthropic`
     gets the text path).

3. **`src/llm/` — provider adapters.** `base.py` defines `ModelBackend.run_turn(system_prompt,
   messages, tools, max_tokens) -> ModelTurn(text, tool_calls: [ToolCall(id, name, arguments)],
   raw)`. `factory.create_backend()` reads `LLM_BACKEND`. `anthropic_backend.py` uses native
   tool use with adaptive thinking. `huggingface_backend.py` runs Transformers in-process
   (no vLLM), appends a JSON-envelope instruction (`{"tool_calls":[...]}`) and tool list to the
   system prompt, greedy-decodes, and parses the JSON back into `ToolCall`s; a parse failure
   yields zero tool calls, which ends the loop with no recommendation.

4. **`src/grade.py` + `tests/test_case.py` — evaluation.** `grade()` re-simulates the agent's
   `stint_plan` (it does not trust the agent's self-reported `total_time_s`) and passes if the
   plan is valid and within `tolerance_pct` (1%) of `optimal_total_time_s`. The test is one
   fixed question (Hamilton, SOFT start, 2021 Spanish GP). The proposal's evaluation plan
   calls for a scenario set measuring stop-count correctness, first-pit-lap error, invalid-plan
   rate, and cost/latency; none of that harness exists yet.

## Things that bite

- FastF1 logging is forced to `ERROR` and warnings are suppressed in `tools.py` so the
  trajectory stays readable; re-enable when debugging data loading.
- `estimate_tyre_degradation` returns an `{"error": ...}` dict (not an exception) when a
  compound has < 5 clean laps; `_deg_table` silently drops such compounds, so a compound can
  be "available" in `get_race_info` but unknown to the simulator. In the 2021 Spanish GP
  data only SOFT and MEDIUM appear at all, so the simulator only ever has two compounds.
- `requirements.txt` pins upper bounds (`fastf1<4`, `anthropic<2`) because the committed
  `.ff1pkl` cache format and Messages API calls depend on them.
