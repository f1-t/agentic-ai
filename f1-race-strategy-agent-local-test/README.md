# F1 Race Strategy Engineer Agent — CSE 598 Capstone Baseline

An **agentic** baseline: Claude acts as a Formula 1 race strategist. Given a driver
and their starting tyre, the agent reasons over **real 2021 Spanish GP telemetry**
to recommend a pit/tyre strategy. It runs a genuine *plan → call tools → observe →
refine* loop — it inspects tyre degradation, proposes candidate one- and two-stop
strategies, simulates each with a deterministic race-time model, compares them, and
submits a final recommendation.

The recommendation is **objectively checkable**: it is compared against the
optimum of the same deterministic simulator (see `data/ground_truth.json`).

## What the baseline does (step by step)

1. **Model** — Anthropic `claude-opus-5` (configurable) via the official `anthropic`
   SDK, driven by a **hand-written tool-use loop** in [`src/agent.py`](src/agent.py)
   (the loop is explicit on purpose, so the agentic cycle is visible).
2. **Tools** the agent can call ([`src/tools.py`](src/tools.py)):
   - `get_race_info` — race length, compounds, pit-loss, min stint length.
   - `get_driver_stints` — a driver's *actual* stints + starting compound (from data).
   - `estimate_tyre_degradation` — fits a linear pace-vs-tyre-age model per compound
     from that race's green-flag laps.
   - `simulate_strategy` — deterministic estimate of total race time for any proposed
     stint plan (validates: laps sum to race length, ≥2 compounds, min stint length).
3. **Data** — real timing data from [FastF1](https://docs.fastf1.dev), pre-downloaded
   into `cache/` and read in **offline mode** → no network needed for race data.

## Setup

```bash
git clone https://github.com/DoomDust7/f1-race-strategy-agent.git
cd f1-race-strategy-agent
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### API key (required)

The agent calls the Anthropic API, so you need an API key with credits.

```bash
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...
# optional: ANTHROPIC_MODEL=claude-sonnet-5   (cheaper/faster than the default claude-opus-5)
```

### No API key handy?

You can still confirm the environment, the offline data, and the deterministic
simulator (no key, no network, no cost):

```bash
python -c "from src import tools; print(tools.optimal_strategy('SOFT'))"
# -> {'num_stops': 1, 'first_pit_lap': 10, ..., 'total_time_s': 5542.1}
```

A full captured agent run (real `python tests/test_case.py` output ending in
`RESULT: PASS`) is saved in [`docs/sample_run.txt`](docs/sample_run.txt), so the
baseline's behavior is visible even without running the model.

## Run

**The concrete test case** (this is the graded example — runs the agent and checks
its answer against the deterministic optimum):

```bash
python tests/test_case.py
```

**Ask the agent your own question:**

```bash
python -m src.run "Hamilton starts on SOFT. One-stop or two-stop, and when should he first pit?"
```

### Input / output

- **Input:** a natural-language strategy question (the test uses a fixed one about
  Hamilton in the 2021 Spanish GP). Race data comes from `cache/`.
- **Output:** the agent's tool-call trajectory printed to stdout, then a final
  recommendation (number of stops, first pit lap, full stint plan, rationale).
  The test additionally prints `PASS`/`FAIL`.

## Expected result

The deterministic optimum (in `data/ground_truth.json`) is ≈ **5542s**. The test
**passes when the agent's recommended strategy is valid and its simulated race time
is within 1% of that optimum** — i.e. the agent used the tools to reach a
near-optimal plan. (The simple model now ranks a one-stop fastest; see
`docs/sample_run.txt` for an earlier captured run against the previous optimum.)

## Reproducibility notes

- **No race-data network dependency:** `cache/` is committed and read with
  `fastf1.Cache.offline_mode(True)`. The only external call is to the Anthropic API.
- **Regenerate the cache** (needs network, optional):
  `python scripts/build_cache.py`.
- **Deterministic ground truth:** `data/ground_truth.json` is produced by
  `tools.optimal_strategy()` (a brute-force over the simulator) and is re-derivable.
- Runtime: a few minutes (one cached `session.load` + the agent's ~5–8 tool calls).

## Known limitations (baseline)

- The strategy simulator is a **simplified oracle**: a linear tyre model, a constant
  pit-loss, no safety cars, traffic, track evolution, fuel effect, or rival response.
  Because stint time is additive it ignores fuel burn, so it slightly over-values the
  softer compound; the test therefore grades on being *within 1% of the optimum*
  rather than on one exact plan.
- Single race, single driver. See the proposal's "Limitations and Next Steps".

## Model backends (Anthropic or local Hugging Face)

The agent loop is provider-neutral. `src/agent.py` owns the F1 plan -> tool -> observe
loop, while `src/llm/` contains interchangeable model adapters. The simulator,
race data, tool definitions, grading, and test case are shared across backends.

### Local Hugging Face on a Mac (no vLLM)

Install the normal project dependencies plus the local-model dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-hf.txt
cp .env.example .env
```

Then set:

```bash
LLM_BACKEND=huggingface
HF_MODEL=Qwen/Qwen3-4B-Instruct-2507
HF_DEVICE=auto
```

`HF_DEVICE=auto` prefers Apple Silicon MPS when available, then CUDA, then CPU.
The model is loaded directly by Transformers in the same Python process; there is
no vLLM server or other inference service. The first run downloads model weights
from Hugging Face, so it requires network access and enough free disk/RAM.

For a smaller Mac, start with a small instruct/tool-capable model. You can change
`HF_MODEL` without touching the F1 agent code. If a model is gated/private, set
`HF_TOKEN`; public models do not require it.

Run the same evaluation entry point for either backend:

```bash
python tests/test_case.py
```

### Anthropic

To switch back to Claude:

```bash
LLM_BACKEND=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-opus-5
```

### Why local tool calls use JSON

Anthropic exposes native tool-call objects, while open-weight models differ in
how they represent tools. The Hugging Face adapter therefore asks the model for a
small provider-neutral JSON envelope (`tool_calls`) and converts it into the same
internal `ToolCall` objects used by the agent. This keeps model-provider formatting
out of the F1 simulator and agent logic.
