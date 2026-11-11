# Broadcast / game-engine layer

This module visualizes the existing strategy-agent experiment; it does **not** replace `src/sim` as the research simulator.

## Run the real configured agent

```bash
python scripts/run_game.py --agent --render --record output/spain_agent.mp4
```

## No API/model credentials: deterministic smoke test

```bash
python scripts/run_game.py --demo-agent --render --record output/spain_demo.mp4
```

Press `q` or Esc to stop the interactive window. Add `--headless` to render an MP4 without opening a window.

The default scenario is `data/snapshots/2023_spanish_HAM_lap30.json`. The circuit loader first tries the committed FastF1 cache and derives the circuit centerline from X/Y telemetry. If unavailable, it uses the bundled offline fallback geometry. Pit lane is a separate path with team-specific pit boxes; a PIT action drives the focus car through PIT_ENTRY -> PIT_LANE -> PIT_BOX -> PIT_EXIT -> RACING.

`RaceGameEngine` is intentionally a visual clock. Strategy estimates/outcome evaluation remain owned by the existing `src/sim` and agent tools so visualization changes cannot silently alter experimental results.
