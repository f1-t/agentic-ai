# Baseline usage-flow prototype

Open `index.html` directly in a browser.
It is a self-contained, throwaway presentation prototype with no server, external assets, or saved browser state.

## Question and observed result

The question is whether a technical viewer can trace one race snapshot through reference search, policy plans, simulation, and regret.
The prototype carries that same state across five connected views and exposes the exact snapshot and selected result after every action.
The demonstrated lap-30 case compares all four policies against the optimizer using the current integrated evaluation path.
The final-lap case retains unavailable policies and their reasons instead of inventing scores.
These observations validate the demonstration flow, without establishing the accuracy of the simulator's physics or the performance of a language-model agent.

## Present the demo

1. Select the usage-flow walkthrough and load Hamilton after lap 30.
2. Follow the search, remaining-stint timeline, and regret comparison.
3. Select stochastic scoring to inspect paired outcomes on shared draws.
4. Select the late-race walkthrough to show when planned stops cease to be feasible.

The default world matches the current integrated runner.
The seeded world is an explicitly separate demonstration configuration with rate events, posterior parameter draws, and residual lap noise.
Its full-race fit scores the plans, while the baseline policies use the belief fit through the decision lap.
Whiskers measure paired Monte Carlo precision, not physical model accuracy.
No live language-model agent is called.

The controls replay captured Python outputs for the five available snapshots, two world configurations, and two reference stop budgets.
They do not rerun the Python optimizer inside the browser.
Fixed policies can retain more stops than the reference budget, matching the current evaluation behavior.
The chart marks those rows explicitly.

## Rebuild the capture

Run from the project directory with its Python environment active.

```bash
python -m src.baselines.prototype_demo.build_demo
```

The generator writes the evidence and the standalone page from the template.
The evidence includes the source revision, complete snapshots, model configurations, candidate counts, per-draw outcomes, and the integrated runner's lap-30 result.
The captured page remains usable without Python.

This prototype belongs on its throwaway branch, outside main.
The presentation video and decoded-frame verification are separate delivery artifacts rather than production project files.
