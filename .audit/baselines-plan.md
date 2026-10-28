# Baselines and optimizer implementation plan

## Definition of done

The optimizer returns the lowest mean remaining simulated time over the supplied seeds among every legal plan with zero, one, or two future stops.
It accepts a mid-race snapshot, preserves current tyre age and completed race history, and permits a stop at the current boundary when legal.
The actual-strategy replay and three heuristic policies produce legal remaining plans or an explicit unavailable result.
An offline command evaluates all four policies and the optimizer against the committed Spanish Grand Prix cache without calling a language model.
Independent small-race enumeration, stochastic counterexamples, boundary cases, and cached-data integration tests verify these claims.

## Scope and rigor

Implement one isolated Python package under the existing project, an offline command, tests, and an integration guide.
The estimated scope is six to eight production modules and three test modules, completed in one working session with checks after each unit.
Use high algorithmic rigor because regret scores rely on the optimum, with modest integration scope because snapshot and stochastic simulator contracts are still owned by teammates.
Do not change the agent, simulator physics, frozen ground truth, data cache, or evaluation ownership.
The rival strategist remains a documented sprint-three stretch goal.

## Grounding

The sprint planning PDF assigns Quan the optimizer, actual-strategy replay, fixed one-stop and two-stop heuristics, and degradation-crossover greedy policy.
The current repository exposes a deterministic full-race simulator in the tool module and has no snapshot contract or stochastic simulator package.
The current Python environment lacks FastF1, pytest, and Ruff.
Use a temporary environment to install verification dependencies without changing the project environment.

## Execution phases

1. Capture the current simulator result and actual driver strategy, and write independent behavioral checks before implementing algorithms.
2. Define immutable snapshot and remaining-plan records with explicit lap boundaries, tyre age, completed compound history, and legality rules.
3. Add exhaustive remaining-plan search and common-seed scoring with deterministic tie breaking and explicit no-plan errors.
4. Add actual-strategy replay, fixed heuristic one-stop and two-stop policies, and a greedy degradation-crossover policy.
5. Add an adapter for the existing cached deterministic simulator and a command that prints a machine-readable comparison.
6. Run the whole offline workflow, inspect results against the current ground truth, document the proposed teammate contracts, and obtain an independent read-only review.

## Design choices

Use a simulator callable that consumes a snapshot, remaining plan, and seed and returns remaining time in seconds.
Optimize the arithmetic mean on a fixed, explicit seed tuple shared by every candidate and baseline.
This is an empirical expected-time optimum for those seeds, not a claim about all stochastic futures.
Use an immutable remaining-plan structure so each simulation receives the same snapshot and plan.
Keep the legacy adapter separate from the core algorithms so Jason can replace it without changing search or policies.
Do not invent Jason's stochastic physics or move Sarah's snapshot builder into Quan's package.
Use one writer for implementation and one independent read-only reviewer at the end.
Skip architectural candidate fan-out because these contracts are reversible and the exhaustive-search structure is concrete.

## Checkpoint

After the domain and search checks pass, run the cached-data path before adding documentation or broadening scope.
If the teammate interfaces are missing, finish the proposed contracts and adapter and report that live integration remains unverified.
Record every accepted unit and any failed experiment in the adjacent decision trail.
