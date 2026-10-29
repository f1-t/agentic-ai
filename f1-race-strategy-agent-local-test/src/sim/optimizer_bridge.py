"""Bridge between the simulator's contracts and the baseline optimizer (src/baselines).

The baseline search enumerates plans on its own small Snapshot/Plan records; this module
converts our richer contracts into those records, scores every candidate in one
`Simulator.simulate_batch` call, and converts the chosen plan back. Both packages follow
the unified rule set in docs/rules.md (R1-R6).

    from_contract(snapshot, sim)        contracts.Snapshot -> baselines Snapshot
    to_contract(plan, snapshot)         baselines Plan -> contracts.Plan, tyre sets assigned
    optimize_batch(snapshot, sim, ...)  the baseline optimizer, scored by simulate_batch
    as_continuation(sim, ...)           a continuation for Simulator.compare
    pace_for_crossover(sim, snapshot)   a pace callback for greedy_crossover
"""
from dataclasses import asdict, dataclass, field

import numpy as np

from ..baselines import domain, policies
from ..baselines.search import Evaluation, OptimizationResult, candidate_plans
from ..contracts import Plan, PlanStint, Snapshot, StintRecord
from .config import compound_class
from .rules import laps_on_set


def from_contract(snapshot: Snapshot, sim) -> domain.Snapshot:
    """The baseline search's view of a snapshot. Physics stay in the simulator (R5):
    pit_loss_s is carried only for the baseline record's legacy field."""
    used = snapshot.compounds_used
    return domain.Snapshot(
        total_laps=snapshot.race.total_laps,
        completed_laps=snapshot.laps_completed,
        current_compound=snapshot.compound,
        tyre_age=snapshot.tyre_age,
        current_stint_laps=laps_on_set(snapshot),
        used_compounds=tuple(sorted(used)),
        available_compounds=tuple(sorted(sim.fitted_for(snapshot).modelled)),
        min_stint_laps=snapshot.race.min_stint_laps,
        pit_loss_s=float(snapshot.race.pit_loss_s),
        # R3: the two-compound rule is waived once a non-slick class is in the history
        require_two_compounds=all(compound_class(sim.config, c) == "SLICK" for c in used),
    )


def _to_contract(plan: domain.Plan, snapshot: Snapshot, sources=None):
    """(contracts.Plan, assignment). `assignment` has one entry per later stint: the set
    drawn for it (R4), or None when the snapshot is unconstrained or no set of that
    compound is left (the simulator's inventory rule then rejects the plan)."""
    first, *later = plan.stints
    if first.laps == 0:
        raise ValueError("The first stint has zero laps; an immediate stop is not a plan "
                         "(docs/rules.md R1): take the snapshot one lap earlier.")
    stints = [PlanStint(compound=first.compound, laps=first.laps)]
    pool = (None if snapshot.sets_available is None else
            [t for t in snapshot.sets_available if sources is None or t.source in sources])
    assignment = []
    for i, s in enumerate(later, start=2):
        same = [t for t in pool or [] if t.compound == s.compound]
        if not same:
            stints.append(PlanStint(compound=s.compound, laps=s.laps))
            assignment.append(None)
            continue
        new = [t for t in same if t.laps_run == 0]
        chosen = new[0] if new else min(same, key=lambda t: t.laps_run)
        pool.remove(chosen)
        stints.append(PlanStint(compound=s.compound, laps=s.laps, start_age=chosen.laps_run,
                                set_id=chosen.set_id))
        assignment.append({"stint": i, "compound": s.compound, "set_id": chosen.set_id,
                           "start_age": chosen.laps_run, "substituted": not new})
    return Plan(stints=stints), assignment


def to_contract(plan: domain.Plan, snapshot: Snapshot, *, sources=None) -> Plan:
    """A baseline plan as a contracts.Plan. With an inventory each later stint draws a
    distinct set: a new one of its compound when available, else the least used one,
    whose laps_run becomes start_age (R4). `sources` limits the provenance tags that
    count, as SimConfig.inventory_sources does."""
    return _to_contract(plan, snapshot, sources)[0]


@dataclass(frozen=True, slots=True)
class BatchOptimizationResult(OptimizationResult):
    """OptimizationResult plus what the batch scorer adds: the chosen plan as a contract
    (sets assigned), the number of candidates the simulator rejected, and diagnostics
    (the best plan's set assignment, seed and n_samples)."""

    plan: Plan | None = None
    invalid: int = 0
    diagnostics: dict = field(default_factory=dict)


def _key(evaluation, snapshot):
    """The baseline optimizer's tie-break: mean, then fewer stops, earlier pit laps,
    compound order (src/baselines/search.py, optimize)."""
    plan = evaluation.plan
    return (evaluation.mean_time_s, plan.num_stops, plan.pit_laps(snapshot),
            tuple(s.compound for s in plan.stints))


def score_batch(snapshot: Snapshot, plans, sim, *, seed=0, n_samples=None, scenario=None, state=None):
    """Score baseline plans in one simulate_batch call under shared draws (R6).

    Returns (evaluations, contract_plans, assignments, batch); an entry is None where the
    plan has a zero-lap first stint or the simulator rejected it. Sample i is
    default_rng([seed, i]), so seeds=(0..n-1) of seed 0 are Quan's seed tuple."""
    sources = sim.config.inventory_sources
    state = state or from_contract(snapshot, sim)
    converted = []
    for p in plans:
        try:
            converted.append(_to_contract(p, snapshot, sources))
        except ValueError:
            converted.append(None)
    todo = [i for i, c in enumerate(converted) if c is not None]
    batch = sim.simulate_batch(snapshot, [converted[i][0] for i in todo], seed=seed,
                               n_samples=n_samples, scenario=scenario)
    n = batch.times.shape[1]
    seeds = tuple(range(n))
    evaluations = [None] * len(plans)
    contract_plans = [None] * len(plans)
    assignments = [None] * len(plans)
    for r, i in enumerate(todo):
        if batch.valid[r]:
            evaluations[i] = Evaluation(plans[i], seeds, tuple(float(t) for t in batch.times[r]))
            contract_plans[i] = batch.plans[r]
            assignments[i] = converted[i][1]
    return evaluations, contract_plans, assignments, batch


def optimize_batch(snapshot: Snapshot, sim, *, seed=0, n_samples=None, max_stops=2,
                   scenario=None) -> BatchOptimizationResult:
    """The baseline optimizer with every candidate scored in one simulate_batch call.

    Candidates come from baselines.candidate_plans; plans with a zero-lap first stint are
    filtered out (R1) until candidate_plans takes a first-stint minimum. Rows the
    simulator rejects are dropped and counted in `invalid`. The best plan follows the
    baseline optimizer's tie-break."""
    state = from_contract(snapshot, sim)
    candidates = list(candidate_plans(state, max_stops))
    plans = [p for p in candidates if p.stints[0].laps > 0]
    evals, contract_plans, assignments, batch = score_batch(
        snapshot, plans, sim, seed=seed, n_samples=n_samples, scenario=scenario, state=state)
    scored = [i for i, e in enumerate(evals) if e is not None]
    if not scored:
        raise domain.NoFeasiblePlan("No legal remaining plan exists within the future-stop limit.")
    best = min(scored, key=lambda i: _key(evals[i], state))
    return BatchOptimizationResult(
        evals[best], len(scored), plan=contract_plans[best], invalid=len(plans) - len(scored),
        diagnostics={"set_assignment": assignments[best], "seed": seed,
                     "n_samples": batch.times.shape[1],
                     "zero_lap_first_stints": len(candidates) - len(plans)},
    )


def as_continuation(sim, **kw):
    """A continuation for Simulator.compare: optimize_batch from the advanced snapshot.

    The first stint continues the fitted set: start_age 0 after a stay, the set's
    laps_run (the advanced tyre_age) after a pit, which is the snapshot whose current
    stint has no laps yet. `kw` goes to optimize_batch."""
    def continuation(snapshot_after: Snapshot) -> Plan:
        plan = optimize_batch(snapshot_after, sim, **kw).plan
        just_pitted = snapshot_after.laps_completed - sum(s.laps for s in snapshot_after.stints) == 0
        first = plan.stints[0].model_copy(
            update={"start_age": snapshot_after.tyre_age if just_pitted else 0})
        return plan.model_copy(update={"stints": [first, *plan.stints[1:]]})
    return continuation


def pace_for_crossover(sim, snapshot: Snapshot):
    """A (compound, tyre_age) -> lap time callback for baselines.greedy_crossover, from
    the snapshot's fit, evaluated on the next lap."""
    model = sim.fitted_for(snapshot).degradation
    lap = [snapshot.laps_completed + 1]

    def pace(compound, age):
        return float(np.asarray(model.lap_times(compound, [age], lap))[0])
    return pace


# --- `python -m src.baselines --scorer batch` -----------------------------------
def legacy_snapshot(context, race) -> Snapshot:
    """The contracts.Snapshot for a baselines.legacy.LegacyContext: the driver's actual
    stints that ended before the current one become the stint history."""
    state = context.snapshot
    before = state.completed_laps - state.current_stint_laps
    stints, lap = [], 0
    for s in context.actual_stints:
        if lap + s.laps > before:
            break
        stints.append(StintRecord(compound=s.compound, laps=s.laps, start_lap=lap + 1))
        lap += s.laps
    if lap != before:
        raise ValueError("The driver's actual stints do not end where the current stint starts.")
    return Snapshot(race=race, driver=context.driver, laps_completed=state.completed_laps,
                    compound=state.current_compound, tyre_age=state.tyre_age, stints=stints)


def batch_comparison(context, seeds, max_stops, sim=None, race=None):
    """baselines.__main__.comparison with every plan scored by simulate_batch.

    The report has the same shape. Samples are 0..n-1 of seed 0 (R6), so `seeds` must be
    0..n-1. `sim` and `race` default to the legacy simulator and race in src.legacy_tools."""
    from .. import legacy_tools as tools

    seeds = tuple(seeds)
    if seeds != tuple(range(len(seeds))) or not seeds:
        raise ValueError("The batch scorer draws samples 0..n-1 of seed 0; pass --seeds 0 1 ... n-1.")
    sim = sim or tools._get_simulator()
    snapshot = legacy_snapshot(context, race or tools._race_params())
    state = from_contract(snapshot, sim)
    n = len(seeds)
    optimum = optimize_batch(snapshot, sim, n_samples=n, max_stops=max_stops)
    reference = optimum.evaluation
    pace = pace_for_crossover(sim, snapshot)
    named = {
        "actual_strategy": lambda: policies.actual_strategy(state, context.actual_stints),
        "fixed_one_stop": lambda: policies.fixed_one_stop(state),
        "fixed_two_stop": lambda: policies.fixed_two_stop(state),
        "greedy_crossover": lambda: policies.greedy_crossover(state, pace, max_stops),
    }
    rows = {}
    for name, policy in named.items():
        try:
            plan = policy()
            if plan.num_stops > max_stops:
                raise domain.PolicyUnavailable("This policy exceeds the optimizer's future-stop limit.")
            if plan.stints[0].laps == 0:
                raise domain.PolicyUnavailable("The plan stops immediately; the batch scorer needs "
                                               "a first stint of at least one lap (docs/rules.md R1).")
            (result,), _, _, batch = score_batch(snapshot, [plan], sim, n_samples=n, state=state)
            if result is None:
                raise domain.PolicyUnavailable("; ".join(batch.errors[0]))
        except domain.PolicyUnavailable as error:
            rows[name] = {"status": "unavailable", "reason": str(error)}
            continue
        rows[name] = {
            "status": "ok",
            **result.to_dict(state),
            "regret_s": result.mean_time_s - reference.mean_time_s,
        }
    return {
        "schema_version": 1,
        "race": context.race,
        "driver": context.driver,
        "simulator": "sim_batch",
        "objective": "minimum_mean_remaining_time_over_supplied_seeds",
        "max_future_stops": max_stops,
        "snapshot": asdict(state),
        "optimizer": {
            **reference.to_dict(state),
            "candidates_evaluated": optimum.candidates_evaluated,
        },
        "baselines": rows,
    }
