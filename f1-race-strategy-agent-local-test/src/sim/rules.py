"""Plan validity rules. Each rule returns a list of error strings."""
from dataclasses import dataclass

from ..contracts import Plan, Snapshot
from .config import compound_class
from .registry import Registry

RULES = Registry("rule")


@dataclass
class RuleContext:
    snapshot: Snapshot
    plan: Plan
    modelled: set
    config: object = None   # SimConfig when available
    stint_limits: dict | None = None   # R8: compound -> oldest tyre age run in this race (stint_limits)


@RULES.register("first_stint_matches")
def first_stint_matches(ctx):
    first = ctx.plan.stints[0].compound
    if first != ctx.snapshot.compound:
        return [f"First stint continues the current set ({ctx.snapshot.compound}), got {first}."]
    return []


@RULES.register("lap_sum")
def lap_sum(ctx):
    have, need = ctx.plan.total_laps, ctx.snapshot.remaining_laps
    if have != need:
        return [f"Stint laps sum to {have}, must equal remaining laps {need}."]
    return []


@RULES.register("known_compounds")
def known_compounds(ctx):
    unknown = sorted({s.compound for s in ctx.plan.stints} - ctx.modelled)
    if unknown:
        return [f"Unknown compound '{unknown[0]}'. Modelled compounds: {sorted(ctx.modelled)}"]
    return []


def laps_on_set(snapshot):
    """Laps already run on the fitted set in this race: the laps since the last recorded
    stop, capped by the set's age when the stint history does not add up."""
    return min(snapshot.tyre_age, snapshot.laps_completed - sum(s.laps for s in snapshot.stints))


@RULES.register("min_stint")
def min_stint(ctx):
    """Enforce the minimum stint length (docs/rules.md, R2).

    The minimum is a planning constraint rather than a legality rule. It counts the laps
    already run on the fitted set in this race: a stop may be taken only once
    laps_on_set + first.laps reaches the minimum, every later stint has at least the
    minimum, and a first stint that runs to the flag has no minimum.
    """
    snap = ctx.snapshot
    floor = snap.race.min_stint_laps
    stints = ctx.plan.stints
    run = laps_on_set(snap)

    def counted(i, s):
        return run + s.laps if i == 0 else s.laps

    def short(i, s):
        if i == 0 and len(stints) == 1:
            return False   # runs to the flag
        return counted(i, s) < floor and not (
            # class-change waiver: a stint on either side of a change of tyre class (inters
            # to slicks after three laps) is a crossover stop, not a short stint. Kept
            # separate from the clause above so the two rules compose.
            _class_change(ctx, i))

    def message(i, s):
        already = f" ({run} already run on the set)" if i == 0 and run else ""
        return f"Stint of {s.laps} laps{already} is below the {floor}-lap minimum."

    return [message(i, s) for i, s in enumerate(stints) if short(i, s)]


def _class_change(ctx, i):
    """Whether stint i has a different tyre class from the stint before or after it."""
    stints = ctx.plan.stints
    cls = compound_class(ctx.config, stints[i].compound)
    return any(compound_class(ctx.config, stints[j].compound) != cls
               for j in (i - 1, i + 1) if 0 <= j < len(stints))


@RULES.register("two_compounds")
def two_compounds(ctx):
    used = ctx.snapshot.compounds_used | {s.compound for s in ctx.plan.stints}
    if any(compound_class(ctx.config, c) != "SLICK" for c in used):
        return []   # a wet-weather plan or history: the two-compound rule is a dry rule
    if len(used) < 2:
        return ["A valid dry strategy must use at least two different compounds."]
    return []


def stint_limits(frame, rank: int = 1) -> dict[str, float]:
    """R8: compound -> the tyre age at least `rank` cars reached on it in `frame`, every lap
    counted: the rank-th largest of the per-car maximum ages (SimConfig.stint_limit_rank).
    Rank 1 is the oldest tyre any car ran. A compound fewer than `rank` cars ran takes its
    smallest per-car maximum (the age every car on it reached); rank above 1 needs a driver
    column. Simulator computes it once from the full race lap frame, so the belief and the
    world (both built on that frame) share it whatever their fit window: a hindsight race
    constant, like the observed pit loss."""
    if frame is None or "compound" not in frame or "tyre_age" not in frame:
        return {}
    f = frame[frame["compound"].notna() & frame["tyre_age"].notna()]
    if rank == 1:
        return {str(c): float(a) for c, a in f.groupby("compound")["tyre_age"].max().items()}
    if "driver" not in f:
        raise ValueError("stint_limit_rank above 1 needs a driver column in the race frame")
    per_car = f.groupby(["compound", "driver"])["tyre_age"].max()
    return {str(c): float(ages.sort_values(ascending=False).iloc[min(rank, len(ages)) - 1])
            for c, ages in per_car.groupby(level="compound")}


def _soonest_stop(snapshot) -> int:
    """Laps the current set must still run before a stop is legal (R1, R2)."""
    return max(1, snapshot.race.min_stint_laps - laps_on_set(snapshot))


def current_set_past_limit(snapshot, limits, margin) -> bool:
    """R8 fallback: the current set cannot reach its soonest legal stop inside its limit."""
    seen = (limits or {}).get(snapshot.compound)
    return margin is not None and seen is not None and snapshot.tyre_age + _soonest_stop(snapshot) > seen + margin


def stint_limit_violations(snapshot, plan, limits, margin) -> list[str]:
    """R8 stint limit (docs/rules.md): every stint ends at most at the oldest tyre age run on
    its compound in this race (`limits`, stint_limits) plus `margin` laps. A compound no car
    ran in the race has no limit. Fallback: when the current set is already past its limit at
    the snapshot (current_set_past_limit), the first stint may run to the soonest legal stop,
    or to the flag when no legal stop fits in the laps left. None `margin` turns R8 off.
    `plan` is a contracts.Plan or anything whose stints have compound and laps (start_age 0
    when absent, a new set)."""
    if margin is None or not limits:
        return []
    out = []
    past = current_set_past_limit(snapshot, limits, margin)
    soonest = _soonest_stop(snapshot)
    for j, s in enumerate(plan.stints):
        seen = limits.get(s.compound)
        if seen is None:
            continue
        end = (snapshot.tyre_age if j == 0 else getattr(s, "start_age", 0)) + s.laps
        if end <= seen + margin:
            continue
        if j == 0 and past and (s.laps <= soonest or (len(plan.stints) == 1 and
                                                      snapshot.remaining_laps < soonest + snapshot.race.min_stint_laps)):
            continue
        hint = f"; this set is already past it, so pit after {soonest} lap(s)" if j == 0 and past else ""
        out.append(f"Stint {j + 1} ({s.compound}) ends at tyre age {end:g}, past the stint limit of "
                   f"{seen + margin:g} (the oldest {s.compound} tyre run in this race, {seen:g}, + {margin}){hint}.")
    return out


@RULES.register("stint_limit")
def stint_limit(ctx):
    return stint_limit_violations(ctx.snapshot, ctx.plan, ctx.stint_limits,
                                  getattr(ctx.config, "stint_limit_margin_laps", None))


def assign_sets(snapshot, plan, sources=None):
    """Match each stint after the first to a distinct set in the inventory.

    Returns (sets, errors). `sets[i]` is the TyreSet for stint i, None for the
    fitted first stint or when the snapshot is unconstrained. `sources` limits
    which provenance tags count as available.
    """
    if snapshot.sets_available is None:
        return [None] * len(plan.stints), []
    pool = [t for t in snapshot.sets_available if sources is None or t.source in sources]
    out, errors = [None], []
    for i, st in enumerate(plan.stints[1:], start=2):
        if st.set_id is not None:
            found = [t for t in pool if t.set_id == st.set_id]
            if not found:
                errors.append(f"Stint {i}: set '{st.set_id}' is not available.")
                out.append(None)
                continue
            t = found[0]
            if t.compound != st.compound:
                errors.append(f"Stint {i}: set '{st.set_id}' is {t.compound}, plan says {st.compound}.")
            if st.start_age not in (0, t.laps_run):
                errors.append(f"Stint {i}: start_age {st.start_age} conflicts with set '{st.set_id}' ({t.laps_run} laps run).")
        else:
            found = [t for t in pool if t.compound == st.compound and t.laps_run == st.start_age]
            if not found:
                ages = sorted(t.laps_run for t in pool if t.compound == st.compound)
                detail = f"available {st.compound} sets have {ages} laps run" if ages else f"no {st.compound} set left"
                errors.append(f"Stint {i}: no {st.compound} set with {st.start_age} laps run; {detail}.")
                out.append(None)
                continue
            t = found[0]
        pool.remove(t)
        out.append(t)
    return out, errors


def _sources(config):
    return getattr(config, "inventory_sources", None)


@RULES.register("inventory")
def inventory(ctx):
    return assign_sets(ctx.snapshot, ctx.plan, _sources(ctx.config))[1]


def apply_inventory(snapshot, plan, config=None):
    """Return the plan with start_age taken from each named set."""
    sets, errors = assign_sets(snapshot, plan, _sources(config))
    if errors or all(t is None for t in sets):
        return plan
    stints = [s if t is None else s.model_copy(update={"start_age": t.laps_run})
              for s, t in zip(plan.stints, sets)]
    return plan.model_copy(update={"stints": stints})


def check(names, ctx):
    return [err for name in names for err in RULES.get(name)(ctx)]
