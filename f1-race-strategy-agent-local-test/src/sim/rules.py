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
