"""Plan validity rules. Each rule returns a list of error strings."""
from dataclasses import dataclass

from ..contracts import Plan, Snapshot
from .registry import Registry

RULES = Registry("rule")


@dataclass
class RuleContext:
    snapshot: Snapshot
    plan: Plan
    modelled: set


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


@RULES.register("min_stint")
def min_stint(ctx):
    floor = ctx.snapshot.race.min_stint_laps
    return [f"Stint of {s.laps} laps is below the {floor}-lap minimum."
            for s in ctx.plan.stints[1:] if s.laps < floor]


@RULES.register("two_compounds")
def two_compounds(ctx):
    used = ctx.snapshot.compounds_used | {s.compound for s in ctx.plan.stints}
    if len(used) < 2:
        return ["A valid dry strategy must use at least two different compounds."]
    return []


def check(names, ctx):
    return [err for name in names for err in RULES.get(name)(ctx)]
