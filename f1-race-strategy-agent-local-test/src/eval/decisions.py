"""The decision unit: an immediate action plus a continuation.

The agent submits one decision at a snapshot (src/agent.py submit_recommendation):

    {"decision": "PIT", "compound": "MEDIUM", "confidence": 0.7, ...}
    {"decision": "STAY_OUT", "reassess_after_laps": 3, "confidence": 0.6, ...}

The eval turns it into a plan for the remaining laps and the world scores that plan
(docs/eval.md "Decisions"):

    PIT c                          -> {"action": "pit", "compound": c}, pit at the end of this lap
    STAY_OUT, reassess_after n     -> {"action": "stay", "laps": n}
    continuation                   =  optimizer_bridge.optimize_batch(advanced, belief_sim, stops left)
                                      (a STAY_OUT n under SC/VSC: the best full plan from the
                                      snapshot with min_first_laps n + 1, docs/eval.md "Decisions")

`max_stops` counts future stops from the decision snapshot, the same space the reference
optimizer searches: after a PIT the continuation may add max_stops - 1 (at least 0), after a
stay max_stops. Every plan from a decision therefore lies inside the reference's space.

Simulator.expand_action joins the action and the continuation, which is planned from the
snapshot after the action on the belief simulator's fit at the decision snapshot (no lap
after the decision enters it). A stay of n at least the laps left runs the current set to
the flag, with no continuation. The continuation draws from seed +
sim.agent_tools.BELIEF_SEED_OFFSET, like every belief-side simulation, so the belief never
plans on the samples the world scores the plan on.

A decision may also name the first stop it keeps (agent protocol v2, optional, kept_plan):

    PIT c, first_stop_lap = lap + 1           -> the best full plan from the snapshot that pits now onto c
    STAY_OUT n, first_stop_lap L, compound c  -> the best full plan whose first stop is at the end of L onto c

searched as belief_optimum searches, so naming belief_optimum's own first stop scores its plan; without
first_stop_lap a decision is expanded as above, unchanged.

For the Brier score, p_pit is the stated probability that pitting now is right: the
confidence for a PIT, 1 - confidence for a STAY_OUT.
"""
from __future__ import annotations

from ..baselines.domain import NoFeasiblePlan
from ..contracts import Plan, Snapshot
from ..sim import agent_tools
from ..sim.optimizer_bridge import as_continuation, optimize_batch

DECISIONS = ("PIT", "STAY_OUT")


class InvalidDecision(ValueError):
    """The submission cannot be turned into a legal plan."""


def kind(decision: dict) -> str:
    """PIT or STAY_OUT. Raises InvalidDecision otherwise."""
    value = str((decision or {}).get("decision", "")).strip().upper()
    if value not in DECISIONS:
        raise InvalidDecision(f"decision must be PIT or STAY_OUT, got {value!r}")
    return value


def to_action(decision: dict) -> dict:
    """The Simulator.expand_action action for a submission."""
    if kind(decision) == "PIT":
        compound = decision.get("compound")
        if not compound:
            raise InvalidDecision("PIT needs a compound")
        return {"action": "pit", "compound": str(compound)}
    laps = decision.get("reassess_after_laps")
    if isinstance(laps, bool) or not isinstance(laps, int) or laps < 1:
        raise InvalidDecision(f"STAY_OUT needs reassess_after_laps of at least 1, got {laps!r}")
    return {"action": "stay", "laps": laps}


def p_pit(decision: dict) -> float | None:
    """confidence for PIT, 1 - confidence for STAY_OUT; None when either is unusable."""
    try:
        decided = kind(decision)
        confidence = float(decision.get("confidence"))
    except (InvalidDecision, TypeError, ValueError):
        return None
    if not 0.0 <= confidence <= 1.0:
        return None
    return confidence if decided == "PIT" else 1.0 - confidence


def to_plan(snapshot: Snapshot, decision: dict, belief_sim, max_stops: int, *, seed: int = 0,
            n_samples: int | None = None) -> Plan:
    """The decision's plan for the remaining laps: its action expanded by belief_sim with an
    optimize_batch continuation (seed + BELIEF_SEED_OFFSET and n_samples go to it). `max_stops` limits the future
    stops of the whole plan, as for the reference: the continuation gets max_stops - 1
    (at least 0) after a PIT and max_stops after a STAY_OUT. Raises
    InvalidDecision when the submission is malformed, the action breaks a rule, or no legal
    continuation exists."""
    action = to_action(decision)
    if action["action"] == "stay" and action["laps"] >= snapshot.remaining_laps:
        return Plan.of([[snapshot.compound, snapshot.remaining_laps]])
    if decision.get("plan") is not None:
        return named_plan(snapshot, decision, action, belief_sim, max_stops)
    if decision.get("first_stop_lap") is not None:
        return kept_plan(snapshot, decision, action, belief_sim, max_stops, seed=seed, n_samples=n_samples)
    # the fit at the decision snapshot plans the continuation too, as Simulator.compare does:
    # refitting at the advanced snapshot would let a to_snapshot belief see the laps stayed out
    fit = belief_sim.fitted_for(snapshot)
    if action["action"] == "pit" and action["compound"].upper() not in fit.modelled:
        raise InvalidDecision(f"PIT onto {action['compound']!r}, which the belief does not model "
                              f"(modelled: {sorted(fit.modelled)})")
    # max_stops counts future stops from the decision snapshot, as the reference does: a PIT
    # spends one, so its continuation may add max_stops - 1; a stay spends none
    left = max(0, max_stops - 1) if action["action"] == "pit" else max_stops
    # belief-side draws: seed + the offset, never the world's samples (agent_tools)
    continuation = as_continuation(fit, seed=seed + agent_tools.BELIEF_SEED_OFFSET, n_samples=n_samples,
                                   max_stops=left)
    try:
        plan, errors = fit.expand_action(snapshot, action, continuation)
    except NoFeasiblePlan as exc:
        raise InvalidDecision(f"no legal continuation after {action}: {exc}") from exc
    if errors:
        raise InvalidDecision("; ".join(errors))
    return plan


def named_plan(snapshot: Snapshot, decision: dict, action: dict, belief_sim, max_stops: int) -> Plan:
    """The plan of a decision that names its whole plan (agent protocol v2free's optional `plan`,
    [[compound, laps], ...] for the remaining laps, the first stint continuing the current set): that
    plan exactly, its later stints drawn from the inventory as any baseline plan's are
    (optimizer_bridge.to_contract). It must agree with the decision: a PIT stops at the end of the next
    lap onto `compound`; a STAY_OUT of k laps makes no stop before the end of lap laps_completed + k + 1;
    with first_stop_lap or compound given, they must be the plan's. At most max_stops stops, every
    compound modelled; the rules are the simulator's (the submission check validates it). Raises
    InvalidDecision otherwise."""
    from ..baselines import domain
    from ..sim.optimizer_bridge import to_contract
    raw = decision.get("plan")
    try:
        pairs = [(str(c).upper(), int(n)) for c, n in raw]
    except (TypeError, ValueError) as exc:
        raise InvalidDecision(f"plan must be [[compound, laps], ...], got {raw!r}") from exc
    if not pairs or any(n < 1 for _, n in pairs):
        raise InvalidDecision("plan needs at least one stint and every stint at least one lap")
    if pairs[0][0] != snapshot.compound:
        raise InvalidDecision(f"the plan's first stint continues the current set ({snapshot.compound}), got {pairs[0][0]}")
    if sum(n for _, n in pairs) != snapshot.remaining_laps:
        raise InvalidDecision(f"the plan's stints sum to {sum(n for _, n in pairs)} laps; "
                              f"{snapshot.remaining_laps} remain")
    if len(pairs) - 1 > max_stops:
        raise InvalidDecision(f"the plan makes {len(pairs) - 1} stops; at most {max_stops}")
    modelled = belief_sim.fitted_for(snapshot).modelled
    if (bad := [c for c, _ in pairs[1:] if c not in modelled]):
        raise InvalidDecision(f"the plan fits {bad[0]}, which the belief does not model (modelled: {sorted(modelled)})")
    done, first = snapshot.laps_completed, pairs[0][1]
    nxt = pairs[1][0] if len(pairs) > 1 else None
    if action["action"] == "pit" and (first != 1 or len(pairs) < 2):
        raise InvalidDecision("PIT stops at the end of the next lap: the plan's first stint must be 1 lap")
    if action["action"] == "stay" and len(pairs) > 1 and first < action["laps"] + 1:
        raise InvalidDecision(f"STAY_OUT {action['laps']} allows no stop before the end of lap {done + action['laps'] + 1}; "
                              f"the plan stops at the end of lap {done + first}")
    if decision.get("compound") and nxt and str(decision["compound"]).upper() != nxt:
        raise InvalidDecision(f"compound {decision['compound']!r} is not the plan's first new compound {nxt}")
    lap = decision.get("first_stop_lap")
    if lap is not None and (nxt is None or lap != done + first):
        raise InvalidDecision(f"first_stop_lap {lap!r} is not the plan's first stop "
                              f"({'none' if nxt is None else done + first})")
    plan = domain.Plan(tuple(domain.Stint(c, n) for c, n in pairs))
    return to_contract(plan, snapshot, sources=belief_sim.config.inventory_sources)


def kept_plan(snapshot: Snapshot, decision: dict, action: dict, belief_sim, max_stops: int, *, seed: int = 0,
              n_samples: int | None = None) -> Plan:
    """The plan of a decision that names the first stop it keeps (agent protocol v2's optional
    first_stop_lap, with compound the compound fitted there): the belief's best full plan from the
    snapshot whose first stop is at the end of lap first_stop_lap onto compound, searched as
    belief_optimum searches (seed + BELIEF_SEED_OFFSET, n_samples, max_stops from the snapshot) with
    that first stop fixed (optimize_batch's min_first_laps = max_first_laps, next_compound). So a
    decision naming belief_optimum's own first stop is scored on belief_optimum's plan, where a bare
    STAY_OUT k is re-planned after the stay. A PIT names the end of the next lap; a STAY_OUT of k laps
    a stop no earlier than the end of lap laps_completed + k + 1. Raises InvalidDecision otherwise."""
    lap, done = decision.get("first_stop_lap"), snapshot.laps_completed
    if isinstance(lap, bool) or not isinstance(lap, int):
        raise InvalidDecision(f"first_stop_lap must be a whole lap number, got {lap!r}")
    compound = decision.get("compound")
    if not compound:
        raise InvalidDecision("first_stop_lap needs the compound fitted at that stop")
    if action["action"] == "pit" and lap != done + 1:
        raise InvalidDecision(f"PIT stops at the end of lap {done + 1}: first_stop_lap must be {done + 1} or left out")
    if action["action"] == "stay" and lap < done + action["laps"] + 1:
        raise InvalidDecision(f"STAY_OUT {action['laps']} allows no stop before the end of lap "
                              f"{done + action['laps'] + 1}: first_stop_lap {lap} needs reassess_after_laps "
                              f"of at most {max(1, lap - done - 1)}")
    fit = belief_sim.fitted_for(snapshot)
    compound = str(compound).upper()
    if compound not in fit.modelled:
        raise InvalidDecision(f"a stop onto {compound!r}, which the belief does not model (modelled: {sorted(fit.modelled)})")
    w = lap - done
    try:
        return optimize_batch(snapshot, fit, seed=seed + agent_tools.BELIEF_SEED_OFFSET, n_samples=n_samples,
                              max_stops=max_stops, min_first_laps=w, max_first_laps=w, next_compound=compound).plan
    except NoFeasiblePlan as exc:
        raise InvalidDecision(f"no legal plan stops at the end of lap {lap} onto {compound}: {exc}") from exc


def belief_optimum(snapshot: Snapshot, belief_sim, max_stops: int, *, seed: int = 0,
                   n_samples: int | None = None) -> Plan:
    """The belief-optimal full plan from the snapshot with no decision imposed (the
    belief_optimizer policy): optimize_batch on the belief's fit at the snapshot, with the same
    belief seed (seed + BELIEF_SEED_OFFSET), samples and future-stop limit as to_plan's
    continuation. Raises InvalidDecision when the belief finds no legal plan."""
    fit = belief_sim.fitted_for(snapshot)
    try:
        return optimize_batch(snapshot, fit, seed=seed + agent_tools.BELIEF_SEED_OFFSET, n_samples=n_samples,
                              max_stops=max_stops).plan
    except NoFeasiblePlan as exc:
        raise InvalidDecision(f"the belief finds no legal plan: {exc}") from exc
