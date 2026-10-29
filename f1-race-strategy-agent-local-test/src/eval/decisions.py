"""The decision unit: an immediate action plus a continuation.

The agent submits one decision at a snapshot (src/agent.py submit_recommendation):

    {"decision": "PIT", "compound": "MEDIUM", "confidence": 0.7, ...}
    {"decision": "STAY_OUT", "reassess_after_laps": 3, "confidence": 0.6, ...}

The eval turns it into a plan for the remaining laps and the world scores that plan
(docs/eval.md "Decisions"):

    PIT c                          -> {"action": "pit", "compound": c}, pit at the end of this lap
    STAY_OUT, reassess_after n     -> {"action": "stay", "laps": n}
    continuation                   =  optimizer_bridge.optimize_batch(advanced, belief_sim, max_stops)

Simulator.expand_action joins the action and the continuation, which is planned from the
snapshot after the action on the belief simulator's fit at the decision snapshot (no lap
after the decision enters it). A stay of n at least the laps left runs the current set to
the flag, with no continuation.

For the Brier score, p_pit is the stated probability that pitting now is right: the
confidence for a PIT, 1 - confidence for a STAY_OUT.
"""
from __future__ import annotations

from ..baselines.domain import NoFeasiblePlan
from ..contracts import Plan, Snapshot
from ..sim.optimizer_bridge import as_continuation

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
    optimize_batch continuation (seed, n_samples and max_stops go to it). Raises
    InvalidDecision when the submission is malformed, the action breaks a rule, or no legal
    continuation exists."""
    action = to_action(decision)
    if action["action"] == "stay" and action["laps"] >= snapshot.remaining_laps:
        return Plan.of([[snapshot.compound, snapshot.remaining_laps]])
    # the fit at the decision snapshot plans the continuation too, as Simulator.compare does:
    # refitting at the advanced snapshot would let a to_snapshot belief see the laps stayed out
    fit = belief_sim.fitted_for(snapshot)
    if action["action"] == "pit" and action["compound"].upper() not in fit.modelled:
        raise InvalidDecision(f"PIT onto {action['compound']!r}, which the belief does not model "
                              f"(modelled: {sorted(fit.modelled)})")
    continuation = as_continuation(fit, seed=seed, n_samples=n_samples, max_stops=max_stops)
    try:
        plan, errors = fit.expand_action(snapshot, action, continuation)
    except NoFeasiblePlan as exc:
        raise InvalidDecision(f"no legal continuation after {action}: {exc}") from exc
    if errors:
        raise InvalidDecision("; ".join(errors))
    return plan
