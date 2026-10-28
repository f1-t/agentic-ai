"""Tyre-set inventory: the physical sets a driver has in the garage during the race.

Rebuilt from the FastF1 Laps tables of the weekend's sessions. TyreLife is the
cumulative lap count of one physical set and keeps counting across garage visits
and across sessions, so a re-fitted set is recognised by its TyreLife carrying on
from where it stopped. A FastF1 Stint is one fitting, not one set.

Each set is a dict {compound, laps_run, source, set_id}, where source is
  observed  seen in the data, and its fate was not in doubt
  inferred  seen, but a guess was needed (a reappearance matched only by the
            off-by-one fallback or not at all, or a return tie that could have
            removed it instead of another set)
  assumed   never seen in any session; filled in from the event allocation
set_id is f"{compound}-{n}" with n running per compound ("SOFT-1", "HARD-2").

Snapshot.sets_available is the garage list: the sets held, without the one fitted
to the car. weekend_sets(race_laps=...) already leaves the fitted set out; for a
result built without race_laps, split_fitted() takes it out.
"""
from collections import Counter

import pandas as pd

SESSION_ORDER = ["FP1", "FP2", "FP3", "SQ", "S", "Q"]

_MISSING_COMPOUNDS = {"", "NAN", "NONE"}
_PUBLIC_KEYS = ("compound", "laps_run", "source", "set_id")


def _compound(value):
    """Normalised compound name, or None when missing (NaN, None, the string "nan")."""
    if not isinstance(value, str):
        if pd.isna(value):
            return None
        value = str(value)
    value = value.strip().upper()
    return None if value in _MISSING_COMPOUNDS else value


def _stints(laps, driver, number=None):
    """One dict per FastF1 Stint (one fitting) of `driver`, in running order.

    With `number`, the laps run under that DriverNumber are taken instead.
    """
    if laps is None or len(laps) == 0:
        return []
    who = laps["Driver"] == driver if number is None else laps["DriverNumber"].astype(str) == number
    d = laps[who & laps["Stint"].notna()]
    out = []
    for _, g in d.groupby("Stint", sort=True):
        g = g.sort_values("LapNumber")
        comps = [c for c in map(_compound, g["Compound"]) if c]
        if not comps:
            continue  # compound unknown: cannot be tied to a set
        life = g["TyreLife"].dropna()  # NaN TyreLife: skip the lap, keep the stint
        fresh = g["FreshTyre"].dropna()
        out.append({
            "compound": comps[0],
            "start": int(life.iloc[0]) if len(life) else None,
            "max": int(life.max()) if len(life) else None,
            "fresh": bool(len(fresh) and fresh.iloc[0] == True),  # noqa: E712
            "first_lap": g["LapNumber"].min(),
            "n_laps": len(g),
        })
    return out


def _car_number(tables, driver):
    """The driver's DriverNumber, from the first table where the code has laps."""
    for laps in tables:
        if laps is None or len(laps) == 0 or "DriverNumber" not in laps:
            continue
        nums = laps.loc[laps["Driver"] == driver, "DriverNumber"].dropna()
        if len(nums):
            return str(nums.iloc[0])
    return None


def _session_stints(book, laps, driver, name, number):
    """The driver's stints in one session, or his car's when a reserve drove it.

    A reserve driver in the nominated driver's car uses that driver's sets
    (2021 Art. 24.4 d), so the car's laps under another code count as his.
    """
    stints = _stints(laps, driver)
    if stints or number is None or laps is None or len(laps) == 0 or "DriverNumber" not in laps:
        return stints
    others = laps.loc[(laps["DriverNumber"].astype(str) == number) & (laps["Driver"] != driver),
                      "Driver"].unique()
    if len(others) == 0:
        return stints  # nobody ran the car: no running, returns come from unseen sets
    book["notes"].append(f"{name}: no {driver} laps, took car {number} laps of "
                         f"{', '.join(map(str, others))}")
    return _stints(laps, driver, number)


def _new_set(book, compound, laps_run, source):
    book["ids"][compound] = book["ids"].get(compound, 0) + 1
    ts = {"compound": compound, "laps_run": laps_run, "source": source,
          "set_id": f"{compound}-{book['ids'][compound]}", "_last": -1, "_seq": len(book["seen"])}
    book["seen"].append(ts)
    book["garage"].append(ts)
    return ts


def _match(garage, compound, laps_run):
    """The most recently fitted held set of `compound` with exactly `laps_run` laps."""
    cands = [s for s in garage if s["compound"] == compound and s["laps_run"] == laps_run]
    return max(cands, key=lambda s: (s["_last"], s["_seq"])) if cands else None


def _refit(book, compound, start, where):
    """Held set for a re-fitting whose TyreLife starts at `start` (k), or None.

    Tried in order: laps_run == k - 1 (exact), == k (refit recorded without the
    increment), == k - 2 (one extra lap counted). A match on either fallback tags
    the set "inferred" and is counted in book["fallbacks"].
    """
    for laps_run in (start - 1, start, start - 2):
        ts = _match(book["garage"], compound, laps_run) if laps_run >= 0 else None
        if ts is None:
            continue
        if laps_run != start - 1:
            ts["source"] = "inferred"
            book["fallbacks"] += 1
            book["notes"].append(f"{where}: {compound} at TyreLife {start} matched "
                                 f"{ts['set_id']} ({laps_run} laps) by fallback")
        return ts
    return None


def _drop(garage, ts):
    garage[:] = [s for s in garage if s is not ts]


def _run_session(book, laps, driver, name, number):
    """Fit every stint of one session onto a set. Returns the sets used, in first-use order."""
    used = []
    for st in _session_stints(book, laps, driver, name, number):
        c = st["compound"]
        if st["start"] is None:
            if not st["fresh"]:
                book["notes"].append(f"{name}: {c} stint with no TyreLife and not fresh, skipped")
                continue
            ts = _new_set(book, c, st["n_laps"], "inferred")
            book["notes"].append(f"{name}: fresh {c} with no TyreLife, laps_run from lap count")
        elif st["fresh"] and st["start"] == 1:
            ts = _new_set(book, c, 0, "observed")
        else:
            ts = _refit(book, c, st["start"], name)
            if ts is None:
                ts = _new_set(book, c, st["start"] - 1, "inferred")
                book["notes"].append(f"{name}: {c} at TyreLife {st['start']} matches no held set, "
                                     f"created {ts['set_id']}")
        if st["max"] is not None:
            ts["laps_run"] = max(ts["laps_run"], st["max"])
        book["order"] += 1
        ts["_last"] = book["order"]
        if all(u is not ts for u in used):
            used.append(ts)
    return used


def _hand_back(book, used, r, name):
    """Hand back after a session. r is an int (any compound) or {compound: count}."""
    if isinstance(r, dict):
        for c, n in r.items():
            c = _compound(c)
            _hand_back_some(book, [s for s in used if s["compound"] == c], n, name, c)
    else:
        _hand_back_some(book, used, r, name, None)


def _hand_back_some(book, cands, r, name, compound):
    """Return the r most-used of `cands` (ties: the one used earliest).

    In hindsight mode, sets the race later runs are kept back first: race demand
    not already covered by held sets outside `cands` protects candidates of the
    same (compound, laps_run).
    """
    if r <= 0:
        return
    if book["demand"] is not None:
        held = Counter((s["compound"], s["laps_run"]) for s in book["garage"]
                       if all(s is not u for u in cands))
        left = book["demand"] - held
        kept = []
        for ts in sorted(cands, key=lambda s: -s["laps_run"]):
            if left[(ts["compound"], ts["laps_run"])] > 0:
                left[(ts["compound"], ts["laps_run"])] -= 1
                kept.append(ts)
        cands = [s for s in cands if all(s is not k for k in kept)]
    cands = sorted(cands, key=lambda s: -s["laps_run"])  # stable, so ties keep first-use order
    for ts in cands[:r]:
        _drop(book["garage"], ts)
        book["returned"].append((name, ts["compound"], ts["laps_run"]))
    if r < len(cands) and cands[r - 1]["laps_run"] == cands[r]["laps_run"]:
        for ts in cands[r:]:
            if ts["source"] == "observed":
                ts["source"] = "inferred"
        book["notes"].append(f"{name}: return tie at {cands[r]['laps_run']} laps")
    short = r - min(r, len(cands))
    if short:
        book["shortfall"] += [compound] * short
        book["returned"] += [(name, compound, None)] * short
        book["notes"].append(f"{name}: {short} of {r} returned {compound or ''}"
                             f"{' ' if compound else ''}sets never seen, taken from unused sets")


def _build(sessions, driver, allocation, returns, race_laps=None, upto_lap=None,
           order=None, hindsight=False):
    """Work behind weekend_sets(). Returns (garage, info) with internal fields kept."""
    if upto_lap is not None and race_laps is None:
        raise ValueError("upto_lap needs race_laps")
    if hindsight and race_laps is None:
        raise ValueError("hindsight needs race_laps")
    order = list(SESSION_ORDER if order is None else order)
    unknown = (set(sessions) | set(returns)) - set(order)
    if unknown:
        raise ValueError(f"sessions {sorted(unknown)} not in the session order {order}")
    book = {"garage": [], "seen": [], "ids": {}, "order": 0, "shortfall": [], "notes": [],
            "fallbacks": 0, "returned": [], "demand": None}
    if hindsight:  # measurement only: the race's own sets, (compound, laps_run) per stint
        book["demand"] = Counter((st["compound"], 0 if st["start"] is None else st["start"] - 1)
                                 for st in _stints(race_laps, driver))
    number = _car_number([sessions.get(n) for n in order] + [race_laps], driver)

    for name in order:
        if name not in sessions and name not in returns:
            continue
        # A missing session is no running, but its returns still happen.
        used = _run_session(book, sessions.get(name), driver, name, number)
        _hand_back(book, used, returns.get(name, 0), name)

    # Sets in the allocation never seen in any session. Returns that had no seen set
    # to hand back come out of these: from the named compound, or from the compound
    # with the most left when the return named none.
    alloc = {_compound(c): n for c, n in allocation.items()}
    gap = {c: max(n - sum(s["compound"] == c for s in book["seen"]), 0) for c, n in alloc.items()}
    for c in book["shortfall"]:
        if c is None and gap:
            c = max(gap, key=gap.get)
        if gap.get(c, 0) == 0:
            book["notes"].append("more sets returned than the allocation allows")
            continue
        gap[c] -= 1
    for c, n in gap.items():
        for _ in range(n):
            _new_set(book, c, 0, "assumed")

    info = {"notes": book["notes"], "seen": book["seen"], "fitted": None, "consumed": [],
            "unmatched": [], "returned": book["returned"]}
    if race_laps is not None:
        for i, st in enumerate(_stints(race_laps, driver)):
            if i > 0 and (upto_lap is None or st["first_lap"] > upto_lap):
                break
            if st["start"] is None and not st["fresh"]:
                info["unmatched"].append(st)
                continue
            if st["start"] is None or (st["fresh"] and st["start"] == 1):
                ts = _match(book["garage"], st["compound"], 0)
            else:
                ts = _refit(book, st["compound"], st["start"], "R")
            if ts is None:
                need = 0 if st["start"] is None else st["start"] - 1
                info["unmatched"].append(st)
                book["notes"].append(f"R: {st['compound']} stint at lap {st['first_lap']} "
                                     f"needs a set with {need} laps, none held")
                continue
            _drop(book["garage"], ts)
            info["consumed"].append(ts)
            if i == 0:
                info["fitted"] = ts
    info["fallbacks"] = book["fallbacks"]
    return book["garage"], info


def _public(ts):
    return None if ts is None else {k: ts[k] for k in _PUBLIC_KEYS}


def weekend_report(sessions, driver, allocation, returns, race_laps=None, upto_lap=None,
                   order=None, hindsight=False):
    """weekend_sets() for one driver, plus how the result was reached.

    Returns {"sets": the weekend_sets() list, "fitted": the race start set or None,
    "fallbacks": count of off-by-one fallback matches, "returned": [(session,
    compound, laps_run), ...] with laps_run None for a never-seen set, "notes": [...]}.
    """
    garage, info = _build(sessions, driver, allocation, returns, race_laps, upto_lap,
                          order, hindsight)
    return {"sets": [_public(s) for s in garage], "fitted": _public(info["fitted"]),
            "fallbacks": info["fallbacks"], "returned": list(info["returned"]),
            "notes": list(info["notes"])}


def weekend_sets(sessions, driver, allocation, returns, race_laps=None, upto_lap=None,
                 order=None, hindsight=False):
    """Sets in `driver`'s garage at the start of the race (or at `upto_lap` of it).

    sessions:   FastF1 Laps tables keyed "FP1", "FP2", "FP3", "SQ", "S", "Q"; missing
                keys are allowed. If the driver has no laps in a session, laps run
                under his DriverNumber by another code (a reserve driver in his car)
                are taken as his.
    allocation: sets per compound for the event, e.g. {"HARD": 2, "MEDIUM": 3, "SOFT": 8}.
    returns:    sets handed back after each session, keyed by session name. A value is
                an int (the most-used sets, any compound) or {compound: count}, e.g.
                "Q": {"SOFT": 1} for the 2021 Q3 return of one set of the Q3 spec.
    race_laps:  the race Laps table. Its first stint picks the fitted start set, which
                is left out of the result. With upto_lap, every stint starting on or
                before that lap also consumes its set. Race laps never decide returns.
    order:      running order of the sessions; default SESSION_ORDER, filtered to the
                keys present. Pass it for a weekend that ran in another order.
    hindsight:  for measurement only, needs race_laps: returns keep back the sets the
                race later runs, to see how often the default inference disagrees.
    Without race_laps the start set is unknown, so nothing is removed: the result is
    the full race-start inventory, start set included (split_fitted takes it out).
    Returns a list of {compound, laps_run, source, set_id}.
    """
    return weekend_report(sessions, driver, allocation, returns, race_laps, upto_lap,
                          order, hindsight)["sets"]


def split_fitted(sets, compound, laps_run):
    """Split the set on the car out of a set list. Returns (fitted, garage).

    fitted is the last set in `sets` with that compound and laps_run, or None when
    there is none (garage is then all of `sets`). garage, the rest, is what
    Snapshot.sets_available takes.
    """
    compound = _compound(compound)
    for i in range(len(sets) - 1, -1, -1):
        if sets[i]["compound"] == compound and sets[i]["laps_run"] == laps_run:
            return sets[i], list(sets[:i]) + list(sets[i + 1:])
    return None, list(sets)
