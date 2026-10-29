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
to the car. weekend_sets(race_laps=...) already leaves the fitted set out, and so
does a Q3 driver's result under q2_start_rule; for any other result built without
race_laps, split_fitted() takes it out.
"""
from collections import Counter

import pandas as pd

SESSION_ORDER = ["FP1", "FP2", "FP3", "SQ", "S", "Q"]
Q_PARTS = ("Q1", "Q2", "Q3")

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


def _stints(laps, driver):
    """One dict per FastF1 Stint (one fitting) of `driver`, in running order."""
    if laps is None or len(laps) == 0:
        return []
    d = laps[(laps["Driver"] == driver) & laps["Stint"].notna()]
    out = []
    for stint, g in d.groupby("Stint", sort=True):
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
            "stint": stint,
        })
    return out


def stand_in(ran, regulars, driver):
    """The code that drove for `driver` in a session, by team, or None.

    `ran` is the codes with laps for the team in the session, `regulars` the team's race
    drivers. When `driver` is the only regular without laps and exactly one other code
    ran for the team, that code drove for him.
    """
    others = set(ran) - set(regulars)
    if set(regulars) - set(ran) == {driver} and len(others) == 1:
        return next(iter(others))
    return None


def _session_stints(book, laps, driver, name):
    """The driver's stints in one session, or the reserve's who drove for him.

    A reserve driver standing in for the nominated driver uses his sets (2021
    Art. 24.4 d). Reserves run their own car numbers (NIS #45, KUB #88 in 2021),
    so when the driver has no laps in the session the reserve is found by name in
    book["reserves"] ({session: {reserve_code: nominated_code}}), or else by team:
    with race laps, a team's regular drivers are its codes in the race, and if the
    driver is the only one of them without laps and exactly one other code ran for
    the team, that code drove for him.
    """
    if laps is None or len(laps) == 0:
        return []
    if (laps["Driver"] == driver).any():
        return _stints(laps, driver)
    codes = [r for r, d in book["reserves"].get(name, {}).items() if d == driver]
    how = "reserves"
    if not codes and book["team"] is not None and "Team" in laps:
        code = stand_in(set(laps.loc[laps["Team"] == book["team"], "Driver"].dropna()), book["regulars"], driver)
        if code is not None:
            codes, how = [code], book["team"]
    if not codes:
        return []  # nobody ran for him: no running, returns come from unseen sets
    book["notes"].append(f"{name}: no {driver} laps, took {', '.join(codes)} laps ({how})")
    return [st for code in codes for st in _stints(laps, code)]


def _new_set(book, compound, laps_run, source):
    book["ids"][compound] = book["ids"].get(compound, 0) + 1
    ts = {"compound": compound, "laps_run": laps_run, "source": source,
          "set_id": f"{compound}-{book['ids'][compound]}", "_last": -1, "_seq": len(book["seen"]),
          "_protected": False}
    book["seen"].append(ts)
    book["garage"].append(ts)
    return ts


def _match(garage, compound, laps_run):
    """The most recently fitted held set of `compound` with exactly `laps_run` laps.

    An equal unprotected set is preferred: the protected set waits for the race start.
    """
    cands = [s for s in garage if s["compound"] == compound and s["laps_run"] == laps_run]
    return max(cands, key=lambda s: (not s["_protected"], s["_last"], s["_seq"])) if cands else None


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


def _run_session(book, laps, driver, name, by_stint=None):
    """Fit every stint of one session onto a set. Returns the sets used, in first-use order.

    by_stint, if given, is filled with {Stint: set} for the session.
    """
    used = []
    for st in _session_stints(book, laps, driver, name):
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
        if by_stint is not None:
            by_stint[st["stint"]] = ts
        if all(u is not ts for u in used):
            used.append(ts)
    return used


def _q_parts(sessions):
    """[(name, laps)] for Q1, Q2 and Q3, or None when Q cannot be split.

    Plain frames under the keys "Q1", "Q2", "Q3" are taken as they are; a FastF1
    Laps under "Q" is split with its split_qualifying_sessions().
    """
    if any(k in sessions for k in Q_PARTS):
        if "Q" in sessions:
            raise ValueError("give Q or Q1/Q2/Q3, not both")
        return [(k, sessions.get(k)) for k in Q_PARTS]
    q = sessions.get("Q")
    if q is None or not hasattr(q, "split_qualifying_sessions"):
        return None
    return list(zip(Q_PARTS, q.split_qualifying_sessions()))


def _run_qualifying(book, parts, driver):
    """_run_session over Q1, Q2 and Q3 in turn. Returns the sets used, in first-use order.

    For a Q3 driver (one with laps in Q3), the set of his fastest Q2 lap is marked
    protected before Q3 runs: 2021 Art. 24.4 j makes him start the race on it, so it
    is never handed back.
    """
    q3 = parts[2][1]
    in_q3 = q3 is not None and len(q3) > 0 and bool((q3["Driver"] == driver).any())
    used = []
    for name, laps in parts:
        by_stint = {}
        for ts in _run_session(book, laps, driver, name, by_stint):
            if all(u is not ts for u in used):
                used.append(ts)
        if name == "Q2" and in_q3:
            ts = _fastest_lap_set(laps, driver, by_stint)
            if ts is None:
                book["notes"].append(f"Q2: no timed {driver} lap on a known set, start set "
                                     f"not protected")
            else:
                ts["_protected"] = True
    return used


def _fastest_lap_set(laps, driver, by_stint):
    """The set of the driver's fastest lap in `laps` (deleted laps skipped), or None."""
    if laps is None or len(laps) == 0:
        return None
    d = laps[(laps["Driver"] == driver) & laps["LapTime"].notna()]
    if "Deleted" in d:
        d = d[d["Deleted"] != True]  # noqa: E712
    if len(d) == 0:
        return None
    return by_stint.get(d.loc[d["LapTime"].idxmin(), "Stint"])


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

    A protected set (a Q3 driver's fastest-Q2 set) is never a candidate. In
    hindsight mode, sets the race later runs are kept back first: race demand not
    already covered by held sets outside `cands` protects candidates of the same
    (compound, laps_run).
    """
    cands = [s for s in cands if not s["_protected"]]
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
           order=None, hindsight=False, q2_start_rule=False, reserves=None):
    """Work behind weekend_sets(). Returns (garage, info) with internal fields kept."""
    if upto_lap is not None and race_laps is None:
        raise ValueError("upto_lap needs race_laps")
    if hindsight and race_laps is None:
        raise ValueError("hindsight needs race_laps")
    order = list(SESSION_ORDER if order is None else order)
    reserves = reserves or {}
    names = {"Q" if q2_start_rule and n in Q_PARTS else n for n in sessions}
    unknown = (names | set(returns) | set(reserves)) - set(order)
    if unknown:
        raise ValueError(f"sessions {sorted(unknown)} not in the session order {order}")
    book = {"garage": [], "seen": [], "ids": {}, "order": 0, "shortfall": [], "notes": [],
            "fallbacks": 0, "returned": [], "demand": None, "reserves": reserves,
            "team": None, "regulars": set()}
    if hindsight:  # measurement only: the race's own sets, (compound, laps_run) per stint
        book["demand"] = Counter((st["compound"], 0 if st["start"] is None else st["start"] - 1)
                                 for st in _stints(race_laps, driver))
    if race_laps is not None and "Team" in race_laps:
        team = race_laps.loc[race_laps["Driver"] == driver, "Team"].dropna()
        if len(team):
            book["team"] = team.iloc[0]
            book["regulars"] = set(race_laps.loc[race_laps["Team"] == book["team"], "Driver"])
    q_parts = _q_parts(sessions) if q2_start_rule else None
    if q2_start_rule and q_parts is None and "Q" in sessions:
        book["notes"].append("Q: not split into Q1/Q2/Q3, q2_start_rule not applied")

    for name in order:
        if name not in names and name not in returns:
            continue
        # A missing session is no running, but its returns still happen.
        if name == "Q" and q_parts is not None:
            used = _run_qualifying(book, q_parts, driver)
        else:
            used = _run_session(book, sessions.get(name), driver, name)
        _hand_back(book, used, returns.get(name, 0), name)

    # Sets in the allocation never seen in any session. Returns that had no seen set
    # to hand back come out of these: from the named compound, or from the compound
    # with the most left when the return named none. More sets seen than allocated
    # means a set was counted twice (a phantom); it is noted, not corrected.
    alloc = {_compound(c): n for c, n in allocation.items()}
    seen = Counter(s["compound"] for s in book["seen"])
    for c, n in alloc.items():
        if seen[c] > n:
            book["notes"].append(f"{c}: {seen[c]} sets seen, allocation {n}")
    gap = {c: max(n - seen[c], 0) for c, n in alloc.items()}
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
    protected = next((s for s in book["garage"] if s["_protected"]), None)
    if race_laps is None and protected is not None:
        _drop(book["garage"], protected)  # a Q3 driver's start set is known before the race
        info["fitted"] = protected
    if race_laps is not None:
        for i, st in enumerate(_stints(race_laps, driver)):
            if i > 0 and (upto_lap is None or st["first_lap"] > upto_lap):
                break
            need = 0 if st["start"] is None else st["start"] - 1
            if st["start"] is None and not st["fresh"]:
                ts = None  # no TyreLife and not fresh: cannot be tied to a set
            elif (i == 0 and protected is not None and protected["compound"] == st["compound"]
                    and protected["laps_run"] == need):
                ts = protected
            elif st["start"] is None or (st["fresh"] and st["start"] == 1):
                ts = _match(book["garage"], st["compound"], 0)
            else:
                ts = _refit(book, st["compound"], st["start"], "R")
            if i == 0 and protected is not None and ts is not protected:
                book["notes"].append(f"R: Q2 start set mismatch, {protected['set_id']} "
                                     f"({protected['laps_run']} laps) protected, race started on "
                                     f"{st['compound']} at TyreLife {st['start']}")
            if ts is None:
                info["unmatched"].append(st)
                if st["start"] is not None or st["fresh"]:
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
                   order=None, hindsight=False, q2_start_rule=False, reserves=None):
    """weekend_sets() for one driver, plus how the result was reached.

    Returns {"sets": the weekend_sets() list, "fitted": the race start set or None,
    "fallbacks": count of off-by-one fallback matches, "returned": [(session,
    compound, laps_run), ...] with laps_run None for a never-seen set, "notes": [...]}.
    """
    garage, info = _build(sessions, driver, allocation, returns, race_laps, upto_lap,
                          order, hindsight, q2_start_rule, reserves)
    return {"sets": [_public(s) for s in garage], "fitted": _public(info["fitted"]),
            "fallbacks": info["fallbacks"], "returned": list(info["returned"]),
            "notes": list(info["notes"])}


def weekend_sets(sessions, driver, allocation, returns, race_laps=None, upto_lap=None,
                 order=None, hindsight=False, q2_start_rule=False, reserves=None):
    """Sets in `driver`'s garage at the start of the race (or at `upto_lap` of it).

    sessions:   FastF1 Laps tables keyed "FP1", "FP2", "FP3", "SQ", "S", "Q"; missing
                keys are allowed. Under q2_start_rule, plain frames keyed "Q1", "Q2",
                "Q3" may stand in for "Q".
    allocation: sets per compound for the event, e.g. {"HARD": 2, "MEDIUM": 3, "SOFT": 8}.
                More sets of a compound seen than allocated is noted, not corrected.
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
    q2_start_rule: era flag, True for seasons through 2021 (Art. 24.4 j): a Q3 driver
                starts the race on the set of his fastest Q2 lap, so that set is never
                handed back. Q is split into Q1/Q2/Q3 (split_qualifying_sessions() on a
                FastF1 Laps under "Q", or the "Q1"/"Q2"/"Q3" keys); a driver with laps
                in Q3 is a Q3 driver.
    reserves:   {session: {reserve_code: nominated_code}}, who drove for whom where the
                nominated driver has no laps. Without an entry, and given race_laps, a
                session's only non-race code of the driver's Team is taken as his
                reserve when he is the team's only race driver without laps there.

    The start set: for a Q3 driver under q2_start_rule it is known without race_laps,
    and is left out of the result (weekend_report's "fitted"). Given race_laps, the
    race's lap-1 set wins; a different set from the protected one adds the note "Q2
    start set mismatch". For non-Q3 drivers under q2_start_rule, and for everyone when
    the rule is off, the start set is unknown without race_laps, so nothing is removed:
    the result is the full race-start inventory, start set included (split_fitted
    takes it out).
    Returns a list of {compound, laps_run, source, set_id}.
    """
    return weekend_report(sessions, driver, allocation, returns, race_laps, upto_lap,
                          order, hindsight, q2_start_rule, reserves)["sets"]


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
