"""Tyre-set inventory: the physical sets a driver has in the garage during the race.

Rebuilt from the FastF1 Laps tables of the weekend's sessions. TyreLife is the
cumulative lap count of one physical set and keeps counting across garage visits
and across sessions, so a re-fitted set is recognised by its TyreLife carrying on
from where it stopped. A FastF1 Stint is one fitting, not one set.

Each set is a dict {compound, laps_run, source, set_id}, where source is
  observed  seen in the data, and its fate was not in doubt
  inferred  seen, but a guess was needed (a reappearance with no matching set,
            or a return tie that could have removed it instead of another set)
  assumed   never seen in any session; filled in from the event allocation
"""
import pandas as pd

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


def _new_set(book, compound, laps_run, source):
    # set_id = compound initial + running index; the index runs per initial so ids
    # stay unique even when two compounds share a letter (2018 HARD / HYPERSOFT).
    prefix = compound[0]
    book["ids"][prefix] = book["ids"].get(prefix, 0) + 1
    ts = {"compound": compound, "laps_run": laps_run, "source": source,
          "set_id": f"{prefix}{book['ids'][prefix]}", "_last": -1, "_seq": len(book["seen"])}
    book["seen"].append(ts)
    book["garage"].append(ts)
    return ts


def _match(garage, compound, laps_run):
    """The most recently fitted held set of `compound` with exactly `laps_run` laps."""
    cands = [s for s in garage if s["compound"] == compound and s["laps_run"] == laps_run]
    return max(cands, key=lambda s: (s["_last"], s["_seq"])) if cands else None


def _drop(garage, ts):
    garage[:] = [s for s in garage if s is not ts]


def _run_session(book, laps, driver, name):
    """Fit every stint of one session onto a set. Returns the sets used, in first-use order."""
    used = []
    for st in _stints(laps, driver):
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
            ts = _match(book["garage"], c, st["start"] - 1)
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
    """Return the r most-used sets of this session (ties: the one used earliest)."""
    if r <= 0:
        return
    cands = sorted(used, key=lambda s: -s["laps_run"])  # stable, so ties keep first-use order
    for ts in cands[:r]:
        _drop(book["garage"], ts)
    if r < len(cands) and cands[r - 1]["laps_run"] == cands[r]["laps_run"]:
        for ts in cands[r:]:
            if ts["source"] == "observed":
                ts["source"] = "inferred"
        book["notes"].append(f"{name}: return tie at {cands[r]['laps_run']} laps")
    short = r - min(r, len(cands))
    if short:
        book["shortfall"] += short
        book["notes"].append(f"{name}: {short} of {r} returned sets never seen, taken from unused sets")


def _build(sessions, driver, allocation, returns, race_laps=None, upto_lap=None):
    """Work behind weekend_sets(). Returns (garage, info) with internal fields kept."""
    if upto_lap is not None and race_laps is None:
        raise ValueError("upto_lap needs race_laps")
    book = {"garage": [], "seen": [], "ids": {}, "order": 0, "shortfall": 0, "notes": []}

    for name, laps in sessions.items():
        used = _run_session(book, laps, driver, name)
        _hand_back(book, used, returns.get(name, 0), name)
    for name, r in returns.items():
        if name not in sessions:  # missing session: no running, the returns still happen
            _hand_back(book, [], r, name)

    # Sets in the allocation never seen in any session. Returns that had no seen set
    # to hand back come out of these, from the compound with the most left.
    alloc = {_compound(c): n for c, n in allocation.items()}
    gap = {c: max(n - sum(s["compound"] == c for s in book["seen"]), 0) for c, n in alloc.items()}
    for _ in range(book["shortfall"]):
        c = max(gap, key=gap.get)
        if gap[c] == 0:
            book["notes"].append("more sets returned than the allocation allows")
            break
        gap[c] -= 1
    for c, n in gap.items():
        for _ in range(n):
            _new_set(book, c, 0, "assumed")

    info = {"notes": book["notes"], "seen": book["seen"], "fitted": None, "consumed": [],
            "unmatched": []}
    if race_laps is not None:
        for i, st in enumerate(_stints(race_laps, driver)):
            if i > 0 and (upto_lap is None or st["first_lap"] > upto_lap):
                break
            if st["start"] is None and not st["fresh"]:
                info["unmatched"].append(st)
                continue
            need = 0 if st["start"] is None else st["start"] - 1
            ts = _match(book["garage"], st["compound"], need)
            if ts is None:
                info["unmatched"].append(st)
                book["notes"].append(f"R: {st['compound']} stint at lap {st['first_lap']} "
                                     f"needs a set with {need} laps, none held")
                continue
            _drop(book["garage"], ts)
            info["consumed"].append(ts)
            if i == 0:
                info["fitted"] = ts
    return book["garage"], info


def weekend_sets(sessions, driver, allocation, returns, race_laps=None, upto_lap=None):
    """Sets in `driver`'s garage at the start of the race (or at `upto_lap` of it).

    sessions:   FastF1 Laps tables keyed "FP1", "FP2", "FP3", "Q" (and "SQ"/"S"),
                given in the order they ran; missing keys are allowed.
    allocation: sets per compound for the event, e.g. {"HARD": 2, "MEDIUM": 3, "SOFT": 8}.
    returns:    sets handed back after each session, keyed by session name.
    race_laps:  the race Laps table. Its first stint picks the fitted start set, which
                is left out of the result. With upto_lap, every stint starting on or
                before that lap also consumes its set.
    Without race_laps the start set is unknown, so nothing is removed: the result is
    the full race-start inventory, start set included.
    Returns a list of {compound, laps_run, source, set_id}.
    """
    garage, _ = _build(sessions, driver, allocation, returns, race_laps, upto_lap)
    return [{k: s[k] for k in _PUBLIC_KEYS} for s in garage]
