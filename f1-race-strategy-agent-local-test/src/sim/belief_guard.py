"""The belief's thin-fit guard (SimConfig.belief_guard, off by default; docs/eval.md "Thin wear fits").

The judge's guard (wear_guard.py) protects its whole-race fit from a compound whose laps come from
a handful of cars. The belief's to_snapshot fit has the same weakness early in a race, and worse: a
compound with min_samples visible clean rows is fitted from them however few cars ran it, so a
compound started by two slow cars looks slow (2019 UK at lap 10: HARD from MAG and GRO, 2.4 s/lap
slower than the judge's). This guard is the belief-side counterpart, on the rows visible at the
snapshot only.

A compound qualifies when its visible clean race rows (the window's own as-of clean flags) number at
least one but come from fewer than `min_drivers` drivers or fewer than `min_rows` rows. Its fitted
wear w (sd s) and its level at the mean visible tyre age a, L = base + w a (sd from the fit's
covariance), are each combined by precision with a target:

    wear target   the proxy's wear blend from the well-covered compounds of this fit (practice step,
                  as-of cross-race step: proxy._blend with proxy.as_of_config), combined by precision
                  with the as-of historical wear (wear_guard.wear_prior, races strictly before this one)
    level target  the proxy's pace blend from the well-covered compounds (practice step, qualifying,
                  as-of cross-race step) plus the wear target times a

With `scale_sd_by_drivers` the fit's sds are widened by sqrt(min_drivers / drivers), as the judge's
guard does. mode "wear" moves the wear only and keeps L (the judge's form); "pace_wear" moves both.
With no well-covered compound there is no pace target and only the wear moves toward the history.
Everything read is at or before the snapshot or from earlier races, so the information contract holds.
"""
from __future__ import annotations

import copy

import numpy as np

from . import proxy as _proxy
from .wear_guard import fit_rows, wear_prior


def _prec(x, sx, t, st):
    """Precision-weighted combination of (x, sx) and (t, st); x when the target is unusable."""
    if not (np.isfinite(t) and np.isfinite(st) and st > 0):
        return x
    if not (np.isfinite(sx) and sx > 0):
        return t
    wx, wt = 1.0 / sx ** 2, 1.0 / st ** 2
    return (x * wx + t * wt) / (wx + wt)


def guarded(model, rows, race_id, guard, *, proxy_cfg=None, practice=None, quali=None,
            wear_lower_bound=None) -> tuple[object, dict]:
    """A copy of `model` (the belief's fit at the snapshot) with its thin compounds guarded (module
    docstring), and {compound: change}; `model` itself when nothing qualifies. `rows` are the race rows
    the fit could see (Simulator.window_rows), `proxy_cfg` the as-of SimConfig.compound_proxy."""
    rep = getattr(model, "fit_report", None) or {}
    if not len(rows) or "lap" not in rows:
        return model, {}
    tab = fit_rows(rows)
    coefs = dict(model.coefs)
    well = [c for c in coefs if c in tab.index and tab.loc[c, "n_drivers"] >= guard.min_drivers
            and tab.loc[c, "n_rows"] >= guard.min_rows]
    sds = rep.get("sd", {})
    cov = _proxy._cov_of(model)
    race_rows = dict(rep.get("n_race_rows_per_compound") or {})
    sources = list(proxy_cfg.sources) if proxy_cfg is not None else []
    changes = {}
    for c, (base, w) in model.coefs.items():
        if c not in tab.index or c not in _proxy.SOFTNESS:
            continue
        r = tab.loc[c]
        if r["n_rows"] < 1 or c in well:
            continue
        n_drv = int(r["n_drivers"])
        a = float(r["mean_age"]) if np.isfinite(r["mean_age"]) else 0.0
        scale = (np.sqrt(guard.min_drivers / max(1, n_drv))
                 if guard.scale_sd_by_drivers and n_drv < guard.min_drivers else 1.0)
        sw = float((sds.get("wear_s_per_lap") or {}).get(c, np.nan))
        sb = float((sds.get("base_s") or {}).get(c, np.nan))
        cbw = 0.0
        if cov is not None:
            at, m = cov
            i, j = at.get(f"base_s[{c}]"), at.get(f"wear_s_per_lap[{c}]")
            if i is not None and j is not None:
                sb, sw, cbw = float(np.sqrt(max(m[i, i], 0.0))), float(np.sqrt(max(m[j, j], 0.0))), float(m[i, j])
        var_l = sb ** 2 + a ** 2 * sw ** 2 + 2 * a * cbw
        sl = float(np.sqrt(var_l)) if np.isfinite(var_l) and var_l > 0 else float("nan")
        sw, sl = sw * scale, sl * scale
        # the targets
        p, psd, psrc = wear_prior(race_id, c)
        tw, tsd, pace = p, psd, None
        anchors = [x for x in well if x != c]
        if anchors and proxy_cfg is not None:
            wb = _proxy._blend("wear", c, anchors, coefs, practice, quali, proxy_cfg, race_rows, sources)
            if wb is not None and np.isfinite(wb["tau"]) and wb["tau"] > 0:
                tw = _prec(wb["value"], wb["tau"], p, psd)
                tsd = float(1.0 / np.sqrt(1.0 / wb["tau"] ** 2 + 1.0 / psd ** 2))
            if guard.mode == "pace_wear":
                pace = _proxy._blend("pace", c, anchors, coefs, practice, quali, proxy_cfg, race_rows, sources)
        w2 = _prec(float(w), sw, tw, tsd)
        if wear_lower_bound is not None:
            w2 = max(w2, float(wear_lower_bound))
        lvl = float(base) + float(w) * a
        lvl2 = lvl
        if pace is not None and np.isfinite(pace["tau"]) and pace["tau"] > 0:
            tl = pace["value"] + tw * a
            tl_sd = float(np.sqrt(pace["tau"] ** 2 + (a * tsd) ** 2))
            lvl2 = _prec(lvl, sl, tl, tl_sd)
        base2 = lvl2 - w2 * a
        if base2 == base and w2 == w:
            continue
        coefs[c] = (base2, w2) if not isinstance(model.coefs[c], list) else [base2, w2]
        changes[c] = {"n_rows": int(r["n_rows"]), "n_drivers": n_drv, "mean_age": a, "wear_s": float(w),
                      "wear_sd": sw, "wear_target": float(tw), "wear_target_sd": float(tsd), "prior_source": psrc,
                      "wear_guarded_s": float(w2), "level_s": lvl, "level_sd": sl,
                      "level_target_s": (float(pace["value"] + tw * a) if pace is not None else None),
                      "level_guarded_s": float(lvl2), "base_shift_s": float(base2 - base),
                      "anchors": anchors}
    if not changes:
        return model, {}
    out = copy.copy(model)
    out.__dict__.pop("_draw_plan", None)
    out.coefs = coefs
    if rep:
        out.fit_report = {**rep, "belief_guard": changes}
        unc = rep.get("unconstrained")
        if unc is not None and "mean" in unc:   # a bounded fit draws around this mean: move it with the point
            names = list(unc["names"])
            mean = list(unc["mean"])
            for c in changes:
                for k, name in enumerate((f"base_s[{c}]", f"wear_s_per_lap[{c}]")):
                    if name in names:
                        i = names.index(name)
                        mean[i] = float(mean[i]) + float(coefs[c][k]) - float(model.coefs[c][k])
            out.fit_report["unconstrained"] = {**unc, "mean": mean}
    return out, changes
