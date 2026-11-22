"""The pre-registered reporting choices (configs/eval/reporting.yaml, docs/eval.md "Pre-registered
reporting"): which comparisons lead, the statistic and its intervals, the verdict words, the tail
threshold, and the held-out set reported three ways (the fresh five, the earlier four, all nine).

The scenarios stage records the file's sha256 in a report set's sampling (record); the tables stage
refuses a report set whose recorded sha256 is not the file's (check), copies the file verbatim with
its sha256 into provenance.json (provenance) and writes the tables (tables): per group of races, the
paired comparisons in the pre-registered order with the (race, driver) cluster interval, the race
cluster interval beside it and the verdict, the tail at the stated threshold, and the per-policy
row counts by status. Races are named in the file by season and event and matched to race ids by
season and the event's stem in the id (a race id is <season>_<round>_<event slug>)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
REPORTING_FILE = ROOT / "configs" / "eval" / "reporting.yaml"
SAMPLING_KEY = "reporting_sha256"
STEMS = {"france": ("franc", "french"), "mexico": ("mexic",), "australia": ("austral",), "belgium": ("belgi",),
         "austria": ("austria",), "monaco": ("monaco",), "sakhir": ("sakhir",), "miami": ("miami",)}
CLUSTER_KEY = {"race_driver": "driver", "race": "race"}


def sha256(path=REPORTING_FILE) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path=REPORTING_FILE) -> dict:
    return yaml.safe_load(Path(path).read_text())


def record(sampling: dict, path=REPORTING_FILE) -> dict:
    """`sampling` with the reporting file's sha256 (stage scenarios, report sets)."""
    return {**sampling, SAMPLING_KEY: sha256(path)}


def check(scenario_set, path=REPORTING_FILE) -> None:
    """Refuse (SystemExit) a report set (development false) whose recorded reporting sha256 is not
    the file's: the choices would have been edited after the set was built. A development set passes."""
    if scenario_set.development:
        return
    recorded, now = (scenario_set.sampling or {}).get(SAMPLING_KEY), sha256(path)
    if recorded != now:
        raise SystemExit(f"{Path(path).name} has sha256 {now[:12]}, but this report set was built under "
                         f"{recorded[:12] if recorded else 'none recorded'}: the pre-registered reporting choices changed after "
                         "the set was built; the tables stage refuses")


def provenance(path=REPORTING_FILE) -> dict:
    """The file verbatim and its sha256, for provenance.json."""
    return {"path": str(Path(path).relative_to(ROOT)) if Path(path).is_relative_to(ROOT) else str(path),
            "sha256": sha256(path), "text": Path(path).read_text()}


def matches(race_id: str, entry: dict) -> bool:
    parts = str(race_id).split("_", 2)
    season, slug = parts[0], parts[-1]
    event = str(entry["event"]).lower()
    return season == str(entry["season"]) and any(s in slug.lower() for s in STEMS.get(event, (event,)))


def groups(races, spec: dict) -> list[tuple[str, str, list[str]]]:
    """(name, label, race ids) per held-out group present in `races`, in the file's order; a set with
    none of the named races is one group, "development (all)". Raises ValueError when a named race
    matches more than one race id, or a race of a set with named races belongs to no named group."""
    races = sorted(set(races))
    named = [g for g in spec["held_out"]["groups"] if g["races"] != "all"]
    out, covered = [], set()
    for g in named:
        ids = []
        for entry in g["races"]:
            hit = [r for r in races if matches(r, entry)]
            if len(hit) > 1:
                raise ValueError(f"{entry} matches {hit}: name it more precisely in the reporting file")
            ids += hit
        if ids:
            out.append((g["name"], g["label"], ids))
            covered |= set(ids)
    if not out:
        return [("development (all)", "development races, labelled development", races)]
    stray = sorted(set(races) - covered)
    if stray:
        raise ValueError(f"races {stray} belong to no held-out group of the reporting file")
    every = next((g for g in spec["held_out"]["groups"] if g["races"] == "all"), None)
    if every is not None:
        out.append((every["name"], every["label"], races))
    return out


def comparators(spec: dict) -> list[tuple[str, str]]:
    """(policy, role) in the pre-registered order: lead, secondary, reference only, footnote."""
    c = spec["comparisons"]
    return ([(p, "lead") for p in c["lead"]] + [(p, "secondary") for p in c["secondary"]]
            + [(p, "reference only") for p in c["reference_only"]] + [(p, "footnote") for p in c.get("footnote", {})])


def verdict(low, high) -> str | None:
    if low is None or high is None or np.isnan(low) or np.isnan(high):
        return None
    return "ahead" if high < 0 else "behind" if low > 0 else "level"


def paired(rows: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Agent minus each comparator (pre-registered order) on the scenarios both scored: the mean
    paired regret difference, the primary cluster interval and verdict, the beside cluster interval."""
    from .compare import paired_table
    st = spec["statistic"]
    present = set(rows["policy"])
    order = [(p, role) for p, role in comparators(spec) if p in present]
    if "agent" not in present or not order:
        return pd.DataFrame()
    kw = dict(n_boot=int(st["resamples"]), level=float(st["level"]), seed=int(st["seed"]))
    primary = paired_table(rows, "agent", [p for p, _ in order], cluster=CLUSTER_KEY[st["primary_cluster"]], **kw)
    beside = paired_table(rows, "agent", [p for p, _ in order], cluster=CLUSTER_KEY[st["beside_cluster"]], **kw)
    out = primary.rename(columns={"ci_low": "ci_low_race_driver", "ci_high": "ci_high_race_driver",
                                  "clusters": "race_driver_clusters"})
    out["ci_low_race"], out["ci_high_race"] = beside["ci_low"].values, beside["ci_high"].values
    out["races"] = beside["clusters"].values
    out["verdict"] = [verdict(lo, hi) for lo, hi in zip(out["ci_low_race_driver"], out["ci_high_race_driver"])]
    out.insert(0, "role", [role for _, role in order])
    return out


def tail(rows: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Per policy: scored, the share over the stated threshold and those beside it, median, worst."""
    t = spec["tail"]
    out = []
    ok = rows[rows["status"] == "ok"].assign(regret_s=lambda d: pd.to_numeric(d["regret_s"], errors="coerce"))
    for pol, g in ok.groupby("policy", sort=True):
        r = g["regret_s"].dropna()
        row = {"policy": pol, "scored": len(r), f"share_over_{t['threshold_s']}s (stated)": float((r > t["threshold_s"]).mean())}
        row.update({f"share_over_{s}s": float((r > s).mean()) for s in t["beside_s"]})
        row.update(median_s=float(r.median()), worst_s=float(r.max()))
        out.append(row)
    return pd.DataFrame(out)


def counts(rows: pd.DataFrame) -> pd.DataFrame:
    """Rows per policy and status (ok, invalid, unavailable, error, no_reference), nothing dropped."""
    return rows.groupby(["policy", "status"]).size().unstack(fill_value=0).reset_index()


def tables(rows: pd.DataFrame, spec: dict, era_of=None) -> dict[str, pd.DataFrame]:
    """{name: table}: for each group (groups) the paired comparisons, the tail and the counts, stacked
    with a group column in the file's order; with held_out.by_era and `era_of`, the paired comparisons
    per era of all the races as well."""
    gs = groups(rows["race_id"].unique(), spec)
    out = {"reporting_paired": [], "reporting_tail": [], "reporting_counts": []}
    for name, label, races in gs:
        g = rows[rows["race_id"].isin(races)]
        for key, table in (("reporting_paired", paired(g, spec)), ("reporting_tail", tail(g, spec)),
                           ("reporting_counts", counts(g))):
            if len(table):
                out[key].append(table.assign(group=name, group_label=label, group_races=len(races)))
    if spec["held_out"].get("by_era") and era_of is not None:
        by_era = []
        for era, g in rows.assign(era=rows["race_id"].map(era_of)).groupby("era", sort=True):
            p = paired(g, spec)
            if len(p):
                by_era.append(p.assign(era=era, era_races=g["race_id"].nunique()))
        out["reporting_paired_by_era"] = by_era
    return {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in out.items()}
