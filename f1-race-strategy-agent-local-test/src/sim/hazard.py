"""Tyre hazard: the per lap probability of a tyre event (a failure) on old tyres.

A hazard model is fitted on the same frame as the degradation model and maps
(compound, effective tyre age) to a per lap event probability:

    fit(laps, min_samples, **params) -> model
    model.per_lap_probability(compounds, ages, names=None) -> array shaped like ages

`compounds` holds compound names, or int codes with `names[code]` the compound name.
`draws` says whether the engine draws hazard randomness at all: "none" draws nothing,
so seeds from before the hazard hook reproduce exactly. What an event costs (DNF or a
big time loss) is the engine's business, set in SimConfig.hazard_effects.
"""
import numpy as np

from .registry import Registry

HAZARD = Registry("hazard model")


@HAZARD.register("none")
class NoHazard:
    draws = False

    @classmethod
    def fit(cls, laps, min_samples=5, **_):
        return cls()

    def per_lap_probability(self, compounds, ages, names=None):
        return np.zeros(np.shape(ages))


@HAZARD.register("age_logistic")
class AgeLogisticHazard:
    """p = p_max * sigmoid((age - a0_c) / scale), a0_c = oldest clean tyre of compound c + margin.

    The fitted frame says how old each compound has been run without failing, so the
    hazard rises past that support. A compound with fewer than min_samples clean laps
    has no support of its own and uses the mean a0 of the others; with no supported
    compound at all, a0 is infinite and the probability is zero. Defaults are provisional.
    """

    draws = True

    def __init__(self, a0, p_max=0.02, scale=2.0):
        self.a0 = a0                # compound -> age at which p = p_max / 2
        self.p_max, self.scale = p_max, scale
        self.default_a0 = float(np.mean(list(a0.values()))) if a0 else float("inf")

    @classmethod
    def fit(cls, laps, min_samples=5, p_max=0.02, margin=3.0, scale=2.0, **_):
        clean = laps[laps["clean"].astype(bool) & laps["tyre_age"].notna()]
        a0 = {c: float(d["tyre_age"].max()) + margin
              for c, d in clean.groupby("compound") if len(d) >= min_samples}
        return cls(a0, p_max, scale)

    def a0_for(self, compound):
        return self.a0.get(compound, self.default_a0)

    def per_lap_probability(self, compounds, ages, names=None):
        compounds = np.asarray(compounds)
        if compounds.dtype.kind in "iu":
            a0 = np.array([self.a0_for(c) for c in names], dtype=float)[compounds]
        else:
            uniq, inv = np.unique(compounds, return_inverse=True)
            a0 = np.array([self.a0_for(c) for c in uniq], dtype=float)[inv].reshape(compounds.shape)
        with np.errstate(over="ignore"):
            return self.p_max / (1.0 + np.exp(-(np.asarray(ages, dtype=float) - a0) / self.scale))
