"""Effective tyre age under SC/VSC, and the tyre hazard hook. Synthetic data only."""
import numpy as np
import pytest

from src.sim import SimConfig, Simulator
from src.sim.config import EventEffects, HazardEffects
from src.sim.degradation import DEGRADATION
from src.sim.events import Scenario
from src.sim.hazard import HAZARD, AgeLogisticHazard, NoHazard

from test_sim import RACE, TRUE, start, synthetic_frame

SOFT_B, SOFT_W = TRUE["SOFT"]
PLAN = [["SOFT", 20], ["MEDIUM", 30]]


@pytest.fixture(scope="module")
def sim():
    return Simulator(synthetic_frame(), SimConfig(degradation={"name": "linear"}))


def hazard_sim(hazard=None, n_samples=1, **effects):
    cfg = SimConfig(degradation={"name": "linear"}, n_samples=n_samples,
                    hazard=hazard or {"name": "age_logistic"}, hazard_effects=HazardEffects(**effects))
    return Simulator(synthetic_frame(), cfg)


def step_hazard(s, soft_a0=15.5):
    """A SOFT hazard that is 0 up to age 15 and 1 from age 16; MEDIUM never fails."""
    s.hazard = AgeLogisticHazard({"SOFT": soft_a0, "MEDIUM": 1e9}, p_max=1.0, scale=0.01)
    return s


# --- effective tyre age ------------------------------------------------------
def test_sc_laps_reduce_age_within_the_stint(sim):
    fx = EventEffects()
    res = sim.simulate(start(), PLAN, scenario=Scenario(sc_laps=[10, 11, 12]), return_laps=True)
    # the age at lap 20 is 20 - 3 * (1 - 0.3) = 17.9
    assert res.lap_times_s[19] == pytest.approx(SOFT_B + SOFT_W * (20 - 3 * (1 - fx.sc_age_increment)))
    sc_lap = fx.sc_lap_factor * sim.ref_lap_s
    soft = (fx.lap1_extra_s + sum(SOFT_B + SOFT_W * k for k in range(1, 10)) + 3 * sc_lap
            + sum(SOFT_B + SOFT_W * (k - 2.1) for k in range(13, 21)))
    assert res.stint_seconds[0] == pytest.approx(soft)


def test_age_increment_restarts_with_a_new_set(sim):
    plain = sim.simulate(start(), [["SOFT", 20], ["MEDIUM", 30, 6]])
    sc = sim.simulate(start(), [["SOFT", 20], ["MEDIUM", 30, 6]], scenario=Scenario(sc_laps=[10, 11, 12]))
    assert sc.stint_seconds[1] == pytest.approx(plain.stint_seconds[1])
    assert sc.stint_seconds[0] != pytest.approx(plain.stint_seconds[0])


def test_stop_on_an_sc_lap_pays_the_reduced_pit_loss(sim):
    res = sim.simulate(start(), PLAN, scenario=Scenario(sc_laps=[20]))
    assert res.pit_loss_total_s == pytest.approx(RACE.pit_loss_s * EventEffects().sc_pit_loss_factor)


def test_vsc_age_increment_and_unit_increments_reproduce_plain_ages():
    unit = Simulator(synthetic_frame(), SimConfig(degradation={"name": "linear"}, effects=EventEffects(
        sc_age_increment=1.0, vsc_age_increment=1.0)))
    plain = unit.simulate(start(), PLAN, return_laps=True)
    vsc = unit.simulate(start(), PLAN, scenario=Scenario(vsc_laps=[5, 6]), return_laps=True)
    assert vsc.lap_times_s[6:] == plain.lap_times_s[6:]
    half = Simulator(synthetic_frame(), SimConfig(degradation={"name": "linear"}))
    res = half.simulate(start(), PLAN, scenario=Scenario(vsc_laps=[5, 6]), return_laps=True)
    assert res.lap_times_s[19] == pytest.approx(SOFT_B + SOFT_W * (20 - 2 * (1 - 0.5)))


def test_mid_race_snapshot_carries_the_set_age_into_the_sc_age(sim):
    from src.contracts import Snapshot
    snap = Snapshot(race=RACE, driver="X", laps_completed=20, compound="SOFT", tyre_age=20, stints=[])
    res = sim.simulate(snap, [["SOFT", 5], ["MEDIUM", 25]], scenario=Scenario(sc_laps=[21]), return_laps=True)
    assert res.lap_times_s[4] == pytest.approx(SOFT_B + SOFT_W * (20 + 0.3 + 4))


# --- hazard model -------------------------------------------------------------
def test_registry_and_default_is_none(sim):
    assert {"none", "age_logistic"} <= set(HAZARD.keys())
    assert isinstance(sim.hazard, NoHazard) and not sim.hazard.draws


def test_age_logistic_fit_reads_the_age_support():
    df = synthetic_frame()
    df.loc[(df["compound"] == "MEDIUM") & (df["tyre_age"] > 15), "clean"] = False
    h = AgeLogisticHazard.fit(df, min_samples=5)
    assert h.a0 == {"SOFT": 23.0, "MEDIUM": 18.0}
    assert h.a0_for("HARD") == pytest.approx(20.5)
    ages = np.array([[23.0, 18.0, 20.5]])
    by_name = h.per_lap_probability(np.array([["SOFT", "MEDIUM", "HARD"]]), ages)
    by_code = h.per_lap_probability(np.array([[2, 1, 0]]), ages, names=["HARD", "MEDIUM", "SOFT"])
    assert by_name == pytest.approx(np.full((1, 3), 0.01))
    assert by_code == pytest.approx(by_name)


def test_compound_under_min_samples_uses_the_mean_a0():
    df = synthetic_frame()
    h = AgeLogisticHazard.fit(df[(df["compound"] == "SOFT") | (df["tyre_age"] <= 1)], min_samples=12, margin=0.0)
    assert h.a0 == {"SOFT": 20.0} and h.a0_for("MEDIUM") == 20.0


def test_hazard_none_draws_nothing_and_keeps_numbers():
    plans = [PLAN, [["SOFT", 15], ["MEDIUM", 20], ["SOFT", 15]]]
    cfg = dict(degradation={"name": "test_noisy_hazard"}, n_samples=40)
    default = Simulator(synthetic_frame(), SimConfig(**cfg)).simulate_batch(start(), plans, seed=5)
    none = Simulator(synthetic_frame(), SimConfig(**cfg, hazard={"name": "none"})).simulate_batch(
        start(), plans, seed=5)
    # a hazard that never fires still draws, after the model: the model draws are unchanged
    silent = Simulator(synthetic_frame(), SimConfig(**cfg, hazard={"name": "age_logistic", "params": {
        "p_max": 0.0}})).simulate_batch(start(), plans, seed=5)
    assert np.array_equal(default.times, none.times) and np.array_equal(default.times, silent.times)
    assert default.times.std(axis=1).min() > 0
    assert not none.dnf.any() and not none.big_loss.any()
    res = none.result(0, start())
    assert res.p_dnf == 0.0 and res.p_big_loss == 0.0


def test_hazard_far_past_support_almost_surely_fires():
    s = hazard_sim({"name": "age_logistic", "params": {"margin": -100.0, "p_max": 0.9}}, n_samples=200)
    res = s.simulate(start(), PLAN, seed=1)
    assert res.p_dnf + res.p_big_loss == pytest.approx(1.0, abs=0.01)
    assert 0.35 < res.p_dnf < 0.65


def test_event_is_exactly_one_of_dnf_or_big_loss():
    s = hazard_sim({"name": "age_logistic", "params": {"margin": -10.0, "p_max": 0.05}}, n_samples=300)
    b = s.simulate_batch(start(), [PLAN, [["SOFT", 35], ["MEDIUM", 15]]], seed=2)
    assert not (b.dnf & b.big_loss).any()
    assert np.array_equal(np.isnan(b.dnf_lap), ~b.dnf)
    assert (b.dnf | b.big_loss).any() and (~(b.dnf | b.big_loss)).any()
    r = b.result(1, start())
    assert r.p_dnf == pytest.approx(b.dnf[1].mean()) and r.p_big_loss == pytest.approx(b.big_loss[1].mean())


def test_plans_with_identical_first_stints_share_event_laps():
    # every event of the first plan lies in its SOFT first stint (MEDIUM never fails), and
    # the second plan runs the same first stint: within it the event laps must coincide
    s = hazard_sim(n_samples=300, dnf_share=1.0)
    s.hazard = AgeLogisticHazard({"SOFT": 10.0, "MEDIUM": 1e9}, p_max=0.2)
    b = s.simulate_batch(start(), [[["SOFT", 25], ["MEDIUM", 25]], [["SOFT", 25], ["MEDIUM", 10], ["SOFT", 15]]],
                         seed=3)
    first = b.dnf[0]
    assert first.any() and (~first).any()
    assert np.array_equal(b.dnf_lap[0][first], b.dnf_lap[1][first])
    assert (b.dnf_lap[0][first] <= 25).all()
    rest = b.dnf_lap[1][~first]               # events of the second plan in its last SOFT stint only
    assert (np.isnan(rest) | (rest > 35)).all()


def test_dnf_penalty_is_added_once_after_the_event(sim):
    base = sim.simulate(start(), PLAN, return_laps=True)
    s = step_hazard(hazard_sim(dnf_share=1.0))
    b = s.simulate_batch(start(), [PLAN], return_laps=True)
    diff = b.lap_times[0] - np.array(base.lap_times_s)
    assert b.dnf[0, 0] and not b.big_loss[0, 0] and b.dnf_lap[0, 0] == 16   # event on lap 16, age 16
    # the remaining laps and stops run as if the car had continued, plus the penalty once
    assert diff[16] == pytest.approx(HazardEffects().dnf_penalty_s)
    assert np.delete(diff, 16) == pytest.approx(np.zeros(49))
    assert b.pit_loss[0, 0] == pytest.approx(base.pit_loss_total_s)
    assert b.times[0, 0] == pytest.approx(base.remaining.mean_s + 300.0)


def test_big_loss_is_added_on_the_event_lap_only(sim):
    base = sim.simulate(start(), PLAN, return_laps=True)
    s = step_hazard(hazard_sim(dnf_share=0.0))
    b = s.simulate_batch(start(), [PLAN], return_laps=True)
    assert b.big_loss[0, 0] and not b.dnf[0, 0]
    diff = b.lap_times[0] - np.array(base.lap_times_s)
    assert diff[15] == pytest.approx(20.0) and np.delete(diff, 15) == pytest.approx(np.zeros(49))
    assert b.times[0, 0] == pytest.approx(base.remaining.mean_s + 20.0)


def test_dnf_on_the_final_lap_books_the_penalty_on_it(sim):
    base = sim.simulate(start("MEDIUM"), [["MEDIUM", 34], ["SOFT", 16]], return_laps=True)
    s = step_hazard(hazard_sim(dnf_share=1.0))
    b = s.simulate_batch(start("MEDIUM"), [[["MEDIUM", 34], ["SOFT", 16]]], return_laps=True)
    assert b.dnf[0, 0] and b.dnf_lap[0, 0] == 50
    assert b.lap_times[0, -1] == pytest.approx(base.lap_times_s[-1] + 300.0)
    assert b.times[0, 0] == pytest.approx(base.remaining.mean_s + 300.0)


def test_sc_age_lowers_the_hazard():
    # a SOFT set 16 laps old on lap 16 is only 13.9 after three SC laps: the event comes later
    s = step_hazard(hazard_sim(dnf_share=1.0))
    b = s.simulate_batch(start(), [PLAN], scenario=Scenario(sc_laps=[10, 11, 12]))
    a = s.simulate_batch(start(), [PLAN])
    assert a.dnf_lap[0, 0] == 16 and b.dnf_lap[0, 0] == 18   # under SC: lap 18, age 15.9


def test_hazard_events_never_make_a_sample_faster():
    plans = [PLAN, [["SOFT", 35], ["MEDIUM", 15]], [["SOFT", 15], ["MEDIUM", 20], ["SOFT", 15]]]
    cfg = dict(degradation={"name": "test_noisy_hazard"}, n_samples=400)
    plain = Simulator(synthetic_frame(), SimConfig(**cfg)).simulate_batch(start(), plans, seed=9)
    hz = Simulator(synthetic_frame(), SimConfig(**cfg, hazard={"name": "age_logistic", "params": {
        "margin": -5.0, "p_max": 0.05}})).simulate_batch(start(), plans, seed=9)
    fx = HazardEffects()
    assert hz.dnf.any() and hz.big_loss.any()
    assert (hz.times[hz.dnf] > plain.times[hz.dnf]).all()
    assert hz.times[hz.dnf] == pytest.approx(plain.times[hz.dnf] + fx.dnf_penalty_s)
    assert hz.times[hz.big_loss] == pytest.approx(plain.times[hz.big_loss] + fx.big_loss_s)
    quiet = ~(hz.dnf | hz.big_loss)
    assert np.array_equal(hz.times[quiet], plain.times[quiet])


def test_longer_stint_is_riskier_under_age_logistic():
    # 20 vs 35 lap SOFT first stint on the synthetic frame (SOFT support to age 20, a0 23)
    s = hazard_sim(n_samples=4000)
    short, long = (s.simulate(start(), p, seed=0) for p in (PLAN, [["SOFT", 35], ["MEDIUM", 15]]))
    assert long.p_dnf > short.p_dnf
    assert long.remaining.cvar_90_s > short.remaining.cvar_90_s


@DEGRADATION.register("test_noisy_hazard")
class _Noisy:
    def __init__(self, inner, offset=0.0):
        self.inner, self.offset, self.compounds = inner, offset, inner.compounds

    @classmethod
    def fit(cls, data, **kw):
        return cls(DEGRADATION.get("linear").fit(data, **kw))

    def draw(self, rng):
        return _Noisy(self.inner, rng.normal(0, 0.2))

    def lap_times(self, compound, ages, laps):
        return self.inner.lap_times(compound, ages, laps) + self.offset
