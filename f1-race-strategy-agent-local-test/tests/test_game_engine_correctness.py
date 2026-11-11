from types import SimpleNamespace
import numpy as np
from game_engine.circuit import Circuit,PitBox,DRSZone
from game_engine.race_state import RaceState,CarState,CarMode
from game_engine.engine import RaceGameEngine
from game_engine.event import load_event,build_field


def tiny_engine():
    c=Circuit('T',2023,4657,np.array([[0,0],[100,0],[100,100],[0,100],[0,0]],float),np.array([[0,10],[100,10]],float),pit_boxes=[PitBox('Mercedes',.5)],drs_zones=[DRSZone('Z',.2,.25,.4)])
    cars=[CarState('VER','Red Bull',1,'MEDIUM',5,10,.50,speed_kph=280,number=1),CarState('HAM','Mercedes',2,'MEDIUM',5,10,.48,speed_kph=280,number=44)]
    s=RaceState(66,cars=cars);return c,s,RaceGameEngine(c,s)


def test_pit_limiter_is_per_car():
    c,s,g=tiny_engine();h=g.car('HAM');v=g.car('VER')
    h.mode=CarMode.PIT_LANE;h.pit_progress=.2
    g.step(.5)
    assert h.speed_kph==c.pit_speed_kph
    assert v.mode==CarMode.RACING
    assert v.speed_kph>200


def test_pit_does_not_reset_or_jump_canonical_distance():
    c,s,g=tiny_engine();h=g.car('HAM');before=h.race_distance_m
    h.mode=CarMode.PIT_LANE;h.pit_progress=.2
    g.step(1.0)
    expected=(c.pit_speed_kph/3.6)
    assert abs((h.race_distance_m-before)-expected)<1e-6
    assert h.race_distance_m>before


def test_overtake_immediately_updates_position():
    c,s,g=tiny_engine();v=g.car('VER');h=g.car('HAM')
    h.race_distance_m=v.race_distance_m+5
    g._sync_all()
    assert h.position==1 and v.position==2


def test_lap_and_progress_are_derived_from_distance():
    c,s,g=tiny_engine();h=g.car('HAM')
    h.race_distance_m=12*c.length_m+0.25*c.length_m
    g._sync_all()
    assert h.lap==13
    assert abs(h.progress-.25)<1e-9


def test_spain_lap30_snapshot_initializes_ver_leader_ham_p2_and_no_lapped_cars():
    event=load_event('spain-2023')
    snap=SimpleNamespace(driver='HAM',laps_completed=30,compound='MEDIUM',tyre_age=6,position=2,gap_ahead_s=12.963,gap_behind_s=3.630)
    cars=build_field(event,snap);state=RaceState(event['laps'],cars=cars);circuit=__import__('game_engine.circuits.barcelona',fromlist=['load_barcelona']).load_barcelona(2023);g=RaceGameEngine(circuit,state)
    assert g.car('VER').position==1
    assert g.car('HAM').position==2
    assert g.car('VER').lap==g.car('HAM').lap
    assert max(x.lap for x in state.cars)-min(x.lap for x in state.cars)==0
