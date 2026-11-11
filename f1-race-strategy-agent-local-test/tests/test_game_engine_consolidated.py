import numpy as np
from game_engine.circuit import Circuit,PitBox,DRSZone
from game_engine.race_state import RaceState,CarState,CarMode
from game_engine.engine import RaceGameEngine
from game_engine.renderer.broadcast import render


def make_engine():
    c=Circuit('T',2023,1000.0,np.array([[0,0],[100,0],[100,100],[0,100],[0,0]],float),np.array([[0,10],[30,10],[70,10],[100,10]],float),pit_boxes=[PitBox('Mercedes',.50)],drs_zones=[DRSZone('Z',.2,.25,.4)],pit_entry_track_s=.90,pit_exit_track_s=.10,pit_speed_limit_s=.08,pit_speed_limit_end_s=.92)
    cars=[]
    for i,(d,t,prog) in enumerate([('VER','Red Bull',.95),('SAI','Ferrari',.93),('NOR','McLaren',.91),('HAM','Mercedes',.90),('ALO','Aston Martin',.87),('RUS','Mercedes',.84)],1):
        cars.append(CarState(d,t,i,'MEDIUM',5,10,prog,speed_kph=250,number=i))
    s=RaceState(66,cars=cars); return c,s,RaceGameEngine(c,s)


def test_interval_is_to_immediate_car_ahead_not_leader():
    c,s,g=make_engine(); ordered=sorted(s.cars,key=lambda x:x.position)
    # P3 interval must use P2->P3 distance, not leader->P3 distance.
    p2,p3=ordered[1],ordered[2]
    expected=(p2.race_distance_m-p3.race_distance_m)/max(p3.speed_kph/3.6,35.)
    assert abs(p3.interval_ahead_s-expected)<1e-9
    assert p3.gap_to_leader_s>p3.interval_ahead_s


def test_lapped_car_reports_whole_lap_state():
    c,s,g=make_engine(); back=g.car('RUS'); leader=g.car('VER')
    back.race_distance_m=leader.race_distance_m-c.length_m-50
    g._sync_all(); assert back.laps_behind==1


def test_pit_path_has_distinct_entry_and_exit_mapping():
    c,s,g=make_engine()
    assert c.pit_entry_track_s != c.pit_exit_track_s
    assert abs(c.pit_equivalent_progress(0)-c.pit_entry_track_s)<1e-9
    assert abs(c.pit_equivalent_progress(1)-c.pit_exit_track_s)<1e-9


def test_position_loss_in_pits_requires_cars_to_pass_canonical_distance():
    c,s,g=make_engine(); h=g.car('HAM'); a=g.car('ALO'); r=g.car('RUS')
    h.mode=CarMode.PIT_BOX; h.pit_progress=.5; h.pit_entry_race_distance_m=h.race_distance_m; h.pit_timer_s=5.; h.speed_kph=0
    initial=h.position
    # No arbitrary reset to the back on pit mode change.
    g._sync_all(); assert h.position==initial
    # Only cars moved beyond HAM's canonical coordinate pass him.
    a.race_distance_m=h.race_distance_m+1; g._sync_all(); assert a.position<h.position
    r.race_distance_m=h.race_distance_m+2; g._sync_all(); assert r.position<h.position
    assert h.position==initial+2


def test_renderer_smoke_consolidated():
    c,s,g=make_engine(); frame=render(s,c); assert frame.shape==(1080,1920,3)
