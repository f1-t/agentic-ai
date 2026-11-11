import numpy as np
from game_engine.circuit import Circuit,PitBox,DRSZone
from game_engine.race_state import RaceState,CarState,CarMode
from game_engine.engine import RaceGameEngine
from game_engine.renderer.broadcast import _tower_rows,render

def setup():
 c=Circuit('T',2023,4657,np.array([[0,0],[100,0],[100,100],[0,100],[0,0]],float),np.array([[0,10],[100,10]],float),pit_boxes=[PitBox('Mercedes',.5)],drs_zones=[DRSZone('Z',.2,.25,.4)])
 cars=[CarState('VER','Red Bull',1,'M',5,10,.50,number=1),CarState('HAM','Mercedes',2,'M',5,10,.48,number=44)]
 s=RaceState(66,cars=cars);return c,s,RaceGameEngine(c,s)

def test_radio_exchange_expires():
 c,s,g=setup();s.radio_exchange([('PIT WALL','BOX, BOX',1.0),('44 HAM','Copy.',1.0)]);assert s.active_radio().source=='PIT WALL';g.step(1.1);assert s.active_radio().source=='44 HAM';g.step(1.0);assert s.active_radio() is None

def test_distance_motion():
 c,s,g=setup();x=g.car('HAM');before=x.race_distance_m;g.step(1.0);assert 20 < x.race_distance_m-before < 100

def test_pit_does_not_reset_race_distance():
 c,s,g=setup();h=g.car('HAM');h.mode=CarMode.PIT_LANE;h.pit_progress=.2;before=h.race_distance_m;g.step(1.0);assert h.race_distance_m>before and h.position<=2

def test_tower_last_page_does_not_wrap():
 c,s,g=setup();s.cars=[]
 for i in range(1,21):s.cars.append(CarState(f'D{i}','Mercedes',i,'M',1,1,(21-i)/21,number=i))
 leader,rows,_=_tower_rows(s,11.0,visible=7);assert [x.position for x in rows]==list(range(16,21))

def test_renderer_smoke():
 c,s,g=setup();frame=render(s,c);assert frame.shape==(1080,1920,3)
