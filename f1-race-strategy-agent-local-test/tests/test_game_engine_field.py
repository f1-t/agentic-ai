from game_engine.event import load_event,build_field
from game_engine.circuits.barcelona import load_barcelona
from game_engine.race_state import RaceState
from game_engine.engine import RaceGameEngine

def test_spain_2023_field_and_drs():
 e=load_event('spain-2023');cars=build_field(e,None);assert len(cars)==20
 assert {c.driver for c in cars if c.mode.value=='PIT_START'}=={'LEC','SAR'}
 c=load_barcelona(2023);assert len(c.drs_zones)==2;assert [b.team for b in c.pit_boxes][:2]==['Red Bull','Ferrari']
 s=RaceState(66,cars=cars);g=RaceGameEngine(c,s);g.step(.1);assert len(s.cars)==20
