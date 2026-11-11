from game_engine.circuits.barcelona import load_barcelona
from game_engine.race_state import RaceState,CarState,CarMode
from game_engine.engine import RaceGameEngine

def test_pit_cycle_changes_tyres_and_rejoins():
    c=load_barcelona(2023,use_fastf1=False)
    car=CarState('HAM','Mercedes',1,'MEDIUM',20,31,.90)
    s=RaceState(66,cars=[car]); e=RaceGameEngine(c,s); e.request_pit('HAM','HARD')
    for _ in range(4000): e.step(.05)
    assert car.compound=='HARD'; assert car.mode==CarMode.RACING; assert car.target_compound is None
    assert any(x['type']=='TYRES_CHANGED' for x in s.events); assert any(x['type']=='PIT_EXIT' for x in s.events)
