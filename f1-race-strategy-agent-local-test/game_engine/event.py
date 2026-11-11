from __future__ import annotations
import json
from pathlib import Path
from .race_state import CarState,CarMode

ASSETS=Path(__file__).resolve().parent/'assets'/'events'
TEAM_PACE={'Red Bull':77.0,'Mercedes':78.0,'Ferrari':78.2,'Aston Martin':78.4,'McLaren':79.0,'Alpine':79.2,'Haas':80.0,'Alfa Romeo':80.1,'AlphaTauri':80.3,'Williams':80.4}

# Calibration only for the visual simulator when the HAM lap-30 snapshot is used.
# HAM's P2, VER ahead by 12.963 s, and 3.630 s to the car behind come from the
# snapshot. The remaining gaps are deliberately labelled estimates, not official timing.
SPAIN_2023_LAP30_CALIBRATION = [
    ('VER', 0.000), ('HAM', 12.963), ('RUS', 16.593), ('PER', 21.5),
    ('SAI', 26.0), ('STR', 30.0), ('ALO', 34.0), ('OCO', 37.0),
    ('GAS', 40.0), ('LEC', 42.0), ('TSU', 44.0), ('ZHO', 46.0),
    ('NOR', 48.0), ('PIA', 50.0), ('HUL', 52.0), ('DEV', 54.0),
    ('MAG', 56.0), ('ALB', 58.0), ('BOT', 60.0), ('SAR', 62.0),
]

def load_event(name='spain-2023'):
    return json.loads((ASSETS/(name.replace('-','_')+'.json')).read_text())

def build_field(event,snapshot=None):
    if snapshot and event.get('event_id')=='spain-2023' and snapshot.driver==event['focus_driver'] and snapshot.laps_completed==30:
        return _build_spain_lap30(event,snapshot)

    lap=(snapshot.laps_completed+1) if snapshot else 1
    cars=[]
    for row in event['drivers']:
        g=row['grid']; pit=row['start']=='PIT_LANE'
        progress=(1.0-(g-1)*.006)%1 if lap<=1 else (1.0-(g-1)*.018)%1
        compound='MEDIUM'; age=max(0,lap-1)
        if snapshot and row['driver']==event['focus_driver']:
            compound=snapshot.compound;age=snapshot.tyre_age
        cars.append(CarState(row['driver'],row['team'],g,compound,age,lap,progress,number=row['number'],grid_position=g,
            mode=CarMode.PIT_START if pit and lap<=1 else CarMode.RACING,base_lap_s=TEAM_PACE.get(row['team'],80.0)+(g%3)*.08))
    return cars

def _build_spain_lap30(event,snapshot):
    rows={r['driver']:r for r in event['drivers']}
    order=dict(SPAIN_2023_LAP30_CALIBRATION)
    # Put the leader well into lap 31. Convert time gaps to distance using a
    # representative 230 km/h local race speed. This creates one coherent distance
    # scale without pretending the non-HAM gaps are official telemetry.
    leader_progress=.72
    speed_mps=230.0/3.6
    length_m=4657.0
    lap=snapshot.laps_completed+1
    cars=[]
    for pos,(driver,gap_s) in enumerate(SPAIN_2023_LAP30_CALIBRATION,1):
        row=rows[driver]
        progress=leader_progress-(gap_s*speed_mps/length_m)
        # Calibration gaps are kept under one lap, so nobody is spuriously lapped.
        progress=max(.02,progress)
        compound='MEDIUM';age=6
        if driver==snapshot.driver:
            compound=snapshot.compound;age=snapshot.tyre_age
        cars.append(CarState(driver,row['team'],pos,compound,age,lap,progress,number=row['number'],grid_position=row['grid'],
            mode=CarMode.RACING,base_lap_s=TEAM_PACE.get(row['team'],80.0)+(row['grid']%3)*.08))
    return cars
