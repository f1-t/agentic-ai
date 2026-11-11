from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from game_engine.circuit import Circuit,PitBox,DRSZone
_ASSET_DIR=Path(__file__).resolve().parent.parent/'assets'/'circuits'
def load_barcelona(year:int=2023)->Circuit:
    if year<2023: raise ValueError('This patch ships the verified 2023 no-chicane asset only.')
    data=json.loads((_ASSET_DIR/'barcelona_2023.json').read_text())
    center=Circuit._resample(data['centerline'],1800,closed=True)
    pit=Circuit._resample(data['pit_lane'],500,closed=False)
    return Circuit(name=data['name'],year=data['year'],length_m=data['length_m'],centerline=center,pit_lane=pit,
        track_width_m=data.get('track_width_m',12.0),pit_speed_kph=data.get('pit_speed_kph',80.0),
        pit_boxes=[PitBox(**x) for x in data.get('pit_boxes',[])],drs_zones=[DRSZone(**x) for x in data.get('drs_zones',[])],
        pit_entry_track_s=data.get('pit_entry_track_s',.965),pit_speed_limit_s=data.get('pit_speed_limit_s',.08),
        pit_speed_limit_end_s=data.get('pit_speed_limit_end_s',.92),pit_exit_track_s=data.get('pit_exit_track_s',.075))
