from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum

class CarMode(str,Enum):
    GRID='GRID'; PIT_START='PIT_START'; RACING='RACING'; PIT_ENTRY='PIT_ENTRY'; PIT_LANE='PIT_LANE'; PIT_BOX='PIT_BOX'; PIT_EXIT='PIT_EXIT'; FINISHED='FINISHED'

@dataclass
class CarState:
    driver:str; team:str; position:int; compound:str; tyre_age:int
    lap:int; progress:float; speed_kph:float=230.0; mode:CarMode=CarMode.RACING
    pit_progress:float=0.0; pit_timer_s:float=0.0; target_compound:str|None=None
    gap_s:float=0.0; number:int=0; grid_position:int|None=None
    drs_eligible:bool=False; drs_active:bool=False; drs_zone:str=''
    base_lap_s:float=80.0; race_distance_m:float=0.0; lateral_offset_m:float=0.0
    pace_factor:float=1.0; battle_state:str=''
    gap_to_leader_s:float=0.0; interval_ahead_s:float=0.0; laps_behind:int=0
    pit_entry_race_distance_m:float|None=None

@dataclass
class RadioLine:
    source:str; text:str; start_s:float; end_s:float

@dataclass
class RaceState:
    total_laps:int; time_s:float=0.0; cars:list[CarState]=field(default_factory=list)
    status:str='GREEN'; agent_state:str='MONITORING'; events:list[dict]=field(default_factory=list)
    focus_driver:str='HAM'; radio_lines:list[RadioLine]=field(default_factory=list)

    def radio_exchange(self, lines:list[tuple[str,str,float]]):
        t=self.time_s
        self.radio_lines=[]
        for source,text,duration in lines:
            self.radio_lines.append(RadioLine(source,text,t,t+duration)); t+=duration

    def active_radio(self):
        for line in self.radio_lines:
            if line.start_s <= self.time_s < line.end_s:return line
        return None
