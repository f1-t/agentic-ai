#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
from src.contracts import Snapshot
from game_engine.circuits.barcelona import load_barcelona
from game_engine.race_state import RaceState
from game_engine.engine import RaceGameEngine
from game_engine.event import load_event,build_field
from game_engine.renderer import render

def demo_decision(snapshot):return {'decision':'PIT','compound':'HARD','confidence':.76,'reassess_after_laps':3,'rationale':'Pit window is open; convert to a long final hard stint.','key_risks':['traffic on rejoin']},[]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--event',default='spain-2023');ap.add_argument('--snapshot',default='data/snapshots/2023_spanish_HAM_lap30.json');g=ap.add_mutually_exclusive_group();g.add_argument('--agent',action='store_true');g.add_argument('--demo-agent',action='store_true');ap.add_argument('--render',action='store_true');ap.add_argument('--record');ap.add_argument('--fps',type=int,default=30);ap.add_argument('--seconds',type=float,default=55);ap.add_argument('--headless',action='store_true');ap.add_argument('--debug-state',action='store_true',help='Print canonical race state once per simulated second');args=ap.parse_args()
 snap=Snapshot.model_validate(json.loads(Path(args.snapshot).read_text()));event=load_event(args.event);circuit=load_barcelona(event['year']);state=RaceState(event['laps'],cars=build_field(event,snap),focus_driver=event['focus_driver']);eng=RaceGameEngine(circuit,state,event['focus_driver']);rec=traj=None
 if args.agent:
  from game_engine.agent_adapter import StrategyAgentAdapter;rec,traj=StrategyAgentAdapter().decide(args.snapshot)
 elif args.demo_agent:rec,traj=demo_decision(snap)
 if rec:
  state.agent_state='BOX THIS LAP' if rec['decision']=='PIT' else 'STAY OUT';state.radio_exchange([('PIT WALL','BOX, BOX' if rec['decision']=='PIT' else 'STAY OUT',2.2),('44 HAM','Copy.' if rec['decision']=='PIT' else 'Understood.',1.6)])
  if rec['decision']=='PIT':eng.request_pit(event['focus_driver'],rec['compound'])
 writer=None
 if args.record:
  out=Path(args.record);out.parent.mkdir(parents=True,exist_ok=True);writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*'mp4v'),args.fps,(1920,1080))
 dt=1/args.fps
 for frame_i in range(int(args.seconds*args.fps)):
  eng.step(dt)
  if args.debug_state and frame_i % args.fps == 0:
   print('\nSTATE t=%.1fs' % state.time_s)
   for c in sorted(state.cars,key=lambda x:x.position):
    print(f'{c.position:>2} {c.driver:<3} {c.mode.value:<9} lap={c.lap:>2} prog={c.progress:5.3f} speed={c.speed_kph:6.1f} dist={c.race_distance_m:9.1f}')
  if args.render or writer:
   frame=render(state,circuit)
   if writer:writer.write(frame)
   if args.render and not args.headless:
    cv2.imshow('PLAN A - F1 Strategy Agent',frame)
    if cv2.waitKey(max(1,int(1000/args.fps)))&0xFF in (27,ord('q')):break
 if writer:writer.release()
 cv2.destroyAllWindows();print(json.dumps({'recommendation':rec,'events':state.events[-30:]},indent=2))
 if args.record:print(f'Recorded: {args.record}')
if __name__=='__main__':main()
