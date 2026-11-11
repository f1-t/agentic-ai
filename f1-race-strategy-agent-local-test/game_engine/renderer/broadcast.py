from __future__ import annotations
import cv2,numpy as np,math
from game_engine.race_state import CarMode
W,H=1920,1080; WHITE=(245,245,245);MUTED=(165,170,178);DARK=(14,16,20);GREEN=(80,210,95);YELLOW=(60,215,245)
TEAM={'Mercedes':(215,210,40),'Red Bull':(50,70,225),'Ferrari':(55,55,230),'McLaren':(35,145,245),'Aston Martin':(80,155,45),'Alpine':(225,105,230),'Haas':(200,200,205),'Williams':(230,110,45),'AlphaTauri':(170,150,80),'Alfa Romeo':(70,70,150)}
TYRE={'SOFT':(55,55,230),'MEDIUM':(60,215,240),'HARD':(235,235,235),'INTERMEDIATE':(65,190,65),'WET':(220,110,40)}
def put(im,s,p,sc=.55,c=WHITE,t=1):cv2.putText(im,str(s),p,cv2.FONT_HERSHEY_SIMPLEX,sc,c,t,cv2.LINE_AA)
def _bounds(c):
 p=np.vstack([c.centerline,c.pit_lane]);mn=p.min(0);mx=p.max(0);return mn,np.maximum(mx-mn,1e-9)
def _map(p,c,x0=430,y0=125,x1=1850,y1=830):
 mn,sp=_bounds(c);q=(np.asarray(p)-mn)/sp;scale=min((x1-x0)/sp[0],(y1-y0)/sp[1]);used=sp*scale;ox=x0+((x1-x0)-used[0])/2;oy=y0+((y1-y0)-used[1])/2;return np.c_[ox+(np.atleast_2d(p)[:,0]-mn[0])*scale,oy+(np.atleast_2d(p)[:,1]-mn[1])*scale]
def _rot(points,x,y,ang,scale=1.0):
 p=np.asarray(points,float)*scale;ca,sa=math.cos(ang),math.sin(ang);R=np.array([[ca,-sa],[sa,ca]]);return (p@R.T+[x,y]).astype(np.int32)
def _car(im,x,y,ang,color,label,drs=False,scale=.58):
 """Compact top-down F1 silhouette; intentionally much narrower than the track."""
 tyre=(20,21,23);carbon=(35,36,38);dark=(12,13,15)
 for cx,cy,rx,ry in [(7,-9,6,3),(7,9,6,3),(-10,-9,7,3),(-10,9,7,3)]:
  pts=_rot([[cx-rx,cy-ry],[cx+rx,cy-ry],[cx+rx,cy+ry],[cx-rx,cy+ry]],x,y,ang,scale);cv2.fillConvexPoly(im,pts,tyre,cv2.LINE_AA)
 for poly,fill in [
  ([[15,-11],[18,-11],[18,11],[15,11],[13,6],[13,-6]],carbon),
  ([[16,-2],[4,-3],[-1,-4],[-7,-6],[-13,-5],[-15,-3],[-15,3],[-13,5],[-7,6],[-1,4],[4,3]],color),
  ([[5,-7],[-8,-8],[-13,-5],[-8,-3],[4,-4]],color),
  ([[5,7],[-8,8],[-13,5],[-8,3],[4,4]],color),
  ([[-13,-7],[-17,-7],[-17,7],[-13,7]],carbon)]:
  cv2.fillConvexPoly(im,_rot(poly,x,y,ang,scale),fill,cv2.LINE_AA)
 cv2.ellipse(im,(int(x),int(y)),(max(2,int(4*scale)),max(2,int(3*scale))),math.degrees(ang),0,360,dark,-1,cv2.LINE_AA)
 if drs:cv2.polylines(im,[_rot([[-17,-7],[-17,7]],x,y,ang,scale)],False,GREEN,1,cv2.LINE_AA)
 put(im,label,(int(x+12),int(y-8)),.24,GREEN if drs else WHITE,1)

def _tower_rows(state,now,visible=7):
 """Linear NASCAR-style pages: never wrap P20 directly into P2 on one page."""
 ordered=sorted(state.cars,key=lambda c:c.position);leader=ordered[0];rest=ordered[1:]
 dwell=4.5;slide=.75;cycle=dwell+slide
 pages=[rest[i:i+visible] for i in range(0,len(rest),visible)] or [[]]
 page=int(now//cycle)%len(pages);phase=now%cycle
 rows=pages[page]
 # transition amount is exposed for future pixel sliding, but page contents stay linear.
 slide_t=0.0 if phase<=dwell else min(1.0,(phase-dwell)/slide)
 return leader,rows,slide_t

def render(state,circuit):
 im=np.full((H,W,3),(196,214,226),np.uint8);cv2.rectangle(im,(0,0),(W,70),DARK,-1);put(im,'SPANISH GRAND PRIX',(405,43),.72,WHITE,2);put(im,circuit.name.upper(),(820,42),.40,MUTED,1)
 focus=next(c for c in state.cars if c.driver==state.focus_driver)
 # Daylight Barcelona world: grass, gravel/runoff, grandstands and crowd texture.
 cv2.rectangle(im,(390,100),(1885,850),(72,132,73),-1)
 cv2.ellipse(im,(1520,280),(210,105),-10,0,360,(125,164,190),-1,cv2.LINE_AA);cv2.ellipse(im,(760,690),(170,80),12,0,360,(125,164,190),-1,cv2.LINE_AA)
 for gx,gy,gw,gh in [(480,145,260,48),(1380,705,330,55),(980,125,250,42)]:
  cv2.rectangle(im,(gx,gy),(gx+gw,gy+gh),(105,110,115),-1);cv2.rectangle(im,(gx,gy),(gx+gw,gy+7),(225,225,220),-1)
  for yy in range(gy+14,gy+gh-4,9):
   for xx in range(gx+8,gx+gw-5,12):cv2.circle(im,(xx,yy),2,((xx+yy)%90+80,(xx*3)%120+70,(yy*2)%120+80),-1)
 tr=_map(circuit.centerline,circuit).astype(np.int32);cv2.polylines(im,[tr],True,(115,150,180),56,cv2.LINE_AA);cv2.polylines(im,[tr],True,(235,235,230),38,cv2.LINE_AA);cv2.polylines(im,[tr],True,(62,64,66),31,cv2.LINE_AA);cv2.polylines(im,[tr],True,(45,46,48),5,cv2.LINE_AA)
 pit=_map(circuit.pit_lane,circuit).astype(np.int32);cv2.polylines(im,[pit],False,(205,205,200),28,cv2.LINE_AA);cv2.polylines(im,[pit],False,(78,79,80),20,cv2.LINE_AA);cv2.polylines(im,[pit],False,(235,235,230),2,cv2.LINE_AA)
 # Distinct pit landmarks.
 ex,ey=pit[0]; ox,oy=pit[-1]; put(im,'PIT ENTRY',(int(ex)-10,int(ey)-22),.30,YELLOW,1);put(im,'PIT EXIT / REJOIN',(int(ox)+10,int(oy)-18),.30,YELLOW,1)
 cv2.circle(im,(int(ex),int(ey)),7,YELLOW,2);cv2.circle(im,(int(ox),int(oy)),7,GREEN,2)
 # Pit wall + garage apron / team bays.
 for i,b in enumerate(circuit.pit_boxes):
  j=min(int(b.s*(len(pit)-1)),len(pit)-1);x,y=pit[j];
  cv2.rectangle(im,(x-18,y-22),(x+18,y+22),(58,60,62),-1);cv2.rectangle(im,(x-15,y-19),(x+15,y-10),TEAM.get(b.team,(150,150,150)),-1)
  put(im,b.team[:3].upper(),(x-13,y-12),.20,WHITE,1)
 # white pit-lane divider gives the lane a proper pit-road read
 cv2.polylines(im,[pit],False,(215,215,210),2,cv2.LINE_AA)
 for b in circuit.pit_boxes:
  i=min(int(b.s*(len(pit)-1)),len(pit)-1);x,y=pit[i];cv2.rectangle(im,(x-10,y-14),(x+10,y+14),(150,150,145),1)
 for z in circuit.drs_zones:
  for s,label in [(z.detection_s,'D'),(z.activation_s,'DRS')]:
   x,y=_map(circuit.point(s),circuit)[0].astype(int);cv2.circle(im,(x,y),7,GREEN,2);put(im,label,(x+10,y-8),.32,GREEN)
 for c in reversed(state.cars):
  usepit=c.mode in {CarMode.PIT_START,CarMode.PIT_ENTRY,CarMode.PIT_LANE,CarMode.PIT_BOX,CarMode.PIT_EXIT};p=circuit.point(min(c.pit_progress,1),pit=True) if usepit else circuit.point(c.progress);x,y=_map(p,circuit)[0];ang=circuit.heading(min(c.pit_progress,1),pit=True) if usepit else circuit.heading(c.progress);_car(im,x,y,ang,TEAM.get(c.team,(190,190,190)),c.driver,c.drs_active)
 # NASCAR-inspired tower: leader pinned, cycling classification, HAM footer if offscreen
 cv2.rectangle(im,(25,78),(375,855),(15,17,21),-1);put(im,'SPANISH GP',(45,112),.48,WHITE,2);put(im,f'LAP {focus.lap} / {state.total_laps}',(45,140),.40,MUTED);cv2.circle(im,(292,126),7,GREEN if state.status=='GREEN' else YELLOW,-1);put(im,state.status,(307,132),.34,WHITE);put(im,'INTERVAL',(45,169),.36,MUTED)
 leader,rows,slide=_tower_rows(state,state.time_s);shown=[leader]+rows
 def row(c,y,pinned=False):
  if c.driver==state.focus_driver:cv2.rectangle(im,(35,y-27),(365,y+17),(38,41,48),-1)
  put(im,c.position,(48,y),.48,MUTED);put(im,c.number,(78,y),.34,MUTED);put(im,c.driver,(113,y),.56,WHITE,2)
  stat='LEADER' if c.position==1 else ('PIT' if c.mode not in {CarMode.RACING,CarMode.GRID} else (f'+{c.laps_behind} LAP' if c.laps_behind==1 else (f'+{c.laps_behind} LAPS' if c.laps_behind>1 else f'+{c.interval_ahead_s:.1f}')))
  put(im,stat,(195,y),.37,YELLOW if stat=='PIT' else WHITE);cv2.circle(im,(302,y-6),8,TYRE.get(c.compound,WHITE),-1);put(im,c.tyre_age,(316,y),.31,MUTED)
  if c.drs_active:put(im,'DRS',(335,y),.28,GREEN,1)
 row(leader,205,True);cv2.line(im,(35,225),(365,225),(70,73,78),1)
 for i,c in enumerate(rows):row(c,263+i*64)
 if focus not in shown:
  cv2.line(im,(35,760),(365,760),(70,73,78),1);row(focus,807,True)
 cv2.rectangle(im,(390,870),(1885,1045),(15,17,21),-1);put(im,str(focus.number),(420,932),1.15,WHITE,2);put(im,focus.driver,(500,922),.72,WHITE,2);put(im,f'P{focus.position}',(500,957),.48,MUTED);put(im,focus.compound,(690,914),.42,MUTED);put(im,f'{focus.tyre_age} LAPS',(690,951),.66,WHITE,2)
 if focus.drs_active:put(im,'DRS ACTIVE',(930,947),.52,GREEN,2)
 elif focus.drs_eligible:put(im,'DRS ELIGIBLE',(930,947),.45,GREEN,1)
 put(im,'AI RACE ENGINEER',(1260,912),.42,MUTED);put(im,state.agent_state,(1260,953),.66,GREEN if 'PIT' in state.agent_state or 'BOX' in state.agent_state else YELLOW,2)
 radio=state.active_radio()
 if radio:
  # F1-style keyed radio: compact top-center strip, then disappears.
  x0,y0,x1,y1=690,78,1500,157
  cv2.rectangle(im,(x0,y0),(x1,y1),(18,20,24),-1);cv2.rectangle(im,(x0,y0),(x0+7,y1),YELLOW,-1)
  put(im,f'RADIO  •  {radio.source}',(x0+25,y0+28),.36,YELLOW,1);put(im,radio.text,(x0+25,y0+61),.48,WHITE,1)
 return im
