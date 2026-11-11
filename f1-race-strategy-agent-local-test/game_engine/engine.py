from __future__ import annotations
import math
from .race_state import RaceState,CarMode

class RaceGameEngine:
    """Visual race engine. race_distance_m is the sole classification coordinate."""
    def __init__(self,circuit,state:RaceState,focus='HAM'):
        self.circuit=circuit; self.state=state; self.focus=focus
        for c in state.cars:
            c.race_distance_m=max(0.0,(c.lap-1+c.progress)*circuit.length_m)
            c.pace_factor=max(.985,min(1.015,78.5/max(c.base_lap_s,1)))
        self._sync_all()
    def car(self,d): return next(c for c in self.state.cars if c.driver==d)
    def request_pit(self,driver,compound):
        c=self.car(driver); c.target_compound=compound
        self.state.events.append({'t':self.state.time_s,'type':'PIT_REQUEST','driver':driver,'compound':compound})
    def _box_s(self,team):
        b=next((b for b in self.circuit.pit_boxes if b.team==team),None); return b.s if b else .5
    def _order(self): return sorted(self.state.cars,key=lambda c:-c.race_distance_m)
    def _ahead(self,c):
        o=self._order(); i=o.index(c); return o[i-1] if i else None
    def _seconds_for_distance(self,meters,reference):
        return max(0.,meters/max(reference.speed_kph/3.6,35.))
    def _gap(self,c,a):
        if not a:return 999.
        return self._seconds_for_distance(a.race_distance_m-c.race_distance_m,c)
    def _profile(self,p):
        p%=1.; target=286+25*math.cos(2*math.pi*(p-.02))
        for center,minimum,width in [(.105,112,.027),(.205,155,.034),(.335,150,.030),(.455,138,.030),(.590,220,.025),(.705,92,.031),(.825,148,.030),(.925,210,.025)]:
            d=min(abs(p-center),1-abs(p-center)); q=math.exp(-.5*(d/width)**2); target=min(target,target*(1-q)+minimum*q)
        return max(85.,min(325.,target))
    def _approach(self,v,target,dt): return min(target,v+20*dt) if v<target else max(target,v-65*dt)
    def _drs(self,c,old,new):
        c.drs_active=False
        if c.mode!=CarMode.RACING or self.state.status!='GREEN':return
        a=self._ahead(c); gap=self._gap(c,a)
        for z in self.circuit.drs_zones:
            crossed=(old<=z.detection_s<new) if new>=old else (z.detection_s>=old or z.detection_s<new)
            if crossed:c.drs_eligible=gap<=1.;c.drs_zone=z.name
            if c.drs_eligible and c.drs_zone==z.name and z.contains(new):c.drs_active=True
            if c.drs_zone==z.name and not z.contains(new) and new>z.end_s:c.drs_eligible=False;c.drs_zone=''
    def _sync_car(self,c):
        L=self.circuit.length_m; completed=max(0,int(c.race_distance_m//L)); c.lap=completed+1; c.progress=(c.race_distance_m%L)/L
        if c.race_distance_m>=self.state.total_laps*L:c.mode=CarMode.FINISHED
    def _sync_all(self):
        for c in self.state.cars:self._sync_car(c)
        ordered=self._order()
        for i,c in enumerate(ordered,1):c.position=i
        if not ordered:return
        leader=ordered[0]
        for i,c in enumerate(ordered):
            delta=max(0.,leader.race_distance_m-c.race_distance_m)
            c.laps_behind=int(delta//self.circuit.length_m)
            c.gap_to_leader_s=0. if i==0 else self._seconds_for_distance(delta,c)
            c.interval_ahead_s=0. if i==0 else self._seconds_for_distance(max(0.,ordered[i-1].race_distance_m-c.race_distance_m),c)
            c.gap_s=c.gap_to_leader_s # compatibility only; renderer uses explicit fields
    def _pit_equivalent_distance(self,c):
        """Equivalent race-course distance for the car's current point on pit road."""
        L=self.circuit.length_m
        entry=c.pit_entry_race_distance_m
        span=((self.circuit.pit_exit_track_s-self.circuit.pit_entry_track_s)%1.0)*L
        if entry is None:
            # Preserve continuity even if a test/debug tool places a car directly on pit road.
            entry=c.race_distance_m-max(0.,min(1.,c.pit_progress))*span
            c.pit_entry_race_distance_m=entry
        return entry+max(0.,min(1.,c.pit_progress))*span
    def _advance_pit(self,c,dt):
        """Advance pit progress so canonical race distance changes by actual pit-speed metres."""
        if c.pit_entry_race_distance_m is None:
            self._pit_equivalent_distance(c)  # establish a continuity-preserving anchor first
        span=((self.circuit.pit_exit_track_s-self.circuit.pit_entry_track_s)%1.0)*self.circuit.length_m
        c.pit_progress=min(1.,c.pit_progress+(c.speed_kph/3.6)*dt/max(span,1e-9))
        c.race_distance_m=self._pit_equivalent_distance(c)
    def step(self,dt):
        st=self.state; st.time_s+=dt
        for c in st.cars:
            if c.mode==CarMode.FINISHED:continue
            if c.mode==CarMode.PIT_START:
                c.speed_kph=self.circuit.pit_speed_kph; self._advance_pit(c,dt)
                if st.time_s>=2.:c.mode=CarMode.PIT_EXIT;c.pit_progress=max(c.pit_progress,.80)
            elif c.mode==CarMode.RACING:
                old=c.progress; a=self._ahead(c); gap=self._gap(c,a); target=self._profile(old)*c.pace_factor
                if a and gap<1.4 and target>220:target+=7.
                if c.drs_active:target+=18.
                target-=max(c.tyre_age-18,0)*.22; c.speed_kph=self._approach(c.speed_kph,min(340.,target),dt)
                c.race_distance_m+=(c.speed_kph/3.6)*dt; new=(c.race_distance_m%self.circuit.length_m)/self.circuit.length_m; self._drs(c,old,new)
                c.lateral_offset_m=(-3.0 if a and gap<.45 and c.speed_kph>a.speed_kph+2 else 0.0)
                # Pit entry is a distinct late-lap landmark.
                if c.target_compound and old<self.circuit.pit_entry_track_s<=new:
                    c.mode=CarMode.PIT_ENTRY;c.pit_progress=0.;c.drs_active=False;c.pit_entry_race_distance_m=c.race_distance_m
                    st.events.append({'t':st.time_s,'type':'PIT_ENTRY','driver':c.driver,'position':c.position})
            elif c.mode==CarMode.PIT_ENTRY:
                c.speed_kph=self._approach(c.speed_kph,self.circuit.pit_speed_kph,dt); self._advance_pit(c,dt)
                if c.pit_progress>=self.circuit.pit_speed_limit_s:c.mode=CarMode.PIT_LANE
            elif c.mode==CarMode.PIT_LANE:
                c.speed_kph=self.circuit.pit_speed_kph; self._advance_pit(c,dt)
                if c.pit_progress>=self._box_s(c.team):c.mode=CarMode.PIT_BOX;c.pit_timer_s=2.45;c.speed_kph=0.
            elif c.mode==CarMode.PIT_BOX:
                c.speed_kph=0.; c.pit_timer_s-=dt; c.race_distance_m=self._pit_equivalent_distance(c)
                if c.pit_timer_s<=0:
                    old=c.compound;c.compound=c.target_compound or c.compound;c.tyre_age=0;c.mode=CarMode.PIT_EXIT
                    st.events.append({'t':st.time_s,'type':'TYRES_CHANGED','driver':c.driver,'from':old,'to':c.compound})
            elif c.mode==CarMode.PIT_EXIT:
                c.speed_kph=self.circuit.pit_speed_kph; self._advance_pit(c,dt)
                if c.pit_progress>=1:
                    c.mode=CarMode.RACING;c.pit_progress=0.;c.target_compound=None;c.pit_entry_race_distance_m=None
                    st.events.append({'t':st.time_s,'type':'PIT_EXIT','driver':c.driver,'position':c.position})
            self._sync_car(c)
        self._sync_all()
