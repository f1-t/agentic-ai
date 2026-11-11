from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np

@dataclass
class PitBox:
    team: str
    s: float

@dataclass
class DRSZone:
    name: str
    detection_s: float
    activation_s: float
    end_s: float

    def contains(self, s: float) -> bool:
        s %= 1.0
        if self.activation_s <= self.end_s:
            return self.activation_s <= s <= self.end_s
        return s >= self.activation_s or s <= self.end_s

@dataclass
class Circuit:
    name: str
    year: int
    length_m: float
    centerline: np.ndarray
    pit_lane: np.ndarray
    track_width_m: float = 12.0
    pit_speed_kph: float = 80.0
    pit_boxes: list[PitBox] = field(default_factory=list)
    drs_zones: list[DRSZone] = field(default_factory=list)
    pit_entry_track_s: float = 0.965
    pit_speed_limit_s: float = 0.08
    pit_speed_limit_end_s: float = 0.92
    pit_exit_track_s: float = 0.075

    def pit_equivalent_progress(self, pit_s: float) -> float:
        """Map pit-lane progress to the equivalent wrapped racing-lap progress."""
        pit_s=max(0.0,min(1.0,float(pit_s)))
        span=(self.pit_exit_track_s-self.pit_entry_track_s)%1.0
        return (self.pit_entry_track_s+pit_s*span)%1.0

    @staticmethod
    def _resample(points, n=1800, closed=True):
        p=np.asarray(points,float)
        if closed and np.linalg.norm(p[0]-p[-1])>1e-9: p=np.vstack([p,p[0]])
        d=np.linalg.norm(np.diff(p,axis=0),axis=1); c=np.r_[0,np.cumsum(d)]
        q=np.linspace(0,c[-1],n,endpoint=not closed); out=[]
        for s in q:
            i=min(max(np.searchsorted(c,s,side='right')-1,0),len(d)-1)
            f=(s-c[i])/max(d[i],1e-12); out.append(p[i]*(1-f)+p[i+1]*f)
        return np.asarray(out)

    def point(self, progress: float, *, pit=False):
        p=self.pit_lane if pit else self.centerline
        x=(progress%1)*(len(p)-1); i=int(x); f=x-i
        return p[i]*(1-f)+p[min(i+1,len(p)-1)]*f

    def heading(self, progress: float, *, pit=False):
        a=self.point(progress-.001,pit=pit); b=self.point(progress+.001,pit=pit)
        return math.atan2(b[1]-a[1],b[0]-a[0])
