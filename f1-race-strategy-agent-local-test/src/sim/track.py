"""Track state: a plan-independent draw per sample and a lap time multiplier by tyre class.

A track model maps to the engine through two calls:

    model.draw(snapshot, rng) -> state          once per sample, after the event,
                                                degradation and hazard draws
    model.factor(state, class_codes, laps) -> multiplier broadcastable to (n_plans, n_laps)

`class_codes` are ints indexing config.class_names(config) (SLICK first). The factor
multiplies the green lap time before the SC/VSC overlay. "dry" draws nothing and returns
1.0, so every seed from before this hook reproduces exactly. This is the seam wet class
curves plug into; none are modelled here.
"""
from .registry import Registry

TRACK = Registry("track model")


@TRACK.register("dry")
class DryTrack:
    def draw(self, snapshot, rng):
        return None

    def factor(self, state, class_codes, laps):
        return 1.0
