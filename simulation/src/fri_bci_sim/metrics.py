"""Aggregate measured episode outcomes; never infer success from a command."""

from dataclasses import dataclass
import math
from statistics import mean


@dataclass(frozen=True)
class Outcome:
    episode: str
    requested: str
    success: bool
    unintended_motion: bool
    dropped: bool
    duration_s: float

    def __post_init__(self):
        if not self.episode or self.requested not in ("hold", "grasp_lift"):
            raise ValueError("Episode and a supported requested behavior are required")
        if not math.isfinite(self.duration_s) or self.duration_s < 0:
            raise ValueError("Duration must be finite and nonnegative")
        if any(type(x) is not bool for x in (self.success, self.unintended_motion, self.dropped)):
            raise ValueError("Outcome flags must be JSON booleans")


def summarize(outcomes: list[Outcome]) -> dict:
    if not outcomes or len({o.episode for o in outcomes}) != len(outcomes):
        raise ValueError("Require nonempty outcomes with unique episode IDs")
    rates = {mode: mean(o.success for o in outcomes if o.requested == mode)
             for mode in ("hold", "grasp_lift") if any(o.requested == mode for o in outcomes)}
    return {"episodes": len(outcomes), "success_by_request": rates,
            "balanced_success": mean(rates.values()) if len(rates) == 2 else None,
            "unintended_motion_fraction": mean(o.unintended_motion for o in outcomes),
            "drop_fraction": mean(o.dropped for o in outcomes),
            "mean_duration_s": mean(o.duration_s for o in outcomes)}
