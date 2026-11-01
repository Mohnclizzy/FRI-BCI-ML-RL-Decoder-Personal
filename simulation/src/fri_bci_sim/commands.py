"""Timestamped, label-free policy input and conservative command acceptance."""

from dataclasses import dataclass
from enum import StrEnum
import math


class Skill(StrEnum):
    HOLD = "hold"
    GRASP_LIFT = "grasp_lift"


@dataclass(frozen=True)
class Estimate:
    source: str
    episode: str
    window_end_s: float
    available_at_s: float
    p_rest: float
    p_hand: float

    def __post_init__(self):
        values = (self.window_end_s, self.available_at_s, self.p_rest, self.p_hand)
        if not self.source or not self.episode or not all(math.isfinite(v) for v in values):
            raise ValueError("Source, episode and finite values are required")
        if self.window_end_s < 0 or self.available_at_s < self.window_end_s:
            raise ValueError("A prediction cannot be available before its window ends")
        if not (0 <= self.p_rest <= 1 and 0 <= self.p_hand <= 1):
            raise ValueError("Probabilities must be in [0, 1]")
        if not math.isclose(self.p_rest + self.p_hand, 1.0, abs_tol=1e-6):
            raise ValueError("Probabilities must sum to one")


@dataclass(frozen=True)
class Request:
    skill: Skill
    reason: str
    issued_at_s: float


class CommandGate:
    """Accept sustained imagery evidence; reset evidence at source/episode boundaries.

    This is an experimental filter, not a physical safety guarantee. HOLD must
    be implemented by the robot executor as position/grasp maintenance, never
    by indiscriminately opening the gripper or setting joint targets to zero.
    """

    def __init__(self, threshold=0.7, persistence_s=0.25, max_age_s=1.0):
        if not all(math.isfinite(v) for v in (threshold, persistence_s, max_age_s)):
            raise ValueError("Gate parameters must be finite")
        if not 0.5 < threshold <= 1 or persistence_s < 0 or max_age_s <= 0:
            raise ValueError("Invalid gate parameters")
        self.threshold, self.persistence_s, self.max_age_s = threshold, persistence_s, max_age_s
        self._key = None
        self._last_time = None
        self._last_window_end = None
        self._candidate_since = None

    def update(self, estimate: Estimate, now_s: float) -> Request:
        if not math.isfinite(now_s):
            raise ValueError("Clock must be finite")
        key = (estimate.source, estimate.episode)
        if key != self._key:
            self._key, self._last_time, self._last_window_end, self._candidate_since = key, None, None, None
        def hold(reason):
            self._candidate_since = None
            return Request(Skill.HOLD, reason, now_s)
        if now_s < estimate.available_at_s:
            return hold("not_yet_available")
        if now_s - estimate.window_end_s > self.max_age_s:
            return hold("stale")
        if self._last_time is not None:
            if estimate.available_at_s <= self._last_time or estimate.window_end_s <= self._last_window_end:
                return hold("duplicate_or_out_of_order")
            if estimate.available_at_s - self._last_time > self.max_age_s:
                self._candidate_since = None
        self._last_time, self._last_window_end = estimate.available_at_s, estimate.window_end_s
        if estimate.p_rest >= 0.5:
            return hold("rest")
        if estimate.p_hand < self.threshold:
            return hold("low_confidence")
        if self._candidate_since is None:
            self._candidate_since = estimate.available_at_s
        if estimate.available_at_s - self._candidate_since < self.persistence_s:
            return Request(Skill.HOLD, "collecting_evidence", now_s)
        return Request(Skill.GRASP_LIFT, "accepted", now_s)
