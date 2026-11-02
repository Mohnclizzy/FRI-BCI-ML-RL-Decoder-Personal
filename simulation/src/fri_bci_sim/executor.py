"""Simulator-independent request state machine for robot skill execution."""

from dataclasses import dataclass
from enum import StrEnum
from copy import deepcopy
from bisect import bisect_right
import json
import math
from pathlib import Path

from .commands import Estimate, Request, Skill


class ExecutorState(StrEnum):
    HOLD = "hold"
    PAUSED = "paused"
    RUNNING = "running"
    COMPLETE = "complete"


@dataclass(frozen=True)
class ExecutorEvent:
    state: ExecutorState
    start_skill: bool
    hold_position: bool


class SkillExecutor:
    """Edge-trigger skills; HOLD freezes the current pose/gripper target.

    The simulator adapter supplies measured joint targets and advances a
    running skill. Repeated GRASP_LIFT decisions never restart a skill.
    """

    def __init__(self):
        self.state = ExecutorState.HOLD
        self.started = False

    def update(self, request: Request, *, skill_complete: bool = False) -> ExecutorEvent:
        if self.state == ExecutorState.COMPLETE or (self.started and skill_complete):
            self.state = ExecutorState.COMPLETE
            return ExecutorEvent(self.state, start_skill=False, hold_position=True)
        if request.skill == Skill.HOLD:
            self.state = ExecutorState.PAUSED if self.started else ExecutorState.HOLD
            return ExecutorEvent(self.state, start_skill=False, hold_position=True)
        start = not self.started
        self.started = True
        self.state = ExecutorState.RUNNING
        return ExecutorEvent(self.state, start_skill=start, hold_position=False)


def _clone(value):
    return value.clone() if hasattr(value, "clone") else deepcopy(value)


class HoldTargetLatch:
    """Latch executed position targets; measured drift never replaces a target."""

    def __init__(self, arm_target, finger_width):
        self.accept(arm_target, finger_width)

    def accept(self, arm_target, finger_width):
        if not math.isfinite(finger_width) or finger_width < 0:
            raise ValueError("Finger target must be finite and nonnegative")
        self._arm = _clone(arm_target)
        self._finger = float(finger_width)

    def hold(self):
        return _clone(self._arm), self._finger

    def resolve(self, holding, proposed_arm_target, proposed_finger_width):
        """Select the actual actuator targets; ignore proposals during HOLD."""
        if not holding:
            self.accept(proposed_arm_target, proposed_finger_width)
        return self.hold()


class DecisionReplay:
    """One recording's validated command trace with a simulator-clock watchdog.

    Requests enable/pause a single skill; they do not restart it. Explicit
    controls are the only modes permitted without an EEG decision trace.
    """

    def __init__(self, path: Path | None, condition: str, max_age_s: float = 1.0):
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("Maximum estimate age must be finite and positive")
        if condition not in ("raw", "confidence", "temporal", "always-hold", "scripted-positive", "scripted-pause"):
            raise ValueError("Unknown command condition")
        self.condition, self.max_age_s = condition, max_age_s
        self.events, self.times = [], []
        if condition in ("always-hold", "scripted-positive", "scripted-pause"):
            if path is not None:
                raise ValueError("Explicit controls must not receive an EEG trace")
            return
        if path is None:
            raise ValueError("EEG command conditions require --decisions")
        key = previous = None
        with Path(path).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    data = row["estimate"]
                    estimate = Estimate(**data)
                    decision = row["decisions"][condition]
                    issued = float(decision["issued_at_s"])
                    skill = Skill(decision["skill"])
                    reason = str(decision.get("reason", "recorded_replay"))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid decision trace line {line_number}: {exc}") from exc
                current_key = (estimate.source, estimate.episode)
                if key is not None and current_key != key:
                    raise ValueError("One simulator episode requires one source/episode trace")
                if not math.isfinite(issued) or issued < estimate.available_at_s:
                    raise ValueError(f"Decision timestamp violates causality on line {line_number}")
                if previous and (estimate.window_end_s <= previous[0]
                                 or estimate.available_at_s < previous[1] or issued < previous[2]):
                    raise ValueError(f"Duplicate or out-of-order decision on line {line_number}")
                key = current_key
                previous = (estimate.window_end_s, estimate.available_at_s, issued)
                self.events.append((issued, estimate.window_end_s, skill, reason))
                self.times.append(issued)
        if not self.events:
            raise ValueError("Decision trace is empty")

    def at(self, time_s: float) -> Request:
        if not math.isfinite(time_s) or time_s < 0:
            raise ValueError("Replay clock must be finite and nonnegative")
        if self.condition == "scripted-positive":
            return Request(Skill.GRASP_LIFT, "scripted_positive_control", time_s)
        if self.condition == "always-hold":
            return Request(Skill.HOLD, "always_hold_control", time_s)
        if self.condition == "scripted-pause":
            active = time_s < 9.0 or 12.0 <= time_s < 28.0
            return Request(Skill.GRASP_LIFT if active else Skill.HOLD, "scripted_pause_control", time_s)
        index = bisect_right(self.times, time_s) - 1
        if index < 0:
            return Request(Skill.HOLD, "awaiting_estimate", time_s)
        _, window_end, skill, reason = self.events[index]
        if time_s - window_end > self.max_age_s:
            return Request(Skill.HOLD, "stale", time_s)
        return Request(skill, reason, time_s)


class ExecutionMonitor:
    """Measure physical drift and lift on every physics step, independently of rewards."""

    def __init__(self, joint_pos, hand_pos, cube_z):
        self.initial_q, self.last_q = tuple(joint_pos), tuple(joint_pos)
        self.initial_hand, self.last_hand = tuple(hand_pos), tuple(hand_pos)
        self.baseline_z, self.max_z = cube_z, cube_z
        self.max_contact = [0.0, 0.0]
        self.max_hand = self.max_joint = self.max_hold_hand = self.max_hold_joint = 0.0
        self.max_hold_target = self.max_hold_finger = 0.0
        self.hold_start = None
        self.sustained_s = self.best_sustained_s = self.drop_s = 0.0
        self.acquired = self.dropped = False

    @staticmethod
    def _joint_distance(a, b):
        return max((abs(x-y) for x, y in zip(a, b)), default=0.0)

    @staticmethod
    def _distance(a, b):
        return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))

    def observe(self, joint_pos, hand_pos, cube_z, contact_n, *, holding,
                arm_target, finger_target, dt):
        self.max_hand = max(self.max_hand, self._distance(hand_pos, self.initial_hand))
        self.max_joint = max(self.max_joint, self._joint_distance(joint_pos, self.initial_q))
        self.max_z = max(self.max_z, cube_z)
        self.max_contact = [max(a, b) for a, b in zip(self.max_contact, contact_n)]
        if holding:
            if self.hold_start is None:
                self.hold_start = (self.last_q, self.last_hand, tuple(arm_target), finger_target)
            q0, hand0, target0, finger0 = self.hold_start
            self.max_hold_joint = max(self.max_hold_joint, self._joint_distance(joint_pos, q0))
            self.max_hold_hand = max(self.max_hold_hand, self._distance(hand_pos, hand0))
            self.max_hold_target = max(self.max_hold_target, self._joint_distance(arm_target, target0))
            self.max_hold_finger = max(self.max_hold_finger, abs(finger_target-finger0))
        else:
            self.hold_start = None
        lifted = cube_z-self.baseline_z >= 0.08 and min(contact_n) >= 0.1
        self.sustained_s = self.sustained_s + dt if lifted else 0.0
        self.best_sustained_s = max(self.best_sustained_s, self.sustained_s)
        self.acquired = self.acquired or self.best_sustained_s + 1e-9 >= 0.5
        self.drop_s = self.drop_s + dt if self.acquired and cube_z-self.baseline_z < 0.04 else 0.0
        self.dropped = self.dropped or self.drop_s + 1e-9 >= 0.25
        self.last_q, self.last_hand = tuple(joint_pos), tuple(hand_pos)

    def summary(self):
        return {
            "baseline_cube_z_m": self.baseline_z, "max_cube_z_m": self.max_z,
            "max_finger_contact_n": self.max_contact,
            "best_sustained_lift_contact_s": self.best_sustained_s,
            "physical_lift_pass": self.acquired,
            "dropped_after_acquisition": self.dropped,
            "drop_definition": "after sustained lift, cube below initial z+0.04 m for >=0.25 s",
            "max_hand_displacement_m": self.max_hand,
            "max_joint_displacement_rad": self.max_joint,
            "max_hold_hand_displacement_m": self.max_hold_hand,
            "max_hold_joint_displacement_rad": self.max_hold_joint,
            "max_hold_target_change_rad": self.max_hold_target,
            "max_hold_finger_target_change_m": self.max_hold_finger,
        }
