"""Offline measured-state scoring for a single recorded robot episode.

Annotations are accepted only here, after simulator execution. All durations
are based on saved samples; no movement or success is inferred from reward or
command acceptance. Contact between samples is not observable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

from .replay_evaluation import AnnotatedTrial


@dataclass(frozen=True)
class PhysicalThresholds:
    lift_m: float = 0.08
    contact_n: float = 0.1
    sustain_s: float = 0.5
    joint_speed_rad_s: float = 0.02
    hand_speed_m_s: float = 0.005
    drop_height_m: float = 0.04
    drop_sustain_s: float = 0.25
    max_sample_gap_s: float = 0.25
    target_change_rad: float = 1e-6
    finger_target_change_m: float = 1e-6

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("Physical thresholds must be finite and positive")
        if self.drop_height_m >= self.lift_m:
            raise ValueError("Drop height must be below the acquisition lift height")


def _number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite numeric data")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite numeric data") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    return value


def _vector(row, name, length=None):
    values = row.get(name)
    if not isinstance(values, list) or not values or (length is not None and len(values) != length):
        raise ValueError(f"{name} has missing or invalid dimensions")
    return [_number(v, name) for v in values]


def _validated_rows(rows):
    result = []
    previous = -1.0
    joint_count = None
    for raw in rows:
        row = dict(raw)
        t = _number(row.get("time_s"), "time_s")
        if t < 0 or t <= previous:
            raise ValueError("Telemetry time_s must be nonnegative and strictly increasing")
        previous = t
        row["time_s"] = t
        for key, size in (("hand_pose_w", 7), ("cube_pose_w", 7), ("finger_contact_n", 2)):
            row[key] = _vector(row, key, size)
        row["joint_pos"] = _vector(row, "joint_pos", joint_count)
        joint_count = len(row["joint_pos"])
        row["joint_target"] = _vector(row, "joint_target", joint_count)
        if row.get("requested_skill") not in ("hold", "grasp_lift"):
            raise ValueError("Telemetry requested_skill must be hold or grasp_lift")
        if min(row["finger_contact_n"]) < 0:
            raise ValueError("Finger contact magnitudes must be nonnegative")
        if "finger_target_width_m" in row:
            row["finger_target_width_m"] = _number(row["finger_target_width_m"], "finger_target_width_m")
        if "interval_start_s" in row:
            start = _number(row["interval_start_s"], "interval_start_s")
            if not 0 <= start <= t:
                raise ValueError("Invalid telemetry interval_start_s")
            row["interval_start_s"] = start
        result.append(row)
    if len(result) < 2:
        raise ValueError("At least two telemetry samples are required")
    return result


def _max_delta(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def evaluate_physical_rows(
    rows: Iterable[dict], *, trial: AnnotatedTrial | None = None,
    thresholds: PhysicalThresholds = PhysicalThresholds(),
    baseline_cube_height_m: float | None = None,
) -> dict:
    """Score one episode; reset-containing/multiple-episode files are rejected.

    A successful sampled lift needs both height and contacts at consecutive
    samples spanning sustain_s. Gaps above max_sample_gap_s reset continuity.
    REST movement uses finite-difference speeds, assigned uniformly to the
    interval between samples and intersected with the annotation intervals.
    """
    rows = _validated_rows(rows)
    if any(r.get("done") or r.get("state_after_automatic_reset") for r in rows):
        raise ValueError("Split episodes and remove post-reset rows before physical scoring")
    if len({r["episode"] for r in rows if "episode" in r}) > 1:
        raise ValueError("Physical scoring accepts one episode at a time")
    if trial is not None:
        from .replay_evaluation import _validate_trials
        _validate_trials([trial])
        rows = [r for r in rows if r["time_s"] <= trial.horizon_s + 1e-9]
        if len(rows) < 2:
            raise ValueError("Insufficient telemetry inside the annotation horizon")
    baseline = rows[0]["cube_pose_w"][2] if baseline_cube_height_m is None else _number(
        baseline_cube_height_m, "baseline_cube_height_m")
    first, last = rows[0], rows[-1]
    good_since = drop_since = None
    first_success = None
    best_sustain = 0.0
    carrying = False
    drops = []
    gaps = []
    rest_duration = rest_moving = rest_path = 0.0
    observed_duration = total_moving = 0.0
    max_hold_target_excursion = max_hold_finger_excursion = 0.0
    hold_anchor = None
    finger_anchor = None
    max_hand_drift = max_joint_drift = 0.0
    max_joint_speed = max_hand_speed = 0.0
    for i, row in enumerate(rows):
        t = row["time_s"]
        max_hand_drift = max(max_hand_drift, math.dist(row["hand_pose_w"][:3], first["hand_pose_w"][:3]))
        max_joint_drift = max(max_joint_drift, _max_delta(row["joint_pos"], first["joint_pos"]))
        good = row["cube_pose_w"][2] - baseline >= thresholds.lift_m and min(row["finger_contact_n"]) >= thresholds.contact_n
        connected = i > 0 and t - rows[i - 1]["time_s"] <= thresholds.max_sample_gap_s + 1e-9
        if not connected:
            good_since = drop_since = None
        if good:
            if good_since is None:
                good_since = t
            duration = t - good_since
            best_sustain = max(best_sustain, duration)
            if duration + 1e-9 >= thresholds.sustain_s:
                carrying = True
                if first_success is None:
                    first_success = t
        else:
            good_since = None
        below_drop_height = row["cube_pose_w"][2] - baseline < thresholds.drop_height_m
        if carrying and below_drop_height:
            if drop_since is None:
                drop_since = t
            if t - drop_since + 1e-9 >= thresholds.drop_sustain_s:
                drops.append(t)
                carrying = False
                drop_since = None
        else:
            drop_since = None
        if row["requested_skill"] == "hold":
            if hold_anchor is None:
                anchor = rows[i - 1] if i else row
                hold_anchor = anchor["joint_target"]
                finger_anchor = anchor.get("finger_target_width_m")
            max_hold_target_excursion = max(max_hold_target_excursion, _max_delta(row["joint_target"], hold_anchor))
            if finger_anchor is not None and "finger_target_width_m" in row:
                max_hold_finger_excursion = max(max_hold_finger_excursion, abs(row["finger_target_width_m"] - finger_anchor))
        else:
            hold_anchor = finger_anchor = None
        if i == 0:
            continue
        prev = rows[i - 1]
        dt = t - prev["time_s"]
        if not connected:
            gaps.append({"start_s": prev["time_s"], "end_s": t})
            continue
        observed_duration += dt
        joint_speed = _max_delta(row["joint_pos"], prev["joint_pos"]) / dt
        hand_speed = math.dist(row["hand_pose_w"][:3], prev["hand_pose_w"][:3]) / dt
        max_joint_speed = max(max_joint_speed, joint_speed)
        max_hand_speed = max(max_hand_speed, hand_speed)
        moving = joint_speed > thresholds.joint_speed_rad_s or hand_speed > thresholds.hand_speed_m_s
        total_moving += dt if moving else 0.0
        if trial is not None:
            for interval in trial.intervals:
                if interval.label != "rest":
                    continue
                overlap = max(0.0, min(t, interval.end_s) - max(prev["time_s"], interval.start_s))
                rest_duration += overlap
                rest_moving += overlap if moving else 0.0
                rest_path += hand_speed * overlap
    declared_horizon = trial.horizon_s if trial else last["time_s"]
    return {
        "measurement_kind": "sampled_simulator_state",
        "thresholds": asdict(thresholds),
        "samples": len(rows), "first_sample_s": first["time_s"], "last_sample_s": last["time_s"],
        "observed_interval_duration_s": observed_duration,
        "unobserved_duration_s": max(0.0, declared_horizon - observed_duration),
        "maximum_sample_interval_s": max(b["time_s"] - a["time_s"] for a, b in zip(rows, rows[1:])),
        "excluded_gaps": gaps,
        "baseline_cube_height_m": baseline,
        "baseline_source": "first_saved_sample" if baseline_cube_height_m is None else "explicit",
        "max_cube_lift_m": max(r["cube_pose_w"][2] - baseline for r in rows),
        "max_finger_contact_n": [max(r["finger_contact_n"][i] for r in rows) for i in range(2)],
        "sampled_physical_lift_pass": first_success is not None,
        "first_sampled_success_time_s": first_success,
        "best_sampled_lift_with_bilateral_contact_s": best_sustain,
        "drops_after_sampled_acquisition": len(drops), "drop_detection_times_s": drops,
        "max_hand_displacement_from_start_m": max_hand_drift,
        "max_joint_displacement_from_start_rad": max_joint_drift,
        "max_hand_speed_m_s": max_hand_speed, "max_joint_speed_rad_s": max_joint_speed,
        "total_moving_duration_s": total_moving,
        "max_hold_target_excursion_rad": max_hold_target_excursion,
        "max_hold_finger_target_excursion_m": max_hold_finger_excursion if all("finger_target_width_m" in r for r in rows) else None,
        "rest_motion": None if trial is None else {
            "observed_rest_duration_s": rest_duration,
            "moving_duration_s": rest_moving,
            "moving_fraction": rest_moving / rest_duration if rest_duration else None,
            "hand_path_length_m": rest_path,
        },
        "limitations": [
            "Samples cannot establish uninterrupted contact or exclude motion between samples.",
            "Motion durations assume each finite-difference speed applies uniformly between samples.",
            "Drop means falling below the drop-height threshold continuously for the configured duration after sustained lift acquisition; contact loss alone is not a drop.",
            "Physical REST motion includes servo settling/drift and cannot be attributed solely to accepted EEG requests.",
        ],
    }


def evaluate_request_target_delays(rows: Iterable[dict], events: Iterable[dict], *,
                                   thresholds: PhysicalThresholds = PhysicalThresholds()) -> dict:
    """First commanded-target change after each HOLD -> GRASP_LIFT request.

    Events have time_s/skill keys and must already be label-free decoded
    decisions. This measures actuator-target dispatch, not motion onset.
    """
    rows = _validated_rows(rows)
    events = [dict(e) for e in events]
    previous_time = -1.0
    for event in events:
        t = _number(event.get("time_s"), "request time_s")
        if t < 0 or t <= previous_time or event.get("skill") not in ("hold", "grasp_lift"):
            raise ValueError("Request events must be ordered, nonnegative, and use the supported skills")
        previous_time = t
        event["time_s"] = t
    spans = []
    active_start = None
    for event in events:
        t = event["time_s"]
        if t > rows[-1]["time_s"]:
            break
        if event["skill"] == "grasp_lift" and active_start is None:
            active_start = t
        elif event["skill"] == "hold" and active_start is not None:
            spans.append((active_start, t, False))
            active_start = None
    if active_start is not None:
        spans.append((active_start, rows[-1]["time_s"], True))
    actions = []
    for prev, row in zip(rows, rows[1:]):
        if row["time_s"] - prev["time_s"] > thresholds.max_sample_gap_s + 1e-9:
            continue
        changed = _max_delta(row["joint_target"], prev["joint_target"]) > thresholds.target_change_rad
        if "finger_target_width_m" in row and "finger_target_width_m" in prev:
            changed |= abs(row["finger_target_width_m"] - prev["finger_target_width_m"]) > thresholds.finger_target_change_m
        if changed and row["requested_skill"] == "grasp_lift":
            actions.append(row.get("interval_start_s", row["time_s"]))
    results = []
    for start, end, censored in spans:
        action = next((t for t in actions if start - 1e-9 <= t < end), None)
        results.append({"request_time_s": start, "active_span_end_s": end,
                        "target_change_time_s": action,
                        "target_change_delay_s": max(0.0, action - start) if action is not None else None,
                        "right_censored": censored and action is None})
    delays = [r["target_change_delay_s"] for r in results if r["target_change_delay_s"] is not None]
    return {"definition": "First saved arm/finger target change in each active request span; target-dispatch timing, not actual motion latency",
            "timing": "interval_start_s where recorded; otherwise post-step sample time (upper-bound timestamp)",
            "active_request_spans": len(results), "spans_with_target_change": len(delays),
            "mean_target_change_delay_s": sum(delays) / len(delays) if delays else None,
            "events": results}
