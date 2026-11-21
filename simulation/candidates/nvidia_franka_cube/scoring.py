"""Timestamp-based scoring, independent of simulation and training reward."""
from __future__ import annotations
import math


def require_valid_reset(initial, first_sample):
    result = reset_continuity(initial, first_sample)
    if not result["reset_continuity_pass"]:
        raise RuntimeError(f"Reset state lost at first physics sample: {result}")
    return result


def reset_continuity(initial, first_sample):
    """Flag state writes that disappear on the first physics update.

    The native reset assigns zero cube velocity. These generous one-tick
    displacement limits are a validity gate, not a pickup metric.
    """
    cube_jump = math.dist(initial["cube_pose_w"][:3], first_sample["cube_pose_w"][:3])
    hand_jump = math.dist(initial["hand_pose_w"][:3], first_sample["hand_pose_w"][:3])
    return {"cube_jump_m": cube_jump, "hand_jump_m": hand_jump,
            "reset_continuity_pass": cube_jump <= 0.02 and hand_jump <= 0.05}


def score(samples, baseline, *, max_gap_s, complete):
    good_since = below_since = success_time = None
    acquired = False
    drops = []
    best = 0.0
    previous = None
    contacts_observed = 0
    sample_gaps = 0
    peak_forces = [None, None]
    for row in samples:
        t = row["time_s"]
        if not math.isfinite(t) or not math.isfinite(row["cube_height_m"]):
            raise ValueError("Non-finite physics timestamp or cube height")
        if previous is not None and t <= previous:
            raise ValueError("Samples must have strictly increasing timestamps")
        if previous is not None and t - previous > max_gap_s:
            sample_gaps += 1
        if previous is None or t - previous > max_gap_s:
            good_since = below_since = None
        previous = t
        forces = row["finger_contact_n"]
        valid = len(forces) == 2 and all(f is not None and math.isfinite(f) and f >= 0 for f in forces)
        contacts_observed += int(valid)
        if valid:
            peak_forces = [f if p is None else max(p, f) for p, f in zip(peak_forces, forces)]
        good = valid and min(forces) >= 0.1 and row["cube_height_m"] - baseline >= 0.08
        if good:
            good_since = t if good_since is None else good_since
            best = max(best, t - good_since)
            if t - good_since >= 0.5 - 1e-9:
                acquired = True
                if success_time is None:
                    success_time = t
        else:
            good_since = None
        if acquired and row["cube_height_m"] - baseline < 0.04:
            below_since = t if below_since is None else below_since
            if t - below_since >= 0.25 - 1e-9:
                drops.append(t)
                acquired = False
                below_since = None
        else:
            below_since = None
    coverage = contacts_observed / len(samples) if samples else 0.0
    valid_measurement = complete and coverage == 1.0 and sample_gaps == 0
    return {
        "status": "completed" if complete else "incomplete",
        "samples": len(samples), "contact_coverage": coverage,
        "max_cube_rise_m": max((r["cube_height_m"] - baseline for r in samples), default=None),
        "best_sampled_sustain_s": best, "first_sampled_success_s": success_time,
        "sampled_lift_observed": success_time is not None,
        "measurement_valid": valid_measurement,
        "physical_lift_pass": success_time is not None if valid_measurement else None,
        "sample_gaps": sample_gaps,
        "max_finger_contact_n": peak_forces,
        "review_flags": ["finger_force_at_or_above_100N"] if any(
            f is not None and f >= 100.0 for f in peak_forces) else [],
        "drop_detection_times_s": drops,
        "definition": "rise>=0.08m and each finger force>=0.1N spanning>=0.5s",
        "limitation": "Sampled contact only; no claim about contact between observations.",
    }
