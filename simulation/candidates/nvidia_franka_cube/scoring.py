"""Timestamp-based scoring, independent of simulation and training reward."""
from __future__ import annotations


def score(samples, baseline, *, max_gap_s, complete):
    good_since = below_since = success_time = None
    acquired = False
    drops = []
    best = 0.0
    previous = None
    contacts_observed = 0
    for row in samples:
        t = row["time_s"]
        if previous is not None and t <= previous:
            raise ValueError("Samples must have strictly increasing timestamps")
        if previous is None or t - previous > max_gap_s:
            good_since = below_since = None
        previous = t
        forces = row["finger_contact_n"]
        valid = len(forces) == 2 and all(f is not None for f in forces)
        contacts_observed += int(valid)
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
    return {
        "status": "completed" if complete else "incomplete",
        "samples": len(samples), "contact_coverage": coverage,
        "max_cube_rise_m": max((r["cube_height_m"] - baseline for r in samples), default=None),
        "best_sampled_sustain_s": best, "first_sampled_success_s": success_time,
        "physical_lift_pass": success_time is not None if coverage == 1.0 else None,
        "drop_detection_times_s": drops,
        "definition": "rise>=0.08m and each finger force>=0.1N spanning>=0.5s",
        "limitation": "Sampled contact only; no claim about contact between observations.",
    }
