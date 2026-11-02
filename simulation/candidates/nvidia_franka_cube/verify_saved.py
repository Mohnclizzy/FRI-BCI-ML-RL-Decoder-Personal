"""Independently rescore serialized telemetry, without simulator dependencies."""
import argparse
import json
import math
from pathlib import Path

from scoring import score


def verify(directory):
    rows = [json.loads(line) for line in (directory / "telemetry.jsonl").read_text().splitlines()]
    saved = json.loads((directory / "summary.json").read_text())
    results = []
    for expected in saved["episodes"]:
        episode = expected["episode"]
        subset = [r for r in rows if r["episode"] == episode]
        initial = [r for r in subset if r["record_type"] == "initial_pre_action"]
        terminal = [r for r in subset if r["record_type"] == "terminal_pre_reset"]
        if len(initial) != 1 or len(terminal) != 1 or subset[0] != initial[0] or subset[-1] != terminal[0]:
            raise ValueError(f"Invalid initial/terminal boundaries in episode {episode}")
        samples = [r for r in subset if r["record_type"] == "sample"]
        intervals = [r["sample_interval_s"] for r in samples]
        result = score(samples, initial[0]["cube_height_m"], max_gap_s=max(intervals) * 1.01, complete=True)
        if result["physical_lift_pass"] != expected["physical_lift_pass"]:
            raise ValueError("Serialized physical result differs from online result")
        if not math.isclose(result["best_sampled_sustain_s"], expected["best_sampled_sustain_s"], abs_tol=1e-8):
            raise ValueError("Serialized sustained interval differs from online result")
        result.update(episode=episode, max_finger_contact_n=[max(r["finger_contact_n"][i] for r in samples) for i in (0, 1)],
                      first_bilateral_contact_s=next((r["time_s"] for r in samples if min(r["finger_contact_n"]) >= .1), None),
                      max_hand_displacement_m=max(math.dist(r["hand_pose_w"][:3], initial[0]["hand_pose_w"][:3]) for r in samples))
        results.append(result)
    if len(results) != saved["episodes_requested"] or saved["error"] is not None:
        raise ValueError("Saved evaluation incomplete or contains errors")
    return {"serialized_rescore_matches": True, "episodes": results,
            "successes": sum(r["physical_lift_pass"] is True for r in results)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(verify(parser.parse_args().directory), indent=2))
