"""Extract early one-sided contact from two selected near-table playbacks."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "outputs/isaac/near_table_progress_eval_local"
BASE = RAW / "near-table-progress-eval-5x12s-seed8-12-20261102T2330Z"
GUARD = {
    8: RAW / "near-table-finger-guard-eval-seed8-20261102T2350Z",
    11: RAW / "near-table-finger-guard-eval-seed11-20261102T2352Z",
}
OUTPUT = ROOT / "results/near_table_contact_trace_2026-11-02.json"
FINGERS = ("panda_finger_joint1", "panda_finger_joint2")


def record(row):
    cube, hand = row["cube_pose_w"], row["hand_pose_w"]
    joints = dict(zip(row["joint_names"], row["robot_joint_pos"]))
    return {
        "time_s": row["episode_time_s"],
        "contact_n": row["finger_contact_n"],
        "cube_minus_hand_xyz_m": [round(cube[i] - hand[i], 5) for i in range(3)],
        "cube_z_m": cube[2],
        "finger_position_m": [joints[name] for name in FINGERS],
        "finger_target_m": [row["joint_pos_target"][name] for name in FINGERS],
    }


def inspect(path, expected_sha256):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"Raw trace differs from verified Brev copy: {path}")
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    start = next(row for row in rows if row["record_type"] == "episode_start")
    physics = [row for row in rows if row["record_type"] == "physics_step"]
    first = next((row for row in physics if max(row["finger_contact_n"]) >= 1), None)
    peak = max(physics, key=lambda row: max(row["finger_contact_n"]))
    return {
        "raw_sha256": digest,
        "start": record(start),
        "first_contact_at_least_1_n": record(first) if first else None,
        "peak_contact": record(peak),
    }


def main():
    baseline = json.loads((ROOT / "results/near_table_progress_screen_2026-11-02.json").read_text())
    guarded = json.loads((ROOT / "results/near_table_finger_guard_diagnostic_2026-11-02.json").read_text())
    base_hashes = baseline["verified_off_host_copy"]["evaluation_files_sha256"]
    guard_hashes = guarded["raw_copy_sha256"]
    def expected(path, hashes):
        return hashes[str(path.relative_to(ROOT))]
    results = []
    for seed in (8, 11):
        original_path = BASE / f"seed{seed}/ppo-telemetry.jsonl"
        guard_path = GUARD[seed] / "ppo-telemetry.jsonl"
        results.append({
            "seed": seed,
            "original": inspect(original_path, expected(original_path, base_hashes)),
            "finger_guard": inspect(guard_path, expected(guard_path, guard_hashes)),
        })
    report = {
        "scope": "Selected seed 8 and 11 trace diagnostic; neither a new rollout nor an unbiased success estimate",
        "contact_threshold_n": 1.0,
        "cases": results,
        "interpretation": "The recorded contact occurs very early with the cube nearly centered under the hand and is predominantly on the right finger. Bounding finger targets did not remove it. Hand pose is the palm body, not a fingertip pose; the trace alone cannot establish collision geometry or a unique cause.",
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
