"""Publish the bounded near-table learning curve and physical screen.

Use --collect only with restored ignored Brev logs in the local input folder.
The default mode redraws the figure from the tracked compact JSON record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "outputs/isaac/near-table-stage1-eval-local"
RESULT = ROOT / "results/near_table_stage1_screen_2026-11-02.json"
FIGURE = ROOT / "results/figures/near_table_stage1_2026-11-02.png"
MODEL_SHA256 = "50713284b5c112bce1269a2a25054c10100b115f7c1075d01f49c152ebd3895e"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect():
    log = RAW / "train_console.log"
    plain = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text(encoding="utf-8", errors="replace"))

    def series(label):
        return [float(value) for value in re.findall(rf"{re.escape(label)}:\s*([-+\d.]+)", plain)]

    reward, success = series("Mean reward"), series("Metrics/success_rate")
    if len(reward) != 250 or len(success) != 250 or "Training time:" not in plain:
        raise ValueError("Expected a completed 250-update run with 250 reward and success metrics")
    train_exit = RAW / "train_exit.json"
    if json.loads(train_exit.read_text(encoding="utf-8"))["exit_code"] != 0:
        raise ValueError("Training exit was not zero")

    trials = []
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in (log, train_exit)}
    for seed in range(8, 13):
        path = RAW / f"seed{seed}_summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary["episodes_requested"] != 1 or len(summary["episode_results"]) != 1:
            raise ValueError(f"Unexpected episode count for seed {seed}")
        exit_path = RAW / f"seed{seed}_exit.txt"
        exit_code = int(exit_path.read_text(encoding="utf-8").strip())
        if exit_code != 0:
            raise ValueError(f"Playback seed {seed} exited {exit_code}")
        episode = summary["episode_results"][0]
        trials.append({
            "seed": seed,
            "exit_code": exit_code,
            "status": episode["status"],
            "termination": episode["termination"],
            "duration_s": episode["duration_s"],
            "physics_samples": episode["physics_samples"],
            "contact_coverage": episode["contact_coverage"],
            "max_cube_lift_m": episode["max_cube_lift_m"],
            "max_finger_contact_n": episode["max_finger_contact_n"],
            "first_bilateral_contact_s": episode["first_contact_time_s"],
            "best_sustained_lift_contact_s": episode["best_sustained_lift_contact_s"],
            "physical_lift_pass": episode["physical_lift_pass"],
            "drop_count": episode["drop_count"],
        })
        hashes[str(path.relative_to(ROOT))] = sha256(path)
        hashes[str(exit_path.relative_to(ROOT))] = sha256(exit_path)

    trace = RAW / "seed8.jsonl"
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    actions = [value for row in rows if row["record_type"] == "policy_step_start"
               for value in row["policy_action"]]
    physics = [row for row in rows if row["record_type"] == "physics_step"]
    if len(actions) != 360 * 9 or len(physics) != 1440:
        raise ValueError("Expected a complete seed-8 policy and physics trace")
    hashes[str(trace.relative_to(ROOT))] = sha256(trace)

    return {
        "scope": "near-table first rung; training reward and separate physical pickup screen",
        "task": "FRI-Franka-Lift-NearTable",
        "training": {"seed": 8, "environments": 256, "rollout_steps": 32, "updates": 250,
                     "transitions": 2048000, "exit_code": 0, "reward_by_update": reward,
                     "task_success_rate_by_update": success, "checkpoint": "model_249.pt",
                     "checkpoint_sha256": MODEL_SHA256},
        "physical_screen": {"seeds": list(range(8, 13)), "seconds_per_episode": 12,
                            "lift_threshold_m": 0.08, "bilateral_force_threshold_n": 0.1,
                            "dwell_threshold_s": 0.5, "trials": trials,
                            "completed": sum(t["status"] == "completed" for t in trials),
                            "passes": sum(t["status"] == "completed" and t["physical_lift_pass"] for t in trials),
                            "recorder_parity_caveat": "Physical screen does not establish parity with the scripted recorder."},
        "seed8_diagnostic": {
            "max_abs_raw_action": max(map(abs, actions)),
            "raw_action_components_abs_gt_1": sum(abs(value) > 1 for value in actions),
            "raw_action_components_total": len(actions),
            "min_hand_z_m": min(row["hand_pose_w"][2] for row in physics),
            "max_hand_z_m": max(row["hand_pose_w"][2] for row in physics),
            "initial_hand_z_m": rows[0]["hand_pose_w"][2],
        },
        "source_sha256": hashes,
    }


def plot(data):
    rewards = data["training"]["reward_by_update"]
    trials = data["physical_screen"]["trials"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
    axes[0].plot(range(1, len(rewards) + 1), rewards, color="#3273a8", linewidth=2)
    axes[0].set(xlabel="PPO update", ylabel="Mean training reward", title="Training reward rose")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].text(.03, .94, f"{rewards[0]:.2f} → {rewards[-1]:.2f}",
                 transform=axes[0].transAxes, va="top", fontsize=10)

    seeds = [trial["seed"] for trial in trials]
    rises = [trial["max_cube_lift_m"] for trial in trials]
    axes[1].bar(seeds, rises, color="#d06c58", width=.7)
    axes[1].axhline(.08, color="#383838", linestyle="--", linewidth=1.4,
                    label="Pickup rise threshold (0.08 m)")
    axes[1].set(xticks=seeds, ylim=(0, .105), xlabel="Fresh-process seed",
                ylabel="Maximum cube rise (m)", title="Physical screen: 0/5 pickups")
    axes[1].grid(axis="y", alpha=.2)
    axes[1].set_axisbelow(True)
    axes[1].legend(frameon=False, loc="upper right", fontsize=8)
    for seed, rise in zip(seeds, rises):
        axes[1].text(seed, rise + .003, f"{rise:.3f}", ha="center", fontsize=8)

    fig.suptitle("Near-table PPO, 2.048M transitions: contact without pickup", fontsize=13)
    fig.text(.5, .005, "Different metrics: reward is optimization feedback; pickup requires cube rise + sustained bilateral contact.",
             ha="center", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, .035, 1, .94))
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if args.collect:
        data = collect()
        RESULT.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    else:
        data = json.loads(RESULT.read_text(encoding="utf-8"))
    plot(data)
    print(FIGURE)


if __name__ == "__main__":
    main()
