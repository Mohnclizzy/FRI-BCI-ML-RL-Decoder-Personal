"""Collect and plot the fresh-process near-table progress physical screen.

Use --collect with restored ignored Brev output. Default redraws the figure
from the tracked compact record without raw telemetry or Isaac Lab.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "outputs/isaac/near_table_progress_eval_local/near-table-progress-eval-5x12s-seed8-12-20261102T2330Z"
RESULT = ROOT / "results/near_table_progress_screen_2026-11-02.json"
FIRST = ROOT / "results/near_table_stage1_screen_2026-11-02.json"
FIGURE = ROOT / "results/figures/near_table_progress_screen_2026-11-02.png"
TRAIN_BACKUP = ROOT / "outputs/isaac/near_table_progress_eval_local/near-table-progress-256x250-seed8-20261102T2305Z"
CHECKPOINT_SHA256 = "46f68c087111b5f713c33a63b7355f3c4c35eeaba26e4240f778667086b443d6"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect() -> dict:
    trials, hashes = [], {}
    for seed in range(8, 13):
        directory = RAW / f"seed{seed}"
        summary_path = directory / "ppo-telemetry.summary.json"
        trace_path = directory / "ppo-telemetry.jsonl"
        exit_path = directory / "exit.txt"
        console_path = directory / "console.log"
        for path in (summary_path, trace_path, exit_path, console_path):
            hashes[str(path.relative_to(ROOT))] = sha256(path)
        exit_code = int(exit_path.read_text(encoding="utf-8").strip())
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if exit_code or summary["playback_error"] or summary["episodes_completed"] != 1:
            raise ValueError(f"Seed {seed} did not complete cleanly")
        if len(summary["episode_results"]) != 1 or summary["incomplete_episodes"]:
            raise ValueError(f"Seed {seed} has an unexpected episode count")
        episode = summary["episode_results"][0]
        rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
        starts = [r for r in rows if r["record_type"] == "episode_start"]
        terminals = [r for r in rows if r["record_type"] == "terminal_pre_reset"]
        physics = [r for r in rows if r["record_type"] == "physics_step"]
        policy = [r for r in rows if r["record_type"] == "policy_step_start"]
        if len(starts) != 1 or len(terminals) != 1 or len(physics) != episode["physics_samples"]:
            raise ValueError(f"Seed {seed} is missing reset, terminal or physics samples")
        if episode["status"] != "completed" or episode["contact_coverage"] != 1 or episode["physics_step_gaps"]:
            raise ValueError(f"Seed {seed} has incomplete physical coverage")
        if summary["physics_sampling_mode"] != "environment_physics_step":
            raise ValueError(f"Seed {seed} used a different sampling mode")
        actions = [value for step in policy for value in step["policy_action"]]
        peak_force_state = max(physics, key=lambda r: max(r["finger_contact_n"]))
        finger_names = ("panda_finger_joint1", "panda_finger_joint2")
        trials.append({
            "seed": seed,
            "exit_code": exit_code,
            "episode_status": episode["status"],
            "termination": episode["termination"],
            "duration_s": episode["duration_s"],
            "physics_samples": episode["physics_samples"],
            "policy_steps": summary["policy_steps"],
            "contact_coverage": episode["contact_coverage"],
            "physics_step_gaps": episode["physics_step_gaps"],
            "baseline_cube_height_m": episode["baseline_cube_height_m"],
            "max_cube_lift_m": episode["max_cube_lift_m"],
            "terminal_cube_height_m": terminals[0]["cube_height_m"],
            "max_finger_contact_n": episode["max_finger_contact_n"],
            "first_bilateral_contact_s": episode["first_contact_time_s"],
            "first_lift_s": episode["first_lift_time_s"],
            "acquisition_time_s": episode["acquisition_time_s"],
            "best_sustained_lift_contact_s": episode["best_sustained_lift_contact_s"],
            "drop_count": episode["drop_count"],
            "drop_times_s": episode["drop_times_s"],
            "physical_lift_pass": episode["physical_lift_pass"],
            "max_abs_raw_policy_action": max(map(abs, actions)),
            "raw_action_components_abs_gt_1": sum(abs(x) > 1 for x in actions),
            "raw_action_components_total": len(actions),
            "minimum_finger_joint_targets_m": [
                min(row["joint_pos_target"][name] for row in physics) for name in finger_names
            ],
            "peak_force_state": {
                "time_s": peak_force_state["episode_time_s"],
                "cube_height_m": peak_force_state["cube_height_m"],
                "finger_contact_n": peak_force_state["finger_contact_n"],
                "finger_joint_pos_m": peak_force_state["robot_joint_pos"][-2:],
                "finger_joint_target_m": [peak_force_state["joint_pos_target"][name] for name in finger_names],
            },
        })

    first = json.loads(FIRST.read_text(encoding="utf-8"))
    if first["physical_screen"]["seeds"] != [8, 9, 10, 11, 12]:
        raise ValueError("First near-table screen used different seeds")
    source_sha = {
        "tasks/near_table.py": "a4fbdf45eb36a004d5769d15b20f623900f3802cb897882469351189a762d632",
        "tasks/near_table_progress.py": "6184906be8444f90ebab5057bc8aae782c68407d0d61a24876e3d798e2bf2e22",
        "tasks/__init__.py": "b57d943c29820d87835d61bed730553368dfd9a202255fc98dee88173afa5a51",
        "curriculum.py": "f454ba1f3d5b69eebf4909e120bdf0fc24614eb7865f9afbdcf736242de6662f",
        "launch.py": "126d2593ad5f74bfa165edd489f22932656c90e1593ee4e847665c081d6e452d",
        "ppo_telemetry.py": "0148706238cb7173a4e24add1e806f93a35ec32d78d5037e53547cc2655677b2",
    }
    provenance = ROOT / "outputs/isaac/progress_eval_provenance"
    source_pairs = {
        "tasks/near_table.py": ("near_table_remote.py", "simulation/src/fri_bci_sim/tasks/near_table.py"),
        "tasks/near_table_progress.py": ("near_table_progress_remote.py", "simulation/src/fri_bci_sim/tasks/near_table_progress.py"),
        "tasks/__init__.py": ("tasks_init_remote.py", "simulation/src/fri_bci_sim/tasks/__init__.py"),
        "curriculum.py": ("curriculum_remote.py", "simulation/src/fri_bci_sim/curriculum.py"),
        "launch.py": ("launch_remote.py", "simulation/src/fri_bci_sim/launch.py"),
        "ppo_telemetry.py": ("ppo_telemetry_remote.py", "simulation/src/fri_bci_sim/ppo_telemetry.py"),
    }
    for key, (remote_name, local_name) in source_pairs.items():
        remote_path, local_path = provenance / remote_name, ROOT / local_name
        remote_bytes, local_bytes = remote_path.read_bytes(), local_path.read_bytes()
        if sha256(remote_path) != source_sha[key]:
            raise ValueError(f"Executed source hash changed: {key}")
        if remote_bytes.replace(b"\r\n", b"\n") != local_bytes.replace(b"\r\n", b"\n"):
            raise ValueError(f"Executed source content differs from published checkout: {key}")
    training_hashes = {
        "logs/rsl_rl/lift_franka/2026-11-02_23-03-47/model_249.pt": CHECKPOINT_SHA256,
        "logs/rsl_rl/lift_franka/2026-11-02_23-03-47/params/env.yaml": "026b7916f0e6ba2b5b71d67608c7988b3a7013283468ff7c81024448364d1998",
        "logs/rsl_rl/lift_franka/2026-11-02_23-03-47/params/agent.yaml": "112fb63b96ccb7eff7a72047ba302da0fa1f5b257f31b3159622b9df1d8c98d6",
        "console.log": "0d1e53df559981d6b052e2a5fe6d3b3e0555da63c13c311b0b7c8351db0fc955",
        "exit.json": "e89df0eae5ec7db81135aed16463968955c578bcc4f2ec43172eff351f080c17",
    }
    for name, expected in training_hashes.items():
        if sha256(TRAIN_BACKUP / name) != expected:
            raise ValueError(f"Training backup hash differs from Brev: {name}")
    return {
        "scope": "fresh-process physical screen of the second near-table reward variant",
        "task": "FRI-Franka-Lift-NearTable-Progress",
        "training_checkpoint": "/home/ubuntu/FRI-BCI-ML-RL-Decoder-Personal/outputs/isaac/near-table-progress-256x250-seed8-20261102T2305Z/logs/rsl_rl/lift_franka/2026-11-02_23-03-47/model_249.pt",
        "evaluation_run_directory": "/home/ubuntu/FRI-BCI-ML-RL-Decoder-Personal/outputs/isaac/near-table-progress-eval-5x12s-seed8-12-20261102T2330Z",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "isaaclab_commit": "e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4",
        "remote_personal_checkout_head": "b3422a0eef9105c4dcdb430474c6e3f3cf5b8828",
        "source_import_root": "/home/ubuntu/FRI-BCI-ML-RL-Decoder-Personal/simulation/src",
        "executed_source_sha256": source_sha,
        "source_content_matches_published_checkout_after_newline_normalization": True,
        "resolved_training_config_sha256": {
            "env.yaml": "026b7916f0e6ba2b5b71d67608c7988b3a7013283468ff7c81024448364d1998",
            "agent.yaml": "112fb63b96ccb7eff7a72047ba302da0fa1f5b257f31b3159622b9df1d8c98d6",
        },
        "evaluation": {
            "seeds": [8, 9, 10, 11, 12],
            "requested_horizon_s": 12,
            "physics_dt_s": 1 / 120,
            "policy_dt_s": 1 / 30,
            "sampling_mode": "environment_physics_step",
            "lift_threshold_m": 0.08,
            "bilateral_force_threshold_n": 0.1,
            "sustained_contact_threshold_s": 0.5,
            "trials": trials,
            "completed_episodes": sum(t["episode_status"] == "completed" for t in trials),
            "physical_lift_passes": sum(t["physical_lift_pass"] for t in trials),
            "passes_without_recorded_drop": sum(t["physical_lift_pass"] and t["drop_count"] == 0 for t in trials),
            "early_terminations": sum(t["termination"]["terminated"] for t in trials),
        },
        "first_variant_result_sha256": sha256(FIRST),
        "raw_source_sha256": hashes,
        "verified_off_host_copy": {
            "local_ignored_root": str(TRAIN_BACKUP.parent.relative_to(ROOT)),
            "training_key_files_sha256": training_hashes,
            "evaluation_files_sha256": hashes,
            "scope": "Current progress training and five evaluation cases only; not all historical Brev outputs",
        },
        "interpretation_limit": "Five fixed near-grasp starts are diagnostic; high forces, early terminations, drop and recorder/physics parity prevent a reliability or controlled scripted comparison claim.",
    }


def plot(data: dict):
    first = json.loads(FIRST.read_text(encoding="utf-8"))
    baseline = first["physical_screen"]["trials"]
    trials = data["evaluation"]["trials"]
    seeds = np.array([t["seed"] for t in trials])
    x = np.arange(len(seeds))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1.2, 1]})
    axes[0].bar(x - .18, [t["max_cube_lift_m"] for t in baseline], width=.36,
                label="First near-table reward", color="#779eb7")
    axes[0].bar(x + .18, [t["max_cube_lift_m"] for t in trials], width=.36,
                label="Progress reward", color="#d78359")
    axes[0].axhline(.08, color="#333333", linestyle="--", linewidth=1.2,
                    label="Pickup rise threshold")
    axes[0].set(xticks=x, xticklabels=seeds, xlabel="Fresh-process seed",
                ylabel="Maximum cube rise (m)", title="Measured cube rise: 0/5 → 2/5")
    axes[0].grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False, fontsize=8)
    for i, trial in enumerate(trials):
        if trial["termination"]["terminated"]:
            axes[0].annotate("early end", (i + .18, trial["max_cube_lift_m"]),
                             xytext=(0, 5), textcoords="offset points",
                             ha="center", va="bottom", fontsize=8)
        elif trial["drop_count"]:
            axes[0].text(i + .18, trial["max_cube_lift_m"] - .012, "drop",
                         ha="center", va="top", fontsize=8, color="#222222")

    axes[1].bar(x - .18, [t["max_finger_contact_n"][0] for t in trials], width=.36,
                label="Left", color="#6288b3")
    axes[1].bar(x + .18, [t["max_finger_contact_n"][1] for t in trials], width=.36,
                label="Right", color="#b96865")
    axes[1].set(xticks=x, xticklabels=seeds, xlabel="Fresh-process seed",
                ylabel="Peak sampled finger force (N)", title="High, asymmetric contact forces")
    axes[1].grid(axis="y", alpha=.2)
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.set_axisbelow(True)
    fig.suptitle("Near-table reward revision: physical screen, 12 s maximum per start", fontsize=13)
    fig.text(.5, .005, "Two sampled acquisitions; one later dropped. Five starts do not establish reliable pickup or safe force.",
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
