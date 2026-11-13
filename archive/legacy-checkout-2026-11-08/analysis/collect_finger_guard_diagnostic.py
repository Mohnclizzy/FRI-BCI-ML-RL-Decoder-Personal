"""Summarize two selected inference-only finger-target-clamp checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "outputs/isaac/near_table_progress_eval_local"
BASELINE = ROOT / "results/near_table_progress_screen_2026-11-02.json"
OUTPUT = ROOT / "results/near_table_finger_guard_diagnostic_2026-11-02.json"
RUNS = {8: "near-table-finger-guard-eval-seed8-20261102T2350Z",
        11: "near-table-finger-guard-eval-seed11-20261102T2352Z"}
REMOTE_HASHES = {
    8: ("ae613682e33187db10b0a47f67402e62ff99dfd01760e7478eb155814a986f76",
        "0fa48b149fc13dd608050ac4e28773364acbb0391452cd9d2b4dfb488885b04a",
        "4152287cc4368f96a3c838afcf4dd44e3def5c6f28768598bbc7da9ab113e9da",
        "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa"),
    11: ("1bc36a7f5b88b8e21dd930fc5ceeb4e135a3a931d7a10379d48751f466fbf9cd",
         "4de0d018aee4b34d72c2a9662e36d12420911d744097446bfbc127790c1f2c92",
         "e0df7038a5d76d21364111d6c6cffbb158d6d2b2a7bb0685c90a50071572de04",
         "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original = json.loads(BASELINE.read_text(encoding="utf-8"))
    baseline = {trial["seed"]: trial for trial in original["evaluation"]["trials"]}
    reset_path = RAW / "near-table-finger-guard-reset-20261102T2345Z/report.json"
    if sha256(reset_path) != "05f10305116544c01fa08ddb50ae6ddb6992df82ac30ed0b5684bf9b34403f06":
        raise ValueError("Guard reset report differs from Brev")
    reset = json.loads(reset_path.read_text(encoding="utf-8"))
    if not reset["pass"] or not all(abs(x - .02) < 1e-6 for x in reset["finger_joint_targets_m"]):
        raise ValueError("Guard reset check did not prove a 0.02 m finger target")
    provenance = ROOT / "outputs/isaac/progress_eval_provenance"
    source_files = (
        ("finger_guard_remote.py", "simulation/src/fri_bci_sim/tasks/finger_guard.py",
         "91c8bed068277f8bd14e72375cc9378bb0aacf0bed59d22d6ec16bd41402fba4"),
        ("tasks_init_guard_remote.py", "simulation/src/fri_bci_sim/tasks/__init__.py",
         "7243b89004790ae5c826934a4f9b5608ff2227dfd2397182ff0151e28605b372"),
        ("inspect_finger_guard_remote.py", "simulation/scripts/inspect_finger_guard.py",
         "dc081ce2738eedc1dd270aa15d4812e775aeb94979f38602d55f9641f6074d97"),
    )
    for remote_name, published_name, expected_hash in source_files:
        remote, published = provenance / remote_name, ROOT / published_name
        if sha256(remote) != expected_hash:
            raise ValueError(f"Executed guard source hash changed: {remote_name}")
        if remote.read_bytes().replace(b"\r\n", b"\n") != published.read_bytes().replace(b"\r\n", b"\n"):
            raise ValueError(f"Executed guard source differs: {published_name}")
    pairs, hashes = [], {str(reset_path.relative_to(ROOT)): sha256(reset_path)}
    for seed, run_name in RUNS.items():
        directory = RAW / run_name
        names = ("ppo-telemetry.jsonl", "ppo-telemetry.summary.json", "console.log", "exit.txt")
        for name, expected in zip(names, REMOTE_HASHES[seed]):
            path = directory / name
            actual = sha256(path)
            if actual != expected:
                raise ValueError(f"Local guard copy differs from Brev: {path}")
            hashes[str(path.relative_to(ROOT))] = actual
        if int((directory / "exit.txt").read_text(encoding="utf-8").strip()) != 0:
            raise ValueError(f"Guarded seed {seed} did not exit cleanly")
        summary = json.loads((directory / "ppo-telemetry.summary.json").read_text(encoding="utf-8"))
        if summary["episodes_completed"] != 1 or summary["incomplete_episodes"]:
            raise ValueError(f"Guarded seed {seed} has incomplete episode coverage")
        episode = summary["episode_results"][0]
        rows = [json.loads(line) for line in (directory / "ppo-telemetry.jsonl").read_text(encoding="utf-8").splitlines()]
        physics = [row for row in rows if row["record_type"] == "physics_step"]
        terminals = [row for row in rows if row["record_type"] == "terminal_pre_reset"]
        if len(physics) != episode["physics_samples"] or len(terminals) != 1:
            raise ValueError(f"Guarded seed {seed} is missing physics/terminal state")
        names = ("panda_finger_joint1", "panda_finger_joint2")
        min_targets = [min(row["joint_pos_target"][name] for row in physics) for name in names]
        if min(min_targets) < .02 - 1e-6:
            raise ValueError(f"Guarded seed {seed} violated target floor")
        peak = max(physics, key=lambda row: max(row["finger_contact_n"]))
        prev = baseline[seed]
        pairs.append({
            "seed": seed,
            "selection": "previous failure" if seed == 8 else "previous acquisition without recorded drop",
            "baseline": {"duration_s": prev["duration_s"], "max_cube_lift_m": prev["max_cube_lift_m"],
                         "max_finger_contact_n": prev["max_finger_contact_n"],
                         "physical_lift_pass": prev["physical_lift_pass"],
                         "terminal_cube_height_m": prev["terminal_cube_height_m"]},
            "guarded": {"duration_s": episode["duration_s"],
                        "max_cube_lift_m": episode["max_cube_lift_m"],
                        "max_finger_contact_n": episode["max_finger_contact_n"],
                        "physical_lift_pass": episode["physical_lift_pass"],
                        "termination": episode["termination"],
                        "terminal_cube_height_m": terminals[0]["cube_height_m"],
                        "physics_samples": episode["physics_samples"],
                        "contact_coverage": episode["contact_coverage"],
                        "physics_step_gaps": episode["physics_step_gaps"],
                        "minimum_finger_joint_targets_m": min_targets,
                        "peak_force_time_s": peak["episode_time_s"],
                        "peak_force_finger_joint_targets_m": [peak["joint_pos_target"][name] for name in names]},
        })

    report = {
        "scope": "selected paired inference diagnostic, not a five-start policy evaluation",
        "checkpoint_sha256": original["checkpoint_sha256"],
        "baseline_task": original["task"],
        "guarded_task": "FRI-Franka-Lift-NearTable-Progress-FingerGuard",
        "training_under_guard": False,
        "changed_control": "Absolute position targets of both finger joints clipped to [0.02, 0.04] m after the original relative action mapping; other task config inherited",
        "guard_reset_check": reset,
        "pairs": pairs,
        "executed_guard_source_sha256": "91c8bed068277f8bd14e72375cc9378bb0aacf0bed59d22d6ec16bd41402fba4",
        "executed_task_registration_sha256": "7243b89004790ae5c826934a4f9b5608ff2227dfd2397182ff0151e28605b372",
        "executed_reset_script_sha256": "dc081ce2738eedc1dd270aa15d4812e775aeb94979f38602d55f9641f6074d97",
        "executed_source_matches_published_checkout_after_newline_normalization": True,
        "raw_copy_sha256": hashes,
        "finding": "The clamp removed negative finger targets but did not reduce the selected high-force failure; seed 11 lost its previous acquisition. Stop this guard as a candidate fix.",
        "limit": "Seeds 8 and 11 were selected from prior outcomes. This is a diagnostic comparison, not an unbiased success-rate estimate. Force may depend on arm motion and contact geometry; the clamp result does not isolate a unique cause.",
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
