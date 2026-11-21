import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

repo = Path(__file__).parents[1] / "FRI-BCI-ML-RL-Decoder"
raw = repo / "outputs/isaac/candidate-integration-20261102"
target = repo / "results/nvidia_candidate_2026-11-02"
target.mkdir(exist_ok=True)
sys.path.insert(0, str(repo / "simulation/candidates/nvidia_franka_cube"))
from verify_saved import verify

for name, source in {
    "evaluation_summary.json": raw / "eval-5ep/summary.json",
    "evaluation_manifest.json": raw / "eval-5ep/manifest.json",
    "startup_summary.json": raw / "startup-04/summary.json",
    "preparation.json": raw / "compat-v3/PREPARATION.json",
    "cpu_probe.json": raw / "probe-01/summary.json",
}.items():
    shutil.copyfile(source, target / name)

verified = verify(raw / "eval-5ep")
(target / "serialized_rescore.json").write_text(json.dumps(verified, indent=2) + "\n")
with (target / "episodes.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["episode", "physical_lift_pass", "max_post_action_rise_m", "best_sustain_s", "max_left_force_n", "max_right_force_n", "first_bilateral_contact_s", "max_hand_displacement_m"])
    for row in verified["episodes"]:
        writer.writerow([row["episode"], row["physical_lift_pass"], row["max_cube_rise_m"], row["best_sampled_sustain_s"], *row["max_finger_contact_n"], row["first_bilateral_contact_s"], row["max_hand_displacement_m"]])

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

checksums = {
    "raw_backup_archive": {"relative_path": "outputs/isaac/candidate-integration-20261102.tar.gz",
                           "sha256": digest(raw.with_suffix(".tar.gz")), "verified_against_brev": True},
    "raw_run_files": [{"path": str(p.relative_to(raw)), "sha256": digest(p)}
                      for p in sorted(raw.rglob("*")) if p.is_file() and "worker-wip" not in p.parts],
    "integration_files": [{"path": str(p.relative_to(repo)), "sha256": digest(p)}
                          for p in sorted((repo / "simulation/candidates/nvidia_franka_cube").glob("*.py"))],
}
(target / "sha256.json").write_text(json.dumps(checksums, indent=2) + "\n")
print(json.dumps({"successes": verified["successes"], "backup_sha256": checksums["raw_backup_archive"]["sha256"], "published_directory": str(target)}))
