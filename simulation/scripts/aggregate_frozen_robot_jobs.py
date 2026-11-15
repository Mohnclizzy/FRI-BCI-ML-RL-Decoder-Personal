"""Aggregate and exposure-join a completed frozen robot replay batch."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fri_bci_sim.robomimic_harness import join_horizon_exposure_and_telemetry  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--robot-results", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/local_scratch_2026-11-11"))
    args = parser.parse_args()
    study, robot = args.study, args.robot_results
    jobs = json.loads((study / "robot_jobs.json").read_text())["jobs"]
    progress = json.loads((robot / "progress.json").read_text())
    if progress["count"] != 270 or len(progress["completed"]) != 270:
        raise ValueError("Robot batch incomplete")
    if len(jobs) != 270:
        raise ValueError("Frozen job manifest is not 270 rows")
    rows = []
    for job in jobs:
        key = "S%03d_seed%d_%s" % (job["subject"], job["seed"], job["condition"])
        folder = robot / key
        summary = folder / "summary.json"
        telemetry = folder / "telemetry.jsonl"
        if not summary.is_file() or not telemetry.is_file():
            raise ValueError("Missing completed robot artifacts: " + key)
        if job["condition"] in ("selected", "raw"):
            condition = "eeg"
            decisions = study / job["trace"].replace("\\", "/")
            gate = "raw"
            schedule = None
        elif job["harness_condition"] == "enable-schedule":
            condition = "enable-schedule"
            decisions, gate = None, None
            schedule = json.loads((study / job["schedule"].replace("\\", "/")).read_text())["spans"]
        else:
            condition = job["harness_condition"]
            decisions, gate, schedule = None, None, None
        cache = args.cache_dir / "cache" / "7d3de6fbf853d44a" / ("S%03dR08.json" % job["subject"])
        intervals = json.loads(cache.read_text())["intervals"]
        joined = join_horizon_exposure_and_telemetry(
            condition=condition, decisions_path=decisions, gate=gate, schedule_spans=schedule,
            telemetry_path=telemetry, summary_path=summary, intervals=intervals,
        )
        physical = json.loads(summary.read_text())
        rows.append(dict(job=job, physical=physical, joined=joined))
    json_path = robot / "joined_metrics.json"
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    metrics = ["terminal_retained", "first_acquisition_step", "requested_enable_steps", "executed_policy_steps",
               "post_acquisition_latch_steps", "requested_rest_enable_fraction",
               "pre_latch_executed_policy_during_rest_s", "total_rest_eef_path_length_m",
               "pre_latch_rest_eef_path_length_m", "total_rest_moving_duration_s"]
    groups = []
    for condition in ("selected", "raw", "always-enable", "always-hold", "contiguous", "shifted"):
        group = [r for r in rows if r["job"]["condition"] == condition]
        data = {}
        for metric in metrics:
            values = []
            for row in group:
                key = metric
                value = row["physical"].get(key) if key in row["physical"] else row["joined"].get(key)
                if metric == "requested_rest_enable_fraction":
                    value = (row["joined"].get("annotation_alignment_20s") or {}).get(metric)
                if metric.startswith("pre_latch_") or metric in ("total_rest_eef_path_length_m", "pre_latch_rest_eef_path_length_m", "total_rest_moving_duration_s"):
                    value = (((row["joined"].get("robot_telemetry_alignment_20s") or {}).get("rest_motion_on_horizon") or {}).get(metric))
                if value is not None:
                    values.append(float(value))
            data[metric] = dict(mean=float(np.mean(values)) if values else None, n=len(values))
        groups.append(dict(condition=condition, n=len(group), retention_count=sum(r["physical"]["terminal_retained"] for r in group), metrics=data))
    # Pair conditions on identical participant/robot-seed starts. Then average
    # robot seeds within participant so the cohort summary still has N=15.
    by_start = {(r["job"]["subject"], r["job"]["seed"], r["job"]["condition"]): r for r in rows}
    paired_metrics = ("terminal_retained", "first_acquisition_step", "requested_enable_steps",
                      "executed_policy_steps", "post_acquisition_latch_steps",
                      "requested_rest_enable_fraction", "total_rest_eef_path_length_m",
                      "total_rest_moving_duration_s")
    paired = []
    for condition in ("raw", "always-enable", "always-hold", "contiguous", "shifted"):
        delta_by_subject = {metric: {} for metric in paired_metrics}
        for subject in range(1, 16):
            for seed in range(3):
                selected = by_start[(subject, seed, "selected")]
                comparison = by_start[(subject, seed, condition)]
                for metric in paired_metrics:
                    def metric_value(row):
                        if metric in ("requested_rest_enable_fraction", "total_rest_eef_path_length_m",
                                      "total_rest_moving_duration_s"):
                            align = row["joined"].get("annotation_alignment_20s", {})
                            telem = row["joined"].get("robot_telemetry_alignment_20s", {})
                            rest = telem.get("rest_motion_on_horizon", {})
                            if metric == "requested_rest_enable_fraction": return align.get(metric)
                            return rest.get(metric)
                        value = row["physical"].get(metric)
                        if value is None: value = row["joined"].get(metric)
                        return value
                    a, b = metric_value(selected), metric_value(comparison)
                    if a is not None and b is not None:
                        delta_by_subject[metric].setdefault(subject, []).append(float(a) - float(b))
        summaries = {}
        for metric, subjects in delta_by_subject.items():
            person_means = [float(np.mean(v)) for v in subjects.values() if v]
            summaries[metric] = dict(selected_minus_comparison_mean=float(np.mean(person_means)) if person_means else None,
                                     participants=len(person_means), participant_means=person_means)
        paired.append(dict(comparison=condition, pairs=45, summaries=summaries))
    summary = dict(n_rollouts=len(rows), groups=groups, paired_selected_minus_comparison=paired,
                   aggregation_note="Paired on participant and robot seed; cohort differences average the three robot-seed differences within each of 15 participants.",
                   joined_metrics_sha256=__import__("hashlib").sha256(json_path.read_bytes()).hexdigest())
    (robot / "aggregate.json").write_text(json.dumps(summary, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [r["condition"] for r in groups]
    retained = [r["retention_count"] / r["n"] for r in groups]
    rest_path = [r["metrics"]["total_rest_eef_path_length_m"]["mean"] or 0.0 for r in groups]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].bar(labels, retained, color="#53789e")
    axes[0].set(ylabel="Terminal retention fraction", ylim=(0, 1), title="Robot outcome")
    axes[1].bar(labels, rest_path, color="#ba7652")
    axes[1].set(ylabel="Mean end-effector path during REST (m)", title="REST-associated motion")
    for ax in axes:
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=.25)
    fig.savefig(robot / "robot_outcomes.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
