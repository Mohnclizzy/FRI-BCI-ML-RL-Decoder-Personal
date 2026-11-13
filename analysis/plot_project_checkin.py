"""Collect existing evidence and plot descriptive EEG/robot check-in figures.

Usage: python analysis/plot_project_checkin.py --collect
       python analysis/plot_project_checkin.py

The collect mode needs the ignored outputs restored locally. Plotting alone uses
the tracked compact JSON, so the figures can be regenerated from a Git checkout.
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
DATA = ROOT / "results/project_checkin_plot_data_2026-11-02.json"
FIGURES = ROOT / "results/figures"


def collect() -> dict:
    sources = {}

    def load(path: str):
        file = ROOT / path
        sources[path] = hashlib.sha256(file.read_bytes()).hexdigest()
        return json.loads(file.read_text(encoding="utf-8"))

    decoder = load("outputs/decoder_validation_001/confirmation_metrics.json")
    commands = []
    for subject in range(6, 11):
        report = load(f"outputs/isaac/S{subject:03d}R12-annotation-24s.json")
        if report["annotation_source"]["subject"] != subject:
            raise ValueError("EEG subject/source mismatch")
        for gate in ("raw", "confidence", "temporal"):
            row = report["evaluation"]["conditions"][gate]
            commands.append({
                "subject": subject, "gate": gate,
                "rest_false_active_pct": 100 * row["false_activation_fraction_of_rest"],
                "imagery_detected_pct": 100 * row["detection_coverage_fraction"],
                "detected_bouts": row["detected_bouts"],
                "imagery_bouts": row["imagery_bouts"],
                "active_duration_s": row["active_duration_s"],
            })

    scripted = []
    for condition in ("scripted-positive", "scripted-pause"):
        path = ("outputs/isaac/eeg-matched-5p-preroll-controls-"
                f"20261102T1500Z-b120s/{condition}/run.log")
        file = ROOT / path
        sources[path] = hashlib.sha256(file.read_bytes()).hexdigest()
        summaries = []
        for line in file.read_text(errors="replace").splitlines():
            if line.startswith('{"output"'):
                entry = json.loads(line)
                if "physical_lift_pass" in entry:
                    summaries.append(entry)
        if len(summaries) != 1 or not summaries[0]["physical_lift_pass"]:
            raise ValueError(f"Missing passing {condition} summary")
        row = summaries[0]
        scripted.append({"condition": condition, "max_cube_rise_m":
                         row["max_cube_z_m"] - row["baseline_cube_z_m"],
                         "physical_lift_pass": True})

    ppo = load("results/ppo_approach_handoff_2026-11-02.json")
    candidate = load("results/nvidia_candidate_2026-11-02/reset_diagnostic.json")
    if ppo["evaluation"]["episodes_completed"] != 5 or candidate["valid_episodes"] != 5:
        raise ValueError("Expected five complete physical screens")
    if not candidate["all_start_continuity_pass"] or not candidate["all_contact_coverage_complete"]:
        raise ValueError("Candidate screen has invalid starts or contact coverage")
    interrupted = load("outputs/isaac/eeg-matched-5p-corrected-20261102T1502Z-b120s/STOPPED_BY_USER.json")
    if not interrupted["interrupted_batch"]:
        raise ValueError("Corrected EEG batch is not marked incomplete")

    return {
        "scope": "descriptive check-in; different robot conditions are not a controlled comparison",
        "source_sha256": sources,
        "decoder": {
            "subjects": "S006-S010 calibrated on runs 4/8, held-out run 12",
            "n_windows": decoder["individual"]["n_windows"],
            "pooled_balanced_accuracy_pct": 100 * decoder["pooled"]["balanced_accuracy"],
            "individual_balanced_accuracy_pct": 100 * decoder["individual"]["balanced_accuracy"],
            "individual_rest_false_activation_pct": 100 * decoder["individual"]["rest_false_activation_fraction"],
            "per_subject": [
                {"subject": subject,
                 "pooled_balanced_accuracy_pct": 100 * decoder["per_subject"][str(subject)]["pooled"]["balanced_accuracy"],
                 "individual_balanced_accuracy_pct": 100 * decoder["per_subject"][str(subject)]["individual"]["balanced_accuracy"],
                 "pooled_rest_false_activation_pct": 100 * decoder["per_subject"][str(subject)]["pooled"]["rest_false_activation_fraction"],
                 "individual_rest_false_activation_pct": 100 * decoder["per_subject"][str(subject)]["individual"]["rest_false_activation_fraction"]}
                for subject in range(6, 11)
            ],
        },
        "commands_24s": commands,
        "scripted_controls": scripted,
        "project_ppo_table": [{"episode": row["episode"],
                               "max_cube_rise_m": row["max_cube_lift_m"],
                               "physical_lift_pass": row["physical_lift_pass"]}
                              for row in ppo["evaluation"]["episodes"]],
        "nvidia_candidate_fresh": [{"seed": row["seed"],
                                    "max_cube_rise_m": row["max_post_action_rise_m"],
                                    "physical_lift_pass": row["physical_lift_pass"],
                                    "duration_s": row["duration_s"],
                                    "termination": row["termination"]}
                                   for row in candidate["episodes"]],
        "corrected_eeg_physical_batch_complete": False,
    }


def eeg_figure(data: dict):
    commands = data["commands_24s"]
    gates = ("raw", "confidence", "temporal")
    colors = ("#536b91", "#8d75a7", "#c0825b")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7), sharey=True,
                             constrained_layout=True)
    metrics = (("rest_false_active_pct", "REST time falsely active ↓"),
               ("imagery_detected_pct", "Imagery bouts detected ↑"))
    for ax, (key, title) in zip(axes, metrics):
        for i, (gate, color) in enumerate(zip(gates, colors)):
            vals = [r[key] for r in commands if r["gate"] == gate]
            if len(vals) != 5:
                raise ValueError("Missing EEG participant result")
            mean = float(np.mean(vals))
            ax.bar(i, mean, color=color, alpha=.52, width=.68)
            ax.scatter(np.full(5, i) + np.linspace(-.18, .18, 5), vals,
                       color=color, edgecolor="white", linewidth=.7, s=43, zorder=3)
            ax.text(i, mean + 2.5, f"{mean:.1f}%", ha="center", fontsize=10)
        ax.set_xticks(range(3), ["Raw", "Confidence", "Temporal"])
        ax.set_ylim(0, 106)
        ax.set_title(title, loc="left", fontsize=12)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Percent of annotated time / bouts")
    fig.suptitle("Recorded EEG command tradeoff · 5 people, first 24 s",
                 x=.04, ha="left", fontsize=14)
    fig.supxlabel("Bars: participant mean · dots: individual participants · exploratory offline replay",
                  fontsize=9)
    return fig


def decoder_figure(data: dict):
    rows = data["decoder"]["per_subject"]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), sharey=True,
                             constrained_layout=True)
    for ax, (key, title) in zip(axes, (
        ("balanced_accuracy_pct", "Balanced accuracy ↑"),
        ("rest_false_activation_pct", "REST windows falsely active ↓"),
    )):
        pooled = [r[f"pooled_{key}"] for r in rows]
        individual = [r[f"individual_{key}"] for r in rows]
        ax.bar(x - .17, pooled, width=.34, label="Pooled CSP/LDA", color="#536b91")
        ax.bar(x + .17, individual, width=.34, label="Per-person CSP/LDA", color="#37856c")
        ax.set_xticks(x, [f"S{r['subject']:03d}" for r in rows])
        ax.set_ylim(0, 105)
        ax.set_title(title, fontsize=12, loc="left")
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        if key == "balanced_accuracy_pct":
            ax.axhline(50, color="#b66f3e", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Percent of held-out windows")
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    fig.suptitle("EEG decoder confirmation · five calibrated people",
                 x=.04, ha="left", fontsize=14)
    fig.supxlabel("Runs 4/8 for training or calibration; run 12 held out. Dashed line: 50% balanced-accuracy reference.",
                  fontsize=9)
    return fig


def robot_figure(data: dict):
    groups = (
        ("Scripted controls", "fixed scene · 24/34 s",
         [r["max_cube_rise_m"] for r in data["scripted_controls"]],
         [r["physical_lift_pass"] for r in data["scripted_controls"]],
         ["positive", "paused"]),
        ("Project PPO table", "Approach · 12 s",
         [r["max_cube_rise_m"] for r in data["project_ppo_table"]],
         [r["physical_lift_pass"] for r in data["project_ppo_table"]],
         [str(r["episode"]) for r in data["project_ppo_table"]]),
        ("NVIDIA candidate", "own scene · ≤3 s · fresh process",
         [r["max_cube_rise_m"] for r in data["nvidia_candidate_fresh"]],
         [r["physical_lift_pass"] for r in data["nvidia_candidate_fresh"]],
         [str(r["seed"]) for r in data["nvidia_candidate_fresh"]]),
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.8), sharey=True,
                             gridspec_kw={"width_ratios": [2, 3, 3]},
                             constrained_layout=True)
    for ax, (title, subtitle, rises, passes, labels) in zip(axes, groups):
        x = np.arange(len(rises))
        ax.bar(x, rises, color=["#37856c" if success else "#718198" for success in passes],
               width=.65)
        ax.axhline(.08, color="#b66f3e", linestyle="--", linewidth=1.2)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, .64)
        ax.set_title(f"{title}\n{subtitle}", fontsize=11)
        ax.grid(axis="y", alpha=.16)
        ax.set_axisbelow(True)
        ax.text(.5, .91, f"{sum(passes)}/{len(passes)} sampled passes",
                transform=ax.transAxes, ha="center", fontsize=10,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9})
        if title == "Project PPO table":
            ax.text(.5, .17, "all rises ≤0.00023 m", transform=ax.transAxes,
                    ha="center", fontsize=9)
    axes[0].set_ylabel("Maximum cube rise (m)")
    fig.suptitle("Robot physical screens · distinct conditions",
                 x=.02, ha="left", fontsize=14)
    fig.supxlabel("Dashed line: 0.08 m height threshold; a pass also needs bilateral force for ≥0.5 s.\n"
                  "Different scenes, resets, controllers and horizons: these panels are not a controlled comparison.",
                  fontsize=9)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true", help="refresh compact data from restored raw outputs")
    args = parser.parse_args()
    if args.collect:
        data = collect()
        DATA.parent.mkdir(parents=True, exist_ok=True)
        DATA.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    else:
        data = json.loads(DATA.read_text(encoding="utf-8"))
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name, figure in (("eeg_decoder_confirmation_2026-11-02", decoder_figure(data)),
                         ("eeg_command_tradeoff_2026-11-02", eeg_figure(data)),
                         ("robot_physical_screens_2026-11-02", robot_figure(data))):
        figure.savefig(FIGURES / f"{name}.png", dpi=180)
        plt.close(figure)
        print(FIGURES / f"{name}.png")


if __name__ == "__main__":
    main()
