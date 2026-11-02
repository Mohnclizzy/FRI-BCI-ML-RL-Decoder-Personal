"""Plot the precomputed S006-S010 command and Franka comparison."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("outputs/isaac")
RUN = ROOT / "eeg-matched-5p-20261102T054610Z"
PERSONS = [f"S{n:03d}" for n in range(6, 11)]
MODES = ["raw", "confidence", "temporal"]
COLORS = {"raw": "#4863a0", "confidence": "#df8b36", "temporal": "#429b76"}

false_active, detected, lifts = {}, {}, {}
for mode in MODES:
    false_active[mode], detected[mode], lifts[mode] = [], [], []
    for person in PERSONS:
        report = json.loads((ROOT / f"{person}R12-annotation-24s.json").read_text())
        metrics = report["evaluation"]["conditions"][mode]
        false_active[mode].append(metrics["false_activation_fraction_of_rest"] * 100)
        detected[mode].append(metrics["detection_coverage_fraction"] * 100)
        log = RUN / f"{person}R12-{mode}" / "run.log"
        summary = None
        for line in reversed(log.read_text(errors="replace").splitlines()):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "physical_lift_pass" in row:
                summary = row
                break
        if summary is None:
            raise ValueError(f"Missing simulator summary: {log}")
        lifts[mode].append(int(summary["physical_lift_pass"]))

fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), constrained_layout=True)
metrics_to_plot = [
    (false_active, "Rest time with active command", "% of annotated rest", 100),
    (detected, "Imagery bouts detected", "% of 3 bouts", 100),
    (lifts, "Measured sustained table lifts", "Successful episodes (of 5)", 5),
]
for ax, (data, title, ylabel, ymax) in zip(axes, metrics_to_plot):
    x = np.arange(len(MODES))
    heights = [sum(data[m]) if ymax == 5 else np.mean(data[m]) for m in MODES]
    ax.bar(x, heights, color=[COLORS[m] for m in MODES], width=.62, zorder=2)
    for i, mode in enumerate(MODES):
        jitter = np.linspace(-.20, .20, len(data[mode]))
        ax.scatter(x[i] + jitter, data[mode], s=32, color="#20252b", edgecolor="white",
                   linewidth=.5, zorder=3)
        label = f"{heights[i]:.1f}" if ymax == 100 else f"{int(heights[i])}/5"
        offset = ymax * (.08 if ymax == 100 else .035)
        ax.text(x[i], heights[i] + offset, label, ha="center", va="bottom", fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": .5})
    ax.set_xticks(x, ["Raw", "Confidence", "Temporal"])
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ymax * 1.17)
    ax.grid(axis="y", alpha=.22, zorder=0)

fig.suptitle("Five-person recorded EEG replay to simulated Franka (exploratory)", fontsize=13)
fig.text(.5, -.015, "S006-S010 · run 12 · fixed scene · 24 s budget · dots are participants · lift = ≥0.08 m rise + ≥0.5 s bilateral contact",
         ha="center", fontsize=8.5)
output = RUN / "S006-S010_matched_results.png"
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=180, bbox_inches="tight")
print(output)
