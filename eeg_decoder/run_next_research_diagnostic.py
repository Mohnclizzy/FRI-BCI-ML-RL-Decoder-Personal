"""Reproduce the frozen stream-interior/transition diagnostic for the 2026 follow-up."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eeg_decoder.command_study_metrics import analyze_transition_vs_interior  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, default=ROOT / "results/bounded_eeg_robot_2026-11-14")
    parser.add_argument("--output", type=Path, default=ROOT / "results/next_research_diagnostic_2026-11-14")
    args = parser.parse_args()
    study, output = args.study, args.output
    output.mkdir(parents=True, exist_ok=True)
    decoders = {"eegnet": [42, 123, 2026], "csp": [None], "riemannian": [None]}
    rows = []
    for subject in range(1, 16):
        intervals = json.loads((study / "scores" / f"S{subject:03d}_annotations.json").read_text())
        for decoder, seeds in decoders.items():
            analyses = []
            for seed in seeds:
                label = "None" if seed is None else str(seed)
                path = study / "scores" / f"{decoder}_seed{label}_S{subject:03d}.npz"
                if not path.is_file():
                    raise FileNotFoundError(f"Required frozen score stream missing: {path}")
                with np.load(path) as saved:
                    analyses.append(analyze_transition_vs_interior(saved["probabilities"], saved["times"], intervals))
            for category in ("pure_rest_interior", "transition_imagery_to_rest", "pure_imagery_interior", "transition_rest_to_imagery"):
                entries = [a[category] for a in analyses]
                for metric in ("mean", "q50", "q80", "q90"):
                    vals = [e[metric] for e in entries if e[metric] is not None]
                    rows.append({"subject": subject, "decoder": decoder, "category": category, "metric": metric,
                                 "value": float(np.mean(vals)) if vals else None,
                                 "window_count_mean_across_training_seeds": float(np.mean([e["count"] for e in entries]))})
            for metric in ("post_imagery_rest_max_in_transition_fraction", "mean_episode_max_all_s",
                           "mean_episode_max_pure_interior_only", "mean_peak_offset_from_rest_onset_s"):
                vals = [a[metric] for a in analyses if a[metric] is not None]
                rows.append({"subject": subject, "decoder": decoder, "category": "post_imagery_rest_episode_maxima",
                             "metric": metric, "value": float(np.mean(vals)) if vals else None,
                             "window_count_mean_across_training_seeds": None})
    with (output / "participant_diagnostic.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    summary = {"study": str(study), "subjects": list(range(1, 16)),
               "seed_rule": "Analyze all three EEGNet seeds, average metric within participant, then summarize N=15 participants; classical models have one fit each.",
               "transition_definition": "A two-second trailing score window is transitional if it crosses a labeled interval boundary; interval labels only classify predictions for offline evaluation.",
               "decoders": {}}
    for decoder in decoders:
        summary["decoders"][decoder] = {}
        for cat in ("pure_rest_interior", "transition_imagery_to_rest", "pure_imagery_interior", "transition_rest_to_imagery"):
            values = [r["value"] for r in rows if r["decoder"] == decoder and r["category"] == cat and r["metric"] == "mean"]
            summary["decoders"][decoder][cat] = {"participant_mean": float(np.mean(values)),
                "participant_sd": float(np.std(values, ddof=1)), "n_participants": len(values)}
        vals = [r["value"] for r in rows if r["decoder"] == decoder and r["metric"] == "post_imagery_rest_max_in_transition_fraction"]
        summary["decoders"][decoder]["post_imagery_rest_max_in_transition_fraction"] = {
            "participant_mean": float(np.mean(vals)), "participant_sd": float(np.std(vals, ddof=1)),
            "n_participants": len(vals)}
    (output / "diagnostic.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cats = ["pure_rest_interior", "transition_imagery_to_rest", "pure_imagery_interior", "transition_rest_to_imagery"]
    labels = ["REST interior", "Imagery→REST\ntransition", "Imagery interior", "REST→imagery\ntransition"]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    x = np.arange(len(cats)); width = .24
    for i, decoder in enumerate(decoders):
        means = [summary["decoders"][decoder][cat]["participant_mean"] for cat in cats]
        sds = [summary["decoders"][decoder][cat]["participant_sd"] for cat in cats]
        ax.bar(x + (i - 1) * width, means, width, yerr=sds, capsize=3, label=decoder)
    ax.set_xticks(x, labels); ax.set_ylabel("Mean hand probability (participant mean ± SD)")
    ax.set_ylim(0, 1); ax.set_title("Continuous EEG score by state interior and transition")
    ax.legend(title="Decoder"); ax.grid(axis="y", alpha=.25)
    fig.savefig(output / "transition_scores.png", dpi=160); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
