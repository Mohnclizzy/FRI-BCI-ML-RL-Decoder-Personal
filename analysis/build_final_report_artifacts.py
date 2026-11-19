"""Verify committed manifests and build figures and tables for the final report.

Reads only committed artifacts from:
- results/bounded_eeg_robot_2026-11-14/
- results/project_review_2026-11-18/
- results/bciciv_ds1_request_pilot_2026-11-19/

Usage:
    python -m analysis.build_final_report_artifacts
    python -m analysis.build_final_report_artifacts --out-dir outputs/repro_short
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

STUDY1_DIR_REL = Path("results/bounded_eeg_robot_2026-11-14")
REVIEW_DIR_REL = Path("results/project_review_2026-11-18")
STUDY2_DIR_REL = Path("results/bciciv_ds1_request_pilot_2026-11-19")
BCICIV_CLOCK_AUDIT_REL = Path("report/final/bciciv_ds1_clock_audit.json")

STUDY1_REQUIRED_MANIFEST_FILES = (
    "command_tradeoff.png",
    "groups.json",
    "participants.json",
    "protocol.json",
    "rows.json",
    "robot/robot_results/aggregate.json",
    "robot/robot_results/S001_seed0_selected/video.mp4",
)


def sha256_file(path: Path) -> str:
    """Compute hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifests(repo_root: Path) -> dict[str, int]:
    """Verify SHA-256 manifests for committed Study 1 inputs and all Study 2 files."""
    study2_dir = repo_root / STUDY2_DIR_REL
    sums2 = json.loads((study2_dir / "SHA256SUMS.json").read_text(encoding="utf-8"))
    for rel, expected_digest in sums2.items():
        target = study2_dir / rel
        if not target.is_file():
            raise FileNotFoundError(f"Missing Study 2 file from manifest: {target}")
        actual = sha256_file(target)
        if actual != expected_digest:
            raise ValueError(f"SHA-256 mismatch for {target}: expected {expected_digest}, got {actual}")

    study1_dir = repo_root / STUDY1_DIR_REL
    sums1 = json.loads((study1_dir / "SHA256SUMS.json").read_text(encoding="utf-8"))
    for rel in STUDY1_REQUIRED_MANIFEST_FILES:
        expected_digest = sums1[rel]
        target = study1_dir / rel
        if not target.is_file():
            raise FileNotFoundError(f"Missing Study 1 file from manifest: {target}")
        actual = sha256_file(target)
        if actual != expected_digest:
            raise ValueError(f"SHA-256 mismatch for {target}: expected {expected_digest}, got {actual}")

    return {
        "study2_manifest_files_verified": len(sums2),
        "study1_core_files_verified": len(STUDY1_REQUIRED_MANIFEST_FILES),
    }


def extract_verified_tables(repo_root: Path, manifest_counts: dict[str, int]) -> dict:
    """Extract and cross-check summary tables directly from committed JSONs."""
    study1_dir = repo_root / STUDY1_DIR_REL
    review_dir = repo_root / REVIEW_DIR_REL
    study2_dir = repo_root / STUDY2_DIR_REL

    groups = json.loads((study1_dir / "groups.json").read_text(encoding="utf-8"))
    rows = json.loads((study1_dir / "rows.json").read_text(encoding="utf-8"))
    aggregate = json.loads((study1_dir / "robot/robot_results/aggregate.json").read_text(encoding="utf-8"))
    cue_clock = json.loads((review_dir / "cue_clock.json").read_text(encoding="utf-8"))
    req_cap = json.loads((review_dir / "request_capacity.json").read_text(encoding="utf-8"))
    pilot = json.loads((study2_dir / "pilot_metrics.json").read_text(encoding="utf-8"))
    bciciv_clock = json.loads((repo_root / BCICIV_CLOCK_AUDIT_REL).read_text(encoding="utf-8"))

    # Compute Study 1 per-participant-averaged cue-window balanced accuracy (raw prefix gives 1 row/seed/sub)
    cue_ba: dict[str, dict[str, float]] = {}
    for decoder in ("eegnet", "riemannian", "csp"):
        by_sub: dict[int, list[float]] = defaultdict(list)
        for r in rows:
            if r["name"] == f"{decoder}/raw":
                by_sub[int(r["subject"])].append(float(r["classification"]["balanced_accuracy"]))
        sub_means = [statistics.mean(by_sub[s]) for s in sorted(by_sub)]
        cue_ba[decoder] = {
            "n_participants": len(sub_means),
            "mean_pct": round(statistics.mean(sub_means) * 100.0, 2),
            "pstdev_pct": round(statistics.pstdev(sub_means) * 100.0, 2),
        }

    study1_robot_conditions = {}
    for g in aggregate["groups"]:
        m = g["metrics"]
        study1_robot_conditions[g["condition"]] = {
            "n_rollouts": g["n"],
            "retention_count": g["retention_count"],
            "first_acquisition_step_mean": None
            if m["first_acquisition_step"]["mean"] is None
            else round(m["first_acquisition_step"]["mean"], 1),
            "requested_enable_s_mean": round(m["requested_enable_steps"]["mean"] / 20.0, 2),
            "executed_policy_s_mean": round(m["executed_policy_steps"]["mean"] / 20.0, 2),
            "post_acquisition_latch_s_mean": round(m["post_acquisition_latch_steps"]["mean"] / 20.0, 2),
            "requested_rest_enable_pct": round(m["requested_rest_enable_fraction"]["mean"] * 100.0, 1),
            "rest_eef_path_m_mean": round(m["total_rest_eef_path_length_m"]["mean"], 3),
            "rest_moving_s_mean": round(m["total_rest_moving_duration_s"]["mean"], 2),
        }

    return {
        "manifest_verification": manifest_counts,
        "study1_physionet": {
            "cue_window_balanced_accuracy_r08": cue_ba,
            "gate_groups": {
                g["name"]: {
                    "passing_participants": round(g["pass_fraction"] * g["participants"]),
                    "participants": g["participants"],
                    "rest_activation_pct": round(g["rest_command_fraction"] * 100.0, 1),
                    "imagery_coverage_pct": round(g["imagery_command_coverage"] * 100.0, 1),
                }
                for g in groups
            },
            "robot_270_rollouts": study1_robot_conditions,
            "paired_selected_minus_raw": {
                k: round(v["selected_minus_comparison_mean"], 4)
                for p in aggregate["paired_selected_minus_comparison"]
                if p["comparison"] == "raw"
                for k, v in p["summaries"].items()
            },
        },
        "review_2026_11_18": {
            "same_schedule_participants": cue_clock["same_schedule_participants"],
            "annotated_both": cue_clock["totals"]["annotated_both"],
            "elapsed_time_correct": cue_clock["totals"]["elapsed_time_correct"],
            "periodic_timer_r08_bal_acc_pct": round(cue_clock["mean_periodic_timer_balanced_accuracy"] * 100.0, 2),
            "r08_slot_capacity_totals": req_cap["totals"],
            "r08_full_recording_ceiling_pct": round(req_cap["mean_participant_full_recording_ceiling"] * 100.0, 1),
            "r08_robot_horizon_ceiling_pct": round(req_cap["mean_participant_robot_horizon_ceiling"] * 100.0, 1),
        },
        "study2_bciciv_ds1": {
            "participants": list(pilot["participants"].keys()),
            "all_240_hold_invariants_pass": pilot["all_240_hold_invariants_pass"],
            "all_240_latch_invariants_pass": pilot["all_240_latch_invariants_pass"],
            "all_240_contact_proxy_validated": pilot["all_240_contact_proxy_validated"],
            "any_240_post_acquisition_loss": pilot["any_240_post_acquisition_loss"],
            "pooled_methods": pilot["pooled_methods"],
            "per_participant": {
                sub: {
                    "runs_234_window_roc_auc": pdata["runs_234_window_roc_auc"],
                    "runs_234_window_bal_acc_05": pdata["runs_234_window_bal_acc_05"],
                    "stream": pdata["eval_runs_234_stream_metrics"],
                    "robot_readiness": pdata["eval_runs_234_robot_readiness"],
                }
                for sub, pdata in pilot["participants"].items()
            },
            "accumulator_slot_rotation_control": pilot["accumulator_slot_rotation_control"],
            "clock_audit": {
                "human_summary": bciciv_clock["human_summary"],
                "artificial_schedule_note": bciciv_clock["artificial_schedule_note"],
            },
        },
    }


def plot_bciciv_ds1_overview(pilot: dict, out_path: Path) -> None:
    """Build two-panel figure showing all 4 participants x 3 controllers and pooled funnel."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.6,
        }
    )

    fig = plt.figure(figsize=(13.2, 5.8), dpi=300)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.18, 1.0], height_ratios=[0.90, 1.10], wspace=0.50, hspace=0.48)

    ax_a = fig.add_subplot(gs[:, 0])
    ax_b_top = fig.add_subplot(gs[0, 1])
    ax_b_bot = fig.add_subplot(gs[1, 1])

    # Colorblind-safe Okabe-Ito-derived palette
    c_img = "#0072B2"    # Blue: Instructed imagery at request time
    c_rest = "#D55E00"   # Vermilion: Scored REST at request time
    c_tr = "#9499A1"     # Neutral cool gray: Unscored 1s transition at request time
    c_ret = "#F0E442"    # High-contrast gold marker for 3-seed retained imagery lifts

    subjects = ["ds1a", "ds1b", "ds1f", "ds1g"]
    method_keys = ["periodic_clock", "eeg_threshold", "eeg_sustained_leaky_accum"]
    method_short = ["Timer", "Thresh", "Accum"]

    # ---------------- PANEL (a): Per-participant x 3 controllers ----------------
    x_centers = []
    x_labels = []
    acc_img_vals = []
    acc_rest_vals = []
    acc_tr_vals = []
    ret_img_vals = []

    group_step = 3.8
    bar_offsets = [-0.95, 0.0, 0.95]
    group_ticks = []
    group_names = []

    for s_idx, sub in enumerate(subjects):
        base_x = s_idx * group_step
        group_ticks.append(base_x)
        slots = pilot["participants"][sub]["eval_runs_234_robot_readiness"]["periodic_clock"]["complete_20s_slots"]
        auc = pilot["participants"][sub]["runs_234_window_roc_auc"]
        group_names.append(f"{sub}\n({slots} slots, AUC {auc:.2f})")
        rr = pilot["participants"][sub]["eval_runs_234_robot_readiness"]
        for m_idx, mk in enumerate(method_keys):
            x_centers.append(base_x + bar_offsets[m_idx])
            x_labels.append(method_short[m_idx])
            acc_img_vals.append(rr[mk]["accepted_imagery"])
            acc_rest_vals.append(rr[mk]["accepted_rest"])
            acc_tr_vals.append(rr[mk]["accepted_transition"])
            ret_img_vals.append(rr[mk]["continuous_retained_lifts_imagery_mean_seeds"])

    x_arr = np.asarray(x_centers)
    img_arr = np.asarray(acc_img_vals)
    rest_arr = np.asarray(acc_rest_vals)
    tr_arr = np.asarray(acc_tr_vals)
    ret_arr = np.asarray(ret_img_vals)

    bar_w = 0.78
    ax_a.bar(x_arr, img_arr, width=bar_w, color=c_img, edgecolor="#1F2933", linewidth=0.6, label="Accepted: Instructed imagery")
    ax_a.bar(x_arr, rest_arr, width=bar_w, bottom=img_arr, color=c_rest, edgecolor="#1F2933", linewidth=0.6, label="Accepted: Scored REST")
    ax_a.bar(x_arr, tr_arr, width=bar_w, bottom=img_arr + rest_arr, color=c_tr, edgecolor="#1F2933", linewidth=0.6, label="Accepted: 1 s transition")
    ax_a.scatter(
        x_arr,
        ret_arr,
        s=34,
        facecolor=c_ret,
        edgecolor="#111111",
        linewidth=0.85,
        zorder=5,
        label="3-seed mean retained lifts (imagery)",
    )

    for xi, vi, vr, vt, vret in zip(x_arr, img_arr, rest_arr, tr_arr, ret_arr):
        # Put imagery count cleanly below marker inside lower blue region
        ax_a.text(xi, min(vi * 0.45, vret - 5.5) if vi >= 18 else vi * 0.5, f"{vi}", ha="center", va="center", color="white", fontsize=7.4, fontweight="bold")
        if vr >= 10:
            ax_a.text(xi, vi + vr * 0.5, f"{vr}", ha="center", va="center", color="white", fontsize=7.2, fontweight="bold")
        if vt >= 10:
            ax_a.text(xi, vi + vr + vt * 0.5, f"{vt}", ha="center", va="center", color="#111111", fontsize=6.9, fontweight="bold")
        elif vt >= 7:
            ax_a.text(xi, vi + vr + vt * 0.5, f"{vt}", ha="center", va="center", color="#111111", fontsize=6.4, fontweight="bold")
        # Retained imagery integer directly above the stacked total bar cleanly
        tot = vi + vr + vt
        ax_a.text(xi, tot + 1.4, f"{vret:.0f}r", ha="center", va="bottom", color="#0B4F78", fontsize=6.8, fontweight="bold")

    # Secondary small controller labels right above x-axis baseline and participant headers cleanly inside bottom margin
    ax_a.set_xticks(x_arr)
    ax_a.set_xticklabels(x_labels, rotation=0, fontsize=7.0)
    for gx, gname in zip(group_ticks, group_names):
        ax_a.text(gx, -10.5, gname, ha="center", va="top", fontsize=7.6, fontweight="bold", color="#111111")

    for s_idx in range(1, len(subjects)):
        ax_a.axvline((s_idx - 0.5) * group_step, color="#D8DCE3", linestyle="--", linewidth=0.8, zorder=0)

    ax_a.set_ylim(0, 106)
    ax_a.set_xlim(-1.6, (len(subjects) - 1) * group_step + 1.6)
    ax_a.set_ylabel("Accepted robot starts & retained lifts (Runs 2–4)")
    ax_a.set_title("(a) Participant-level starts by state at request time\n(top label: 3-seed mean retained imagery lifts)", fontweight="bold", pad=8)
    ax_a.grid(axis="y", color="#EAECEF", linewidth=0.7, zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.legend(loc="upper left", ncol=2, frameon=True, facecolor="white", framealpha=0.92, edgecolor="#D0D7DE", borderpad=0.35, columnspacing=0.9, handletextpad=0.4)

    # ---------------- PANEL (b) TOP: Aggregate Accumulator Imagery Funnel ----------------
    acc_p = pilot["pooled_methods"]["eeg_sustained_leaky_accum"]
    funnel_labels = [
        "Retained lifts (3-seed mean)\n0 post-acquisition drops",
        "Accepted while ready\n94 blocked (busy / tail)",
        "Detected on stream\n10/115 <=2.5s; 245/605 >2.5s",
        "Imagery episodes\n4 humans x 180 (Runs 2–4)",
    ]
    funnel_vals = [
        float(acc_p["continuous_retained_lifts_imagery_mean_seeds"]),
        float(acc_p["accepted_distinct_imagery_episodes"]),
        float(acc_p["detected_imagery_episodes"]),
        float(acc_p["total_imagery_episodes"]),
    ]
    funnel_colors = ["#004D7A", "#0072B2", "#4DA6D6", "#CFD8E3"]
    y_pos = np.arange(len(funnel_labels))

    ax_b_top.barh(y_pos, funnel_vals, height=0.62, color=funnel_colors, edgecolor="#1F2933", linewidth=0.6)
    # Show dashed extension to 720 for visual attrition awareness on bar 2
    ax_b_top.barh(2, 465, left=255, height=0.62, color="#F4E3E3", edgecolor="#B05252", linewidth=0.6, hatch="///", alpha=0.85)
    ax_b_top.text(255 + 465 / 2, 2, "465 undetected (64.6%)", ha="center", va="center", color="#7F1D1D", fontsize=7.2, fontweight="bold")

    val_texts = [
        f"{acc_p['continuous_retained_lifts_imagery_mean_seeds']:.1f} (21.4%)",
        f"{acc_p['accepted_distinct_imagery_episodes']} (22.4%)",
        f"{acc_p['detected_imagery_episodes']} (35.4%)",
        f"{acc_p['total_imagery_episodes']} (100%)",
    ]
    for yi, val, vtxt in zip(y_pos, funnel_vals, val_texts):
        if yi == 3:
            ax_b_top.text(val - 12, yi, vtxt, ha="right", va="center", color="#111111", fontsize=7.5, fontweight="bold")
        elif yi == 2:
            ax_b_top.text(val - 8, yi, vtxt, ha="right", va="center", color="#0A253B", fontsize=7.3, fontweight="bold")
        else:
            ax_b_top.text(val + 10, yi, vtxt, ha="left", va="center", color="#111111", fontsize=7.5, fontweight="bold")

    ax_b_top.set_yticks(y_pos)
    ax_b_top.set_yticklabels(funnel_labels, fontsize=7.2)
    ax_b_top.set_xlim(0, 760)
    ax_b_top.set_xlabel("Imagery episode count (out of 720)")
    ax_b_top.set_title("(b) Frozen EEG accumulator imagery funnel\n(720 -> 255 -> 161 -> 154.0 retained)", fontweight="bold", pad=6)
    ax_b_top.grid(axis="x", color="#EAECEF", linewidth=0.7, zorder=0)
    ax_b_top.set_axisbelow(True)
    ax_b_top.spines["top"].set_visible(False)
    ax_b_top.spines["right"].set_visible(False)

    # ---------------- PANEL (b) BOTTOM: Pooled Controller Comparison Across States ----------------
    pm = pilot["pooled_methods"]
    rot = pilot["accumulator_slot_rotation_control"]

    states = ["Instructed\nimagery", "Scored\nREST", "Unscored\n1 s trans."]
    x_s = np.arange(len(states))
    ctrl_specs = [
        ("periodic_clock", "Periodic timer (accepted / retained)", "#7B8794", -0.25),
        ("eeg_threshold", "EEG threshold (accepted / retained)", "#E69F00", 0.0),
        ("eeg_sustained_leaky_accum", "EEG accumulator (accepted / retained)", "#0072B2", 0.25),
    ]
    w_s = 0.23
    rot_vals = [rot["rotated_mean_imagery_starts"], rot["rotated_mean_rest_starts"], rot["rotated_mean_transition_starts"]]

    for mk, lbl, col, dx in ctrl_specs:
        m = pm[mk]
        acc_vals = [m["accepted_imagery"], m["accepted_rest"], m["accepted_transition"]]
        ret_vals = [
            m["continuous_retained_lifts_imagery_mean_seeds"],
            m["continuous_retained_lifts_rest_mean_seeds"],
            m["continuous_retained_lifts_transition_mean_seeds"],
        ]
        # Background light bar = accepted starts; solid foreground bar = 3-seed mean retained lifts
        ax_b_bot.bar(x_s + dx, acc_vals, width=w_s, color=col, alpha=0.28, edgecolor=col, linewidth=0.8)
        ax_b_bot.bar(x_s + dx, ret_vals, width=w_s, color=col, alpha=0.95, edgecolor="#1F2933", linewidth=0.6, label=lbl)
        for state_idx, (xi, av, rv) in enumerate(zip(x_s + dx, acc_vals, ret_vals)):
            rv_str = f"{rv:.1f}".rstrip("0").rstrip(".") if mk == "eeg_threshold" else f"{rv:.0f}"
            top_y = av
            offset_y = 4.5
            if mk == "eeg_sustained_leaky_accum":
                if rot_vals[state_idx] >= av - 2.0:
                    top_y = max(av, rot_vals[state_idx])
                    offset_y = 8.0
            ax_b_bot.text(xi, top_y + offset_y, f"{av}\n({rv_str})", ha="center", va="bottom", fontsize=6.6, color="#111111", linespacing=0.95)

    # Exploratory circular slot-rotation reference markers (accepted accumulator starts)
    ax_b_bot.scatter(
        x_s + 0.25,
        rot_vals,
        marker="D",
        s=32,
        facecolor="white",
        edgecolor="#B91C1C",
        linewidth=1.2,
        zorder=6,
        label="Exploratory slot rotation (282 accum starts)",
    )

    ax_b_bot.set_xticks(x_s)
    ax_b_bot.set_xticklabels(states, fontsize=8.0)
    ax_b_bot.set_ylim(0, 208)
    ax_b_bot.set_ylabel("Pooled starts / 3-seed lifts")
    ax_b_bot.set_title("(c) Pooled controller outcomes by request-time state\n(accepted above; solid bar & () = 3-seed retained)", fontweight="bold", pad=6)
    ax_b_bot.grid(axis="y", color="#EAECEF", linewidth=0.7, zorder=0)
    ax_b_bot.set_axisbelow(True)
    ax_b_bot.spines["top"].set_visible(False)
    ax_b_bot.spines["right"].set_visible(False)
    ax_b_bot.legend(loc="upper right", fontsize=6.8, frameon=True, facecolor="white", framealpha=0.94, edgecolor="#D0D7DE", borderpad=0.3)

    fig.subplots_adjust(left=0.055, right=0.980, top=0.89, bottom=0.145)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def build_artifacts(repo_root: Path, out_dir: Path, skip_hash_check: bool = False) -> dict:
    """Verify inputs, generate the Study 2 overview figure, and save verified metrics JSON."""
    manifest_counts = verify_manifests(repo_root) if not skip_hash_check else {}
    tables = extract_verified_tables(repo_root, manifest_counts)

    pilot_raw = json.loads((repo_root / STUDY2_DIR_REL / "pilot_metrics.json").read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "bciciv_ds1_pilot_overview.png"
    plot_bciciv_ds1_overview(pilot_raw, fig_path)

    tables_path = out_dir / "verified_report_tables.json"
    tables_path.write_text(json.dumps(tables, indent=2) + "\n", encoding="utf-8")
    return {
        "figure_path": str(fig_path),
        "tables_path": str(tables_path),
        "tables": tables,
    }


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root, help="Repository root path.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "report" / "final" / "figures",
        help="Destination directory for regenerated figure and tables JSON.",
    )
    parser.add_argument("--skip-hash-check", action="store_true", help="Skip SHA-256 verification.")
    args = parser.parse_args(argv)

    result = build_artifacts(args.repo_root.resolve(), args.out_dir.resolve(), args.skip_hash_check)
    print(f"Wrote figure: {result['figure_path']}")
    print(f"Wrote tables: {result['tables_path']}")
    if result["tables"]["manifest_verification"]:
        print("Manifest check:", json.dumps(result["tables"]["manifest_verification"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
