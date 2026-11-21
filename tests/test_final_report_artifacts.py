"""Tests verifying committed manifests and reproducible final-report figures/tables."""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.build_final_report_artifacts import build_artifacts


def test_build_final_report_artifacts_from_committed_records(tmp_path):
    result = build_artifacts(REPO_ROOT, tmp_path, skip_hash_check=False)

    fig_path = Path(result["figure_path"])
    tables_path = Path(result["tables_path"])
    assert fig_path.is_file() and fig_path.stat().st_size > 20_000
    assert tables_path.is_file()

    t = result["tables"]
    assert t["manifest_verification"]["study2_manifest_files_verified"] == 6
    assert t["manifest_verification"]["study1_core_files_verified"] == 7

    # Study 1 headline values
    s1 = t["study1_physionet"]
    assert s1["cue_window_balanced_accuracy_r08"]["eegnet"]["mean_pct"] == 68.15
    assert s1["cue_window_balanced_accuracy_r08"]["csp"]["mean_pct"] == 62.44
    assert s1["cue_window_balanced_accuracy_r08"]["riemannian"]["mean_pct"] == 64.89
    assert s1["robot_270_rollouts"]["selected"]["retention_count"] == 42
    assert s1["robot_270_rollouts"]["raw"]["retention_count"] == 45
    assert s1["robot_270_rollouts"]["selected"]["rest_eef_path_m_mean"] == 0.175
    assert s1["robot_270_rollouts"]["raw"]["rest_eef_path_m_mean"] == 0.209

    # Clock & slot capacity review values
    rev = t["review_2026_11_18"]
    assert rev["same_schedule_participants"] == 15
    assert rev["annotated_both"] == 7278
    assert rev["elapsed_time_correct"] == 7278
    assert rev["r08_slot_capacity_totals"]["complete_slots"] == 90
    assert rev["r08_slot_capacity_totals"]["imagery_full_recording"] == 225

    # Study 2 headline values
    s2 = t["study2_bciciv_ds1"]
    assert s2["any_240_post_acquisition_loss"] is False
    acc = s2["pooled_methods"]["eeg_sustained_leaky_accum"]
    assert acc["total_imagery_episodes"] == 720
    assert acc["detected_imagery_episodes"] == 255
    assert acc["undetected_imagery_episodes"] == 465
    assert acc["detected_but_unaccepted_imagery_episodes"] == 94
    assert acc["accepted_imagery"] == 161
    assert acc["accepted_rest"] == 62
    assert acc["accepted_transition"] == 59
    assert acc["continuous_retained_lifts_imagery_mean_seeds"] == 154.0
    assert acc["continuous_retained_lifts_rest_mean_seeds"] == 59.0
    assert acc["continuous_retained_lifts_transition_mean_seeds"] == 57.0
    assert acc["short_imagery_episodes_le_2_5s"] == 115
    assert acc["short_detected_episodes"] == 10
    assert acc["long_imagery_episodes_gt_2_5s"] == 605
    assert acc["long_detected_episodes"] == 245

    clk = s2["clock_audit"]["human_summary"]
    assert clk["n_human_participants"] == 4
    assert clk["total_human_mi_episodes"] == 960
    assert clk["mean_run1_to_runs234_continuous_cue_bal_acc"] == 0.5063
    assert clk["mean_periodic_4_1s_bal_acc"] == 0.5001
    assert s2["clock_audit"]["artificial_schedule_note"]["ds1c_run1_vs_run3_exact_match"] is True
