"""Tests for the 2026-11-11 reproducibility fixes and robomimic v2 control harness."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import unittest

import eeg_decoder
from eeg_decoder.evaluate_robot_replay import _executed_request_events
import fri_bci_sim
from fri_bci_sim.executor import DecisionReplay
from fri_bci_sim.robomimic_harness import (
    HARNESS_VERSION,
    VALID_GATES,
    CausalDecisionReplay,
    build_step_schedule,
    derive_matched_control_schedules,
    join_horizon_exposure_and_telemetry,
    resolve_step_request,
    spans_to_active_steps,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class ReproducibilityAndHarnessTests(unittest.TestCase):
    def test_imports_resolve_to_this_checkout(self) -> None:
        eeg_path = Path(eeg_decoder.__file__).resolve()
        sim_path = Path(fri_bci_sim.__file__).resolve()
        self.assertTrue(
            eeg_path.is_relative_to(REPO_ROOT),
            f"eeg_decoder imported from outside repo root: {eeg_path}",
        )
        self.assertTrue(
            sim_path.is_relative_to(REPO_ROOT),
            f"fri_bci_sim imported from outside repo root: {sim_path}",
        )

    def test_historical_phase6_latch6_rollout_hash_preserved(self) -> None:
        latch6_path = (
            REPO_ROOT
            / "results"
            / "robomimic_latch6_zoffset_2026-11-04"
            / "latch6_rollout.py"
        )
        digest = hashlib.sha256(latch6_path.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            "53b96476fa91d452817d7d5f82cc8430cf659ac16e5f15bc5c80f6776dc9dd5b",
        )

    def test_artifact_hashes_count_and_backup_verification_record(self) -> None:
        hashes_path = (
            REPO_ROOT
            / "results"
            / "eegnet_training_2026-11-11"
            / "artifact_hashes.json"
        )
        hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
        self.assertEqual(len(hashes), 155)

        backup_path = (
            REPO_ROOT
            / "results"
            / "eegnet_training_2026-11-11"
            / "backup_restore_verification.json"
        )
        backup = json.loads(backup_path.read_text(encoding="utf-8"))
        self.assertEqual(backup["manifest_hashes_verified_on_restore"], 155)
        self.assertTrue(backup["all_155_hashes_match"])
        self.assertTrue(backup["offhost_drive_backup"]["roundtrip_verified"])
        self.assertEqual(backup["offhost_drive_backup"]["restored_manifest_hashes_verified"], 155)

    def test_v2_harness_rejects_ambiguous_pause_and_separates_enable_vs_hold_window(self) -> None:
        self.assertEqual(HARNESS_VERSION, "robomimic-latch6-v2.0.0")
        with self.assertRaisesRegex(ValueError, "Ambiguous condition 'pause'"):
            resolve_step_request(1, "pause", horizon=400)

        enable_sched = build_step_schedule(
            "enable-window",
            horizon=400,
            window_start=30,
            window_steps=150,
        )
        hold_sched = build_step_schedule(
            "hold-window",
            horizon=400,
            window_start=30,
            window_steps=150,
        )
        enable_steps = [r["step"] for r in enable_sched if r["requested_mode"] == "policy"]
        hold_policy_steps = [r["step"] for r in hold_sched if r["requested_mode"] == "policy"]

        self.assertEqual(enable_steps, list(range(30, 180)))
        self.assertEqual(len(enable_steps), 150)
        self.assertEqual(len(hold_policy_steps), 250)
        self.assertEqual(set(enable_steps) & set(hold_policy_steps), set())
        self.assertEqual(set(enable_steps) | set(hold_policy_steps), set(range(1, 401)))

    def test_matched_control_schedules_conserve_steps_and_bouts(self) -> None:
        decisions_path = (
            REPO_ROOT
            / "results"
            / "robomimic_eeg_s002s003_2026-11-06"
            / "S002"
            / "S002R12_decisions.jsonl"
        )
        matched = derive_matched_control_schedules(
            decisions_path,
            gate="raw",
            horizon=400,
            control_freq=20.0,
            shuffle_seeds=(0, 1, 2),
        )
        eeg_s = matched["eeg_schedule"]
        self.assertEqual(eeg_s["requested_enable_steps"], 110)
        self.assertEqual(eeg_s["bout_count"], 5)

        contig = matched["contiguous_duration_matched"]
        self.assertEqual(contig["requested_enable_steps"], 110)
        self.assertEqual(contig["window_steps"], 110)

        expected_bouts = Counter(eeg_s["bout_lengths_steps"])
        for shuf in matched["bout_shuffled_matched"]:
            self.assertEqual(shuf["requested_enable_steps"], 110)
            self.assertEqual(Counter(shuf["bout_lengths_steps"]), expected_bouts)
            active = spans_to_active_steps(shuf["spans"], 400)
            self.assertEqual(len(active), 110)

    def test_join_horizon_exposure_and_telemetry_matches_saved_rollouts(self) -> None:
        intervals_path = (
            REPO_ROOT
            / "results"
            / "project_audit_2026-11-11"
            / "r12_annotation_intervals.json"
        )
        intervals_by_subject = json.loads(intervals_path.read_text(encoding="utf-8"))

        # Check S002 raw seed 0 (22 <20s events, 110 requested steps, 52 executed policy steps, latch at 298)
        joined = join_horizon_exposure_and_telemetry(
            condition="eeg",
            gate="raw",
            horizon=400,
            control_freq=20.0,
            decisions_path=REPO_ROOT
            / "results/robomimic_eeg_s002s003_2026-11-06/S002/S002R12_decisions.jsonl",
            telemetry_path=REPO_ROOT
            / "results/robomimic_eeg_s002s003_2026-11-06/S002_raw_seed0/telemetry.jsonl",
            summary_path=REPO_ROOT
            / "results/robomimic_eeg_s002s003_2026-11-06/S002_raw_seed0/summary.json",
            intervals=intervals_by_subject["S002"]["intervals"],
        )
        self.assertEqual(joined["trace_accounting"]["full_trace_enable_events"], 150)
        self.assertEqual(joined["trace_accounting"]["events_issued_strictly_before_horizon"], 22)
        self.assertEqual(joined["requested_on_horizon"]["requested_enable_steps"], 110)
        telem = joined["robot_telemetry_alignment_20s"]
        self.assertEqual(telem["executed_policy_steps"], 52)
        self.assertEqual(telem["latch_step"], 298)
        self.assertEqual(telem["post_acquisition_latch_steps"], 102)
        self.assertEqual(
            telem["executed_policy_steps"]
            + telem["executed_hold_steps"]
            + telem["post_acquisition_latch_steps"],
            400,
        )

    def test_crc_hysteresis_supported_across_replay_interfaces(self) -> None:
        self.assertIn("crc_hysteresis", VALID_GATES)
        tmp_trace = REPO_ROOT / "results" / "project_audit_2026-11-11" / "r12_annotation_intervals.json"
        self.assertTrue(tmp_trace.exists())
        # Verify DecisionReplay and CausalDecisionReplay accept crc_hysteresis on synthetic JSONL row
        import tempfile

        row = {
            "estimate": {
                "source": "S001R12",
                "episode": "S001R12",
                "window_end_s": 2.0,
                "available_at_s": 2.0,
                "p_rest": 0.507,
                "p_hand": 0.493,
            },
            "decisions": {
                "raw": {"skill": "hold", "reason": "below_hold_threshold", "issued_at_s": 2.0, "state": "hold"},
                "confidence": {"skill": "hold", "reason": "below_hold_threshold", "issued_at_s": 2.0, "state": "hold"},
                "temporal": {"skill": "hold", "reason": "below_hold_threshold", "issued_at_s": 2.0, "state": "hold"},
                "crc_hysteresis": {"skill": "hold", "reason": "below_hold_threshold", "issued_at_s": 2.0, "state": "hold"},
            },
        }
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "trace.jsonl"
            p.write_text(json.dumps(row) + "\n", encoding="utf-8")
            dr = DecisionReplay(p, condition="crc_hysteresis")
            self.assertEqual(dr.at(2.1).skill.value, "hold")
            cdr = CausalDecisionReplay(p, gate="crc_hysteresis")
            self.assertEqual(cdr.at(2.1)[0], "hold")
            events = _executed_request_events(
                [{"row_kind": "post_step", "interval_start_s": 2.1, "requested_skill": "hold"}],
                p,
                "crc_hysteresis",
            )
            self.assertEqual(events, [{"time_s": 2.1, "skill": "hold"}])

    def test_notebook_and_runner_safety_invariants(self) -> None:
        nb_text = (REPO_ROOT / "notebooks" / "FRI_BCI_Colab_Master.ipynb").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('"reset", "--hard"', nb_text)
        self.assertNotIn("unopened prospective", nb_text)

        runner_text = (REPO_ROOT / "run_agent_task.sh").read_text(encoding="utf-8")
        exec_lines = "\n".join(
            line for line in runner_text.splitlines() if not line.lstrip().startswith("#")
        )
        self.assertNotIn("|| true", exec_lines)
        self.assertNotIn('--condition "pause"', exec_lines)

    def test_dev_decoder_gate_study_respects_r04_r08_only_split(self) -> None:
        study_path = (
            REPO_ROOT
            / "results"
            / "decoder_gate_study_2026-11-11"
            / "dev_decoder_gate_study.json"
        )
        study = json.loads(study_path.read_text(encoding="utf-8"))
        self.assertFalse(study["target_r12_touched"])
        self.assertEqual(study["target_fit_run"], 4)
        self.assertEqual(study["development_scoring_run"], 8)
        self.assertEqual(study["development_subjects"], [f"S{s:03d}" for s in range(1, 11)])
        dec = study["decoder_comparison_r08"]
        self.assertAlmostEqual(dec["transfer_ea_eegnet_r04"]["mean_balanced_accuracy"], 0.6633333333, places=4)
        self.assertAlmostEqual(dec["riemannian_lda_r04"]["mean_balanced_accuracy"], 0.6266666667, places=4)
        self.assertAlmostEqual(dec["csp_lda_r04"]["mean_balanced_accuracy"], 0.6100000000, places=4)
        self.assertAlmostEqual(dec["scratch_ea_eegnet_r04"]["mean_balanced_accuracy"], 0.5600000000, places=4)


if __name__ == "__main__":
    unittest.main()
