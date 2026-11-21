"""Build the 20-second horizon-aligned EEG exposure, annotation, and robot telemetry audit.

Joins:
1. All 42 saved Robomimic rollout summaries + step-by-step telemetry (`S001` controls +
   `S001`, `S002`, `S003`, `S006` x `raw`, `confidence`, `temporal` x seeds `0..2`).
2. PhysioNet `R12` ground-truth `REST` / `HAND_IMAGERY` annotation intervals clipped to `[0, 20.0)` s.
3. Matched `enable-window`, `hold-window`, and bout-shuffled `enable-schedule` control specifications
   derived on the 400-step (20 Hz) robot episode clock.
4. 20-second horizon exposure comparison across the November 11 `S001–S015` `EA-EEGNet` `R12` traces
   (`raw`, `confidence`, `temporal`, `crc_hysteresis`).
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "simulation" / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "simulation" / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fri_bci_sim.robomimic_harness import (  # noqa: E402
    HARNESS_VERSION,
    derive_matched_control_schedules,
    join_horizon_exposure_and_telemetry,
)

AUDIT_DIR = Path(__file__).resolve().parent
CACHE_DIR = REPO_ROOT / "outputs" / "local_scratch_2026-11-11" / "cache" / "7d3de6fbf853d44a"
INTERVALS_PATH = AUDIT_DIR / "r12_annotation_intervals.json"
TELEMETRY_AUDIT_JSON = AUDIT_DIR / "horizon_aligned_telemetry_audit.json"
TELEMETRY_AUDIT_MD = AUDIT_DIR / "horizon_aligned_telemetry_audit.md"
MATCHED_CONTROLS_JSON = AUDIT_DIR / "matched_control_schedules_20s.json"


ROLLOUT_SPECS: list[dict[str, Any]] = [
    {
        "subject": "S001",
        "cohort": "controls",
        "condition": "always-enable",
        "gate": None,
        "decisions_path": None,
        "run_dir_template": "results/robomimic_controls_s001r12_2026-11-06/always-enable_seed{seed}",
    },
    {
        "subject": "S001",
        "cohort": "controls",
        "condition": "always-hold",
        "gate": None,
        "decisions_path": None,
        "run_dir_template": "results/robomimic_controls_s001r12_2026-11-06/always-hold_seed{seed}",
    },
    *[
        {
            "subject": "S001",
            "cohort": "csp_lda_replay",
            "condition": "eeg",
            "gate": gate,
            "decisions_path": "results/robomimic_eeg_s001r12_heldout_2026-11-05/S001R12-decisions.jsonl",
            "run_dir_template": f"results/robomimic_eeg_s001r12_heldout_2026-11-05/{gate}_seed{{seed}}",
        }
        for gate in ("raw", "confidence", "temporal")
    ],
    *[
        {
            "subject": subject,
            "cohort": "csp_lda_replay",
            "condition": "eeg",
            "gate": gate,
            "decisions_path": f"results/robomimic_eeg_s002s003_2026-11-06/{subject}/{subject}R12_decisions.jsonl",
            "run_dir_template": f"results/robomimic_eeg_s002s003_2026-11-06/{subject}_{gate}_seed{{seed}}",
        }
        for subject in ("S002", "S003")
        for gate in ("raw", "confidence", "temporal")
    ],
    *[
        {
            "subject": "S006",
            "cohort": "csp_lda_replay",
            "condition": "eeg",
            "gate": gate,
            "decisions_path": "outputs/isaac/S006R12-decisions.jsonl",
            "run_dir_template": f"results/robomimic_eeg_s006r12_2026-11-04/{gate}_seed{{seed}}",
        }
        for gate in ("raw", "confidence", "temporal")
    ],
]


def load_or_export_r12_intervals() -> dict[str, dict[str, Any]]:
    """Load R12 annotation intervals from scratch cache (if present) or committed JSON."""
    existing: dict[str, dict[str, Any]] = {}
    if INTERVALS_PATH.exists():
        existing = json.loads(INTERVALS_PATH.read_text(encoding="utf-8"))

    if CACHE_DIR.exists():
        for subject_id in range(1, 16):
            sid = f"S{subject_id:03d}"
            cache_file = CACHE_DIR / f"{sid}R12.json"
            if cache_file.exists():
                payload = json.loads(cache_file.read_text(encoding="utf-8"))
                existing[sid] = {
                    "subject": sid,
                    "run": 12,
                    "source_edf_sha256": payload["source_edf_sha256"],
                    "npz_sha256": payload["npz_sha256"],
                    "class_counts": payload["class_counts"],
                    "intervals": payload["intervals"],
                }
        INTERVALS_PATH.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    return existing


def summarize_condition_group(runs: list[dict[str, Any]]) -> dict[str, Any]:
    first = runs[0]
    n = len(runs)
    req = first["requested_on_horizon"]
    trace = first["trace_accounting"] or {}
    ann = first["annotation_alignment_20s"]
    telem_list = [r["robot_telemetry_alignment_20s"] for r in runs]
    retained = sum(1 for t in telem_list if t["completion_state"] == "complete_retained")
    unretained = sum(1 for t in telem_list if t["completion_state"] == "complete_unretained")
    incomplete = sum(1 for t in telem_list if t["completion_state"] not in ("complete_retained", "complete_unretained"))
    acq_steps = [t["first_acquisition_step"] for t in telem_list if t["first_acquisition_step"] is not None]
    latch_steps = [t["latch_step"] for t in telem_list if t["latch_step"] is not None]
    bout_lengths = [end - start for start, end in req["requested_spans"]]
    return {
        "subject": first["subject"],
        "cohort": first["cohort"],
        "condition": first["condition"],
        "gate": trace.get("gate"),
        "seeds": n,
        "retained_count": retained,
        "unretained_count": unretained,
        "incomplete_count": incomplete,
        "retained_rate": retained / n,
        "full_trace_enable_events": trace.get("full_trace_enable_events"),
        "events_issued_before_20s": trace.get("events_issued_strictly_before_horizon"),
        "requested_enable_steps_20s": req["requested_enable_steps"],
        "requested_enable_seconds_20s": req["requested_enable_duration_s"],
        "requested_bout_count_20s": req["requested_bout_count"],
        "requested_bout_lengths_steps_20s": bout_lengths,
        "executed_policy_steps_mean": sum(t["executed_policy_steps"] for t in telem_list) / n,
        "executed_policy_steps_by_seed": [t["executed_policy_steps"] for t in telem_list],
        "latched_override_steps_mean": sum(t["post_acquisition_latch_steps"] for t in telem_list) / n,
        "post_latch_requested_enable_steps_mean": sum(t["post_latch_overridden_enable_steps"] for t in telem_list) / n,
        "post_latch_requested_hold_steps_mean": sum(t["post_latch_overridden_hold_steps"] for t in telem_list) / n,
        "first_acquisition_step_by_seed": [t["first_acquisition_step"] for t in telem_list],
        "first_acquisition_step_mean": (sum(acq_steps) / len(acq_steps)) if acq_steps else None,
        "latch_step_by_seed": [t["latch_step"] for t in telem_list],
        "latch_step_mean": (sum(latch_steps) / len(latch_steps)) if latch_steps else None,
        "rest_requested_enable_duration_s_20s": ann["requested_rest_enable_duration_s"],
        "rest_requested_enable_rate_20s": ann["requested_rest_enable_fraction"],
        "rest_pre_latch_executed_policy_s_mean": sum(
            t["rest_motion_on_horizon"]["pre_latch_executed_policy_during_rest_s"] for t in telem_list
        ) / n,
        "rest_eef_path_length_m_mean": sum(
            t["rest_motion_on_horizon"]["total_rest_eef_path_length_m"] for t in telem_list
        ) / n,
        "rest_pre_latch_eef_path_length_m_mean": sum(
            t["rest_motion_on_horizon"]["pre_latch_rest_eef_path_length_m"] for t in telem_list
        ) / n,
        "imagery_requested_enable_duration_s_20s": ann["requested_imagery_enable_duration_s"],
        "imagery_missed_duration_s_20s": ann["requested_imagery_missed_duration_s"],
        "imagery_missed_rate_20s": ann["requested_imagery_missed_fraction"],
        "imagery_bouts_in_horizon": ann["imagery_bouts_in_horizon"],
        "detected_imagery_bouts_in_horizon": ann["detected_imagery_bouts_in_horizon"],
        "imagery_onset_delays_s": ann["requested_onset_delays_s"],
    }


def audit_nov11_eegnet_horizon(intervals_by_subject: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute 20-second horizon exposure for November 11 S001-S015 EA-EEGNet R12 traces."""
    scratch_root = REPO_ROOT / "outputs" / "local_scratch_2026-11-11" / "scratch"
    rows: list[dict[str, Any]] = []
    if not scratch_root.exists():
        if TELEMETRY_AUDIT_JSON.exists():
            prev = json.loads(TELEMETRY_AUDIT_JSON.read_text(encoding="utf-8"))
            return prev.get("nov11_eegnet_20s_horizon_exposure", [])
        return rows

    for subject_id in range(1, 16):
        sid = f"S{subject_id:03d}"
        decisions_path = scratch_root / sid / "evaluation" / f"{sid}R12-decisions.jsonl"
        if not decisions_path.exists():
            continue
        intervals = intervals_by_subject.get(sid, {}).get("intervals")
        for gate in ("raw", "confidence", "temporal", "crc_hysteresis"):
            joined = join_horizon_exposure_and_telemetry(
                decisions_path=decisions_path,
                condition="eeg",
                gate=gate,
                horizon=400,
                control_freq=20.0,
                intervals=intervals,
            )
            req = joined["requested_on_horizon"]
            trace = joined["trace_accounting"]
            ann = joined["annotation_alignment_20s"]
            rows.append(
                {
                    "subject": sid,
                    "cohort": "dev_S001_S010" if subject_id <= 10 else "heldout_opened_2026_11_11_S011_S015",
                    "gate": gate,
                    "full_trace_enable_events": trace["full_trace_enable_events"],
                    "events_issued_before_20s": trace["events_issued_strictly_before_horizon"],
                    "requested_enable_steps_20s": req["requested_enable_steps"],
                    "requested_enable_seconds_20s": req["requested_enable_duration_s"],
                    "requested_bout_count_20s": req["requested_bout_count"],
                    "rest_requested_enable_rate_20s": ann["requested_rest_enable_fraction"],
                    "imagery_requested_enable_rate_20s": 1.0 - ann["requested_imagery_missed_fraction"],
                    "imagery_missed_rate_20s": ann["requested_imagery_missed_fraction"],
                    "imagery_onset_delays_s": ann["requested_onset_delays_s"],
                }
            )
    return rows


def main() -> None:
    intervals_by_subject = load_or_export_r12_intervals()

    per_run_records: list[dict[str, Any]] = []
    grouped_runs: dict[tuple[str, str, str | None], list[dict[str, Any]]] = defaultdict(list)

    for spec in ROLLOUT_SPECS:
        subject = spec["subject"]
        intervals = intervals_by_subject[subject]["intervals"]
        decisions_path = (REPO_ROOT / spec["decisions_path"]) if spec["decisions_path"] else None
        for seed in (0, 1, 2):
            run_dir = REPO_ROOT / spec["run_dir_template"].format(seed=seed)
            summary_path = run_dir / "summary.json"
            telemetry_path = run_dir / "telemetry.jsonl"
            joined = join_horizon_exposure_and_telemetry(
                decisions_path=decisions_path,
                condition=spec["condition"],
                gate=spec["gate"],
                horizon=400,
                control_freq=20.0,
                telemetry_path=telemetry_path,
                summary_path=summary_path,
                intervals=intervals,
            )
            joined["subject"] = subject
            joined["cohort"] = spec["cohort"]
            joined["seed"] = seed
            joined["run_dir"] = run_dir.relative_to(REPO_ROOT).as_posix()
            per_run_records.append(joined)
            grouped_runs[(subject, spec["condition"], spec["gate"])].append(joined)

    condition_summaries = [
        summarize_condition_group(runs)
        for runs in grouped_runs.values()
    ]

    # Build matched control schedules for the 12 EEG replay conditions
    matched_schedules: list[dict[str, Any]] = []
    for spec in ROLLOUT_SPECS:
        if spec["condition"] != "eeg":
            continue
        subject = spec["subject"]
        gate = spec["gate"]
        decisions_path = REPO_ROOT / spec["decisions_path"]
        sched = derive_matched_control_schedules(
            decisions_path,
            gate=gate,
            horizon=400,
            control_freq=20.0,
            shuffle_seeds=(0, 1, 2),
        )
        matched_schedules.append(
            {
                "subject": subject,
                "gate": gate,
                "decisions_path": spec["decisions_path"],
                **sched,
            }
        )

    diagnostic_enable_windows = [
        {
            "label": f"enable_window_{steps:03d}_steps",
            "condition": "enable-window",
            "window_start": 1,
            "window_steps": steps,
            "enabled_seconds": steps / 20.0,
            "rationale": rationale,
        }
        for steps, rationale in (
            (36, "Matches S002 confidence 20s-horizon exposure (36 steps = 1.80s, 7 events, 0/3 retained)"),
            (48, "2.40s contiguous enable window (near single-seed acquisition boundary ~42-58 policy calls)"),
            (60, "3.00s contiguous enable window (just above always-enable acquisition steps [58, 54, 47])"),
            (80, "4.00s contiguous enable window (intermediate between 36-step failure and 110-step S002 raw success)"),
            (110, "Matches S002 raw 20s-horizon exposure (110 steps = 5.50s, 22 events, 3/3 retained)"),
            (130, "Matches S003 temporal 20s-horizon exposure (130 steps = 6.50s, 26 events, 3/3 retained)"),
            (212, "Matches S003 confidence 20s-horizon exposure (212 steps = 10.60s, 43 events, 3/3 retained)"),
            (218, "Matches S006 raw 20s-horizon exposure (218 steps = 10.90s, 44 events, 3/3 retained)"),
        )
    ]

    matched_controls_payload = {
        "harness_version": HARNESS_VERSION,
        "generated_date": "2026-11-11",
        "horizon_steps": 400,
        "control_freq_hz": 20.0,
        "episode_duration_s": 20.0,
        "diagnostic_contiguous_enable_windows": diagnostic_enable_windows,
        "eeg_matched_schedules": matched_schedules,
    }
    MATCHED_CONTROLS_JSON.write_text(
        json.dumps(matched_controls_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    nov11_eegnet_rows = audit_nov11_eegnet_horizon(intervals_by_subject)

    audit_payload = {
        "harness_version": HARNESS_VERSION,
        "generated_date": "2026-11-11",
        "horizon_steps": 400,
        "control_freq_hz": 20.0,
        "episode_duration_s": 20.0,
        "total_saved_rollouts_audited": len(per_run_records),
        "condition_summaries": condition_summaries,
        "per_run_records": per_run_records,
        "nov11_eegnet_20s_horizon_exposure": nov11_eegnet_rows,
    }
    TELEMETRY_AUDIT_JSON.write_text(
        json.dumps(audit_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    # Render Markdown report
    lines = [
        "# 20-Second Horizon-Aligned Exposure, Annotation & Telemetry Audit (2026-11-11)",
        "",
        f"- **Harness Version**: `{HARNESS_VERSION}` (`simulation/src/fri_bci_sim/robomimic_harness.py`)",
        "- **Robot Episode Clock**: `400` steps (`1..400`) at `20.0 Hz` (`[0.0, 20.0)` s)",
        "- **Saved Rollouts Audited**: `42` (`14` subject/condition groups $\\times$ `3` seeds `0..2`)",
        "- **Key Distinction**:",
        "  1. **Full-recording enable events** count 4 Hz EEG windows over the entire `123.0–125.0 s` EDF file.",
        "  2. **20s-horizon requested enable steps** count steps in `1..400` where `CausalDecisionReplay` requests `ENABLE` (`issued_at_s < 20.0` s).",
        "  3. **Executed policy steps** count steps where the robot policy actually ran (`mode == \"policy\"`) before the post-acquisition completion latch (`mode == \"latch\"`) took over.",
        "",
        "## 1. Horizon-Aligned Summary Across All 42 Saved Robomimic Rollouts",
        "",
        "| Subject | Condition / Gate | Retained | Full-Trace Events (123–125s) | `<20s` Events | `<20s` Requested Steps (`s`) | `<20s` Bouts | Executed Policy Steps (seeds `0,1,2`) | Latch Step (seeds `0,1,2`) | Post-Latch Overridden Enable / Hold Steps (mean) | `<20s` `REST` Requested Rate | `<20s` Pre-Latch `REST` EEF Path (m, mean) | `<20s` `HAND_IMAGERY` Missed Rate |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: |",
    ]

    for row in condition_summaries:
        cond_label = row["condition"] if row["gate"] is None else f"eeg / {row['gate']}"
        full_ev = "-" if row["full_trace_enable_events"] is None else str(row["full_trace_enable_events"])
        ev20 = "-" if row["events_issued_before_20s"] is None else str(row["events_issued_before_20s"])
        req_str = f"{row['requested_enable_steps_20s']} ({row['requested_enable_seconds_20s']:.2f}s)"
        pol_str = ", ".join(str(x) for x in row["executed_policy_steps_by_seed"])
        latch_str = ", ".join("-" if x is None else str(x) for x in row["latch_step_by_seed"])
        post_str = (
            f"{row['post_latch_requested_enable_steps_mean']:.1f} / "
            f"{row['post_latch_requested_hold_steps_mean']:.1f}"
        )
        lines.append(
            f"| `{row['subject']}` | `{cond_label}` | `{row['retained_count']}/{row['seeds']}` | "
            f"{full_ev} | {ev20} | {req_str} | {row['requested_bout_count_20s']} | "
            f"`[{pol_str}]` | `[{latch_str}]` | {post_str} | "
            f"{row['rest_requested_enable_rate_20s']:.3f} | "
            f"{row['rest_pre_latch_eef_path_length_m_mean']:.3f} | "
            f"{row['imagery_missed_rate_20s']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 2. Why the Earlier `(59, 120]` Command-Threshold Claim Was Retracted",
            "",
            "- The earlier report (`results/robomimic_eeg_s002s003_2026-11-06/summary.md`) observed that `S002 confidence` (`59` full-recording commands over `123.0 s`) failed (`0/3` retained) while `S006 raw` (`120` full-recording commands over `123.0 s`) succeeded (`3/3` retained), and inferred a critical command-count threshold in `(59, 120]`.",
            "- Aligning the decision traces to the actual `20.0 s` (`400`-step) robot episode horizon shows that the robot never sees commands issued at `t >= 20.0 s`:",
            "  - `S002 confidence` (`59` full-recording events) issued only **7** enable events (`36` requested steps = `1.80 s`, `3` bouts) inside `[0, 20.0)` s, executing `[36, 36, 36]` policy calls — fewer than the `42–58` policy calls required for grasp acquisition.",
            "  - `S002 raw` (`150` full-recording events) issued only **22** enable events (`110` requested steps = `5.50 s`, `5` bouts) inside `[0, 20.0)` s (`128` of its `150` events occurred at `t >= 20.0 s`), yet succeeded (`3/3` retained) with `[52, 44, 48]` executed policy calls (`latch_step` at `[298, 201, 294]`).",
            "  - `S003 temporal` (`308` full-recording events) issued **26** enable events (`130` requested steps = `6.50 s`, `5` bouts) inside `[0, 20.0)` s and succeeded (`3/3` retained) with `[49, 44, 45]` executed policy calls (`latch_step` at `[191, 186, 187]`).",
            "  - `S006 raw` (`120` full-recording events) issued **44** enable events (`218` requested steps = `10.90 s`, `4` bouts) inside `[0, 20.0)` s and succeeded (`3/3` retained) with `[50, 42, 44]` executed policy calls (`latch_step` at `[136, 128, 130]`).",
            "- Thus, on the actual `20.0 s` robot horizon, the transition occurs between **7** `<20s` enable events (`36` steps = `1.80 s`, `S002 confidence`, `0/3`) and **22** `<20s` enable events (`110` steps = `5.50 s`, `S002 raw`, `3/3`), with every successful condition requiring `42–58` executed policy calls (`2.10–2.90 s`) before the completion latch takes over.",
            "",
            "## 3. Pre-Generated Matched Control Schedules (`matched_control_schedules_20s.json`)",
            "",
            "| Subject | Gate | `<20s` Requested Steps (`s`) | `<20s` Bouts | Bout Lengths (steps) | Contiguous Duration-Matched `enable-window` (`start, steps`) |",
            "| --- | --- | ---: | ---: | --- | --- |",
        ]
    )

    for m in matched_schedules:
        eeg_s = m["eeg_schedule"]
        ew = m["contiguous_duration_matched"]
        bouts_str = ", ".join(str(b) for b in eeg_s["bout_lengths_steps"]) or "-"
        lines.append(
            f"| `{m['subject']}` | `{m['gate']}` | "
            f"{eeg_s['requested_enable_steps']} ({eeg_s['requested_enable_duration_s']:.2f}s) | "
            f"{eeg_s['bout_count']} | `[{bouts_str}]` | "
            f"`({ew['window_start']}, {ew['window_steps']})` |"
        )

    lines.extend(
        [
            "",
            "## 4. November 11 `EA-EEGNet` `R12` 20-Second Horizon Exposure (`S001–S015`)",
            "",
            "Note: `S011–S015` `R12` were evaluated on 2026-11-11 and are no longer an unopened confirmation cohort.",
            "",
            "| Subject | Partition | `raw` `<20s` Steps (`REST` / `IMAGERY` Rate) | `confidence` `<20s` Steps (`REST` / `IMAGERY` Rate) | `temporal` `<20s` Steps (`REST` / `IMAGERY` Rate) | `crc_hysteresis` `<20s` Steps (`REST` / `IMAGERY` Rate) |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )

    by_subj_gate = {(r["subject"], r["gate"]): r for r in nov11_eegnet_rows}
    for subject_id in range(1, 16):
        sid = f"S{subject_id:03d}"
        if (sid, "raw") not in by_subj_gate:
            continue
        part = "Dev (`S001–S010`)" if subject_id <= 10 else "Evaluated 2026-11-11 (`S011–S015`)"
        cells = []
        for g in ("raw", "confidence", "temporal", "crc_hysteresis"):
            r = by_subj_gate[(sid, g)]
            cells.append(
                f"`{r['requested_enable_steps_20s']}` "
                f"(`{r['rest_requested_enable_rate_20s']:.2f}` / `{r['imagery_requested_enable_rate_20s']:.2f}`)"
            )
        lines.append(f"| `{sid}` | {part} | {' | '.join(cells)} |")

    TELEMETRY_AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {TELEMETRY_AUDIT_JSON.relative_to(REPO_ROOT).as_posix()}")
    print(f"Wrote {MATCHED_CONTROLS_JSON.relative_to(REPO_ROOT).as_posix()}")
    print(f"Wrote {TELEMETRY_AUDIT_MD.relative_to(REPO_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
