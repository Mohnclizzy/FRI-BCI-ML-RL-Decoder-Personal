# Stopping point — 2026-11-02

Work paused at the user's request. Source remains in the existing local checkout,
HEAD 376927c1a147363e056c50c15d6e2ebbd6cc025a plus uncommitted changes.
No push, fork, or new branch was made. No new PPO training started in this session.

## What is now measured

- Fixed HOLD's target-following bug: the original 24 s negative control drifted
  0.140571 m. Its physical comparison claims remain withdrawn.
- First repaired HOLD kept targets constant but settled 0.028386 m, failing the
  frozen 0.02 m limit. The arm was stable from 2–24 s. A common 2 s physical
  initialization is now separately logged; it never relatches the sagged pose.
- Corrected controls then passed: always-HOLD max hand displacement after
  initialization 5.82e-11 m; uninterrupted scripted table lift 0.179008 m with
  5.083 s sustained bilateral contact; paused/resumed table lift 0.179009 m with
  12.083 s sustained contact. Neither positive control dropped the cube, and
  held arm/finger targets stayed unchanged. These are fixed-scene controls,
  not EEG success rates or learned-policy results.
- S006R12 temporal-filtered recorded EEG ran for 120 s and passed offline
  per-step request/trace alignment and complete telemetry coverage checks.
  It supplied only 0.692 s of active simulation steps, remaining inside the
  initial one-active-second settle phase. No arm motion or lift occurred.
  Annotation scoring detected 1/15 imagery bouts and missed 14, with zero
  rest activation. This is one exploratory integration case, not a matched
  comparison or evidence that temporal filtering improves completion.
- Existing table PPO checkpoint received correctly located cube observations:
  12 captured point-cloud bounding-box centers agreed with the robot-base cube
  coordinates to 3.73e-9 m. It still made no contact and no lift in this 12 s
  audit. Mean weighted reach reward was 0.000599 versus action penalty -0.001464
  before dt scaling. This supports trying the reward variant; it does not
  prove reward shaping will solve table pickup.

## Evidence folders under ignored outputs/isaac

- eeg-matched-5p-corrected-controls-20261102T1450Z-b120s — first failed HOLD.
- eeg-matched-5p-preroll-controls-20261102T1500Z-b120s — all three passing controls.
- eeg-integration-120s-20261102T1500Z — single EEG case, initialization, telemetry,
  source manifest, and local annotation/physical metrics.
- table-observation-audit-20261102T1455Z — actual PPO observations, reward terms,
  source/checkpoint hashes, and observation_audit.json.
- eeg-matched-5p-corrected-20261102T1502Z-b120s — interrupted expanded batch;
  STOPPED_BY_USER.json marks it incomplete. Do not use it as the final comparison.

## What remains

1. Run the corrected 120 s, five-person, three-gate matched batch. The bounded
   runner and strict collector are ready; a partial batch cannot produce the
   final report. Preserve the same traces, scene, thresholds and initialization.
2. Optional separately measured PPO follow-up approved by the design/code review:
   64 environments, 100 additional iterations, seed 8, Approach reward variant,
   starting from the original pre-grasp model_99.pt. Then five measured table
   episodes using the same reset and thresholds as the baseline. If there is
   still no contact, stop PPO expansion and report the failure. This has NOT run.
3. Finish the corrected paired results table/plot, report and short Isaac Lab
   demonstration. No corrected filtering benefit or table-PPO success is claimed.

CPU checks last completed: 55 pytest tests plus 8 subtests. New initialization
was additionally compiled and physically validated by the three controls.
The results collector compiled and passed helper checks but has not consumed
an entire corrected batch yet.

See completion_protocol_2026-11-02.md for the frozen protocol and
simulation/scripts/collect_corrected_matched_results.py for the collector.
Before resuming, verify Git status and live Brev status. Keep work local.
