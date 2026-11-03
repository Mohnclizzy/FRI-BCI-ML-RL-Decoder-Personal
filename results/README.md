# Results index

Start with the [project check-in and figures](../docs/PROJECT_CHECK_IN_2026-11-02.md).
Use the [data and status map](../docs/DATA_AND_STATUS_MAP_2026-11-02.md) for storage,
backup limits, and the latest observed training completion.

| Workstream | Readable evidence | Compact data | Interpretation |
|---|---|---|---|
| EEG decoder and command replay | [Check-in](../docs/PROJECT_CHECK_IN_2026-11-02.md) | [Plot data and input hashes](project_checkin_plot_data_2026-11-02.json), [figures](figures/) | Offline calibrated EEG and request traces; no learned EEG-to-robot result |
| Scripted EEG comparison | [Completion protocol](../docs/completion_protocol_2026-11-02.md) | Earlier raw artifacts are indexed by the [release handoff](../docs/ARTIFACT_HANDOFF.md) | Corrected matched physical batch remains incomplete; historical 24-s comparison is invalid |
| Project Approach PPO | [Run handoff](../docs/BREV_HANDOFF_2026-11-02.md) | [Summary](ppo_approach_handoff_2026-11-02.json), [episodes](ppo_approach_episode_results_2026-11-02.csv), [hashes](ppo_approach_sha256_2026-11-02.json) | 100 updates, 204,800 transitions, 0/5 measured table lifts; recorder execution-path caveat |
| NVIDIA candidate: valid screen | [Reset diagnostic](../docs/NVIDIA_CANDIDATE_RESET_DIAGNOSTIC_2026-11-02.md) | [Fresh-process results](nvidia_candidate_2026-11-02/reset_diagnostic.json) | 1/5 valid sampled lifts; high forces/corrections; early terminations retained |
| NVIDIA candidate: superseded screen | [Correction](../docs/NVIDIA_CANDIDATE_INTEGRATION_2026-11-02.md) | [Reset audit](nvidia_candidate_2026-11-02/reset_audit.json) | `episodes.csv`, `evaluation_summary.json`, and `serialized_rescore.json` describe the original screen with four invalid starts. Do not treat them as a valid 1/5 estimate |
| Candidate recorder guards | [Guard changes](../docs/NVIDIA_CANDIDATE_GUARDS_2026-11-02.md) | [Offline checks](nvidia_candidate_2026-11-02/guard_review.json) | CPU tests and saved-trace rescoring only; no new physical run |
| Near-table learning | [Curriculum and physical screens](../docs/NEAR_TABLE_CURRICULUM_2026-11-02.md) | [Reset screen](near_table_reset_screen_2026-11-02.json), [first policy](near_table_stage1_screen_2026-11-02.json), [progress policy](near_table_progress_screen_2026-11-02.json), [figure](figures/near_table_progress_screen_2026-11-02.png) | Two 2.048M-transition variants in the same fixed near-grasp scene: 0/5 then 2/5 sampled acquisitions; one drop, two early ends, high asymmetric forces. Neither qualifies as reliable pickup |
| Finger target guard diagnostic | [Near-table interpretation](../docs/NEAR_TABLE_CURRICULUM_2026-11-02.md#target-bound-inference-diagnostic) | [Selected paired cases and hashes](near_table_finger_guard_diagnostic_2026-11-02.json) | Inference-only clamp eliminated negative targets but high force remained; prior successful seed lost acquisition. Two selected seeds, no success-rate estimate |
| First-contact trace | [Next bounded diagnostic](../docs/NEAR_TABLE_CURRICULUM_2026-11-02.md#target-bound-inference-diagnostic) | [Verified seed 8 and 11 samples](near_table_contact_trace_2026-11-02.json) | Right-finger contact begins within 0.05–0.083 s with actual fingers still near open; compare a fixed-open playback before further training |
| Fixed-open finger playback | [Protocol and interpretation](../docs/FIXED_OPEN_PLAYBACK_2026-11-03.md) | [Trace audit](fixed_open_playback_20261103/evidence.json) | Two selected one-second runs retained early right-only contact with 0.04 m finger targets. The raw runs are backed up locally; this does not estimate pickup reliability |
| Corrected EEG scripted replay | [Exploratory paired summary](corrected_eeg_2026-11-03-complete/summary.md) | [CSV](corrected_eeg_2026-11-03-complete/summary.csv), [figure](corrected_eeg_2026-11-03-complete/matched_results.png), [collector manifest](corrected_eeg_2026-11-03-complete/collector_manifest.json), [66-file SHA256 manifest](corrected_eeg_2026-11-03-complete/provenance_sha256.json) | All 15 gate/participant cases and 3 frozen controls completed. Physical lifts: raw 5/5, confidence 3/5, temporal 2/5; rest command activity decreases as missed imagery rises. Simulated scripted skill only; no learned PPO or live EEG |
| Other policy reuse | [Panda PPO review](../docs/PANDA_PPO_REUSE_REVIEW_2026-11-02.md) | [Candidate manifest](policy_candidate_manifest_2026-11-02.json) | Source/compatibility review; not a measured pickup result |

The [artifact inventory](artifact_inventory_2026-11-02.json) lists 36 local run
folders, their sizes, available exit records, and hashes for 193 key files.
It is a timestamped snapshot before the progress physical screen, not a backup
or a performance scorecard.
Raw checkpoints and telemetry stay in ignored outputs; compact records belong here.
Do not aggregate screens with different resets, horizons, controllers, or validity.
