# Project data and status map — 2026-11-02

## Where we are

The project has useful components and measured diagnostics, but reliable learned
table pickup and EEG control of a qualified learned policy remain unproven.
The EEG decoder has offline calibrated-person measurements. Scripted physical
controls work in the fixed scene, while the corrected paired EEG batch remains
incomplete. Approach PPO measured 0/5 table pickups. The NVIDIA candidate's valid
fresh-process screen measured 1/5 sampled lifts, with unresolved forces and
reset behavior. Keep those different conditions separate.

This review starts from Personal commit
`d7d8426de41a0f3d050c25629d9d3f30b2224e15`. Its near-table document still says
launched. Inspection of the existing local run now confirms **training completed**:
exit 0, final logged iteration 249/250, total steps 2,048,000, and model_249.pt.
A physical pickup screen was not verified by this review; training completion is
not pickup success. The existing graphs remain frozen to their documented inputs.

Later update: the first near-table checkpoint measured 0/5 physical pickups;
the separately versioned progress-reward checkpoint measured 2/5 sampled
acquisitions, with one drop, two early terminations and high finger forces.
See the [current result index](../results/README.md) and
[near-table protocol/result](NEAR_TABLE_CURRICULUM_2026-11-02.md). The inventory
below is a snapshot from before those screens and does not enumerate the new
directories. The progress training directory and all five raw evaluations
have since been copied off Brev with matching key-file hashes. Full backup
coverage of older outputs is still unverified.

## Find and manage the evidence

- [Results index](../results/README.md): links to authoritative summaries,
  figures, validity corrections and compact machine-readable records.
- [Project check-in](PROJECT_CHECK_IN_2026-11-02.md): cross-workstream narrative
  and existing plots. Read this map's completion note for the newer run status.
- [Local artifact inventory](../results/artifact_inventory_2026-11-02.json):
  observation time, 36 run folders, sizes, exit records, 193 key artifact hashes,
  and the near-table launch/checkpoint evidence. It covers the two existing
  output roots below. It does not catalog unavailable Windows EEG caches or
  prove every historical artifact exists.

| Location | What belongs there | Preservation / backup status |
|---|---|---|
| Personal `codex/robot-learning-design` | Source, protocols, human-readable docs, compact JSON/CSV summaries and plots | Published Git source of record; never push to Angel/upstream |
| `/home/ubuntu/FRI-BCI-ML-RL-Decoder-Personal/outputs/` | New raw logs, traces, checkpoints, prepared third-party candidate copies | Ignored by Git; files remain in place |
| `/home/ubuntu/FRI-project-8b9db42/outputs/` | Original project checkpoints and historical raw runs | Preserved archive; full backup coverage is unverified |
| `data/` and earlier release archive | Cached public EEG and original saved study artifacts | See the [release inventory](ARTIFACT_HANDOFF.md); this is an earlier snapshot, not a backup of every later run |
| Windows `outputs/isaac/candidate-*.tar.gz` | Second copies of candidate integration and reset diagnostic | Matching hashes previously documented; this review rehashed the Brev archives and matched those records, but did not access Windows |

The candidate archive hashes still match:

- Integration: `0a19b51b72a2b2565264becf529885bff248be82d6618cb679959e7dfd9c8a0b`.
- Reset diagnostic: `8a282ee1fb074506cffd9b2c515d68a709c5f81a61c891760e9eb2b3f53ae080`.

Near-table checkpoint:
`/home/ubuntu/FRI-BCI-ML-RL-Decoder-Personal/outputs/isaac/near-table-stage1-256x250-seed8-20261102T2250Z/logs/rsl_rl/lift_franka/2026-11-02_22-49-57/model_249.pt`

SHA256: `50713284b5c112bce1269a2a25054c10100b115f7c1075d01f49c152ebd3895e`.
Its off-host backup is not established by this review. Later Approach raw data
also remains outside the scope of the earlier release backup guarantee.

## Management rules

Use fresh run directories and retain failed/invalid attempts. Label completion,
physical validity and backup status separately. Keep source/config hashes with
each launch, because the shared Brev checkout has synthetic local history and
coordinator changes; local Git status alone does not identify the published
remote tree or actual imported module. Preserve those shared changes.

For each future completed run, publish a small summary and checksum manifest,
link it from the results index, and record a verified second-copy location before
retiring the VM. Never overwrite historical evidence to make a newer result look
cleaner. The original NVIDIA screen is retained but explicitly superseded by its
reset audit. No raw files, runtime settings, code under active development, or
experiment processes were changed by this organization review.
