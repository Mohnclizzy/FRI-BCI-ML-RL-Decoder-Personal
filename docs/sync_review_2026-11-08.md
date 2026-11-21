# Sync review — 2026-11-08

Reviewed commit: 978b8522b0bdb7fa32d3ffa7444f6154f3fc273a.
The remote codex/robot-learning-design branch and clean local clone matched this commit at review time.

## Verified artifacts

- S006–S010 decision streams match results/corrected_eeg_2026-11-03-complete/provenance_sha256.json. The root and eeg-batch-inputs copies are identical.
- S001 matches results/connected_bottleneck_stage1_2026-11-05/manifest.json and the held-out S001 result records.
- The pretrained checkpoint matches results/robomimic_continuous_2026-11-04/sha256_manifest.json.
- All six report/midterm files were downloaded from Drive and matched local SHA-256 checksums. File IDs and hashes are recorded in results/sync_audit_2026-11-08.json.

## Historical manifest and remaining limitations

outputs/isaac/eeg-batch-inputs/source_manifest.json is a historical deployment record, not a manifest of the current checkout. It records commit 376927c1a147363e056c50c15d6e2ebbd6cc025a, a dirty working tree, and different deployed/local hashes for run_eeg_matched_batch.sh and deploy_eeg_matched_batch.sh. It is preserved unchanged. Independent verification of its original AntiHermes bytes remains outstanding.

The claimed inventory of 198 PPO checkpoints, approximately 3.6 GB, and more than 70,000 environment/cache files was not independently verified. Excluding those files from Git does not establish that they are disposable. No deletion was performed or authorized by this review.

## Ignore-rule correction

The eeg-batch-inputs exception now admits only S0*R12-decisions.jsonl and source_manifest.json. Other files and nested directories remain ignored. Verified using git check-ignore against representative checkpoints, telemetry, nested files, and the intended allowlisted paths.
