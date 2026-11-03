# Robot-learning handoff — 2026-11-03

## Source and preservation

- Source of record: `Mohnclizzy/FRI-BCI-ML-RL-Decoder-Personal`, branch `codex/robot-learning-design`. Work only in that repository; do not modify Angel's repository.
- Brev instance: `compact-green-alligator`, NVIDIA L4 (driver 580.178.04). Isaac Lab is `/home/ubuntu/IsaacLab-fri` at `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`.
- Preserve `/home/ubuntu/FRI-project-8b9db42`, all old output directories, and the original pregrasp `model_99.pt` (SHA256 `6a9c5ff66ff813c5b3a72596b620f9729953f91febeb564a88421c59c564ee9b`). The Brev Personal checkout has substantial dirty work and synthetic local history; do not reset, clean, or force-push it. Published branch content was written through the GitHub connector because direct shell GitHub HTTPS authentication was unavailable.
- As of this handoff the L4 was idle (`nvidia-smi`: 0 MiB used). No training/evaluation process was left running by this handoff.

## What has actually been measured

| Phase | Physical result | Record |
|---|---|---|
| Earlier in-hand PPO playback | 5/5 lifts from in-hand starts | [PPO playback report](ppo_lift_evaluation_2026-11-02.md) |
| Older Approach screen | 0/5 pickups from a farther ready pose | [Approach trace review](APPROACH_TABLE_TRACE_REVIEW_2026-11-03.md) |
| Anti-push reward, 50 added PPO updates | 2/5 near-table acquisitions; one later dropped | [Protocol](ANTIPUSH_PROTOCOL_2026-11-03.md), [result](ANTIPUSH_RESULT_2026-11-03.md) |
| Unilateral-contact penalty, 50 more updates | 3/5 fixed near-table acquisitions with sustained bilateral lift; all three met the predicate in the final 60 sampled physics steps, with no recorded post-acquisition drop | [Protocol](UNILATERAL_PROTOCOL_2026-11-03.md), [result](UNILATERAL_RESULT_2026-11-03.md), [trace review](UNILATERAL_TRACE_REVIEW_2026-11-03.md), [metrics](../results/unilateral_2026-11-03/metrics_published.json) |
| Frozen seed-11 replay of unilateral checkpoint | Reproduced transient lift and loss of contact; no acquisition | [Diagnostic result](RETENTION_DIAGNOSTIC_RESULT_2026-11-03.md), [metrics](../results/retention_diagnostic_2026-11-03/metrics.json) |
| Half-arm inference rule on same unilateral checkpoint | 0/5 acquisitions or 8-cm rises, three early terminations; reject this rule | [Protocol](HALFARM_PROTOCOL_2026-11-03.md), [result](HALFARM_RESULT_2026-11-03.md), [metrics](../results/halfarm_2026-11-03/metrics_published.json) |

The historical 0/5 Approach screen is **not a paired comparison** with the later fixed near-table screens: starts differ. The closer pre-intervention Progress screen had 2/5 acquisitions with matching recorded seed labels and initial poses, but latent simulator states are not proven identical. Five episodes from one training seed are a diagnostic, not a reliability estimate. No learned controller has passed a separate farther-ready-pose table-pickup screen. EEG is not connected to PPO. Learned fixed-target HOLD invariance and resume remain untested.

The corrected PPO recorder includes pre-action cube height, pre-reset terminal physical state, incomplete-episode status, acquisition/drop timing, contact coverage, and explicit sampling provenance. Physics is sampled at 120 Hz and policy actions at 30 Hz. Published compact traces at 30 Hz cannot reproduce every physics-step threshold event. Full 120-Hz unrounded traces and checkpoint copies remain in fresh, ignored Brev output directories; paths and SHA256 hashes are recorded in each linked metrics file. Off-host backup of those new raw binary artifacts is **unverified**.

## Additional seed-9 check from saved 120-Hz unilateral trace

A read-only check of the existing unilateral evaluation telemetry (no new simulator run) compared seeds 8, 9, and 11. All three started with hand center aligned in y with the cube center and finger joints at about 40 mm open. At first right-only contact, the hand had shifted toward +y by about 17–19 mm while the fingers remained about 38–40 mm open. The right fingertip **body-frame origin** was about 23 mm on the -y side of the cube center; the left frame was about 53–55 mm on the +y side. These are body origins, not collider surfaces or measured contact points.

For failing seed 9, first right-only contact was at 0.2083 s (right sensor 3.7 N), the cube reached 2 cm horizontal displacement while low at 0.3167 s, and right sensor peak was 99.5 N at 0.7083 s. At that peak, cube XY displacement was 0.252 m, hand y minus cube y was +0.047 m, and both fingertip body-frame origins were on the cube's +y side (+0.063/+0.023 m). It never rose 8 cm. Seeds 8 and 11 showed the same early one-sided geometry but different later trajectories. This supports investigating approach centering and contact geometry; it does not isolate a single cause or measure true contact points.

Seed-9 raw telemetry: `/home/ubuntu/FRI-BCI-ML-RL-Decoder-Personal/outputs/isaac/unilateral-eval-5x12s-20261103T2310Z/seed9/ppo-telemetry.jsonl`, SHA256 `017223aa512ae4a6b2b25334f0e39038a7da333265b88713d7a4c91faa5cb9ce`. It has 1,440/1,440 valid physics contact samples and a terminal pre-reset state. The current unilateral checkpoint `model_347.pt` is SHA256 `37eb85242539f7d2f0c36a916f3aa8e44d2fe6a70030a9da18963d2506e673f7`; its training directory is `outputs/isaac/unilateral-50-seed8-20261103T2310Z/`. The retention replay is in `outputs/isaac/retention-diagnostic-seed11-20261103T2333Z/`; the rejected half-arm screen is in `outputs/isaac/halfarm-eval-5x12s-20261103T2333Z/`.

## Next bounded decision

Stop extending the current PPO update count and do not revive the failed half-arm rule. Before another training budget, inspect seed 9's fingertip/cube alignment and the arm targets driving the early +y shift. The earlier two-action ablation found arm joints 1 and 3 both contribute to early lateral motion in one Progress-policy seed-8 start, while a full-episode early clamp later failed; see [arm ablation](NEAR_TABLE_ARM_ACTION_ABLATION_2026-11-03.md). Freeze one geometry-aware approach or contact intervention only after identifying its measurable pre-contact target. Run it in a new directory with a frozen physical protocol and report acquisition, terminal hold, drops, early terminations, displacement, contact, and source/checkpoint hashes. Separately validate the learned enable/HOLD/resume adapter and evaluate from the farther ready pose before any end-to-end EEG-to-robot or table-pickup claim.
