# Corrected completion protocol — 2026-11-02

The primary question is the completion/rest-motion/delay tradeoff of three EEG
gates driving the same Franka skill. The scripted skill is the integration
baseline; PPO is a separate measured condition. Pre-grasp lifts are never
counted as table pickups.

## Command contract

Use **continuous enable/pause**: GRASP_LIFT advances one trajectory; HOLD pauses
it and preserves the last commanded arm and finger targets. Active requests
resume without restarting; completion is absorbing until episode reset.
Invalid/missing EEG traces fail closed; stale inputs issue HOLD. This is not
a one-pulse autonomous task-selection experiment. Report active duration so
its relationship with completion remains visible.

## Frozen control checks

Before corrected EEG outcomes, use one environment in the existing table scene:

The first corrected always-HOLD attempt is preserved as a failed control: hand
displacement reached 0.0284 m despite constant commanded targets. Inspection
localized it to initial servo settling; displacement from 2–24 s was only
3.7e-9 m. The 0.02 m criterion is unchanged. The adapter now gives every
condition the same deterministic 2 s physical pre-roll to the configured ready
arm target with open fingers, logs it separately in `initialization.jsonl`, and
starts EEG/simulation evaluation time at zero only after pre-roll. The initial
HOLD latch is the commanded pre-roll target, not the settled measured pose.

- Always HOLD for 24 s: no lift; max hand displacement <=0.02 m, max per-joint
  displacement <=0.1 rad; held target changes <=1e-6 rad/m.
- Scripted positive for 24 s: sustained physical lift, completed trajectory,
  no drop and unchanged targets whenever holding.
- Scripted pause for 34 s: enable 0–9 s, HOLD 9–12 s, enable 12–28 s,
  then HOLD. Require lift, completion, no drop and unchanged held targets.

Lift: cube rise >=0.08 m and both object-contact forces >=0.1 N for >=0.5 s.
Drop: after acquisition, cube below its initial height +0.04 m for >=0.25 s.
Log every physics step; distinguish servo settling from target changes. Failed
controls stop the batch; do not silently loosen tolerances after results.

## Corrected matched comparison

First validate one EEG case. Then evaluate S006–S010 run 12 with the same saved
CSP estimates, scene, trajectory, and raw/confidence/temporal gates over the
first 120 s of each recording. Confidence remains 0.7, persistence 0.5 s,
max estimate age 1 s. Labels remain confined to the offline evaluator.

The common horizon uses most of each recording and allows pauses in the
24-active-second skill clock. In the joint-waypoint script, that clock includes
21 s of trajectory phases and 3 s of final hold/stabilization; pauses suspend
the clock. This choice follows inspected results and the HOLD
audit, so the rerun is exploratory rather than fresh confirmation. Historical
24 s physical results remain debugging evidence because their HOLD control
was invalid and lacked the corrected common pre-roll; command-only scores
remain separate evidence.

Report per person/condition: first sampled physical-success time, trajectory
completion time, drops, actual annotated-rest motion, executed rest-command
activation, imagery detection/misses, request-active duration integrated from
per-step executor requests over the full simulator horizon (not capped by the
24 s skill clock), request switches/rate (including the initial HOLD baseline),
and latency endpoints. Rest command activation is the overlap of actually
executed GRASP_LIFT request intervals with annotated rest intervals; it is
reported separately from physical motion during rest. Target
dispatch is the first commanded-target change; physical-action latency is the
first sampled motion interval above 0.02 rad/s joint speed or 0.005 m/s hand
speed after an active request, so it is an upper bound on motion onset. A
request received after absorbing completion is labeled unavailable; a request
with no motion before HOLD or the horizon is censored as nonresponding, never
assigned a large artificial delay. Report telemetry coverage and sample-based
limitations. Decoder balanced accuracy is separate from all robot metrics.

## PPO decision

Inspect exact pinned reward functions and actual pre-action observations before
more training. The reach reward is scaled by 0.1 before bilateral contact; the
original weight 0.05 therefore contributes at most 0.005 before dt scaling.
Progress and success are contact-gated. This supports an approach-shaping
experiment but does not verify any new weights. The Approach variant changes
three weights and is an experimental condition, not a single-term ablation.

After frame checks, run only a bounded pilot and measured table playback.
If table PPO still fails, report it and complete the main comparison using
the scripted skill, as authorized in the accepted project plan.

## Deliverables and scope

Preserve source hashes, configs, traces, checkpoints and telemetry in ignored
outputs. Finish a report with corrected/historical distinctions, paired tables
and plots, an actual Isaac Lab replay, reproduction commands and limitations.
Work locally on the existing branch. No second dataset, live EEG or hardware
is in the core. Stop Brev after GPU work and artifact copying.

Reward source: [pinned Isaac Lab implementation](https://github.com/isaac-sim/IsaacLab/blob/e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4/source/isaaclab_tasks/isaaclab_tasks/core/lift/mdp/rewards.py).
Inspected source copies are in outputs/isaac/pinned_source_audit/.
