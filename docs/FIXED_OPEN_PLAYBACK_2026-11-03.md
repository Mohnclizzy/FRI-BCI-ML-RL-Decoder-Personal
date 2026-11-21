# Fixed-open playback diagnostic — 2026-11-03

## Purpose and scope

This is a selected contact-mechanism diagnostic for the existing near-table
progress checkpoint. The policy still receives the same 100-value policy
observation and emits the same nine joint actions. The seven arm action values
retain the task's relative-position mapping. The two finger action values are
ignored by this opt-in task, which sends both finger position targets to exactly
0.04 m on every policy action. No model was trained under this action mapping.
This is not a pickup-success estimate and is not evidence of reliable learned
pickup.

The task ID is `FRI-Franka-Lift-NearTable-Progress-FixedOpen`. It inherits the
near-table progress reset, observation, reward and physics configuration; the
only runtime intervention is the action term. Training code does not select
this task automatically.

## Interface check

The one-step L4 check used a nonzero nine-value action. It verified the action
shape and joint order, compared the seven applied arm targets to the unchanged
relative-action calculation, and read the two applied finger targets before
physics plus the finger targets after one step.

- Action shape: `[1, 9]`.
- Arm targets: all seven matched the existing action mapping (absolute error
  below `1e-6` m/rad).
- Finger targets before and after the step: `[0.0399999991, 0.0399999991]` m.
- Report: `outputs/isaac/fixed-open-action-check-20261103T0200Z/report.json`.
- Report SHA256: `8e15b911c576e8af281fed7d53a2ce90c936bfdd09ffd5b2c2f2c9ee3103d9fe`.

## Playback protocol

Each seed ran in its own fresh Isaac Lab process, with one environment, the
specified progress checkpoint, seed 8 or 11, and a cap of 30 policy steps
(1.0 s at 30 Hz). No training was run. The existing physics-step recorder
captured pre-action baseline, physics samples, actual joint targets, both
finger sensors, hand and cube poses, and an explicit incomplete-end state.

Both episodes reached the 30-step cap without environment termination. They
are explicitly incomplete 1-second diagnostics, not completed episodes. Each
has 120 physics samples, 120/120 contact samples, no missing contact samples,
no physics-step gaps, and an end-state record. Both recorded finger targets were
0.04 m on every physics sample (zero violations).

| Seed | First right-finger force >=1 N | Peak left/right force (N) | Max cube-center displacement (m) | End cube-center displacement (m) | Max cube rise (m) | End status |
|---:|---:|---:|---:|---:|---:|---|
| 8 | 0.0667 s | 0 / 92.35 | 0.8399 | 0.8399 | 0.0165 | Incomplete at cap |
| 11 | 0.1000 s | 0 / 169.28 | 0.5533 | 0.5533 | 0.0027 | Incomplete at cap |

Neither seed had left-finger contact, bilateral acquisition, or a physical lift.
The large center displacement reflects the cube moving off its initial
position; height rise remained small. Right-finger force peaks were high in
these short diagnostics. They do not demonstrate a useful grasp or safe force.

## Comparison to the published progress trace

The comparison traces are the ordinary progress-task runs from the five-seed
12-second screen, using the same checkpoint and seeds. The comparison is aligned
to the available first 1 second: seed 8's ordinary run terminated after 112
physics steps (0.933 s), so all 112 are compared; seed 11 contributes its first
120 physics steps. This is a short, selected comparison, not a paired success
experiment.

| Seed | Condition | Samples | First right >=1 N | Peak right force (N) | Max cube displacement (m) | End displacement (m) | End cube center (m) | End hand position (m) |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 8 | Fixed open | 120 | 0.0667 s | 92.35 | 0.8399 | 0.8399 | (-0.189, 0.858, 0.264) | (-0.138, 0.618, 0.452) |
| 8 | Progress policy | 112 | 0.0500 s | 153.51 | 0.7254 | 0.7254 | (-0.101, 0.668, 0.239) | (-0.129, 0.574, 0.488) |
| 11 | Fixed open | 120 | 0.1000 s | 169.28 | 0.5533 | 0.5533 | (-0.193, 0.523, 0.281) | (-0.163, 0.499, 0.375) |
| 11 | Progress policy | 120 | 0.0750 s | 107.01 | 0.5501 | 0.5501 | (-0.234, 0.545, 0.344) | (-0.213, 0.514, 0.445) |

The intervention did not produce bilateral contact in either run. The change in
right-finger force and pose differs by seed, and the two short traces do not
support a causal or reliability conclusion.

## Reproduction and artifact paths

Checkpoint:

`outputs/isaac/near-table-progress-256x250-seed8-20261102T2305Z/logs/rsl_rl/lift_franka/2026-11-02_23-03-47/model_249.pt`

SHA256: `46f68c087111b5f713c33a63b7355f3c4c35eeaba26e4240f778667086b443d6`.

- Seed 8: `outputs/isaac/fixed-open-progress-playback-seed8-20261103T0210Z/`
- Seed 11: `outputs/isaac/fixed-open-progress-playback-seed11-20261103T0212Z/`
- Compact audit, aligned trace metrics and source hashes:
  `results/fixed_open_playback_20261103/evidence.json`.
- Runtime: Isaac Lab `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`, NVIDIA L4,
  `newton_mjwarp`; explicit `PYTHONPATH` selected the Personal source worktree.

Each raw run directory retains `launch.json`, `console.log`, `exit.json`,
`ppo-telemetry.jsonl`, and `ppo-telemetry.summary.json`. The action check and
both raw runs were copied to the ignored local
`outputs/isaac/near_table_progress_eval_local/` backup. Both copied telemetry
SHA256 values match the compact evidence, and the local trace audit passed all
four coverage and target checks. The Brev originals remain intact.
