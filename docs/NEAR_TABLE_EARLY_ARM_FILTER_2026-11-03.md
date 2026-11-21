# Near-table early arm filter: one complete seed-8 episode — 2026-11-03

## Decision

The [two-action ablation](NEAR_TABLE_ARM_ACTION_ABLATION_2026-11-03.md)
showed that joint-1 and joint-3 action components jointly cause the selected
seed-8 early lateral shift. An **opt-in inference-only** filter then zeroed
those two components on policy steps 1 and 2, passing every later policy
action unchanged. The recorder saved both original policy and applied actions.
The original progress `model_249.pt` was copied byte-for-byte into a fresh
run directory before loading; no optimization occurred. One environment,
seed 8, and the original 12-second cap were used. EEG remained disconnected.

The episode ended **early at 0.967 s without a physical lift**. The filter
delayed first right >=1 N contact from the saved unmodified seed-8 run's
0.050 s to 0.125 s. It did not prevent a later right-dominant push: maximum
right sensor force was 130.81 N, maximum cube-center XY displacement was
0.672 m, and maximum cube rise was only 0.00683 m. The terminal pre-reset
cube center was `[-0.1091, 0.6072, 0.2282]` m; its z was below the task's
0.24-m lower bound. The recorded terminal flag was true and time-out false.
These are sampled sensor proxies and physical state; reward and process exit
are not pickup evidence. The one-episode screen fails the physical readiness
gate, so **no five-episode expansion or training run was launched**.

| Seed 8 | Original policy | Two-step arm filter |
|---|---:|---:|
| Episode duration | 0.933 s, early terminal | 0.967 s, early terminal |
| First right >=1 N | 0.050 s | 0.125 s |
| Peak right sensor force | 153.51 N | 130.81 N |
| Max cube XY shift | 0.724 m | 0.672 m |
| Max cube rise | 0.0190 m | 0.00683 m |
| Frozen physical lift pass | No | No |

The baseline is the preserved original seed-8 run from the completed five
12-second progress-policy screen. The first **raw** policy action in the new
run matches it exactly. Later raw policy outputs can change because the
filter changes the state after the first action. This is one selected seed,
not a reliability estimate or evidence that the filter improves pickup.

## Recorder and provenance

The filter is disabled by default in
[`ppo_telemetry.py`](../simulation/src/fri_bci_sim/ppo_telemetry.py). It accepts
only `FRI_PPO_EARLY_ARM_FILTER=joint1_joint3_first_two`, checks the nine-action
shape, clones the policy output, and logs `policy_action` and
`applied_action` on every policy step. The frozen task/reset/reward and
physical thresholds did not change. The new runner validates the first raw
action, all applied actions, pre-action reset, terminal state, fingertip poses,
116/116 contact samples and zero physics gaps. The simulator reported
`environment_physics_step`; sample coverage does not establish continuous
contact between samples. Contact-force norms are filtered sensor readings,
not measured bilateral grasp forces.

Pinned Isaac Lab was clean at
`e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`. The NVIDIA L4 ran with
Python 3.12.16 and PyTorch `2.11.0+cu130`. The original and isolated copy of
`model_249.pt` both have SHA256
`46f68c087111b5f713c33a63b7355f3c4c35eeaba26e4240f778667086b443d6`.
The copied checkpoint also confines Isaac Lab's exported inference files to
the fresh run directory. The launcher records source hashes, process
environment and exact argv in `launch.json`; the [compact result and input
hashes](../results/near_table_early_arm_filter_seed8_2026-11-03.json) are
tracked. The executed dirty local task registry has a different byte hash
from the published registry because the latter also registers FixedOpen;
the Progress registration and task configuration match. See the earlier
[module provenance note](FINGERTIP_GEOMETRY_DIAGNOSTIC_2026-11-03.md#provenance-and-runbook).

Raw output is retained at
`/home/ubuntu/FRI-BCI-ML-RL-Decoder-Personal/outputs/isaac/near-table-early-arm-filter-seed8-20261103T0341Z/`.
It includes `launch.json`, `console.log`, `exit.json`, `validation.json`,
`ppo-telemetry.jsonl`, `ppo-telemetry.summary.json`, and `input/model_249.pt`.
It is ignored by Git. **Off-host backup of this new raw run is not verified.**
The original checkpoint and archive were not changed.

## Exact command and checks

Executed from the Personal repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/IsaacLab-fri/.venv/bin/python \
  simulation/scripts/run_near_table_early_arm_filter.py \
  --output-dir outputs/isaac/near-table-early-arm-filter-seed8-20261103T0341Z
```

The runner refuses to reuse a directory. Eleven focused `unittest` tests
passed, including filter shape/nonmutation behavior; the runner and collector
compile. No package was installed. Offline compact result regeneration:

```bash
python analysis/analyze_near_table_early_arm_filter.py \
  --run-dir outputs/isaac/near-table-early-arm-filter-seed8-20261103T0341Z \
  --baseline-dir outputs/isaac/near-table-progress-eval-5x12s-seed8-12-20261102T2330Z/seed8 \
  --output results/near_table_early_arm_filter_seed8_2026-11-03.json
```

## Next learning step

The simple first-two-step filter postpones unilateral contact but does not
prevent the later high-force sideways push. The next policy change should
address approach alignment and contact throughout the early trajectory,
while retaining bilateral lift and force checks. Freeze one versioned task or
action rule, test its isolated behavior, and allocate a bounded PPO budget
only with a predefined full-episode screen. Do not use the short ablation or
this failed single episode as a pickup success or connect EEG to PPO yet.
