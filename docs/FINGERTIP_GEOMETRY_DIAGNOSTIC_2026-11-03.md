# Near-table fingertip geometry diagnostic — 2026-11-03

## Result and decision

Two **selected contact-mechanism** replays of the saved near-table progress
policy were run, seed 8 and seed 11, in separate fresh Isaac Lab processes.
Each used one environment and at most 30 policy steps (1 s). No training,
reward change, EEG connection, or scripted-controller change was made. The
new opt-in recorder field captured both sensor-bearing fingertip body poses at
every saved physics step, the pre-action reset and the end state.

Both replays reproduced the earlier screen's first action exactly and its
first right-finger contact time. At that moment, the hand center was already
12–14 mm toward the cube's +y side. The right fingertip **body frame** had
moved inward to 23.6–24.6 mm on the −y side of the cube center; the left frame
was 47.6–49.6 mm on the +y side. The cube's half width is 25 mm. Both actual
finger joints were still near the 40-mm open reset. This supports an early
lateral-arm-path/right-only-contact explanation for the sideways cube push.
Body frames are not collider surfaces or contact points, so the precise
collision geometry remains unverified.

An independent seed-8 **zero-action control** used the same near-table reset,
physics preset, one environment, 30-step cap, and recorder. Its initial cube,
hand, and joint state matched the seed-8 policy replay exactly. At the policy
run's 0.05-s first contact, hand y minus cube y was −0.00019 m under zero
action versus +0.01372 m under the policy. Across the full control second,
absolute hand-y drift stayed below 0.00073 m and neither finger registered
contact. Reset servo settling alone therefore does not explain the selected
policy run's early lateral shift. The control does not identify the responsible
arm joint or establish behavior across other seeds.

The seed-8 policy's first action included −0.8906 on arm joint 1 and −1.1442
on joint 3. By first contact, their processed position targets were −0.1664
and −0.2409 rad versus reset positions −0.0744 and −0.1074 rad. The trace
also records all seven arm targets and actual joints. Several joints move
together, so these values do not attribute the lateral displacement to one
joint without a controlled ablation.

| Seed | End status | First right >=1 N (s) | Hand y minus cube y at contact (m) | Left/right body-frame y minus cube y (m) | Peak right force (N) | Max cube XY shift (m) | Max cube rise (m) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 8 | Early terminal, 0.833 s | 0.050 | +0.0137 | +0.0496 / −0.0236 | 175.74 | 0.722 | 0.0122 |
| 11 | Incomplete at 1-s cap | 0.075 | +0.0121 | +0.0476 / −0.0246 | 107.05 | 0.544 | 0.0026 |

Seed 8 ended with cube z=0.206 m, below the task's 0.24-m lower z bound;
its saved termination flag is true. Seed 11 reached the cap and has an
explicit `episode_incomplete_end` record. Both had valid contact values on
every physics sample and zero recorded sample gaps (100/100 and 120/120).
Neither acquired a lift. These are short selected diagnostics, **not** two
completed 12-s pickup trials or a success-rate estimate. The large force
peaks and horizontal displacements are failure evidence, not safe behavior.

## Comparison and limits

At reset, the left/right body frames were approximately +0.040/−0.040 m in
y and +0.0416 m in z relative to the cube center. The saved scripted-positive
`grasp` waypoint has finger frames at +0.03955/−0.03796 m in y and +0.0416 m
in z. This static geometry confirms that the near-table reset begins close
to the scripted grasp waypoint's open-finger geometry. It does **not** show
that the learned arm follows the successful scripted trajectory.

The new and earlier progress traces had identical first policy actions and
first-contact times for both seeds. At first contact, cube and hand poses
agreed within `1.6e-8` m; after contact, the runs diverged. This supports the
first-contact geometry comparison while limiting later pointwise claims.
The earlier fixed-open diagnostic retained early right-only contact with both
finger targets at 0.04 m. Thus gripper closing commands alone do not explain
the observed contact. The zero-action control separately rules out reset
settling as the sole source of the seed-8 0.05-s shift. The actual fingertip collision shapes and surface
contact points were not logged; the finger-frame locations must not be
described as mesh intersections or force directions.

The object-contact sensor filter is scoped to the cube in the pinned Franka
configuration. The recorder logs norms of the filtered force matrix at each
available physics update. It reports `environment_physics_step` in these three
runs; 100% coverage describes the saved samples, not uninterrupted contact
between samples or parity with the scripted controller.

## Provenance and runbook

The official project source of record is Personal
`codex/robot-learning-design`. Each launch put
`/home/ubuntu/FRI-BCI-ML-RL-Decoder-Personal/simulation/src/` first on
`PYTHONPATH`. Same-venv import checks before and after playback resolved
`fri_bci_sim.ppo_telemetry` and `fri_bci_sim.tasks.near_table_progress` there.
The runner records exact source hashes and process overrides in each
`launch.json`. The post-run check recorded imported module paths/hashes in the
new parent-level
`import_provenance.json`; the collector checks those hashes against both
policy launch records and the zero-action launch. Pinned Isaac Lab was clean at
`e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`; Newton MJWarp ran on the
NVIDIA L4 with Python 3.12.16 and PyTorch `2.11.0+cu130`. The unchanged source checkpoint was `model_249.pt`, SHA256
`46f68c087111b5f713c33a63b7355f3c4c35eeaba26e4240f778667086b443d6`.

The executed checkout has synthetic local history and a dirty task registry.
Its imported `tasks/__init__.py` SHA256 is
`7243b89004790ae5c826934a4f9b5608ff2227dfd2397182ff0151e28605b372`.
The published Personal registry is a different Git blob
(`d668b38779247768b6b84b8b05e8127f331243de`) because it also registers
the independent FixedOpen task. The selected Progress task registration and
its `near_table.py` and `near_table_progress.py` contents match the published
versions; the exact executed registry bytes remain recorded in the raw launch
and import manifests. Do not represent the remote commit as a byte-identical
snapshot of every imported module.

Raw directories (ignored by Git, retained on Brev; off-host backup **not
verified**):

- `outputs/isaac/fingertip-geometry-progress-20261103T0314Z/seed8/`
- `outputs/isaac/fingertip-geometry-progress-20261103T0314Z/seed11/`
- `outputs/isaac/fingertip-zero-action-control-20261103T0323Z/seed8/`

The policy directories each have `launch.json`, `console.log`, `exit.json`,
`ppo-telemetry.jsonl`, `ppo-telemetry.summary.json`, and `validation.json`.
The zero-action directory has the same telemetry, launch, exit, summary, and
validation files; its console log is in the parent directory. The first zero-action
attempt failed at CLI parsing (unsupported `physics=` override) before
simulator launch; its log is preserved at
`outputs/isaac/fingertip-zero-action-control-20261103T0322Z/console.log`.
The compact record with
input and source hashes is
[`results/fingertip_geometry_progress_2026-11-03.json`](../results/fingertip_geometry_progress_2026-11-03.json).
The recorder change and runner are in
[`ppo_telemetry.py`](../simulation/src/fri_bci_sim/ppo_telemetry.py) and
[`run_fingertip_geometry_diagnostic.py`](../simulation/scripts/run_fingertip_geometry_diagnostic.py).
Eight focused telemetry tests passed before launch and again after collection
with the pinned venv's `unittest` runner. The three Python scripts compile;
the venv does not include `pytest`.

Executed commands, with the same output parent and one fresh subdirectory per
seed:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/IsaacLab-fri/.venv/bin/python \
  simulation/scripts/run_fingertip_geometry_diagnostic.py --seed 8 \
  --output-dir outputs/isaac/fingertip-geometry-progress-20261103T0314Z/seed8
PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/IsaacLab-fri/.venv/bin/python \
  simulation/scripts/run_fingertip_geometry_diagnostic.py --seed 11 \
  --output-dir outputs/isaac/fingertip-geometry-progress-20261103T0314Z/seed11
PYTHONPATH=/home/ubuntu/FRI-BCI-ML-RL-Decoder-Personal/simulation/src \
  FRI_PPO_RECORD_FINGERTIP_BODY_POSES=1 PYTHONDONTWRITEBYTECODE=1 \
  /home/ubuntu/IsaacLab-fri/.venv/bin/python \
  simulation/scripts/run_near_table_zero_action_geometry.py \
  --seed 8 --device cuda:0 \
  --output-dir outputs/isaac/fingertip-zero-action-control-20261103T0323Z/seed8
```

The runner refuses to reuse an output directory. The exact `isaaclab play`
argv and per-process variables are in each `launch.json`. The compact record
can be regenerated offline with:

```bash
python analysis/analyze_fingertip_geometry.py \
  --run-root outputs/isaac/fingertip-geometry-progress-20261103T0314Z \
  --prior-root outputs/isaac/near-table-progress-eval-5x12s-seed8-12-20261102T2330Z \
  --scripted-log outputs/isaac/eeg-matched-5p-continuation-20261103T0300Z-b120s/scripted-positive/run.log \
  --import-provenance outputs/isaac/fingertip-geometry-progress-20261103T0314Z/import_provenance.json \
  --zero-action-dir outputs/isaac/fingertip-zero-action-control-20261103T0323Z/seed8 \
  --failed-cli-log outputs/isaac/fingertip-zero-action-control-20261103T0322Z/console.log \
  --output results/fingertip_geometry_progress_2026-11-03.json
```

## Next learning gate

The evidence localizes the failure to early unilateral contact during the
learned arm path, followed by a large sideways cube push. It does not isolate
a single joint action, collider face, or reward term. The next policy
intervention should be versioned separately and frozen before a bounded
train-and-screen run; it should explicitly address approach centering or
one-sided force while retaining the existing bilateral lift criterion. A
short geometry/arm-target causality check is warranted before spending a PPO
budget. Do not infer pickup from reward, exit code, or these one-second traces,
and do not connect EEG until a learned controller passes a separate physical
readiness screen and HOLD/resume validation.
