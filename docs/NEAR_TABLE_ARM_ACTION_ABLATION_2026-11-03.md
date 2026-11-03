# Near-table first-action arm ablation — 2026-11-03

## Result

The [fingertip geometry diagnostic](FINGERTIP_GEOMETRY_DIAGNOSTIC_2026-11-03.md)
found an early policy-driven lateral shift and right-only contact. To identify
which saved arm commands contribute, four fresh-process runs used the **same
fixed seed-8 reset** and only the first two recorded nine-component policy
actions. They stopped after two policy steps (0.066667 s, eight physics
samples). The policy checkpoint was **not** loaded and no training or EEG ran.
Each run retained the task, reward, contact physics, reset and recorder.

| Frozen action variant | Hand y − cube y at 0.05 s | Right sensor at 0.05 s | First right >=1 N by 0.0667 s |
|---|---:|---:|---:|
| Both saved actions unchanged | +13.72 mm | 9.88 N | 0.050 s |
| Zero joint-1 component in both actions | +8.12 mm | 0 N | None |
| Zero joint-3 component in both actions | +5.34 mm | 0 N | None |
| Zero both joint-1 and joint-3 components | −0.41 mm | 0 N | None |

The unchanged two-action control reproduced the earlier policy trace's first
right contact at 0.05 s. Across all eight physics samples, its maximum
absolute differences were 0.55 µm for cube pose, 0.06 µm for hand pose,
and `2.24e-8` rad for joint positions. All four runs had a matching reset,
8/8 valid contact samples, zero physics gaps, and an explicit incomplete end.
This establishes that both arm components contribute to the selected early
lateral motion. It does not make either component a safe permanent clamp or
prove that avoiding contact for 0.067 s yields a lift later. The actions are
open-loop copies of two outputs; the learned policy would change its later
outputs as the state evolves. The finger frames are link origins, not
collision surfaces or measured contact points.

## Provenance and artifacts

The source policy trace is
`outputs/isaac/fingertip-geometry-progress-20261103T0314Z/seed8/ppo-telemetry.jsonl`,
SHA256 `0d2f8372fb803b2c32e17a83428562d9370a7d87c9a562e48efabebe30b470b0`.
It came from the unchanged progress `model_249.pt`, SHA256
`46f68c087111b5f713c33a63b7355f3c4c35eeaba26e4240f778667086b443d6`.
Each launch records the exact actions, source/config hashes and fresh output
path. The [compact result](../results/near_table_arm_action_ablation_2026-11-03.json)
checks first-action values, reset equality, full-control reproduction, status,
coverage and file hashes. The process used the Personal `simulation/src`
ahead of site packages, pinned Isaac Lab
`e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`, Python 3.12.16,
PyTorch `2.11.0+cu130` and the NVIDIA L4. The imported task registry in
this dirty checkout differs from the published registry only by the latter's
independent FixedOpen registration; see the geometry diagnostic provenance
note. The executed source hashes are recorded per launch.

Raw output root, ignored by Git and retained on Brev:
`/home/ubuntu/FRI-BCI-ML-RL-Decoder-Personal/outputs/isaac/near-table-arm-action-ablation-20261103T0333Z/`.
Its `full-v2/`, `joint1_off/`, `joint3_off/`, and `joint1_joint3_off/`
subdirectories contain `launch.json`, `ppo-telemetry.jsonl`,
`ppo-telemetry.summary.json`, `validation.json`, and `exit.json`. Their console
logs are in the root. The first `full` command included unsupported
`--headless`, failed at CLI parsing with exit 2 before simulator launch,
and its log is preserved as `full.console.log`; `full-v2` is the valid control.
Off-host backup of this new raw output is **not verified**.

## Reproduction

From the Personal repository root, choose a **new** output root and run the
four commands sequentially with the pinned venv. The first command must
reproduce first right contact at 0.05 s before interpreting the other three.
Do not reuse the historical directories.

```bash
PYTHONPATH="$PWD/simulation/src" PYTHONDONTWRITEBYTECODE=1 \
  /home/ubuntu/IsaacLab-fri/.venv/bin/python \
  simulation/scripts/run_near_table_arm_action_ablation.py \
  --variant full --device cuda:0 \
  --source-telemetry outputs/isaac/fingertip-geometry-progress-20261103T0314Z/seed8/ppo-telemetry.jsonl \
  --output-dir outputs/isaac/NEW-ARM-ABLATION/full
```

Repeat with `--variant joint1_off`, `joint3_off`, and `joint1_joint3_off`,
using distinct directories. The exact historical arguments are in each
`launch.json`. Offline collection command:

```bash
python analysis/analyze_near_table_arm_action_ablation.py \
  --run-root outputs/isaac/near-table-arm-action-ablation-20261103T0333Z \
  --source-telemetry outputs/isaac/fingertip-geometry-progress-20261103T0314Z/seed8/ppo-telemetry.jsonl \
  --failed-cli-log outputs/isaac/near-table-arm-action-ablation-20261103T0333Z/full.console.log \
  --output results/near_table_arm_action_ablation_2026-11-03.json
```

The ablation runner's two focused `unittest` cases and the recorder's eight
focused cases pass. All collection scripts compile. The pinned venv lacks
`pytest`; no package was installed.

## Next bounded decision

An inference-only, versioned early-action intervention can now be tested
against the unchanged progress policy in **one complete 12-s seed-8 episode**
with the recorder and original physical thresholds. Freeze its exact rule
before launch, retain all contacts and early failures, and compare to the
saved unmodified seed-8 trajectory. Do not train on the 0.067-s observation
alone or claim pickup from an exit code. A five-episode screen would follow
only if the one-episode intervention passes telemetry, contact-force and
physical-lift review. EEG remains disconnected; the scripted reference and
enable/HOLD/resume protocol remain separate.
