# Fixed-cube physical lift diagnostic — 2026-11-02

Stopped after the user's request. No renderer or video capture was run. This is
a scripted, privileged robot-control reference, not a trained policy or an EEG
experiment. The script commands Franka arm/finger joint targets; it does not
write the cube pose, attach the cube, or use an artificial grasp constraint.

## Runtime and exact reference invocation

- Brev instance: `compact-green-alligator`, one NVIDIA L4.
- Isaac Lab checkout: `/home/ubuntu/IsaacLab-fri`, clean tracked source at
  `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4` (`v3.0.0`).
- Project: `/home/ubuntu/FRI-BCI-ML-RL-Decoder`, editable-installed in that
  checkout's `.venv`; Python 3.12, Newton `newton_mjwarp`.
- No explicit random seed was set for the direct scene reference. It is one
  deterministic fixed-scene trial, not a repeatability or generalization study.
- The tested script SHA256 was
  `25d5917d406e45d2f31ca44dfc732f0ff1096d32f3bea3f69a74a86b8781a4dc`.
  The tested task config SHA256 was
  `19187b0b45539a45a0364c70db1b1fca485dfce854fe3886c66bcc7f124318eb`.
  These hashes were checked against remote files after the run. Exact source
  copies are retained under ignored `outputs/isaac/controlled-lift-008-source/`.

```bash
cd /home/ubuntu/IsaacLab-fri
timeout 180s .venv/bin/python \
  /home/ubuntu/FRI-BCI-ML-RL-Decoder/simulation/scripts/controlled_lift.py \
  --joint-waypoints \
  --output /home/ubuntu/FRI-BCI-ML-RL-Decoder/outputs/isaac/controlled-lift-008.jsonl \
  > /home/ubuntu/FRI-BCI-ML-RL-Decoder/outputs/isaac/controlled-lift-008.log 2>&1
```

The script initialized the Franka's configured ready joint pose, verified the
hand Jacobian against 0.01-radian joint perturbations, solved above/grasp/lift
joint waypoints kinematically, restored the ready pose, then interpolated joint
targets over 24 seconds of simulated physics. Cube dimensions were 0.05 m each;
table top was z=0.255 m; cube center began near (-0.55, 0.10, 0.28) m. Full
gravity was enabled. The hand closed its two fingers against the cube at about
0.025 m joint position each before lifting.

## Measured physical result

The 120 Hz in-process pass condition was cube center at least 0.08 m above its
settled baseline **and** each filtered finger-to-cube normal force at least
0.1 N for 0.5 continuous seconds. Run 008 reported:

| Measurement | Value |
|---|---:|
| Settled cube center z | 0.279892 m |
| Maximum cube center z | 0.458891 m |
| Maximum height gain | 0.1790 m |
| Best sustained height + bilateral contact | 5.075 s |
| Peak left / right finger contact | 7.22 / 8.35 N |
| Maximum horizontal cube displacement | 0.0117 m |
| Final cube center z after 24 simulated seconds | 0.452626 m |

The copied 10 Hz JSONL contains 50 consecutive samples from about 19.0 to
23.9 s with both contacts and at least 0.08 m lift. The 120 Hz summary was
computed inside the script before exit. See ignored local
`outputs/isaac/controlled-lift-008.jsonl` and `.log`. The log also contains
the solved joint waypoints and Jacobian finite-difference check.

Earlier scripted trials failed because a standalone scene did not initialize
the configured Franka ready joint pose, receding joint targets allowed gravity
sag, and online task-space descent swept the cube sideways. Those failed
traces are retained locally as `controlled-lift-001` through `-007` where
copied. They are diagnostics, not independent success-rate trials.

## Corrected task smoke

The external `FRI-Franka-Lift` task was also started with one PPO iteration and
four environments. This checks task discovery, scene/reset construction, full
gravity and training loop integration. It does not show a learned grasp.

```bash
cd /home/ubuntu/FRI-BCI-ML-RL-Decoder
timeout 180s /home/ubuntu/IsaacLab-fri/.venv/bin/python -m fri_bci_sim.cli train \
  --isaaclab-root /home/ubuntu/IsaacLab-fri --num-envs 4 --iterations 1 \
  --run-dir /home/ubuntu/FRI-BCI-ML-RL-Decoder/outputs/isaac/fixed-cube-smoke-001 \
  --execute
```

`exit.json` recorded exit code 0; object-out-of-bound termination was 0 in the
reported iteration and task success was 0. Its resolved `env.yaml` confirms a
single 0.05 m cube at (-0.55, 0.10, 0.28), fixed -9.81 m/s² gravity, z lower
bound 0.24, disabled curriculum, table-contact clearance -0.003, and command
position (.55, -.10, .43) in the rotated Franka base frame. The full smoke
folder, including resolved configs, console, checkpoint and run metadata, is
copied to ignored local `outputs/isaac/fixed-cube-smoke-001/`.

The launcher manifest's `reset_protocol: table_only` label predates the
fixed-cube correction; inspect the resolved `env.yaml` for the actual setup.
No video, multiple-scene validation, policy training beyond the smoke, HOLD
executor, EEG coupling, or paired EEG command comparison was completed.
