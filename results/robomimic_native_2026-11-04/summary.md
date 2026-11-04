# Native robomimic Lift PH BC-RNN diagnostic — 2026-11-04

## Provenance and runtime

Official [model-zoo entry](https://robomimic.github.io/docs/model_zoo/robomimic_v0.1.html): Lift PH low-dimensional BC-RNN, linked `lift_ph_low_dim_epoch_1000_succ_100.pth`. The checkpoint was downloaded from that exact link. SHA256 `3ee222cab41f78ba27afca7ef6f70f9f21bd3cc0a37bd79c0b2351dbede51605`. This is the authors' pretrained model; this project did not train it. The official page reports approximately 100% native success, with an explicit approximation caveat.

Fresh isolated work area: `/home/ubuntu/robomimic-native-phase2-20261104`. robomimic `v0.1` at `2804dd97dd1625ec861298a35cb677129d3bfacc`; robosuite `offline_study` at `d9baa17d5328d7b0ec15ecda54c76c61bb983124`. Python 3.8.20, torch 1.13.1+cpu, MuJoCo 2.1.0, mujoco-py 2.1.2.14, CPU OSMesa video rendering. **Compatibility deviation:** the old robosuite `setup.py` specifies mujoco-py 2.0.2.9 and `requirements.txt` specifies 2.0.2.13; both differ from the installed 2.1.2.14. The pinned source branches and checkpoint are exact, but this is not a byte-identical historical dependency environment. The branch source and local pip freeze are in the raw archive. Isaac outputs and source were untouched.

Checkpoint metadata: `Lift`, Panda, `OSC_POSE`, 7 actions, 20 Hz control, pose deltas capped at ±0.05 m and ±0.5 rad, low-dimensional observations (`robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos`, `object`), no observation normalization, 400-step native horizon, and a 10-step LSTM history. Native robosuite success is cube center above table top +0.04 m. We did not change that criterion.

## Measurement and parity

The recorder ran all 400 actions (20 seconds), sampled cube position, reward, action, terminal state, direct MuJoCo contact pairs, and robosuite's built-in `_check_grasp` at every 20 Hz control step. Physics timestep is 0.002 s (500 Hz); contact telemetry is **20 Hz**, so substep contact interruptions are unobserved. Bilateral contact means cube geom `cube_g0` touches both `gripper0_finger1_pad_collision` and `gripper0_finger2_pad_collision`. This direct-contact proxy exactly matched `_check_grasp` at all 401 sampled states in every completed trace. A 0.5-second contact run means at least 10 consecutive sampled steps; it is a sampled estimate, not a 500 Hz guarantee.

The unmodified official `run_trained_agent.py` for seed 0 stopped on native success after 50 actions (Return 1.0). The aligned instrumented seed 0 used the official seed placement after environment creation, produced the **same first 50 action vectors exactly** as the official HDF5 trajectory, and first crossed native success at step 50. Its saved 80-frame, 4 Hz video shows pickup followed by loss before the terminal state. The recorder's first attempt (`seed0`) completed video but failed to write telemetry because its cleanup called a nonexistent wrapper `close`; the video is preserved. The subsequent `seed0b` and seeds 1–4 were exploratory runs with pre-construction seeding, before the parity correction. All attempts are in the archive.

## Aligned five-start diagnostic

Each start used the same saved policy and native 400-step horizon, with seed set after environment construction. Cube center began at z≈0.830–0.832 m on the 0.8 m table. The project diagnostic requires ≥0.08 m rise from that episode's baseline, bilateral sampled contact for ≥0.5 s, terminal retention, and no post-acquisition drop. Native success and this stronger criterion are separate.

| Seed | Cube start x,y (m) | First native success step | Max rise (m) | Longest bilateral contact (steps) | Terminal cube z (m) | Terminal retained |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.026, -0.026 | 50 | 0.480 | 63 | 0.821 | No |
| 1 | 0.023, -0.028 | 42 | 1.137 | 327 | 0.021 | No |
| 2 | 0.002, -0.022 | 41 | 0.773 | 107 | 0.821 | No |
| 3 | 0.005, -0.029 | 49 | 0.885 | 150 | 0.821 | No |
| 4 | 0.027, -0.004 | 40 | 0.876 | 122 | 0.822 | No |

Native success at any step: **5/5**. Sampled sustained bilateral contact: **5/5**. At least 8 cm rise: **5/5**. Post-acquisition drop: **5/5**. Terminal retention: **0/5**. This is a bounded diagnostic, not a reliability estimate or a completed learned pickup. The official playback's early termination explains why its success result does not establish retention over the full horizon. No result here is scored from reward or process exit.

## Artifacts and next decision

Local raw traces, video, scripts, official playback HDF5, package freeze, and all attempts are in `/home/ubuntu/robomimic-native-phase2-20261104/run/`; compact per-run metrics are in `analysis.json`. Raw archive SHA256 `bbf92cc1d59d2bfe6c19d4c22235d67326e3c0cd84707f68b7f28c6e3cd04f5f`; seed 0 aligned video SHA256 `b600bcb0e16ec25e937872e750b55ac492e1bb99e82f948a1794f3d27e7bc72e`; aligned seed 0 telemetry SHA256 `6a88e535401b5d0cf79d2baaa97bd379eb34546946f91b68435760984373f704`.

Private Drive [Phase 2 folder](https://drive.google.com/drive/folders/1mCICv4tEWmoD1x2AGUZ7_FEiz55A-OrA) holds the checkpoint ([file](https://drive.google.com/file/d/1rjOVjoJemFwSUlE6OBh9Ygzzm4rKZF6G/view)) and all-attempt archive ([file](https://drive.google.com/file/d/1c9rSGemg5I73P2dRjuQ_48uG8DxZkA_8/view)). Connector readback confirmed both file sizes and private owner-only metadata; byte-for-byte download verification remains outstanding.

Next: inspect the drop timing/action/contact traces and check MuJoCo dependency compatibility before a targeted retention intervention. Under the handoff fallback, a native actor that briefly lifts but fails retention needs a measured adaptation target. Do not count this policy as the final learned skill or proceed to EEG/HOLD qualification on the basis of native reward.
