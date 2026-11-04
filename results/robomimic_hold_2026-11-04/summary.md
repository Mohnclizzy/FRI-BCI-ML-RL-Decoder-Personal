# Phase 3 native BC-RNN acquisition HOLD diagnostic — 2026-11-04

## Protocol

The unmodified official Lift PH low-dimensional BC-RNN checkpoint (SHA256 `3ee222cab41f78ba27afca7ef6f70f9f21bd3cc0a37bd79c0b2351dbede51605`) controlled the same five aligned seeds as Phase 2. The original checkpoint, simulator physics, 20 Hz control, 500 Hz physics, and 400-step (20-second) deadline were unchanged. At the first sampled state with bilateral fingerpad/cube contact and cube rise at least 0.08 m above that episode's initial cube height, the controller captured the current end-effector position and orientation. From the next action onward, HOLD set that same absolute pose goal on every control step and sent action `[0,0,0,0,0,0,+1]` (+1 closes the Panda gripper). This is one minimal controller intervention; there was no training or checkpoint edit. The always-HOLD runs used seeds 0–4. A separate seed-0 probe held through step 200 and resumed the BC-RNN at step 201. The policy continued to receive observations only while active; its LSTM was not reset on resume.

Telemetry records all 401 sampled states per run, cube position, direct contact pairs, robosuite grasp check, action, end-effector pose, controller goal and acquisition target. The contact proxy matched the built-in grasp check at every sampled state. Retention requires the acquired cube to remain at least 0.08 m above its initial height, with bilateral contact at the terminal state and no sampled post-acquisition drop. Native Lift success (cube center above table +0.04 m) remains a separate measure. Contact is sampled at 20 Hz, so interruptions within a 0.05-second control interval are unobserved.

## Trace diagnosis and compatibility

The Phase 2 policy acquired at steps 47–58. First bilateral-contact loss occurred at steps 105, 362, 145, 195, and 160 for seeds 0–4, respectively; crossing back below 0.08 m occurred at steps 111, 364, 162, 213, and 168. Before loss, several trajectories carried the cube far from its starting x/y and above 0.48–1.14 m of rise. The policy sometimes issued gripper-open (`-1`) actions while carrying. Contact loss preceded the height-threshold drop in every trace. These observations motivated a fixed pose and close-gripper HOLD at acquisition; they do not isolate a single causal failure mechanism.

The pinned historical robosuite source lists conflicting `mujoco-py` requirements: `2.0.2.9` in `setup.py` and `2.0.2.13` in `requirements.txt`. The runnable environment uses `mujoco-py 2.1.2.14` with MuJoCo 2.1.0; no MuJoCo 2.0 installation was present locally. Exact historical dependency parity was therefore not physically tested. This limits attribution of the original retention failure to the checkpoint alone.

## Physical results

| Run | Acquisition step | HOLD steps | Minimum rise after acquisition (m) | Sampled contact losses after acquisition | Terminal rise (m) | Retained at 400 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Always HOLD seed 0 | 58 | 342 | 0.0845 | 0 | 0.0845 | Yes |
| Always HOLD seed 1 | 54 | 346 | 0.0853 | 0 | 0.0853 | Yes |
| Always HOLD seed 2 | 47 | 353 | 0.0842 | 0 | 0.0842 | Yes |
| Always HOLD seed 3 | 57 | 343 | 0.0838 | 0 | 0.0838 | Yes |
| Always HOLD seed 4 | 47 | 353 | 0.0887 | 0 | 0.0887 | Yes |
| HOLD then resume seed 0 | 58 | 142 | 0.0869 | 0 | 1.0980 | Yes |

Always-HOLD terminal retention was **5/5**, compared with **0/5** under the uninterrupted Phase 2 policy. The single resume probe retained **1/1**; it is exploratory and does not establish resume reliability. All six ran the complete 400 actions. During HOLD, the recorded position and orientation *goals* exactly matched the fixed acquisition target (maximum recorded goal error 0); observed end-effector position deviated from that target by at most 0.0106 m. The terminal margin over the 0.08 m rise threshold was narrow in the always-HOLD runs (0.0038–0.0087 m). This is a measured controller equilibrium, not a learned HOLD skill or EEG qualification.

## Status and artifacts

Bounded Phase 3 diagnostic complete. The fixed acquisition pose with continued close command prevents sampled post-acquisition loss in these five aligned starts. Next research work should test a larger held-out seed set and perturbations before treating retention or resume as reliable; no long training was started here. Raw telemetry, per-run summaries, execution logs, and the exact diagnostic script are under `/home/ubuntu/robomimic-native-phase3-20261104/run/`. Existing Phase 2 and Isaac artifacts were not modified.

The new raw archive is SHA256 `76b320138491df379f7ed2047711013ac2a1d447842e03f2e91edabf7c9ffdc4` (341,724 bytes). It was uploaded to the private [Phase 3 Drive folder](https://drive.google.com/drive/folders/1BjC9bhuEL52lAWsZ6D72gpQI-jInziNh) as [phase3_raw_2026-11-04.tar.gz](https://drive.google.com/file/d/1MCtylPtCqN402ZB6oxTOLe9p_dlg5zUf/view). Connector readback confirmed size, folder parent, and owner-only permission metadata; downloaded-byte SHA256 verification remains outstanding.
