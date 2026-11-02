# Start the next project chat here

We are building a simulation-first project: replay recorded EEG, turn it into a
simple **HOLD / GRASP_LIFT** request, and test whether filtering noisy requests
helps a simulated Franka lift a cube more reliably. The contribution is the
measured system comparison, not a new neural network or RL algorithm.

## What the terms mean

- **Decoder:** reads EEG and estimates whether the person is resting or
  imagining movement. The current dataset labels fist/foot movements, not a
  specific grasp.
- **Policy:** the robot's learned movement rule. It sees simulator state and
  chooses joint movements.
- **RL:** the way we train that movement rule through many simulated attempts.
  Failed attempts give feedback; a reward score encourages better task behavior.
- **PPO:** one common RL training algorithm. It updates the policy gradually
  using batches of recent attempts. PPO is the training method, not the project
  goal. We are not committed to PPO if the task setup or evidence suggests
  another approach.
- **Pre-grasp stage:** an easier training setup where the free cube starts
  between the fingers. It teaches closing/lifting, but success here does not
  mean the robot can pick up a cube from the table.

## What is actually verified

- A scripted, non-learned controller lifted a free cube 17.9 cm and kept both
  fingers in contact for 5.075 seconds in one fixed Isaac Lab Newton scene. It
  never teleported or attached the cube. Full run details:
  `simulation/cloud/controlled_lift_2026-11-02.md`.
- The corrected table task initialized on an L4 for a four-environment,
  one-iteration smoke.
- A pre-grasp PPO pilot ran on an L4: 64 environments, seed 8, 100 iterations.
  The training success metric peaked around 11.5%, fell to 9.2% by the last
  reported iteration, and out-of-bounds terminations approached 39%. This is a
  weak training signal, not proof of physical lifting. No checkpoint playback
  or table transfer was run. See `simulation/cloud/pregrasp_pilot_2026-11-02.md`.
- Decoder comparison/evaluation code exists. On a small fresh group of five
  participants, per-person CSP achieved 59.3% balanced accuracy. No EEG command
  has controlled the simulated robot yet.
- Simulation CPU tests passed (17 tests) before the last cloud pilot.

## Current state and next task

Brev's `compact-green-alligator` L4 was stopped after the pilot; check its live
status before using it. Project source and short-run outputs are in the local
repo, under ignored `outputs/isaac/`. Check Git status first. The remote run
folder is `/home/ubuntu/FRI-project-8b9db42` if still retained.

Do not start a long training job yet. First inspect the pre-grasp reset,
out-of-bounds terminations, resolved configs and saved checkpoint. Make a
bounded one-environment policy playback with physical cube height and bilateral
contact logging. Decide from that evidence whether the failure is the policy,
reward/observation, reset geometry, or checkpoint/playback setup. Only then fix
the specific issue and rerun a small pilot. Move to table starts after repeated
measured pre-grasp lifts. Keep pre-grasp and table-pickup scores separate.

Once a learned table lift works, connect the decoder to the same fixed robot
policy and compare raw predictions, confidence filtering and temporal filtering
on matched recordings/scenes. Measure task completion, movement during rest,
missed commands and delay. Hold must preserve the gripper state, especially
while carrying a cube. No hardware integration, generated EEG, second dataset,
or claims of general control are in the current core experiment.

## Practical notes

- GitHub branch: `codex/isaaclab-project-scaffold`, draft PR #3. Latest known
  commit is `2924b0a`; verify before assuming.
- Pinned simulator: Isaac Lab `v3.0.0`, commit
  `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`, Newton backend, Python 3.12.
- Brev cost is separate from ChatGPT usage. The L4 was previously displayed at
  about $1.08 per running hour. Stop the instance when idle.
- Use a lighter model for bounded coding/log inspection; ask a stronger model
  to review scientific design or an ambiguous failure. Don’t spend turns
  repeating the plan: run one small experiment, preserve its outputs, and state
  exactly what the result supports.
