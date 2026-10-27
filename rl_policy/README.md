# RL policy and proxy gripper

This directory is Mario's first measurable robot-learning milestone. A PyBullet
proxy gripper receives `REST` or `HAND_IMAGERY`. The mapping is **REST → hold**
and **HAND_IMAGERY → attempt a cube grasp and lift**. The latter is an experiment
mapping: PhysioNet did not record a specific grasp-imagery label.

The policy sees the decoded command and robot state, never the true EEG label.
The label is used only to score task completion and to form the training reward.
Episodes are balanced between hold and grasp requests. An eight-step episode
holds one command fixed; each EEG test prediction yields one separate episode.
The simulation uses a kinematic parallel gripper and a fixed constraint when
its fingers enclose the cube. It is a *proxy*, not a realistic contact model or
the proposed Isaac Lab / Tesollo simulation.

## Run

Install Python with PyTorch, then:

```powershell
python -m pip install -r requirements-rl.txt
python -m rl_policy.train --steps 20000 --seed 0 --output outputs/rl_policy/ppo_clean.zip
python -m rl_policy.evaluate --model outputs/rl_policy/ppo_clean.zip --predictions outputs/baseline/test_predictions.json
python -m rl_policy.demo --model outputs/rl_policy/ppo_clean.zip
```

The evaluation command requires Angel's exported decoder predictions. It saves
`outputs/rl_policy/evaluation.json` with per-trial results and four comparisons:
the learned policy with decoded commands, the same policy with correct commands,
an always-hold controller, and an always-grasp controller. Check **grasp success**
and **rest compliance** separately; a policy that grasps every time can get
grasp success while failing all rest trials. The balanced score averages the
two success rates.

The demo command saves an animated `outputs/rl_policy/demo.gif` showing the
learned policy obeying correct hold and grasp commands in the proxy gripper.

## First local run

One PPO run with seed 0 and 20,000 steps completed on CPU. Evaluation used the
existing CSP+LDA export for 150 windows from subjects 1–5, held-out run 12
(75 rest and 75 hand imagery). No model selection used these test predictions.

| Controller / command source | Rest compliance | Grasp success | Balanced success |
|---|---:|---:|---:|
| Learned PPO, decoded command | 57.3% | 41.3% | 49.3% |
| Same PPO, correct command | 100% | 100% | 100% |
| Always hold | 100% | 0% | 50% |
| Always grasp | 0% | 100% | 50% |

The clean-command result checks that this small robot task and policy work.
The decoded-command result mirrors the decoder's current near-chance accuracy:
the robot cannot infer true intent from a wrong, fixed command when the scene
contains no other intent information. This is a **single-seed prototype result**,
not evidence that training with command errors improves performance.

To explore policy training with synthetic command errors, set `--error-rest`
and `--error-imagery` between zero and one, and use a different `--output` path.
These rates are explicit experimental parameters. Do **not** derive them from
the held-out `test_predictions.json`; doing so would contaminate evaluation.
Real decoder-based error rates require predictions from training/validation
recordings, which the current EEG pipeline does not export. A clean and noisy
policy comparison should use multiple seeds, matched scenes, and untouched
held-out predictions before drawing conclusions.

## What this establishes

This code makes the command-to-policy interface, a trainable RL task, and
task-level metrics concrete. It does not establish that EEG can reliably steer
the robot. The current decoder's balanced accuracy is around chance, and this
prototype's one prediction per episode cannot measure continuous command delay,
switches, error bursts, or temporal filtering. The next simulator milestone is
porting this command and evaluation interface to an Isaac Lab gripper or hand;
the next data milestone is validation-only decoder predictions to characterize
command errors without looking at test outcomes.
