# Near-table anti-push PPO diagnostic — 2026-11-03

The frozen [protocol](ANTIPUSH_PROTOCOL_2026-11-03.md) produced **2/5 sustained physical lifts**, one of which later dropped. Only seed 8 had both acquisition and no recorded drop. This is a table-supported near-grasp reset, not pickup from the farther ready pose. The intervention has not solved reliable grasping and is not ready for EEG integration. One training seed and five screens do not establish reliability or generalization.

| Evaluation seed | Duration s | Max rise m | Acquired at s | Drop at s | Max XY shift m | Peak left/right N | End |
|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | 12.000 | 0.2033 | 1.000 | — | 0.409 | 73.0 / 91.8 | timeout |
| 9 | 12.000 | 0.0114 | — | — | 0.526 | 0 / 295.4 | timeout |
| 10 | 12.000 | 0.0126 | — | — | 0.532 | 2.3 / 114.8 | timeout |
| 11 | 1.033 | 0.0136 | — | — | 0.707 | 2.9 / 113.8 | early termination |
| 12 | 12.000 | 0.1912 | 1.450 | 3.017 | 0.591 | 5.9 / 126.4 | timeout |

Peak contact is a filtered sensor magnitude, not a force safety guarantee. XY displacement is the largest sampled displacement at any height, including during lift; it is not exclusively a table-push measure. Seed 11's pre-reset z was 0.2339 m, below the configured 0.24 m lower bound. Seeds 9–11 never satisfied the acquisition predicate; their zero drops mean no *post-acquisition* drop could be recorded, not that they grasped. Seed 12 passed the lift predicate before its drop. High right-side forces and large excursions remain and deserve investigation. Reset pose was fixed; different seeds are not spatial generalization.

All five episodes have pre-action cube height 0.280000 m, pre-reset terminal records, 100% sampled contact coverage, zero reported physics gaps, and no incomplete episodes. The recorder used `environment_physics_step`, 120 Hz (1/120 s); the policy step was 30 Hz (1/30 s), four physics samples per action. Its 30 Hz compact trace is a subsample of the raw physics trace and cannot independently reproduce every threshold crossing. Physics-step sensor coverage does not establish between-sample force continuity. The PPO playback ran continuously enabled: **HOLD invariance was not exercised or established**. The separate scripted HOLD control does not validate this learned policy.

Training used 256 environments, 32 steps per environment and 50 additional iterations, or 409,600 environment transitions. The runner restored actor, critic, iteration and optimizer by default (the pinned `train_rsl_rl.py` uses `resume_optimizer_state=True`); logged iterations were 249–298. The original progress checkpoint SHA256 was `46f68c087111b5f713c33a63b7355f3c4c35eeaba26e4240f778667086b443d6`; the new `model_298.pt` is `e80a46414db8482cf780c56f785154d93ac479b2e2a48ac3d7b1567e3d84729f`. The isolated training copy and each evaluation copy were hash checked. Isaac Lab commit was `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`; host GPU was NVIDIA L4, driver 580.178.04. Resolved configs, launch manifests, console and exit records are in the ignored raw output directories below.

- Training: `outputs/isaac/antipush-50-seed8-20261103T2305Z/`.
- Physical screens and full 120 Hz traces: `outputs/isaac/antipush-eval-5x12s-20261103T2305Z/`.
- [Metrics, full-trace hashes, terminal states](../results/antipush_2026-11-03/metrics.json) and [compact 30 Hz trace](../results/antipush_2026-11-03/trace_30hz.csv).
- Collection: `python analysis/collect_antipush_evaluation.py` (requires a fresh `results/antipush_2026-11-03/` destination; preserve the existing one).

The local checkout at execution had uncommitted pre-existing work and a task registry lacking the independently published FixedOpen registration. The executed source hashes are in training `launch.json`; the published registry must retain FixedOpen in addition to this task. Raw artifacts and checkpoint remain on this host; off-host backup is unverified. Process exit and reward are not evidence of pickup. No farther-start evaluation, pause/resume test, or EEG experiment was run.
