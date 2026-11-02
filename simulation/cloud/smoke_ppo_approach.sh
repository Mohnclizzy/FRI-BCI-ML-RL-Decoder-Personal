#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/home/ubuntu/FRI-project-8b9db42
ISAACLAB_ROOT=/home/ubuntu/IsaacLab-fri
PY="$ISAACLAB_ROOT/.venv/bin/python"
CHECKPOINT="$REPO/outputs/isaac/pregrasp-64x100-seed8/logs/rsl_rl/lift_franka/2026-11-02_00-39-11/model_99.pt"
RUN_DIR="$REPO/outputs/isaac/table-approach-smoke-20261102"

test -x "$PY"
test -f "$CHECKPOINT"
test ! -e "$RUN_DIR"
cd "$REPO"

"$PY" -c 'import gymnasium as gym; import fri_bci_sim.tasks; from fri_bci_sim.tasks.approach import ApproachLiftEnvCfg; print(gym.spec("FRI-Franka-Lift-Approach").kwargs["env_cfg_entry_point"]); print(ApproachLiftEnvCfg.__name__)'

"$PY" -m fri_bci_sim.curriculum \
  --isaaclab-root "$ISAACLAB_ROOT" --stage table-approach --num-envs 4 \
  --seed 8 --iterations 2 --checkpoint "$CHECKPOINT" --run-dir "$RUN_DIR"

"$PY" -m fri_bci_sim.curriculum \
  --isaaclab-root "$ISAACLAB_ROOT" --stage table-approach --num-envs 4 \
  --seed 8 --iterations 2 --checkpoint "$CHECKPOINT" --run-dir "$RUN_DIR" --execute

cat "$RUN_DIR/exit.json"
test "$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["exit_code"])' "$RUN_DIR/exit.json")" = 0
echo "PPO_APPROACH_SMOKE_OK $RUN_DIR"
