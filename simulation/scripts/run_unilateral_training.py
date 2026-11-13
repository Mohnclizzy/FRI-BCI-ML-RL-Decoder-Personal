"""Execute the frozen 50-update anti-push diagnostic in a fresh run directory."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[2]
run = root / 'outputs/isaac/unilateral-50-seed8-20261103T2310Z'
checkpoint = run / 'input/model_298.pt'
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == 'e80a46414db8482cf780c56f785154d93ac479b2e2a48ac3d7b1567e3d84729f'
assert not (run / 'launch.json').exists()
env = os.environ.copy()
env['PYTHONPATH'] = str(root / 'simulation/src')
env['PYTHONDONTWRITEBYTECODE'] = '1'
cmd = ['/home/ubuntu/IsaacLab-fri/.venv/bin/isaaclab', 'train', '--task', 'FRI-Franka-Lift-NearTable-Unilateral', '--rl_library', 'rsl_rl', '--num_envs', '256', '--seed', '8', '--max_iterations', '50', '--checkpoint', str(checkpoint), 'physics=newton_mjwarp', '--viz', 'none']
files = ['simulation/src/fri_bci_sim/tasks/near_table_unilateral.py', 'simulation/src/fri_bci_sim/tasks/near_table_antipush.py', 'simulation/src/fri_bci_sim/tasks/near_table_progress.py', 'simulation/src/fri_bci_sim/tasks/near_table.py', 'simulation/src/fri_bci_sim/tasks/__init__.py', 'simulation/src/fri_bci_sim/ppo_telemetry.py', 'docs/UNILATERAL_PROTOCOL_2026-11-03.md', 'simulation/scripts/run_unilateral_training.py']
(run / 'launch.json').write_text(json.dumps({'argv':cmd, 'cwd':str(run), 'source_hashes':{p:hashlib.sha256((root / p).read_bytes()).hexdigest() for p in files}, 'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(), 'PYTHONPATH':env['PYTHONPATH'], 'isaaclab_head':subprocess.check_output(['git','-C','/home/ubuntu/IsaacLab-fri','rev-parse','HEAD'],text=True).strip()}, indent=2)+'\n')
with (run / 'console.log').open('w') as out:
    result = subprocess.run(cmd, cwd=run, env=env, stdout=out, stderr=subprocess.STDOUT, timeout=1200)
(run / 'exit.json').write_text(json.dumps({'exit_code':result.returncode})+'\n')
raise SystemExit(result.returncode)
