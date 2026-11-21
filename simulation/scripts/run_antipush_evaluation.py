"""Five seeded physical screens of the frozen anti-push PPO diagnostic."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

root = Path(__file__).resolve().parents[2]
train = root / 'outputs/isaac/antipush-50-seed8-20261103T2305Z'
out = root / 'outputs/isaac/antipush-eval-5x12s-20261103T2305Z'
if json.loads((train / 'exit.json').read_text())['exit_code'] != 0:
    raise RuntimeError('training failed')
models = list((train / 'logs').glob('rsl_rl/lift_franka/*/model_*.pt'))
models = [p for p in models if p.name != 'model_249.pt']
if not models:
    raise RuntimeError('no new trained checkpoint')
checkpoint = max(models, key=lambda p:int(p.stem.split('_')[-1]))
out.mkdir(exist_ok=False)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = {'source_checkpoint':str(checkpoint), 'checkpoint_sha256':sha(checkpoint), 'seeds':list(range(8,13)), 'max_policy_steps':360, 'episodes_per_seed':1, 'task':'FRI-Franka-Lift-NearTable-AntiPush'}
(out / 'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for seed in range(8,13):
    run = out / f'seed{seed}'
    run.mkdir()
    cp = run / 'input/model.pt'
    cp.parent.mkdir()
    shutil.copy2(checkpoint, cp)
    assert sha(cp) == manifest['checkpoint_sha256']
    env = os.environ.copy()
    env.update({'PYTHONPATH':str(root / 'simulation/src'), 'PYTHONDONTWRITEBYTECODE':'1', 'FRI_PPO_TELEMETRY':str(run / 'ppo-telemetry.jsonl'), 'FRI_PPO_MAX_STEPS':'360', 'FRI_PPO_EPISODES':'1', 'FRI_PPO_RECORD_FINGERTIP_BODY_POSES':'1'})
    env.pop('FRI_PPO_EARLY_ARM_FILTER',None)
    cmd = ['/home/ubuntu/IsaacLab-fri/.venv/bin/isaaclab','play','--rl_library','rsl_rl','--task',manifest['task'],'--num_envs','1','--seed',str(seed),'--checkpoint',str(cp),'--external_callback=fri_bci_sim.ppo_telemetry.register','physics=newton_mjwarp','--viz','none']
    (run / 'launch.json').write_text(json.dumps({'argv':cmd,'env_overrides':{k:v for k,v in env.items() if k.startswith('FRI_PPO_') or k == 'PYTHONPATH'}},indent=2)+'\n')
    try:
        with (run / 'console.log').open('w') as log:
            res = subprocess.run(cmd,cwd=run,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        code = res.returncode
    except subprocess.TimeoutExpired:
        code = None
    (run / 'exit.json').write_text(json.dumps({'exit_code':code})+'\n')
    print(f'seed {seed}: exit {code}', flush=True)
