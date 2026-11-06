"""One exact-condition seed-11 reproduction of the unilateral policy."""
import hashlib,json,os,shutil,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2]
src=root/'outputs/isaac/unilateral-50-seed8-20261103T2310Z/logs/rsl_rl/lift_franka/2026-11-03_23-11-39/model_347.pt'
out=root/'outputs/isaac/retention-diagnostic-seed11-20261103T2333Z'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
expected='37eb85242539f7d2f0c36a916f3aa8e44d2fe6a70030a9da18963d2506e673f7'
assert sha(src)==expected
out.mkdir(exist_ok=False); cp=out/'input/model.pt'; cp.parent.mkdir();shutil.copy2(src,cp);assert sha(cp)==expected
env=os.environ.copy();env.update({'PYTHONPATH':str(root/'simulation/src'),'PYTHONDONTWRITEBYTECODE':'1','FRI_PPO_TELEMETRY':str(out/'ppo-telemetry.jsonl'),'FRI_PPO_MAX_STEPS':'360','FRI_PPO_EPISODES':'1','FRI_PPO_RECORD_FINGERTIP_BODY_POSES':'1'});env.pop('FRI_PPO_EARLY_ARM_FILTER',None)
cmd=['/home/ubuntu/IsaacLab-fri/.venv/bin/isaaclab','play','--rl_library','rsl_rl','--task','FRI-Franka-Lift-NearTable-Unilateral','--num_envs','1','--seed','11','--checkpoint',str(cp),'--external_callback=fri_bci_sim.ppo_telemetry.register','physics=newton_mjwarp','--viz','none']
files=['docs/RETENTION_DIAGNOSTIC_PROTOCOL_2026-11-03.md','simulation/scripts/run_retention_diagnostic.py','simulation/src/fri_bci_sim/ppo_telemetry.py','simulation/src/fri_bci_sim/tasks/__init__.py','simulation/src/fri_bci_sim/tasks/near_table_unilateral.py','simulation/src/fri_bci_sim/tasks/near_table_antipush.py']
(out/'launch.json').write_text(json.dumps({'argv':cmd,'checkpoint_sha256':sha(cp),'source_sha256':{p:sha(root/p) for p in files},'env_overrides':{k:v for k,v in env.items() if k.startswith('FRI_PPO_') or k=='PYTHONPATH'}},indent=2)+'\n')
try:
 with (out/'console.log').open('w') as log: res=subprocess.run(cmd,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
 code=res.returncode
except subprocess.TimeoutExpired: code=None
(out/'exit.json').write_text(json.dumps({'exit_code':code})+'\n')
if code!=0: raise SystemExit(1)
