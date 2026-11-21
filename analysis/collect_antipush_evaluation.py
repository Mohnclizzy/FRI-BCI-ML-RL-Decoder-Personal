"""Collect auditable physical outcomes and compact 30 Hz trace from five screens."""
import csv
import hashlib
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parents[1]
run = root / 'outputs/isaac/antipush-eval-5x12s-20261103T2305Z'
out = root / 'results/antipush_2026-11-03'
out.mkdir(exist_ok=False)
manifest = json.loads((run / 'manifest.json').read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
results = []
with (out / 'trace_30hz.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['seed','time_s','cube_x','cube_y','cube_z','left_contact_n','right_contact_n','hand_x','hand_y','hand_z'])
    for seed in manifest['seeds']:
        d = run / f'seed{seed}'
        summ = json.loads((d / 'ppo-telemetry.summary.json').read_text())
        rows = [json.loads(line) for line in (d / 'ppo-telemetry.jsonl').open()]
        start = [x for x in rows if x['record_type'] == 'episode_start']
        end = [x for x in rows if x['record_type'] in ('terminal_pre_reset','episode_incomplete_end')]
        samples = [x for x in rows if x['record_type'] == 'physics_step']
        assert len(start) == len(end) == 1 and len(summ['episode_results']) == 1
        ep = summ['episode_results'][0]
        assert len(samples) == ep['physics_samples'] and all(x['contact_valid'] for x in samples)
        assert ep['physics_step_gaps'] == ep['contact_missing_samples'] == 0
        assert sha(d / 'input/model.pt') == manifest['checkpoint_sha256']
        assert all(abs((b['episode_time_s']-a['episode_time_s'])-summ['physics_dt_s']) < 2e-6 for a,b in zip(samples,samples[1:]))
        for i, row in enumerate(samples):
            if i % 4 == 3 or i == len(samples)-1:
                writer.writerow([seed,row['episode_time_s'],*row['cube_pose_w'][:3],*row['finger_contact_n'],*row['hand_pose_w'][:3]])
        xy0 = start[0]['cube_pose_w'][:2]
        maxxy = max(math.dist(xy0, x['cube_pose_w'][:2]) for x in samples)
        results.append({'seed':seed, **ep, 'start_cube_pose_w':start[0]['cube_pose_w'], 'end_record_type':end[0]['record_type'], 'end_cube_pose_w':end[0]['cube_pose_w'], 'max_cube_xy_displacement_m':maxxy, 'sampling_mode':summ['physics_sampling_mode'], 'physics_dt_s':summ['physics_dt_s'], 'policy_dt_s':summ['policy_dt_s'], 'exit':json.loads((d/'exit.json').read_text()), 'telemetry_sha256':sha(d/'ppo-telemetry.jsonl'), 'summary_sha256':sha(d/'ppo-telemetry.summary.json')})
(out / 'metrics.json').write_text(json.dumps({'manifest':manifest,'episodes':results,'trace_note':'Every fourth recorded physics sample plus final sample; full 120 Hz raw traces retained in ignored outputs.', 'trace_sha256':sha(out/'trace_30hz.csv')},indent=2)+'\n')
print(json.dumps(results,indent=2))
